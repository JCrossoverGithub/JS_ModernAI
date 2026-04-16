import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Copy,
  Check,
  Bookmark,
  BookmarkCheck,
  FileText,
  Globe,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Download,
  Zap,
  BookOpen,
  MessageSquare,
  ArrowDown,
  GraduationCap,
} from "lucide-react";
import { useSignalR } from "../hooks/useSignalR";
import MarkdownRenderer from "./MarkdownRenderer";
import PaperResults from "./PaperResults";
import WelcomeScreen from "./WelcomeScreen";
import type { Message, Conversation, SearchMode, Paper } from "../types";

/**
 * Replace numbered citations like [1], [2] in the text with markdown links
 * pointing to the paper's download URL (or view URL as fallback).
 */
function linkifyCitations(content: string, papers: Paper[]): string {
  return content.replace(/\[(\d+)\]/g, (match, numStr) => {
    const idx = parseInt(numStr, 10) - 1;
    if (idx < 0 || idx >= papers.length) return match;
    const paper = papers[idx];
    const url = paper.downloadUrl || paper.url;
    if (!url) return match;
    return `[\\[${numStr}\\]](${url})`;
  });
}

interface ChatWindowProps {
  conversation: Conversation | null;
  mode: SearchMode;
  researchSources: string[];
  conversations: Conversation[];
  onAddMessage: (convId: string, msg: Message) => void;
  onUpdateLastMessage: (
    convId: string,
    updater: (m: Message) => Message
  ) => void;
  onToggleBookmark: (convId: string, msgId: string) => void;
  onExport: (id: string) => void;
  onNewResearch: (mode?: string) => void;
  onSelectConversation: (id: string) => void;
  folderContext: string;
}

const MODE_LABELS: Record<string, string> = {
  default: "Default",
  strict: "Strict",
  chat: "Chat",
  web: "Web",
  research: "Research",
};

const MODE_ICONS: Record<string, typeof Zap> = {
  default: Zap,
  strict: BookOpen,
  chat: MessageSquare,
  web: Globe,
  research: GraduationCap,
};

const MODE_BADGE_COLORS: Record<string, string> = {
  default: "bg-indigo-500/15 text-indigo-300 border-indigo-500/20",
  strict: "bg-amber-500/15 text-amber-300 border-amber-500/20",
  chat: "bg-violet-500/15 text-violet-300 border-violet-500/20",
  web: "bg-emerald-500/15 text-emerald-300 border-emerald-500/20",
  research: "bg-cyan-500/15 text-cyan-300 border-cyan-500/20",
};

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 px-5 py-3 animate-fade-in">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500/20 to-violet-500/20 flex items-center justify-center">
        <Sparkles size={13} className="text-indigo-400" />
      </div>
      <div className="flex items-center gap-1.5 px-4 py-2.5 bg-white/[0.04] rounded-2xl">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  );
}

function SourceBadge({ source }: { source: string }) {
  const isWeb =
    source.toLowerCase() === "duckduckgo" || source.startsWith("http");
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-white/[0.04] border border-white/[0.06] rounded-lg text-xs text-slate-400 hover:bg-white/[0.07] transition-colors">
      {isWeb ? <Globe size={11} /> : <FileText size={11} />}
      <span className="truncate max-w-[140px]">{source}</span>
    </span>
  );
}

function CopyMessageButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="p-1.5 rounded-md text-slate-500 hover:text-slate-300 hover:bg-white/[0.06] transition-colors"
      title="Copy message"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

export default function ChatWindow({
  conversation,
  mode,
  researchSources,
  conversations,
  onAddMessage,
  onUpdateLastMessage,
  onToggleBookmark,
  onExport,
  onNewResearch,
  onSelectConversation,
  folderContext,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamingMsgIdRef = useRef<string | null>(null);
  const pendingPapersRef = useRef<Paper[] | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    if (!showScrollBtn) scrollToBottom();
  }, [conversation?.messages?.length, scrollToBottom, showScrollBtn]);

  // Track scroll position
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const handler = () => {
      const { scrollTop, scrollHeight, clientHeight } = el;
      setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100);
    };
    el.addEventListener("scroll", handler, { passive: true });
    return () => el.removeEventListener("scroll", handler);
  }, []);

  // Focus input when conversation changes
  useEffect(() => {
    inputRef.current?.focus();
  }, [conversation?.id]);

  const { isConnected, send } = useSignalR({
    onToken: (content) => {
      if (!conversation) return;
      const id = streamingMsgIdRef.current;
      if (id) {
        onUpdateLastMessage(conversation.id, (m) =>
          m.id === id ? { ...m, content: m.content + content } : m
        );
      } else {
        const newId = crypto.randomUUID();
        streamingMsgIdRef.current = newId;
        onAddMessage(conversation.id, {
          id: newId,
          role: "assistant",
          content,
          timestamp: Date.now(),
          papers: pendingPapersRef.current || undefined,
        });
      }
    },
    onSources: (sources) => {
      if (!conversation) return;
      onUpdateLastMessage(conversation.id, (m) => ({ ...m, sources }));
    },
    onPapers: (papers) => {
      if (!conversation) return;
      // Store papers for when the assistant message is created
      pendingPapersRef.current = papers;
      // If assistant message already exists, attach papers to it
      const id = streamingMsgIdRef.current;
      if (id) {
        onUpdateLastMessage(conversation.id, (m) =>
          m.id === id ? { ...m, papers } : m
        );
      }
    },
    onStatus: (message) => {
      if (!conversation) return;
      onAddMessage(conversation.id, {
        id: crypto.randomUUID(),
        role: "status",
        content: message,
        timestamp: Date.now(),
      });
    },
    onDone: () => {
      setIsStreaming(false);
      streamingMsgIdRef.current = null;
      pendingPapersRef.current = null;
    },
  });

  const sendMessage = useCallback(async () => {
    if (!input.trim() || !conversation || isStreaming) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
      mode,
      timestamp: Date.now(),
    };

    onAddMessage(conversation.id, userMsg);
    setInput("");
    setIsStreaming(true);
    streamingMsgIdRef.current = null;

    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }

    try {
      await send(
        userMsg.content,
        mode,
        mode === "research" ? researchSources : undefined,
        folderContext || undefined
      );
    } catch (err: any) {
      onAddMessage(conversation.id, {
        id: crypto.randomUUID(),
        role: "status",
        content: `Error: ${err.message}`,
        timestamp: Date.now(),
      });
      setIsStreaming(false);
    }
  }, [input, conversation, isStreaming, mode, researchSources, folderContext, send, onAddMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ─── Welcome screen when no conversation is selected ───
  if (!conversation) {
    return (
      <div className="flex-1 flex flex-col bg-[#0a0b12]">
        <WelcomeScreen
          onNewResearch={onNewResearch}
          recentConversations={conversations}
          onSelectConversation={onSelectConversation}
        />
      </div>
    );
  }

  const messages = conversation.messages;
  const ModeIcon = MODE_ICONS[mode] || Zap;

  return (
    <div className="flex-1 flex flex-col bg-[#0a0b12] h-full relative">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-white/[0.06] bg-[#0a0b12]/80 backdrop-blur-md">
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-medium text-white truncate">
            {conversation.title}
          </h2>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-slate-500">
              {messages.filter((m) => m.role !== "status").length} messages
            </span>
            {!isConnected && (
              <span className="text-[10px] text-amber-500">• Reconnecting…</span>
            )}
          </div>
        </div>
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium border ${MODE_BADGE_COLORS[mode]}`}
        >
          <ModeIcon size={11} />
          {MODE_LABELS[mode]}
        </div>
        <button
          onClick={() => onExport(conversation.id)}
          className="p-2 rounded-lg text-slate-500 hover:text-white hover:bg-white/[0.06] transition-colors"
          title="Export as Markdown"
        >
          <Download size={15} />
        </button>
      </div>

      {/* Messages */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto"
      >
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-1">
          {messages.length === 0 && (
            <div className="flex items-center justify-center py-20">
              <div className="text-center text-slate-500">
                <ModeIcon size={24} className="mx-auto mb-3 opacity-30" />
                <p className="text-sm">
                  Start your research in{" "}
                  <span className="text-slate-300">{MODE_LABELS[mode]}</span>{" "}
                  mode
                </p>
              </div>
            </div>
          )}

          {messages.map((msg, idx) => {
            if (msg.role === "status") {
              return (
                <div
                  key={msg.id}
                  className="flex items-center gap-2 py-2 animate-fade-in"
                >
                  <div className="h-px flex-1 bg-white/[0.04]" />
                  <span className="text-[10px] text-slate-500 italic px-2">
                    {msg.content}
                  </span>
                  <div className="h-px flex-1 bg-white/[0.04]" />
                </div>
              );
            }

            const isUser = msg.role === "user";

            return (
              <div
                key={msg.id}
                className={`group py-4 ${isUser ? "" : ""} animate-fade-in`}
              >
                {/* Avatar + role */}
                <div className="flex items-center gap-2.5 mb-2">
                  {isUser ? (
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-[10px] font-semibold text-white">
                      Y
                    </div>
                  ) : (
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500/30 to-violet-500/30 flex items-center justify-center">
                      <Sparkles size={11} className="text-indigo-400" />
                    </div>
                  )}
                  <span className="text-xs font-medium text-slate-400">
                    {isUser ? "You" : "Assistant"}
                  </span>
                  {msg.mode && isUser && (
                    <span className="text-[10px] text-slate-600">
                      · {MODE_LABELS[msg.mode] || msg.mode}
                    </span>
                  )}
                  <span className="text-[10px] text-slate-600 ml-auto">
                    {new Date(msg.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>

                {/* Content */}
                <div className="pl-8">
                  {isUser ? (
                    <p className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                      {msg.content}
                    </p>
                  ) : (
                    <div className="text-sm text-slate-200 leading-relaxed">
                      <MarkdownRenderer
                        content={
                          msg.papers && msg.papers.length > 0
                            ? linkifyCitations(msg.content, msg.papers)
                            : msg.content
                        }
                      />
                    </div>
                  )}

                  {/* Papers (research mode) */}
                  {msg.papers && msg.papers.length > 0 && (
                    <PaperResults papers={msg.papers} />
                  )}

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && !msg.papers?.length && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {msg.sources.map((s, i) => (
                        <SourceBadge key={i} source={s} />
                      ))}
                    </div>
                  )}

                  {/* Actions */}
                  {!isUser && (
                    <div className="flex items-center gap-1 mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      <CopyMessageButton text={msg.content} />
                      <button
                        onClick={() =>
                          onToggleBookmark(conversation.id, msg.id)
                        }
                        className={`p-1.5 rounded-md transition-colors ${
                          msg.bookmarked
                            ? "text-amber-400 bg-amber-500/10"
                            : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.06]"
                        }`}
                        title={
                          msg.bookmarked
                            ? "Remove bookmark"
                            : "Bookmark this response"
                        }
                      >
                        {msg.bookmarked ? (
                          <BookmarkCheck size={13} />
                        ) : (
                          <Bookmark size={13} />
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {isStreaming && !streamingMsgIdRef.current && <TypingIndicator />}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Scroll to bottom */}
      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-28 right-8 p-2 bg-slate-800 border border-white/[0.08] rounded-full shadow-lg hover:bg-slate-700 transition-colors z-10"
        >
          <ArrowDown size={16} className="text-slate-300" />
        </button>
      )}

      {/* Input */}
      <div className="border-t border-white/[0.06] bg-[#0a0b12]">
        <div className="max-w-3xl mx-auto px-6 py-4">
          <div className="relative bg-white/[0.04] border border-white/[0.08] rounded-2xl focus-within:border-indigo-500/40 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a research question…"
              rows={1}
              disabled={isStreaming}
              className="w-full px-5 pt-4 pb-12 bg-transparent text-sm text-white placeholder-slate-500 outline-none resize-none disabled:opacity-50"
              style={{ minHeight: "56px", maxHeight: "200px" }}
              onInput={(e) => {
                const t = e.currentTarget;
                t.style.height = "auto";
                t.style.height = Math.min(t.scrollHeight, 200) + "px";
              }}
            />
            {/* Bottom toolbar */}
            <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-2.5">
              <div className="flex items-center gap-2">
                <div
                  className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium ${MODE_BADGE_COLORS[mode]}`}
                >
                  <ModeIcon size={10} />
                  {MODE_LABELS[mode]}
                </div>
              </div>
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isStreaming}
                className="flex items-center justify-center w-8 h-8 bg-indigo-500 hover:bg-indigo-600 disabled:bg-white/[0.06] disabled:text-slate-500 text-white rounded-xl transition-all disabled:cursor-not-allowed active:scale-95"
              >
                <Send size={14} />
              </button>
            </div>
          </div>
          <p className="text-[10px] text-slate-600 text-center mt-2">
            Press Enter to send · Shift+Enter for new line · Ctrl+K for commands
          </p>
        </div>
      </div>
    </div>
  );
}
