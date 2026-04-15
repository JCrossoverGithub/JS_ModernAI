import { useState, useEffect, useRef, useMemo } from "react";
import {
  Search,
  X,
  MessageSquare,
  ArrowRight,
  Globe,
  BookOpen,
  Zap,
  FileText,
  Download,
  Trash2,
  GraduationCap,
} from "lucide-react";
import type { Conversation } from "../types";
import { SEARCH_MODES } from "../types";

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  conversations: Conversation[];
  onSelectConversation: (id: string) => void;
  onNewResearch: (mode?: string) => void;
  onExport: (id: string) => void;
  onDelete: (id: string) => void;
  onSetMode: (mode: string) => void;
}

interface PaletteItem {
  id: string;
  icon: React.ReactNode;
  label: string;
  description?: string;
  section: string;
  action: () => void;
}

const MODE_ICONS: Record<string, React.ReactNode> = {
  default: <Zap size={14} />,
  strict: <BookOpen size={14} />,
  chat: <MessageSquare size={14} />,
  web: <Globe size={14} />,
  research: <GraduationCap size={14} />,
};

export default function CommandPalette({
  open,
  onClose,
  conversations,
  onSelectConversation,
  onNewResearch,
  onExport,
  onDelete,
  onSetMode,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const items = useMemo<PaletteItem[]>(() => {
    const all: PaletteItem[] = [];

    // Actions
    SEARCH_MODES.forEach((m) => {
      all.push({
        id: `mode-${m.value}`,
        icon: MODE_ICONS[m.value],
        label: `New ${m.label} Research`,
        description: m.description,
        section: "Actions",
        action: () => {
          onNewResearch(m.value);
          onClose();
        },
      });
    });

    all.push({
      id: "set-mode-default",
      icon: <Zap size={14} />,
      label: "Switch to Default mode",
      section: "Modes",
      action: () => { onSetMode("default"); onClose(); },
    });
    all.push({
      id: "set-mode-strict",
      icon: <BookOpen size={14} />,
      label: "Switch to Strict mode",
      section: "Modes",
      action: () => { onSetMode("strict"); onClose(); },
    });
    all.push({
      id: "set-mode-web",
      icon: <Globe size={14} />,
      label: "Switch to Web mode",
      section: "Modes",
      action: () => { onSetMode("web"); onClose(); },
    });
    all.push({
      id: "set-mode-chat",
      icon: <MessageSquare size={14} />,
      label: "Switch to Chat mode",
      section: "Modes",
      action: () => { onSetMode("chat"); onClose(); },
    });
    all.push({
      id: "set-mode-research",
      icon: <GraduationCap size={14} />,
      label: "Switch to Research mode",
      section: "Modes",
      action: () => { onSetMode("research"); onClose(); },
    });

    // Recent conversations
    conversations.slice(0, 8).forEach((c) => {
      all.push({
        id: `conv-${c.id}`,
        icon: <MessageSquare size={14} />,
        label: c.title,
        description: `${c.messages.length} messages · ${c.mode}`,
        section: "Conversations",
        action: () => {
          onSelectConversation(c.id);
          onClose();
        },
      });
    });

    // Export recent
    conversations.slice(0, 4).forEach((c) => {
      all.push({
        id: `export-${c.id}`,
        icon: <Download size={14} />,
        label: `Export "${c.title}"`,
        section: "Export",
        action: () => {
          onExport(c.id);
          onClose();
        },
      });
    });

    if (!query.trim()) return all;

    const q = query.toLowerCase();
    return all.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        item.description?.toLowerCase().includes(q) ||
        item.section.toLowerCase().includes(q)
    );
  }, [query, conversations, onNewResearch, onSelectConversation, onExport, onClose, onSetMode]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Scroll selected into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const el = list.children[selectedIndex] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, items.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        items[selectedIndex]?.action();
        break;
      case "Escape":
        onClose();
        break;
    }
  };

  if (!open) return null;

  // Group items by section
  const sections = new Map<string, PaletteItem[]>();
  items.forEach((item) => {
    const list = sections.get(item.section) || [];
    list.push(item);
    sections.set(item.section, list);
  });

  let globalIndex = -1;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      onClick={onClose}
      style={{ animation: "fade-backdrop 0.15s ease-out" }}
    >
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-lg mx-4 bg-slate-900 border border-white/[0.08] rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
        style={{ animation: "scale-in 0.2s ease-out" }}
      >
        {/* Search */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-white/[0.06]">
          <Search size={18} className="text-slate-400 shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search commands, conversations…"
            className="flex-1 bg-transparent text-white text-sm placeholder-slate-500 outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center px-2 py-0.5 text-[10px] text-slate-500 bg-white/[0.05] rounded border border-white/[0.08]">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div
          ref={listRef}
          className="max-h-80 overflow-y-auto py-2"
        >
          {items.length === 0 && (
            <div className="px-5 py-8 text-center text-sm text-slate-500">
              No results for "{query}"
            </div>
          )}
          {Array.from(sections.entries()).map(([section, sectionItems]) => (
            <div key={section}>
              <div className="px-5 py-1.5 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                {section}
              </div>
              {sectionItems.map((item) => {
                globalIndex++;
                const idx = globalIndex;
                return (
                  <button
                    key={item.id}
                    onClick={item.action}
                    onMouseEnter={() => setSelectedIndex(idx)}
                    className={`w-full flex items-center gap-3 px-5 py-2.5 text-left transition-colors ${
                      selectedIndex === idx
                        ? "bg-indigo-500/10 text-white"
                        : "text-slate-300 hover:bg-white/[0.03]"
                    }`}
                  >
                    <span className="text-slate-400 shrink-0">{item.icon}</span>
                    <span className="flex-1 text-sm truncate">{item.label}</span>
                    {item.description && (
                      <span className="text-xs text-slate-500 shrink-0">
                        {item.description}
                      </span>
                    )}
                    {selectedIndex === idx && (
                      <ArrowRight size={12} className="text-slate-500 shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Footer hint */}
        <div className="flex items-center gap-4 px-5 py-2.5 border-t border-white/[0.06] text-[10px] text-slate-500">
          <span>↑↓ Navigate</span>
          <span>↵ Select</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
}
