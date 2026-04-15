import {
  BookOpen,
  Globe,
  MessageSquare,
  Upload,
  Sparkles,
  ArrowRight,
  GraduationCap,
} from "lucide-react";
import type { Conversation } from "../types";

interface WelcomeScreenProps {
  onNewResearch: (mode?: string) => void;
  recentConversations: Conversation[];
  onSelectConversation: (id: string) => void;
}

const QUICK_ACTIONS = [
  {
    icon: BookOpen,
    label: "Library Search",
    description: "Search your uploaded documents",
    mode: "strict",
    color: "from-indigo-500/20 to-indigo-600/5",
    iconColor: "text-indigo-400",
  },
  {
    icon: Globe,
    label: "Web Research",
    description: "Search the internet with AI",
    mode: "web",
    color: "from-emerald-500/20 to-emerald-600/5",
    iconColor: "text-emerald-400",
  },
  {
    icon: MessageSquare,
    label: "Chat",
    description: "Conversational AI assistant",
    mode: "chat",
    color: "from-violet-500/20 to-violet-600/5",
    iconColor: "text-violet-400",
  },
  {
    icon: Upload,
    label: "Upload & Analyze",
    description: "Add documents to your library",
    mode: "default",
    color: "from-amber-500/20 to-amber-600/5",
    iconColor: "text-amber-400",
  },
  {
    icon: GraduationCap,
    label: "Research Papers",
    description: "Find and download academic papers",
    mode: "research",
    color: "from-cyan-500/20 to-cyan-600/5",
    iconColor: "text-cyan-400",
  },
];

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function WelcomeScreen({
  onNewResearch,
  recentConversations,
  onSelectConversation,
}: WelcomeScreenProps) {
  const recent = recentConversations.slice(0, 4);

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl w-full animate-slide-up">
        {/* Hero */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 border border-indigo-500/20 mb-6">
            <Sparkles className="text-indigo-400" size={28} />
          </div>
          <h1 className="text-3xl font-bold text-white mb-3 tracking-tight">
            LibraryAI Research
          </h1>
          <p className="text-slate-400 text-lg max-w-md mx-auto leading-relaxed">
            Your AI-powered research assistant. Upload documents, search the
            web, and build your knowledge base.
          </p>
        </div>

        {/* Quick actions */}
        <div className="grid grid-cols-2 gap-3 mb-10">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.mode}
              onClick={() => onNewResearch(action.mode)}
              className={`group relative text-left p-5 rounded-xl bg-gradient-to-br ${action.color} border border-white/[0.06] hover:border-white/[0.12] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]`}
            >
              <action.icon className={`${action.iconColor} mb-3`} size={22} />
              <div className="font-medium text-white text-sm mb-1">
                {action.label}
              </div>
              <div className="text-xs text-slate-400">{action.description}</div>
              <ArrowRight
                size={14}
                className="absolute top-5 right-5 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity"
              />
            </button>
          ))}
        </div>

        {/* Recent conversations */}
        {recent.length > 0 && (
          <div>
            <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3 px-1">
              Recent Research
            </h3>
            <div className="space-y-1">
              {recent.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onSelectConversation(c.id)}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-white/[0.04] transition-colors text-left group"
                >
                  <div className="w-8 h-8 rounded-lg bg-white/[0.04] flex items-center justify-center shrink-0">
                    <MessageSquare size={14} className="text-slate-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-slate-200 truncate">
                      {c.title}
                    </div>
                    <div className="text-xs text-slate-500">
                      {c.messages.length} messages · {timeAgo(c.updatedAt)}
                    </div>
                  </div>
                  <ArrowRight
                    size={14}
                    className="text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
