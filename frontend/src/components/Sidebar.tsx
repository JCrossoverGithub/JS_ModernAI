import { useState, useEffect, useMemo, useRef } from "react";
import {
  Plus,
  Search,
  Pin,
  MoreHorizontal,
  Trash2,
  Download,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  LogOut,
  FileText,
  Brain,
  Upload,
  Eraser,
  AlertTriangle,
  Settings,
  BookOpen,
  Globe,
  Zap,
  Command,
  X,
  GraduationCap,
  FolderOpen,
  FolderClosed,
  GripVertical,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import {
  listFacts,
  saveFact,
  deleteFact,
  listDocuments,
  uploadDocument,
  removeDocument,
  wipeMemory,
  clearBuffer,
} from "../services/api";
import type { Conversation, Folder, SearchMode } from "../types";
import { SEARCH_MODES, RESEARCH_SOURCES } from "../types";

interface SidebarProps {
  conversations: Conversation[];
  folders: Folder[];
  activeId: string | null;
  mode: SearchMode;
  researchSources: string[];
  onResearchSourcesChange: (sources: string[]) => void;
  onSelectConversation: (id: string) => void;
  onNewConversation: (mode?: string, folderId?: string) => void;
  onDeleteConversation: (id: string) => void;
  onTogglePin: (id: string) => void;
  onExport: (id: string) => void;
  onModeChange: (mode: SearchMode) => void;
  onOpenCommandPalette: () => void;
  onCreateFolder: (name: string, convIdA: string, convIdB: string) => string;
  onRenameFolder: (folderId: string, name: string) => void;
  onToggleFolderExpanded: (folderId: string) => void;
  onMoveToFolder: (convId: string, folderId: string) => void;
  onRemoveFromFolder: (convId: string) => void;
  onDeleteFolder: (folderId: string) => void;
  onReorder: (convId: string, newSortOrder: number) => void;
}

const MODE_ICONS: Record<string, typeof Zap> = {
  default: Zap,
  strict: BookOpen,
  chat: MessageSquare,
  web: Globe,
  research: GraduationCap,
};

const MODE_COLORS: Record<string, string> = {
  default: "text-indigo-400",
  strict: "text-amber-400",
  chat: "text-violet-400",
  web: "text-emerald-400",
  research: "text-cyan-400",
};

function timeLabel(ts: number): string {
  const now = new Date();
  const d = new Date(ts);
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / 86400000);

  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 Days";
  if (days < 30) return "Previous 30 Days";
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

function groupConversations(
  convos: Conversation[]
): Map<string, Conversation[]> {
  const groups = new Map<string, Conversation[]>();
  const pinned = convos.filter((c) => c.pinned);
  const unpinned = convos.filter((c) => !c.pinned);

  if (pinned.length > 0) groups.set("Pinned", pinned);

  for (const c of unpinned) {
    const label = timeLabel(c.updatedAt);
    const list = groups.get(label) || [];
    list.push(c);
    groups.set(label, list);
  }

  return groups;
}

// ─── Sub-panels ───

function FactsPanel({ onClose }: { onClose: () => void }) {
  const [facts, setFacts] = useState<string[]>([]);
  const [newFact, setNewFact] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const res = await listFacts();
      setFacts(res.facts || []);
    } catch {
      /* empty */
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleSave = async () => {
    if (!newFact.trim()) return;
    await saveFact(newFact.trim());
    setNewFact("");
    load();
  };

  const handleDelete = async (keyword: string) => {
    await deleteFact(keyword);
    load();
  };

  return (
    <div className="flex flex-col h-full animate-slide-in-left">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Brain size={15} className="text-violet-400" />
          <span className="text-sm font-medium text-white">Memory Facts</span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md hover:bg-white/[0.06] text-slate-400 transition-colors"
        >
          <X size={14} />
        </button>
      </div>

      <div className="p-3">
        <div className="flex gap-2">
          <input
            value={newFact}
            onChange={(e) => setNewFact(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSave()}
            placeholder="Add a fact…"
            className="flex-1 px-3 py-2 bg-white/[0.04] border border-white/[0.06] rounded-lg text-sm text-white placeholder-slate-500 outline-none focus:border-indigo-500/50 transition-colors"
          />
          <button
            onClick={handleSave}
            disabled={!newFact.trim()}
            className="px-3 py-2 bg-indigo-500/20 text-indigo-300 rounded-lg text-sm font-medium hover:bg-indigo-500/30 disabled:opacity-30 transition-colors"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {loading ? (
          <p className="text-xs text-slate-500 p-2">Loading…</p>
        ) : facts.length === 0 ? (
          <p className="text-xs text-slate-500 p-2">
            No facts saved yet. Add facts to give your AI personal context.
          </p>
        ) : (
          <div className="space-y-1.5">
            {facts.map((f, i) => (
              <div
                key={i}
                className="flex items-start gap-2 px-3 py-2 bg-white/[0.03] rounded-lg group"
              >
                <span className="text-xs text-slate-300 flex-1 leading-relaxed">
                  {f}
                </span>
                <button
                  onClick={() => handleDelete(f.split(" ")[0])}
                  className="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all shrink-0 mt-0.5"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-3 border-t border-white/[0.06] space-y-1.5">
        <button
          onClick={() => clearBuffer()}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-400 hover:bg-white/[0.04] transition-colors"
        >
          <Eraser size={12} />
          Clear conversation buffer
        </button>
        <button
          onClick={() => {
            if (confirm("Wipe ALL memory? This cannot be undone."))
              wipeMemory();
          }}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-red-400/70 hover:bg-red-500/10 transition-colors"
        >
          <AlertTriangle size={12} />
          Wipe all memory
        </button>
      </div>
    </div>
  );
}

function DocsPanel({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [uploadPercent, setUploadPercent] = useState(0);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const res = await listDocuments();
      setDocs(res.documents || []);
    } catch {
      /* empty */
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadProgress("Uploading file...");
    setUploadPercent(0);
    try {
      await uploadDocument(file, (message, percent) => {
        setUploadProgress(message);
        setUploadPercent(percent);
      });
      load();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setUploading(false);
      setUploadProgress("");
      setUploadPercent(0);
      e.target.value = "";
    }
  };

  const handleRemove = async (filename: string) => {
    await removeDocument(filename);
    load();
  };

  return (
    <div className="flex flex-col h-full animate-slide-in-left">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <FileText size={15} className="text-amber-400" />
          <span className="text-sm font-medium text-white">
            Document Library
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md hover:bg-white/[0.06] text-slate-400 transition-colors"
        >
          <X size={14} />
        </button>
      </div>

      <div className="p-3">
        <label className="block cursor-pointer">
          <div className="flex items-center justify-center gap-2 px-4 py-3 border border-dashed border-indigo-500/30 bg-indigo-500/5 rounded-xl text-indigo-300 text-sm hover:bg-indigo-500/10 hover:border-indigo-500/50 transition-all">
            <Upload size={15} />
            {uploading ? "Processing…" : "Upload PDF, TXT, or DOCX"}
          </div>
          <input
            type="file"
            accept=".pdf,.txt,.docx,.doc"
            onChange={handleUpload}
            disabled={uploading}
            className="hidden"
          />
        </label>
        {uploading && (
          <div className="mt-2 space-y-1.5">
            <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all duration-300"
                style={{ width: `${uploadPercent}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 text-center">{uploadProgress}</p>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {loading ? (
          <p className="text-xs text-slate-500 p-2">Loading…</p>
        ) : docs.length === 0 ? (
          <p className="text-xs text-slate-500 p-2">
            No documents uploaded yet. Upload files to search them with AI.
          </p>
        ) : (
          <div className="space-y-1.5">
            {docs.map((d) => (
              <div
                key={d}
                className="flex items-center gap-2 px-3 py-2.5 bg-white/[0.03] rounded-lg group"
              >
                <FileText size={13} className="text-slate-500 shrink-0" />
                <span className="text-xs text-slate-300 flex-1 truncate">
                  {d}
                </span>
                <button
                  onClick={() => handleRemove(d)}
                  className="text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all shrink-0"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Context menu ───

function ConversationMenu({
  conversation,
  onPin,
  onExport,
  onDelete,
  onClose,
}: {
  conversation: Conversation;
  onPin: () => void;
  onExport: () => void;
  onDelete: () => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const handler = () => onClose();
    window.addEventListener("click", handler, { once: true });
    return () => window.removeEventListener("click", handler);
  }, [onClose]);

  return (
    <div
      className="absolute right-2 top-8 z-30 w-44 py-1.5 bg-slate-800 border border-white/[0.08] rounded-xl shadow-xl"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        onClick={() => {
          onPin();
          onClose();
        }}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.06] transition-colors"
      >
        <Pin size={12} />
        {conversation.pinned ? "Unpin" : "Pin to top"}
      </button>
      <button
        onClick={() => {
          onExport();
          onClose();
        }}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.06] transition-colors"
      >
        <Download size={12} />
        Export as Markdown
      </button>
      <div className="my-1 border-t border-white/[0.06]" />
      <button
        onClick={() => {
          onDelete();
          onClose();
        }}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors"
      >
        <Trash2 size={12} />
        Delete
      </button>
    </div>
  );
}

// ─── Main sidebar ───

export default function Sidebar({
  conversations,
  folders,
  activeId,
  mode,
  researchSources,
  onResearchSourcesChange,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onTogglePin,
  onExport,
  onModeChange,
  onOpenCommandPalette,
  onCreateFolder,
  onRenameFolder,
  onToggleFolderExpanded,
  onMoveToFolder,
  onRemoveFromFolder,
  onDeleteFolder,
  onReorder,
}: SidebarProps) {
  const { user, logout } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [panel, setPanel] = useState<"none" | "facts" | "docs">("none");
  const [menuId, setMenuId] = useState<string | null>(null);
  const [modeOpen, setModeOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // ── Drag state ──
  const [dragId, setDragId] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<{ id: string; type: "conv" | "folder" | "root" } | null>(null);
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const folderNameRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingFolderId && folderNameRef.current) {
      folderNameRef.current.focus();
      folderNameRef.current.select();
    }
  }, [editingFolderId]);

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return conversations;
    const q = searchQuery.toLowerCase();
    return conversations.filter(
      (c) =>
        c.title.toLowerCase().includes(q) ||
        c.messages.some((m) => m.content.toLowerCase().includes(q))
    );
  }, [conversations, searchQuery]);

  // Build an ordered list: loose conversations sorted by updatedAt,
  // with folder groups placed at the position of their most recently updated child.
  const listItems = useMemo(() => {
    const loose = filtered.filter((c) => !c.folderId).sort((a, b) => b.updatedAt - a.updatedAt);
    const folderConvs = new Map<string, Conversation[]>();
    for (const c of filtered.filter((c) => c.folderId)) {
      const list = folderConvs.get(c.folderId!) || [];
      list.push(c);
      folderConvs.set(c.folderId!, list);
    }
    // Sort within each folder
    for (const [, list] of folderConvs) {
      list.sort((a, b) => b.updatedAt - a.updatedAt);
    }

    type Item = { type: "conv"; conv: Conversation } | { type: "folder"; folder: Folder; convs: Conversation[] };
    const items: Item[] = [];

    // Merge loose convos and folders by most recent activity
    const folderEntries: { folder: Folder; convs: Conversation[]; latest: number }[] = [];
    for (const f of folders) {
      const convs = folderConvs.get(f.id) || [];
      if (convs.length === 0) continue;
      const latest = Math.max(...convs.map((c) => c.updatedAt));
      folderEntries.push({ folder: f, convs, latest });
    }
    folderEntries.sort((a, b) => b.latest - a.latest);

    let li = 0;
    let fi = 0;
    while (li < loose.length || fi < folderEntries.length) {
      const looseTs = li < loose.length ? loose[li].updatedAt : -1;
      const folderTs = fi < folderEntries.length ? folderEntries[fi].latest : -1;
      if (looseTs >= folderTs && li < loose.length) {
        items.push({ type: "conv", conv: loose[li] });
        li++;
      } else if (fi < folderEntries.length) {
        items.push({ type: "folder", folder: folderEntries[fi].folder, convs: folderEntries[fi].convs });
        fi++;
      }
    }
    return items;
  }, [filtered, folders]);

  // ── Drag handlers ──
  const handleDragStart = (convId: string) => (e: React.DragEvent) => {
    setDragId(convId);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", convId);
  };

  const handleDragOver = (id: string, type: "conv" | "folder" | "root") => (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    if (id !== dragId) setDropTarget({ id, type });
  };

  const handleDragLeave = () => setDropTarget(null);

  const handleDrop = (targetId: string, targetType: "conv" | "folder" | "root") => (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const srcId = e.dataTransfer.getData("text/plain") || dragId;
    setDragId(null);
    setDropTarget(null);
    if (!srcId || srcId === targetId) return;

    if (targetType === "folder") {
      // Drop onto a folder → move into it
      onMoveToFolder(srcId, targetId);
    } else if (targetType === "conv") {
      // Drop a conversation onto another conversation → create a new folder
      const src = conversations.find((c) => c.id === srcId);
      const tgt = conversations.find((c) => c.id === targetId);
      if (!src || !tgt) return;

      if (tgt.folderId) {
        // Target is already in a folder → just move src into that folder
        onMoveToFolder(srcId, tgt.folderId);
      } else {
        // Both are loose → create a new folder
        const folderName = `${tgt.title.slice(0, 20)}…`;
        onCreateFolder(folderName, srcId, targetId);
      }
    } else if (targetType === "root") {
      // Drop to root area → remove from folder
      onRemoveFromFolder(srcId);
    }
  };

  const handleDragEnd = () => {
    setDragId(null);
    setDropTarget(null);
  };

  if (panel === "facts") return <FactsPanel onClose={() => setPanel("none")} />;
  if (panel === "docs") return <DocsPanel onClose={() => setPanel("none")} />;

  const ModeIcon = MODE_ICONS[mode] || Zap;

  // ── Render a single conversation row ──
  const renderConvRow = (c: Conversation, indent = false) => {
    const isActive = c.id === activeId;
    const isDragging = dragId === c.id;
    const isDropTarget = dropTarget?.id === c.id && dropTarget.type === "conv";
    const CIcon = MODE_ICONS[c.mode] || Zap;

    return (
      <div
        key={c.id}
        className="relative"
        draggable
        onDragStart={handleDragStart(c.id)}
        onDragOver={handleDragOver(c.id, "conv")}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop(c.id, "conv")}
        onDragEnd={handleDragEnd}
      >
        <button
          onClick={() => onSelectConversation(c.id)}
          className={`w-full flex items-center gap-2 py-2 rounded-lg text-left transition-all group ${
            indent ? "pl-8 pr-3" : "px-3"
          } ${
            isDropTarget
              ? "bg-indigo-500/20 border border-indigo-500/40 border-dashed"
              : isDragging
              ? "opacity-40"
              : isActive
              ? "bg-white/[0.07] text-white"
              : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
          }`}
        >
          <GripVertical size={11} className="text-slate-600 shrink-0 opacity-0 group-hover:opacity-60 cursor-grab" />
          <CIcon
            size={13}
            className={`shrink-0 ${
              isActive ? MODE_COLORS[c.mode] || "text-slate-400" : "text-slate-600"
            }`}
          />
          <span className="flex-1 text-sm truncate">{c.title}</span>
          {c.folderId && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemoveFromFolder(c.id);
              }}
              className="p-0.5 rounded text-slate-600 hover:text-slate-300 opacity-0 group-hover:opacity-70 shrink-0"
              title="Remove from folder"
            >
              <X size={10} />
            </button>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMenuId(menuId === c.id ? null : c.id);
            }}
            className={`p-0.5 rounded text-slate-600 hover:text-slate-300 transition-opacity shrink-0 ${
              menuId === c.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
            }`}
          >
            <MoreHorizontal size={13} />
          </button>
        </button>
        {menuId === c.id && (
          <ConversationMenu
            conversation={c}
            onPin={() => onTogglePin(c.id)}
            onExport={() => onExport(c.id)}
            onDelete={() => onDeleteConversation(c.id)}
            onClose={() => setMenuId(null)}
          />
        )}
      </div>
    );
  };

  return (
    <div className="w-72 bg-[#0f1019] border-r border-white/[0.06] flex flex-col h-full">
      {/* Header */}
      <div className="p-4 pb-3">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-base font-semibold text-white tracking-tight">
            LibraryAI
          </h1>
        </div>
        <button
          onClick={() => onNewConversation(mode)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/20 hover:border-indigo-500/40 text-indigo-300 rounded-xl text-sm font-medium transition-all active:scale-[0.98]"
        >
          <Plus size={15} />
          New Research
        </button>
      </div>

      {/* Search */}
      <div className="px-4 pb-3">
        <div className="relative">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
          />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search threads…"
            className="w-full pl-9 pr-3 py-2 bg-white/[0.04] border border-white/[0.06] rounded-lg text-xs text-white placeholder-slate-500 outline-none focus:border-indigo-500/40 transition-colors"
          />
        </div>
      </div>

      {/* Conversations with drag-and-drop */}
      <div
        className="flex-1 overflow-y-auto px-2"
        onDragOver={handleDragOver("root", "root")}
        onDrop={handleDrop("root", "root")}
      >
        {filtered.length === 0 ? (
          <div className="px-3 py-6 text-center">
            <p className="text-xs text-slate-500">
              {searchQuery ? "No matching threads" : "Start a new research thread"}
            </p>
          </div>
        ) : (
          listItems.map((item) => {
            if (item.type === "conv") {
              return renderConvRow(item.conv);
            }
            // Folder
            const f = item.folder;
            const convs = item.convs;
            const isFolderDrop = dropTarget?.id === f.id && dropTarget.type === "folder";
            return (
              <div
                key={f.id}
                className={`mb-1 rounded-lg transition-all ${
                  isFolderDrop ? "bg-indigo-500/10 ring-1 ring-indigo-500/30" : ""
                }`}
                onDragOver={handleDragOver(f.id, "folder")}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop(f.id, "folder")}
              >
                {/* Folder header */}
                <div className="flex items-center gap-1.5 px-2 py-1.5 group">
                  <button
                    onClick={() => onToggleFolderExpanded(f.id)}
                    className="p-0.5 rounded text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    {f.expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  </button>
                  {f.expanded ? (
                    <FolderOpen size={13} className="text-amber-500/70 shrink-0" />
                  ) : (
                    <FolderClosed size={13} className="text-amber-500/70 shrink-0" />
                  )}
                  {editingFolderId === f.id ? (
                    <input
                      ref={folderNameRef}
                      defaultValue={f.name}
                      onBlur={(e) => {
                        const val = e.target.value.trim();
                        if (val) onRenameFolder(f.id, val);
                        setEditingFolderId(null);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                        if (e.key === "Escape") setEditingFolderId(null);
                      }}
                      className="flex-1 text-xs text-white bg-white/[0.06] border border-white/[0.1] rounded px-1.5 py-0.5 outline-none focus:border-indigo-500/50"
                    />
                  ) : (
                    <span
                      className="flex-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider truncate cursor-pointer"
                      onDoubleClick={() => setEditingFolderId(f.id)}
                      title="Double-click to rename"
                    >
                      {f.name}
                    </span>
                  )}
                  <span className="text-[9px] text-slate-600 tabular-nums">{convs.length}</span>
                  <button
                    onClick={() => onNewConversation(mode, f.id)}
                    className="p-0.5 rounded text-slate-600 hover:text-slate-300 opacity-0 group-hover:opacity-100 transition-all shrink-0"
                    title="New chat in folder"
                  >
                    <Plus size={10} />
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete folder "${f.name}"? Chats will be moved to top level.`))
                        onDeleteFolder(f.id);
                    }}
                    className="p-0.5 rounded text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all shrink-0"
                    title="Delete folder"
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
                {/* Folder children */}
                {f.expanded && (
                  <div className="ml-1 border-l border-white/[0.04] mb-1">
                    {convs.map((c) => renderConvRow(c, true))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Bottom bar */}
      <div className="border-t border-white/[0.06]">
        {/* Management buttons */}
        <div className="flex items-center px-3 py-2 gap-1">
          <button
            onClick={() => setPanel("docs")}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs text-slate-400 hover:text-white hover:bg-white/[0.06] transition-colors"
          >
            <FileText size={13} />
            Library
          </button>
          <button
            onClick={() => setPanel("facts")}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs text-slate-400 hover:text-white hover:bg-white/[0.06] transition-colors"
          >
            <Brain size={13} />
            Memory
          </button>
          <div className="flex-1" />
          <button
            onClick={onOpenCommandPalette}
            className="flex items-center gap-1 px-2 py-1.5 rounded-md text-xs text-slate-500 hover:text-white hover:bg-white/[0.06] transition-colors"
            title="Command palette (Ctrl+K)"
          >
            <Command size={12} />
            <kbd className="text-[9px]">K</kbd>
          </button>
        </div>

        {/* Mode selector */}
        <div className="px-3 pb-2 relative">
          <button
            onClick={() => setModeOpen(!modeOpen)}
            className="w-full flex items-center gap-2 px-3 py-2 bg-white/[0.03] border border-white/[0.06] rounded-lg hover:border-white/[0.1] transition-colors"
          >
            <ModeIcon size={13} className={MODE_COLORS[mode]} />
            <span className="text-xs text-slate-300 flex-1 text-left">
              {SEARCH_MODES.find((m) => m.value === mode)?.label || mode} Mode
            </span>
            <ChevronDown
              size={12}
              className={`text-slate-500 transition-transform ${
                modeOpen ? "rotate-180" : ""
              }`}
            />
          </button>
          {modeOpen && (
            <div className="absolute bottom-full left-3 right-3 mb-1 bg-slate-800 border border-white/[0.08] rounded-xl shadow-xl py-1 z-20">
              {SEARCH_MODES.map((m) => {
                const Icon = MODE_ICONS[m.value] || Zap;
                return (
                  <button
                    key={m.value}
                    onClick={() => {
                      onModeChange(m.value);
                      setModeOpen(false);
                    }}
                    className={`w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                      mode === m.value
                        ? "bg-indigo-500/10 text-white"
                        : "text-slate-300 hover:bg-white/[0.06]"
                    }`}
                  >
                    <Icon size={13} className={MODE_COLORS[m.value]} />
                    <div className="flex-1">
                      <div className="text-xs font-medium">{m.label}</div>
                      <div className="text-[10px] text-slate-500">
                        {m.description}
                      </div>
                    </div>
                    {mode === m.value && (
                      <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Research Sources — visible only in research mode */}
        {mode === "research" && (
          <div className="px-3 pb-2">
            <button
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-white/[0.04] transition-colors"
            >
              <Settings size={12} className="text-cyan-400" />
              <span className="text-[11px] text-slate-400 flex-1 text-left">
                Sources
              </span>
              <span className="text-[10px] text-cyan-400/70 font-medium tabular-nums">
                {researchSources.length}/{RESEARCH_SOURCES.length}
              </span>
              <ChevronDown
                size={11}
                className={`text-slate-500 transition-transform ${sourcesOpen ? "rotate-180" : ""}`}
              />
            </button>
            {sourcesOpen && (
              <div className="mt-1.5 bg-white/[0.02] border border-white/[0.06] rounded-xl p-2 space-y-2 animate-fade-in">
                {(["academic", "discovery"] as const).map((cat) => {
                  const items = RESEARCH_SOURCES.filter((s) => s.category === cat);
                  return (
                    <div key={cat}>
                      <div className="text-[9px] font-semibold uppercase tracking-wider text-slate-500 px-1 mb-1">
                        {cat}
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {items.map((src) => {
                          const active = researchSources.includes(src.id);
                          return (
                            <button
                              key={src.id}
                              onClick={() => {
                                onResearchSourcesChange(
                                  active
                                    ? researchSources.filter((s) => s !== src.id)
                                    : [...researchSources, src.id]
                                );
                              }}
                              title={src.description}
                              className={`px-2 py-1 rounded-md text-[10px] font-medium border transition-all ${
                                active
                                  ? "bg-cyan-500/15 text-cyan-300 border-cyan-500/30 shadow-[0_0_6px_rgba(6,182,212,0.15)]"
                                  : "bg-white/[0.02] text-slate-500 border-white/[0.06] hover:border-white/[0.12] hover:text-slate-400"
                              }`}
                            >
                              {src.name}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
                <div className="flex gap-1 pt-1 border-t border-white/[0.04]">
                  <button
                    onClick={() =>
                      onResearchSourcesChange(RESEARCH_SOURCES.map((s) => s.id))
                    }
                    className="flex-1 text-[9px] text-slate-500 hover:text-cyan-400 py-0.5 transition-colors"
                  >
                    Select All
                  </button>
                  <button
                    onClick={() => onResearchSourcesChange([])}
                    className="flex-1 text-[9px] text-slate-500 hover:text-cyan-400 py-0.5 transition-colors"
                  >
                    Clear All
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* User */}
        <div className="flex items-center gap-2 px-4 py-3 border-t border-white/[0.06]">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center text-[11px] font-semibold text-white">
            {user?.username?.charAt(0).toUpperCase() || "?"}
          </div>
          <span className="text-xs text-slate-400 flex-1 truncate">
            {user?.username}
          </span>
          <button
            onClick={logout}
            className="p-1.5 rounded-md text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            title="Sign out"
          >
            <LogOut size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}
