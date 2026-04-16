import { useState, useCallback, useEffect } from "react";
import type { Conversation, Folder, Message } from "../types";

const STORAGE_KEY = "libraryai_conversations";
const FOLDERS_KEY = "libraryai_folders";

function loadConvos(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const convos: Conversation[] = raw ? JSON.parse(raw) : [];
    // migrate: ensure sortOrder exists
    return convos.map((c, i) => ({
      ...c,
      sortOrder: c.sortOrder ?? i,
    }));
  } catch {
    return [];
  }
}

function loadFolders(): Folder[] {
  try {
    const raw = localStorage.getItem(FOLDERS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function persistConvos(convos: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos));
}

function persistFolders(folders: Folder[]) {
  localStorage.setItem(FOLDERS_KEY, JSON.stringify(folders));
}

function titleFrom(text: string): string {
  const clean = text.replace(/\n/g, " ").trim();
  return clean.length > 50 ? clean.slice(0, 47) + "…" : clean;
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConvos);
  const [folders, setFolders] = useState<Folder[]>(loadFolders);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    persistConvos(conversations);
  }, [conversations]);

  useEffect(() => {
    persistFolders(folders);
  }, [folders]);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  const create = useCallback((mode: string = "default", folderId?: string) => {
    const id = crypto.randomUUID();
    const c: Conversation = {
      id,
      title: "New Research",
      messages: [],
      mode,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      sortOrder: Date.now(),
      ...(folderId ? { folderId } : {}),
    };
    setConversations((prev) => [c, ...prev]);
    setActiveId(id);
    return id;
  }, []);

  const addMessage = useCallback((convId: string, msg: Message) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        const msgs = [...c.messages, msg];
        const title =
          c.messages.length === 0 && msg.role === "user"
            ? titleFrom(msg.content)
            : c.title;
        return { ...c, messages: msgs, title, updatedAt: Date.now() };
      })
    );
  }, []);

  const updateLastMessage = useCallback(
    (convId: string, updater: (m: Message) => Message) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          const msgs = [...c.messages];
          if (msgs.length > 0) {
            msgs[msgs.length - 1] = updater(msgs[msgs.length - 1]);
          }
          return { ...c, messages: msgs, updatedAt: Date.now() };
        })
      );
    },
    []
  );

  const remove = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) setActiveId(null);
    },
    [activeId]
  );

  const togglePin = useCallback((id: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, pinned: !c.pinned } : c))
    );
  }, []);

  const toggleBookmark = useCallback((convId: string, msgId: string) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        return {
          ...c,
          messages: c.messages.map((m) =>
            m.id === msgId ? { ...m, bookmarked: !m.bookmarked } : m
          ),
        };
      })
    );
  }, []);

  const setMode = useCallback((id: string, mode: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, mode } : c))
    );
  }, []);

  // ── Folder operations ──

  const createFolder = useCallback(
    (name: string, convIdA: string, convIdB: string) => {
      const folderId = crypto.randomUUID();
      setFolders((prev) => [
        ...prev,
        { id: folderId, name, expanded: true, sortOrder: Date.now() },
      ]);
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === convIdA || c.id === convIdB) {
            return { ...c, folderId };
          }
          return c;
        })
      );
      return folderId;
    },
    []
  );

  const renameFolder = useCallback((folderId: string, name: string) => {
    setFolders((prev) =>
      prev.map((f) => (f.id === folderId ? { ...f, name } : f))
    );
  }, []);

  const toggleFolderExpanded = useCallback((folderId: string) => {
    setFolders((prev) =>
      prev.map((f) => (f.id === folderId ? { ...f, expanded: !f.expanded } : f))
    );
  }, []);

  const moveToFolder = useCallback((convId: string, folderId: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, folderId } : c))
    );
  }, []);

  const removeFromFolder = useCallback((convId: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === convId ? { ...c, folderId: undefined } : c))
    );
  }, []);

  const deleteFolder = useCallback((folderId: string) => {
    // unhome all conversations, then delete the folder
    setConversations((prev) =>
      prev.map((c) =>
        c.folderId === folderId ? { ...c, folderId: undefined } : c
      )
    );
    setFolders((prev) => prev.filter((f) => f.id !== folderId));
  }, []);

  // Clean up empty folders
  useEffect(() => {
    const usedFolderIds = new Set(
      conversations.filter((c) => c.folderId).map((c) => c.folderId!)
    );
    setFolders((prev) => {
      const cleaned = prev.filter((f) => usedFolderIds.has(f.id));
      if (cleaned.length !== prev.length) return cleaned;
      return prev;
    });
  }, [conversations]);

  /** Build shared context string from sibling chats in the same folder. */
  const getFolderContext = useCallback(
    (convId: string): string => {
      const conv = conversations.find((c) => c.id === convId);
      if (!conv?.folderId) return "";
      const siblings = conversations.filter(
        (c) => c.folderId === conv.folderId && c.id !== convId && c.messages.length > 0
      );
      if (siblings.length === 0) return "";

      const lines: string[] = [];
      for (const sib of siblings) {
        lines.push(`[Context from "${sib.title}"]`);
        // Take last 4 meaningful messages from each sibling
        const recent = sib.messages
          .filter((m) => m.role !== "status")
          .slice(-6);
        for (const m of recent) {
          const role = m.role === "user" ? "User" : "AI";
          lines.push(`${role}: ${m.content.slice(0, 1500)}`);
        }
        lines.push("");
      }
      return lines.join("\n");
    },
    [conversations]
  );

  const reorder = useCallback((convId: string, newSortOrder: number) => {
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId ? { ...c, sortOrder: newSortOrder } : c
      )
    );
  }, []);

  const exportAsMarkdown = useCallback(
    (id: string) => {
      const c = conversations.find((x) => x.id === id);
      if (!c) return;

      let md = `# ${c.title}\n\n`;
      md += `*Mode: ${c.mode} · ${new Date(c.createdAt).toLocaleDateString()}*\n\n---\n\n`;

      for (const m of c.messages) {
        if (m.role === "status") continue;
        md += `### ${m.role === "user" ? "You" : "Assistant"}\n\n`;
        md += `${m.content}\n\n`;
        if (m.sources?.length) {
          md += `> Sources: ${m.sources.join(", ")}\n\n`;
        }
        md += `---\n\n`;
      }

      const blob = new Blob([md], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${c.title.replace(/[^a-zA-Z0-9 ]/g, "").replace(/\s+/g, "_")}.md`;
      a.click();
      URL.revokeObjectURL(url);
    },
    [conversations]
  );

  const search = useCallback(
    (query: string): Conversation[] => {
      if (!query.trim()) return conversations;
      const q = query.toLowerCase();
      return conversations.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.messages.some((m) => m.content.toLowerCase().includes(q))
      );
    },
    [conversations]
  );

  const clearAll = useCallback(() => {
    setConversations([]);
    setFolders([]);
    setActiveId(null);
  }, []);

  return {
    conversations,
    folders,
    active,
    activeId,
    setActiveId,
    create,
    addMessage,
    updateLastMessage,
    remove,
    togglePin,
    toggleBookmark,
    setMode,
    createFolder,
    renameFolder,
    toggleFolderExpanded,
    moveToFolder,
    removeFromFolder,
    deleteFolder,
    reorder,
    getFolderContext,
    exportAsMarkdown,
    search,
    clearAll,
  };
}
