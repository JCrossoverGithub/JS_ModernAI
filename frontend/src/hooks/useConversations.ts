import { useState, useCallback, useEffect } from "react";
import type { Conversation, Message } from "../types";

const STORAGE_KEY = "libraryai_conversations";

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function persist(convos: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos));
}

function titleFrom(text: string): string {
  const clean = text.replace(/\n/g, " ").trim();
  return clean.length > 50 ? clean.slice(0, 47) + "…" : clean;
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(load);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    persist(conversations);
  }, [conversations]);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  const create = useCallback((mode: string = "default") => {
    const id = crypto.randomUUID();
    const c: Conversation = {
      id,
      title: "New Research",
      messages: [],
      mode,
      createdAt: Date.now(),
      updatedAt: Date.now(),
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
    setActiveId(null);
  }, []);

  return {
    conversations,
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
    exportAsMarkdown,
    search,
    clearAll,
  };
}
