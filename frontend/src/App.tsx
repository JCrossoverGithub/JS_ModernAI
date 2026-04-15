import { useState, useEffect, useCallback } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { useConversations } from "./hooks/useConversations";
import LoginPage from "./components/LoginPage";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import CommandPalette from "./components/CommandPalette";
import type { SearchMode } from "./types";
import { DEFAULT_RESEARCH_SOURCES } from "./types";

function AppContent() {
  const { user } = useAuth();
  const [mode, setMode] = useState<SearchMode>("default");
  const [researchSources, setResearchSources] = useState<string[]>(DEFAULT_RESEARCH_SOURCES);
  const [cmdOpen, setCmdOpen] = useState(false);

  const {
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
    setMode: setConvMode,
    exportAsMarkdown,
    search,
  } = useConversations();

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setCmdOpen((o) => !o);
      }
      if (e.key === "Escape" && cmdOpen) {
        setCmdOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [cmdOpen]);

  const handleNewResearch = useCallback(
    (m?: string) => {
      const newMode = (m as SearchMode) || mode;
      setMode(newMode);
      create(newMode);
    },
    [mode, create]
  );

  const handleModeChange = useCallback(
    (m: SearchMode) => {
      setMode(m);
      if (active) setConvMode(active.id, m);
    },
    [active, setConvMode]
  );

  const handleSetModeFromPalette = useCallback(
    (m: string) => {
      setMode(m as SearchMode);
      if (active) setConvMode(active.id, m);
    },
    [active, setConvMode]
  );

  if (!user) return <LoginPage />;

  return (
    <div className="flex h-screen bg-[#0a0b12] text-slate-100">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        mode={mode}
        researchSources={researchSources}
        onResearchSourcesChange={setResearchSources}
        onSelectConversation={setActiveId}
        onNewConversation={handleNewResearch}
        onDeleteConversation={remove}
        onTogglePin={togglePin}
        onExport={exportAsMarkdown}
        onModeChange={handleModeChange}
        onOpenCommandPalette={() => setCmdOpen(true)}
      />
      <ChatWindow
        conversation={active}
        mode={mode}
        researchSources={researchSources}
        conversations={conversations}
        onAddMessage={addMessage}
        onUpdateLastMessage={updateLastMessage}
        onToggleBookmark={toggleBookmark}
        onExport={exportAsMarkdown}
        onNewResearch={handleNewResearch}
        onSelectConversation={setActiveId}
      />
      <CommandPalette
        open={cmdOpen}
        onClose={() => setCmdOpen(false)}
        conversations={conversations}
        onSelectConversation={(id) => {
          setActiveId(id);
          setCmdOpen(false);
        }}
        onNewResearch={(m) => {
          handleNewResearch(m);
          setCmdOpen(false);
        }}
        onExport={exportAsMarkdown}
        onDelete={remove}
        onSetMode={handleSetModeFromPalette}
      />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
