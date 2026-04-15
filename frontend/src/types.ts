export interface AuthResponse {
  token: string;
  username: string;
  userId: string;
}

export interface ChatEvent {
  type: "status" | "debug" | "token" | "sources" | "papers" | "done";
  message?: string;
  standalone_query?: string;
  content?: string;
  sources?: string[];
  papers?: Paper[];
}

export interface Paper {
  title: string;
  authors: string[];
  year: number | null;
  abstract: string;
  url: string;
  venue: string;
  citationCount: number;
  downloadUrl: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "status";
  content: string;
  sources?: string[];
  mode?: string;
  timestamp: number;
  bookmarked?: boolean;
  papers?: Paper[];
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  mode: string;
  createdAt: number;
  updatedAt: number;
  pinned?: boolean;
}

export type SearchMode = "default" | "strict" | "chat" | "web" | "research";

export interface ModeInfo {
  value: SearchMode;
  label: string;
  description: string;
}

export const SEARCH_MODES: ModeInfo[] = [
  { value: "default", label: "Default", description: "Library + Memory" },
  { value: "strict", label: "Strict", description: "Library only" },
  { value: "chat", label: "Chat", description: "Conversational" },
  { value: "web", label: "Web", description: "Internet search" },
  { value: "research", label: "Research", description: "Academic papers" },
];
