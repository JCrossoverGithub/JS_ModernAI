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
  folderId?: string;
  sortOrder: number;
}

export interface Folder {
  id: string;
  name: string;
  expanded: boolean;
  sortOrder: number;
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

// ── Research source configuration ──
export interface ResearchSource {
  id: string;
  name: string;
  category: "academic" | "discovery";
  description: string;
}

export const RESEARCH_SOURCES: ResearchSource[] = [
  { id: "semantic_scholar", name: "Semantic Scholar", category: "academic", description: "200M+ papers with citation data" },
  { id: "arxiv", name: "arXiv", category: "academic", description: "Open-access preprints" },
  { id: "openalex", name: "OpenAlex", category: "academic", description: "Open catalog of scholarly works" },
  { id: "pubmed", name: "PubMed", category: "academic", description: "Biomedical & life sciences" },
  { id: "core", name: "CORE", category: "academic", description: "200M+ open-access articles" },
  { id: "crossref", name: "CrossRef", category: "academic", description: "DOI metadata & references" },
  { id: "europe_pmc", name: "Europe PMC", category: "academic", description: "Biomedical full-text access" },
  { id: "papers_with_code", name: "Papers With Code", category: "discovery", description: "ML papers + implementations" },
  { id: "duckduckgo", name: "DuckDuckGo", category: "discovery", description: "General web search" },
  { id: "wikipedia", name: "Wikipedia", category: "discovery", description: "Encyclopedia background" },
];

export const DEFAULT_RESEARCH_SOURCES = ["semantic_scholar", "arxiv", "openalex"];
