# Frontend — React + TypeScript + Tailwind CSS

The frontend is a single-page application that provides a chat interface, document management, and memory controls for LibraryAI.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Components](#components)
- [Services & State](#services--state)
- [SignalR Integration](#signalr-integration)
- [Styling](#styling)
- [Development](#development)
- [Production Build](#production-build)
- [Docker](#docker)

---

## Overview

| Component       | Technology                                 |
| --------------- | ------------------------------------------ |
| Framework       | React 18 (with hooks, no class components) |
| Language        | TypeScript (strict mode)                   |
| Build Tool      | Vite 6                                     |
| CSS             | Tailwind CSS v4 (Vite plugin)              |
| Real-time       | `@microsoft/signalr` WebSocket client      |
| Routing         | React Router DOM v6                        |
| HTTP Client     | Native `fetch` API                         |

---

## Architecture

```
src/
├── main.tsx              Entry point — React root, BrowserRouter
├── App.tsx               Root component — routes auth vs. main UI
├── index.css             Tailwind import
├── types.ts              Shared TypeScript interfaces
├── vite-env.d.ts         Vite type declarations
│
├── context/
│   └── AuthContext.tsx    Global auth state (JWT + user info)
│
├── services/
│   └── api.ts            REST client — all fetch calls to .NET
│
└── components/
    ├── LoginPage.tsx      Login / register form
    ├── ChatWindow.tsx     Main chat area with SignalR streaming
    └── Sidebar.tsx        Mode selector, facts, documents, actions
```

---

## Components

### `App.tsx`

The root component. Wraps everything in `AuthProvider`. If no user is authenticated, renders `LoginPage`. Otherwise, renders the main layout: `Sidebar` + `ChatWindow` side by side.

**State:** `mode` — the current search mode (`default`, `strict`, `chat`, `web`), passed to both Sidebar and ChatWindow.

### `LoginPage.tsx`

A dual-purpose form for login and registration.

| Feature | Detail |
| ------- | ------ |
| Toggle | "Sign in" / "Create account" switch |
| Fields | Username (register only), Email, Password |
| Validation | HTML5 `required`, `minLength={6}`, `type="email"` |
| Error display | Red banner from API error messages |
| On success | Calls `setUser()` from AuthContext, which stores token in `localStorage` |

### `ChatWindow.tsx`

The main chat area. Manages the SignalR connection lifecycle and renders the message stream.

**Connection lifecycle:**
1. On mount, reads the JWT from `localStorage`
2. Builds a `HubConnection` to `/hubs/chat` with the token
3. Registers a `ReceiveEvent` handler that dispatches by event type
4. On unmount, stops the connection

**Message handling:**

| Event Type | Action |
| ---------- | ------ |
| `token` | Append to the last assistant message (or create a new one). This is how streaming works — each token event extends the growing response. |
| `sources` | Attach the sources array to the last assistant message |
| `status` | Render as a gray italic status line |
| `done` | Re-enable the input and stop the loading indicator |

**Input behavior:**
- Enter sends (Shift+Enter for newline)
- Textarea auto-resizes up to 150px
- Disabled while streaming

### `Sidebar.tsx`

A 288px-wide sidebar with three tabs:

**Mode tab:**
- Four mode buttons with visual selection state
- "Clear Buffer" button — clears the short-term conversation memory
- "Wipe All Memory" button — with a `confirm()` dialog (destructive action)

**Facts tab:**
- Text input + "+" button to save a new fact
- List of existing facts with delete (✕) buttons
- Auto-fetches facts when the tab is opened

**Library (Docs) tab:**
- File upload zone (`.pdf`, `.txt`, `.docx`, `.doc`)
- List of uploaded documents with delete buttons
- Auto-fetches documents when the tab is opened

---

## Services & State

### `api.ts`

All HTTP communication with the .NET backend. Every authenticated request reads the JWT from `localStorage` and attaches it as a `Bearer` token.

**Functions:**

| Function | HTTP | Endpoint | Auth |
| -------- | ---- | -------- | ---- |
| `login(email, password)` | POST | `/api/auth/login` | No |
| `register(username, email, password)` | POST | `/api/auth/register` | No |
| `saveFact(fact)` | POST | `/api/memory/facts` | Yes |
| `listFacts()` | GET | `/api/memory/facts` | Yes |
| `deleteFact(keyword)` | DELETE | `/api/memory/facts?keyword=` | Yes |
| `wipeMemory()` | POST | `/api/memory/wipe` | Yes |
| `clearBuffer()` | POST | `/api/memory/clear-buffer` | Yes |
| `uploadDocument(file)` | POST | `/api/documents/upload` | Yes |
| `listDocuments()` | GET | `/api/documents` | Yes |
| `removeDocument(filename)` | DELETE | `/api/documents/{name}` | Yes |

### `AuthContext.tsx`

React Context providing global authentication state.

**Interface:**
```typescript
{
  user: AuthResponse | null;   // { token, username, userId }
  setUser: (user) => void;     // Store user + persist to localStorage
  logout: () => void;          // Clear user + remove from localStorage
}
```

**Persistence:** On load, reads `user` and `token` from `localStorage`. On `setUser`, writes to both `localStorage` keys. On `logout`, removes both.

### `types.ts`

Shared TypeScript interfaces:

```typescript
interface AuthResponse {
  token: string;
  username: string;
  userId: string;
}

interface ChatEvent {
  type: "status" | "debug" | "token" | "sources" | "done";
  message?: string;
  standalone_query?: string;
  content?: string;
  sources?: string[];
}

interface Message {
  id: string;
  role: "user" | "assistant" | "status";
  content: string;
  sources?: string[];
  mode?: string;
}
```

---

## SignalR Integration

The chat uses SignalR for real-time bidirectional communication. This is necessary because the AI generates tokens one at a time — a regular HTTP request would need to wait for the full response.

### Connection Setup

```typescript
const conn = new HubConnectionBuilder()
  .withUrl("/hubs/chat", {
    accessTokenFactory: () => localStorage.getItem("token")!,
  })
  .withAutomaticReconnect()
  .configureLogging(LogLevel.Warning)
  .build();
```

### Sending Messages

```typescript
await connection.invoke("SendMessage", query, mode);
```

### Receiving Events

```typescript
conn.on("ReceiveEvent", (jsonStr: string) => {
  const event: ChatEvent = JSON.parse(jsonStr);
  // Handle based on event.type
});
```

### Token Streaming Pattern

The key pattern for streaming: when a `token` event arrives, the component checks if the last message has `role: "assistant"`. If yes, it appends the token content. If no, it creates a new assistant message. This produces the character-by-character typing effect.

---

## Styling

The app uses **Tailwind CSS v4** via the Vite plugin. The entire theme is dark mode:

- Background: `bg-gray-950` (near-black)
- Surfaces: `bg-gray-900` (sidebar), `bg-gray-800` (inputs, cards)
- Borders: `border-gray-800` / `border-gray-700`
- Accent: `bg-blue-600` (buttons, active states)
- Text: `text-gray-100` (primary), `text-gray-400/500` (secondary)

No custom CSS files — all styling is done with Tailwind utility classes.

---

## Development

**Prerequisites:** Node.js 20+, .NET backend running on port 5000.

```bash
cd frontend

# Install dependencies
npm install

# Start dev server with HMR
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies:
- `/api/*` → `http://localhost:5000` (REST endpoints)
- `/hubs/*` → `http://localhost:5000` (WebSocket upgrade)

This proxy is configured in `vite.config.ts` so there are no CORS issues during development.

---

## Production Build

```bash
npm run build
```

Outputs to `dist/`. The build is a static SPA that needs a reverse proxy (nginx) to:
1. Serve the static files
2. Route `/api/*` and `/hubs/*` to the .NET backend

The included `nginx.conf` handles this.

---

## Docker

The Dockerfile uses a multi-stage build:
1. **Build stage:** `node:20-slim` — installs deps, runs `npm run build`
2. **Runtime stage:** `nginx:alpine` — copies `dist/` and `nginx.conf`

```bash
docker build -t libraryai-frontend ./frontend
docker run -p 3000:80 libraryai-frontend
```

The nginx config proxies `/api` and `/hubs` to `http://backend:5000` (Docker service name).
