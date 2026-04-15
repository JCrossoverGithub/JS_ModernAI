# Architecture Decision Records

This document records the key architectural decisions made in the LibraryAI project and the reasoning behind them.

---

## ADR-001: Three-Tier Architecture (React → .NET → Python)

**Status:** Accepted

**Context:**  
The original `js_ai.py` is a Python CLI tool that combines UI (terminal I/O), business logic (command routing), and AI processing (LLM, embeddings, vector search) in a single file. We needed to make this accessible via a web browser with user authentication.

**Decision:**  
Split the application into three services:
1. **React frontend** — UI layer
2. **ASP.NET Core backend** — API gateway, auth, session management
3. **Python FastAPI service** — AI engine

**Rationale:**
- Python's AI ecosystem (LangChain, HuggingFace, ChromaDB) is significantly more mature than C# equivalents. Rewriting in C# would lose features and incur maintenance burden.
- .NET excels at enterprise concerns: authentication, authorization, CORS, and HTTP pipeline management.
- Separating the services allows independent scaling — the AI service can run on a GPU machine while the .NET backend runs on a cheaper VM.

**Alternatives considered:**
- **React → Python only (2-tier):** Simpler, but loses .NET's auth ecosystem and enterprise tooling.
- **Pure .NET rewrite:** Would require Semantic Kernel or LangChain.NET, which are less mature. ChromaDB's C# client is limited.

---

## ADR-002: SignalR for Chat Streaming

**Status:** Accepted

**Context:**  
The AI generates responses token-by-token. Users expect a "typing" effect where text appears progressively, like ChatGPT.

**Decision:**  
Use SignalR WebSockets between React and .NET, and Server-Sent Events (SSE) between .NET and Python.

**Rationale:**
- **SignalR** provides automatic reconnection, connection management, and fallback transports (WebSocket → SSE → Long Polling). It integrates natively with ASP.NET Core auth.
- **SSE** on the Python side is simpler than WebSockets for a unidirectional stream and works naturally with FastAPI's `StreamingResponse`.
- The .NET backend reads the SSE stream from Python and forwards each event to the React client via SignalR, acting as a bridge.

**Alternatives considered:**
- **WebSockets end-to-end:** More complex on the Python side with no real benefit (the stream is unidirectional).
- **HTTP polling:** Would lose the real-time streaming effect and add latency.
- **gRPC streaming:** Overkill for this use case and adds complexity to the frontend.

---

## ADR-003: JWT Bearer Tokens for Authentication

**Status:** Accepted

**Context:**  
The application needs user authentication. Each user's memory, facts, and conversation history must be isolated.

**Decision:**  
Use ASP.NET Core Identity with JWT Bearer tokens. Tokens are issued on login/register and sent with every request.

**Rationale:**
- **Stateless auth** — no server-side session storage needed. Scales horizontally.
- **SignalR compatibility** — JWT is passed via query string for WebSocket upgrade requests (standard pattern).
- **Identity framework** — handles password hashing, user management, and can be extended with roles/claims.

**Token details:**
- Algorithm: HMAC-SHA256
- Lifetime: 24 hours (configurable)
- Claims: `sub`, `email`, `nameidentifier`, `name`

---

## ADR-004: ChromaDB for Vector Storage

**Status:** Accepted (inherited from js_ai.py)

**Context:**  
The application needs to store document embeddings and perform similarity search.

**Decision:**  
Keep ChromaDB as the vector database, running as an embedded (in-process) database within the Python service.

**Rationale:**
- Already used by the original `js_ai.py` — no migration needed.
- Embedded mode (no separate server) simplifies deployment.
- Persistent directory means data survives restarts.

**Trade-offs:**
- ChromaDB embedded only supports single-process access. Don't run multiple Python service instances against the same `chroma_db/` directory.
- For production multi-instance deployment, switch to ChromaDB client-server mode or Qdrant/Weaviate.

---

## ADR-005: SQLite for User Authentication Data

**Status:** Accepted

**Context:**  
User accounts need persistent storage. The authentication data (emails, hashed passwords) is relational.

**Decision:**  
Use SQLite via Entity Framework Core.

**Rationale:**
- Zero-configuration — no database server to install or manage.
- Sufficient for the expected user count (single-digit to hundreds).
- Auto-created on first startup via `EnsureCreated()`.

**Trade-offs:**
- SQLite has limited concurrent write support. If deploying at scale, migrate to PostgreSQL.
- The `EnsureCreated()` approach doesn't support schema migrations. For future schema changes, switch to `Database.Migrate()` with EF migrations.

---

## ADR-006: Per-User Memory Isolation

**Status:** Accepted

**Context:**  
Multiple users share the same ChromaDB instance. Users must not see each other's facts or conversation history.

**Decision:**  
Tag all memory entries with a `user_id` metadata field. All queries filter by `where={"user_id": uid}`.

**Rationale:**
- Simple to implement with ChromaDB's metadata filtering.
- No need for separate collections per user (would not scale).

**Note:**
- The document library is **shared** across all users. This is intentional — uploaded documents form a shared knowledge base.
- If per-user document isolation is needed in the future, add `user_id` metadata to document chunks too.

---

## ADR-007: Dark Theme UI

**Status:** Accepted

**Context:**  
AI chat applications are typically used for extended periods. A dark theme reduces eye strain.

**Decision:**  
All-dark Tailwind CSS theme using `gray-950` (background), `gray-900` (surfaces), `gray-800` (inputs).

**Rationale:**
- Matches user expectations from ChatGPT, Claude, and similar tools.
- Tailwind v4 makes theming straightforward with utility classes.
- No light mode toggle — keeping it simple.
