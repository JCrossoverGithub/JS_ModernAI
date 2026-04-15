# LibraryAI

A three-tier web application that turns a local Python AI assistant into a full-stack product with user authentication, real-time streaming chat, document management, and persistent memory.

**Original tool:** [`js_ai.py`](js_ai.py) — a terminal-based RAG chatbot using LangChain, Ollama, and ChromaDB.  
**What this project does:** Wraps that tool in a React frontend + .NET API gateway + Python microservice, adding JWT auth, SignalR streaming, multi-user support, and Docker deployment.

---

## Architecture

```
┌──────────────┐  WebSocket   ┌──────────────────┐  HTTP/SSE  ┌───────────────────┐
│  React UI    │─────────────▶│  .NET Backend     │───────────▶│  Python FastAPI    │
│  (Vite/TS)   │◀─────────────│  (Auth + SignalR) │◀───────────│  (LangChain/AI)   │
│  port 5173   │  SignalR     │  port 5000        │  SSE       │  port 8100         │
└──────────────┘              └──────────────────┘            └───────────────────┘
                                      │                                │
                                 SQLite (users)              ┌─────────┴──────────┐
                                                             │   ChromaDB          │
                                                             │   Ollama LLM        │
                                                             │   HuggingFace Embed  │
                                                             │   DuckDuckGo Search  │
                                                             └────────────────────┘
```

| Layer        | Technology Stack                                              |
| ------------ | ------------------------------------------------------------- |
| **Frontend** | React 18, TypeScript, Vite 6, Tailwind CSS v4, SignalR client |
| **Backend**  | ASP.NET Core 8, JWT Bearer, Identity, SignalR hub, EF Core SQLite |
| **AI Service** | FastAPI, LangChain, HuggingFace `bge-large-en-v1.5`, ChromaDB, Ollama, DuckDuckGo |

### Data Flow

1. User types a message in the React chat UI
2. React sends it via **SignalR WebSocket** to the .NET backend
3. .NET authenticates the JWT, extracts the user ID, and POSTs to Python's `/chat/stream`
4. Python runs the **RAG pipeline**: rephrase → retrieve (memory + documents) → stream LLM tokens
5. Python emits **Server-Sent Events (SSE)** back to .NET
6. .NET forwards each event to React via **SignalR's `ReceiveEvent`** callback
7. React renders tokens as they arrive (live typing effect)

---

## Prerequisites

| Tool | Version | Required For |
| ---- | ------- | ------------ |
| [Node.js](https://nodejs.org/) | 20+ | Frontend |
| [Python](https://www.python.org/) | 3.11+ | AI Service |
| [.NET SDK](https://dotnet.microsoft.com/download/dotnet/8.0) | 8.0+ | Backend |
| [Ollama](https://ollama.ai/) | Latest | LLM inference |
| [Docker](https://www.docker.com/) | Latest | Optional (containerized deployment) |
| NVIDIA GPU + CUDA | Any | Optional (for fast embeddings) |

---

## Quick Start

### Option A: Local Development (3 terminals)

```powershell
# Terminal 1 — Ollama
ollama serve
ollama pull qwen3-coder:30b

# Terminal 2 — Python AI Service
cd ai-service
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8100

# Terminal 3 — .NET Backend
cd backend
dotnet run --urls http://localhost:5000

# Terminal 4 — React Frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Register an account, upload a document, and start chatting.

### Option B: Docker Compose

```powershell
# Ollama must listen on all interfaces for Docker to reach it
$env:OLLAMA_HOST="0.0.0.0"
ollama serve

# Start all services
docker compose up --build
```

Open **http://localhost:3000**.

---

## Features

| Feature | Description |
| ------- | ----------- |
| **Streaming Chat** | Token-by-token response streaming (like ChatGPT) via SignalR |
| **RAG Pipeline** | Retrieval-Augmented Generation over your uploaded documents |
| **4 Search Modes** | Default (library + memory), Strict (library only), Chat (memory only), Web (DuckDuckGo) |
| **Auto Fallback** | If the library can't answer, automatically tries a web search |
| **Document Library** | Upload PDF, TXT, DOCX files — auto-chunked and embedded |
| **Personal Memory** | Save facts about yourself; the AI remembers across sessions |
| **Short-term Buffer** | Last 6 exchanges used for context (pronoun resolution, follow-ups) |
| **JWT Authentication** | Register/login with email + password; per-user data isolation |
| **Dark Mode UI** | Clean dark-themed chat interface with Tailwind CSS |

---

## Project Structure

```
JS_ModernAI/
├── ai-service/                 # Python FastAPI — AI engine
│   ├── library_ai.py           #   Core LibraryAI class (refactored from js_ai.py)
│   ├── main.py                 #   FastAPI endpoints + SSE streaming
│   ├── requirements.txt        #   Python dependencies
│   ├── Dockerfile
│   └── README.md               #   ← Full AI service documentation
│
├── backend/                    # ASP.NET Core 8 — API gateway
│   ├── Program.cs              #   App configuration (DI, auth, CORS, SignalR)
│   ├── Controllers/
│   │   ├── AuthController.cs   #     Register + login (JWT issuance)
│   │   ├── MemoryController.cs #     Facts CRUD + wipe/clear
│   │   └── DocumentsController.cs #  Upload, list, delete documents
│   ├── Hubs/
│   │   └── ChatHub.cs          #   SignalR hub (bridges SSE → WebSocket)
│   ├── Services/
│   │   └── PythonAIService.cs  #   Typed HttpClient to Python FastAPI
│   ├── Data/
│   │   └── AppDbContext.cs     #   EF Core DbContext (Identity tables)
│   ├── Models/                 #   Request/response DTOs
│   ├── appsettings.json        #   Configuration (JWT, CORS, Python URL)
│   ├── Dockerfile
│   └── README.md               #   ← Full backend documentation
│
├── frontend/                   # React + Vite + TypeScript
│   ├── src/
│   │   ├── App.tsx             #   Root component (auth gate)
│   │   ├── components/
│   │   │   ├── LoginPage.tsx   #     Login / register form
│   │   │   ├── ChatWindow.tsx  #     SignalR streaming chat
│   │   │   └── Sidebar.tsx     #     Mode, facts, documents tabs
│   │   ├── context/
│   │   │   └── AuthContext.tsx  #    JWT + user state management
│   │   ├── services/
│   │   │   └── api.ts          #    REST client (fetch + Bearer token)
│   │   └── types.ts            #    Shared TypeScript interfaces
│   ├── nginx.conf              #   Production reverse proxy config
│   ├── Dockerfile
│   └── README.md               #   ← Full frontend documentation
│
├── docs/
│   └── TUTORIAL.md             # Step-by-step beginner tutorial
│
├── docker-compose.yml          # Orchestrates all 3 services
├── .gitignore
├── js_ai.py                    # Original CLI tool (reference)
├── pre-info.txt                # Architecture planning notes
└── README.md                   # ← You are here
```

---

## Documentation Index

| Document | Description |
| -------- | ----------- |
| [**docs/TUTORIAL.md**](docs/TUTORIAL.md) | Step-by-step beginner tutorial (install → first chat) |
| [**ai-service/README.md**](ai-service/README.md) | AI service: API reference, streaming protocol, RAG pipeline, configuration |
| [**backend/README.md**](backend/README.md) | .NET backend: auth flow, SignalR hub, service layer, database schema |
| [**frontend/README.md**](frontend/README.md) | React frontend: component guide, SignalR integration, state management |

---

## API Quick Reference

### Authentication (no token)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/auth/register` | Create account → JWT |
| POST | `/api/auth/login` | Sign in → JWT |

### Memory (JWT required)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/memory/facts` | Save a fact |
| GET | `/api/memory/facts` | List all facts |
| DELETE | `/api/memory/facts?keyword=` | Delete facts by keyword |
| POST | `/api/memory/wipe` | Delete all memory (irreversible) |
| POST | `/api/memory/clear-buffer` | Clear short-term buffer only |

### Documents (JWT required)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/api/documents/upload` | Upload PDF/TXT/DOCX (multipart) |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/{name}` | Remove a document |

### Chat (SignalR WebSocket)

Connect: `/hubs/chat?access_token=<jwt>`  
Send: `SendMessage(query, mode)` — modes: `default`, `strict`, `chat`, `web`  
Receive: `ReceiveEvent(json)` — event types: `debug`, `status`, `token`, `sources`, `done`

---

## Configuration

### Environment Variables

| Variable | Service | Default | Description |
| -------- | ------- | ------- | ----------- |
| `CHROMA_DIR` | Python | `./chroma_db` | ChromaDB storage path |
| `OLLAMA_BASE_URL` | Python | `http://localhost:11434` | Ollama server URL |
| `CORS_ORIGIN` | Python | `http://localhost:5000` | Allowed CORS origin |
| `UPLOAD_DIR` | Python | `./uploads` | Temp upload directory |
| `Jwt__Key` | .NET | *(placeholder)* | JWT signing key (**change in prod**) |
| `Jwt__ExpirationMinutes` | .NET | `1440` | Token lifetime (24h) |
| `PythonAI__BaseUrl` | .NET | `http://localhost:8100` | Python service URL |
| `AllowedOrigins` | .NET | `http://localhost:5173` | CORS origins (comma-separated) |

---

## Security Notes

- **Change the JWT key** in `backend/appsettings.json` before any deployment beyond localhost
- All memory and document operations are scoped to the authenticated user's ID
- File uploads are UUID-prefixed to prevent path traversal
- Temporary upload files are deleted after ingestion
- The Python AI service should not be exposed directly to the internet — only the .NET backend should be public-facing

---

## License

This project is provided as-is for educational and personal use.
