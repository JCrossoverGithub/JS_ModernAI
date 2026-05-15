# JS_ModernAI

A full-stack AI application built to match the architecture of the **LLM Engineer's Handbook**, combining a React chat UI, ASP.NET Core API gateway, and a Python AI service that supports RAG, fine-tuning (SFT + DPO), evaluation, and monitored inference.

---

## Architecture

```
┌──────────────┐  WebSocket   ┌──────────────────┐  HTTP/SSE  ┌─────────────────────┐
│  React UI    │─────────────▶│  .NET Backend     │───────────▶│  Python FastAPI      │
│  (Vite/TS)   │◀─────────────│  (Auth + SignalR) │◀───────────│  (LibraryAI engine) │
│  port 5173   │  SignalR     │  port 5000        │  SSE       │  port 8100           │
└──────────────┘              └──────────────────┘            └─────────────────────┘
                                      │                                │
                                 SQLite (users)          ┌─────────────┴────────────┐
                                                         │  Qdrant  (port 6333)      │
                                                         │  MongoDB (port 27017)     │
                                                         │  Ollama / HF pipeline     │
                                                         │  bge-large-en-v1.5 embed  │
                                                         └──────────────────────────┘
```

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS v4, SignalR client |
| **Backend** | ASP.NET Core 8, JWT Bearer, SignalR, EF Core SQLite |
| **AI Service** | FastAPI, LangChain, HuggingFace, Qdrant, MongoDB, Ollama, ZenML |
| **Vector DB** | Qdrant 1.12 — cosine distance, 1024-dim, per-user filter |
| **Data Warehouse** | MongoDB 7 (`llm_twin` database, `raw_documents` collection) |
| **Embeddings** | `BAAI/bge-large-en-v1.5` — 1024-dim, normalized, CUDA |
| **LLM (default)** | Ollama `mannix/llama3.1-8b-abliterated` |
| **LLM (fine-tuned)** | `meta-llama/Llama-3.1-8B-Instruct` + QLoRA SFT + DPO adapter |

### Data Flow

1. User sends a message via **SignalR WebSocket** → .NET backend
2. .NET authenticates the JWT, extracts `user_id`, POSTs to Python `/chat/stream`
3. Python: rephrase query → retrieve Qdrant (memory + documents) → stream LLM tokens
4. Python emits **Server-Sent Events (SSE)** → .NET → **SignalR `ReceiveEvent`** → React
5. Uploaded documents are written to **MongoDB** (for training pipelines) and embedded into **Qdrant** (for RAG retrieval)

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Node.js](https://nodejs.org/) | 20+ | Frontend |
| [Python](https://www.python.org/) | 3.11 | AI Service |
| [.NET SDK](https://dotnet.microsoft.com/download/dotnet/8.0) | 8.0+ | Backend |
| [Poetry](https://python-poetry.org/) | 1.8+ | Python dependency management |
| [Ollama](https://ollama.ai/) | Latest | LLM inference |
| [Docker + Compose](https://www.docker.com/) | Latest | Qdrant + MongoDB (required) |
| NVIDIA GPU + CUDA 12.6 | RTX 3070 Ti / 8 GB+ | Embeddings + fine-tuning |

---

## Quick Start

### 1. Start infrastructure (Qdrant + MongoDB)

```powershell
docker compose up qdrant mongodb -d
```

### 2. Start Ollama

```powershell
ollama serve
ollama pull mannix/llama3.1-8b-abliterated
```

### 3. Python AI Service

```powershell
cd ai-service

# First time: install with Poetry
pip install poetry
poetry install

# Copy and edit environment (see Configuration section)
Copy-Item .env.example .env

# Start the service
poetry run uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

### 4. .NET Backend

```powershell
cd backend
dotnet run --urls http://localhost:5000
```

### 5. React Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — register an account, upload a document, and start chatting.

---

### Option B: Full Docker Compose

```powershell
# Ollama must be reachable from Docker
$env:OLLAMA_HOST = "0.0.0.0"
ollama serve

docker compose up --build
```

Open **http://localhost:3000**.

> **GPU passthrough** is configured in `docker-compose.yml` under the `ai-service` deploy block. Requires NVIDIA Container Toolkit.

---

## Configuration

All Python settings live in `ai-service/.env` (loaded via pydantic-settings `config.py`).

### Core settings

```env
# --- Inference ---
INFERENCE_MODE=ollama           # "ollama" or "huggingface"
OLLAMA_BASE_URL=http://localhost:11434

# --- Vector DB ---
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=library_ai
QDRANT_MEMORY_COLLECTION=chat_memory

# --- Data Warehouse ---
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=llm_twin

# --- Embeddings ---
EMBED_DEVICE=cuda               # "cpu" if no GPU

# --- CORS ---
CORS_ORIGIN=http://localhost:5000
```

### Switching to your fine-tuned model

After running the training pipeline, point the inference engine at the DPO adapter:

```env
INFERENCE_MODE=huggingface
BASE_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
FINETUNED_MODEL_ID=./models/dpo/final_adapter
HUGGINGFACE_ACCESS_TOKEN=hf_...
```

### Training (optional)

```env
SFT_OUTPUT_DIR=./models/sft
SFT_MAX_SAMPLES=1000
SFT_EPOCHS=3

DPO_OUTPUT_DIR=./models/dpo
DPO_MAX_SAMPLES=500
DPO_BETA=0.1
```

### Opik monitoring (optional)

```env
OPIK_API_KEY=your_key
OPIK_PROJECT_NAME=llm-twin
```

### .NET backend (`backend/appsettings.json`)

```json
"Jwt": { "Key": "CHANGE_ME_32_CHARS_MIN", "ExpirationMinutes": 1440 },
"PythonAI": { "BaseUrl": "http://localhost:8100" },
"AllowedOrigins": "http://localhost:5173"
```

---

## Training Pipelines

All pipelines run from `ai-service/` with Poetry active.

### Feature pipeline (index documents into Qdrant)

```powershell
cd ai-service
poetry run python -m pipelines.feature_pipeline --author-id global
```

### SFT — Supervised Fine-Tuning

```powershell
# With ZenML tracking
poetry run python -m pipelines.training_pipeline --finetuning-type sft

# Without ZenML (faster, local only)
poetry run python -m pipelines.training_pipeline --finetuning-type sft --no-zenml
```

Outputs adapter to `./models/sft/final_adapter`.

### DPO — Preference Alignment

```powershell
poetry run python -m pipelines.training_pipeline \
  --finetuning-type dpo \
  --model-id ./models/sft/final_adapter \
  --no-zenml
```

Outputs adapter to `./models/dpo/final_adapter`.

### Evaluation

```powershell
# RAG quality (RAGAS: context precision, recall, faithfulness, relevancy)
poetry run python -m pipelines.evaluation_pipeline --eval-type rag --no-zenml

# Generation quality (ROUGE + BERTScore) against the fine-tuned model
poetry run python -m pipelines.evaluation_pipeline \
  --eval-type gen \
  --model-id ./models/dpo/final_adapter \
  --no-zenml

# Both at once
poetry run python -m pipelines.evaluation_pipeline --eval-type all --no-zenml
```

Results written to `./evals/rag_eval.json` and `./evals/gen_eval.json`.

---

## Project Structure

```
JS_ModernAI/
├── ai-service/
│   ├── library_ai.py               # Core LibraryAI class (RAG, memory, ingest)
│   ├── main.py                     # FastAPI endpoints + SSE streaming
│   ├── config.py                   # pydantic-settings — all env config
│   ├── pyproject.toml              # Poetry dependencies
│   ├── templates/                  # Jinja2 prompt templates
│   │   ├── rephrase.j2
│   │   ├── qa.j2
│   │   ├── research_query.j2
│   │   └── research_synthesis.j2
│   ├── db/
│   │   └── mongo.py                # MongoDB client singleton
│   ├── models/
│   │   ├── documents.py            # RawDocument, CleanedDocument, DocumentChunk
│   │   └── conversations.py        # ChatMessage, RetrievedContext
│   ├── steps/
│   │   ├── data_collection/
│   │   │   └── loaders.py          # MongoDB + file loaders
│   │   ├── feature_engineering/
│   │   │   ├── cleaning.py         # clean_text()
│   │   │   ├── chunking.py         # RecursiveCharacterTextSplitter
│   │   │   ├── embedding.py        # bge-large-en-v1.5 on CUDA
│   │   │   └── qdrant_loader.py    # Qdrant upsert + collection init
│   │   ├── training/
│   │   │   ├── sft.py              # build_instruction_dataset + run_sft (QLoRA)
│   │   │   ├── dpo.py              # build_preference_dataset + run_dpo
│   │   │   └── hub.py              # push_to_huggingface (local or HF Hub)
│   │   └── evaluation/
│   │       ├── rag_eval.py         # RAGAS metrics
│   │       └── gen_eval.py         # ROUGE + BERTScore
│   └── pipelines/
│       ├── feature_pipeline.py     # ZenML feature pipeline
│       ├── training_pipeline.py    # ZenML SFT + DPO pipelines
│       └── evaluation_pipeline.py  # ZenML eval pipeline
│
├── backend/                        # ASP.NET Core 8
│   ├── Controllers/
│   │   ├── AuthController.cs       # Register + login (JWT)
│   │   ├── MemoryController.cs     # Facts CRUD
│   │   └── DocumentsController.cs  # Upload / list / delete
│   ├── Hubs/ChatHub.cs             # SignalR — bridges SSE → WebSocket
│   └── Services/PythonAIService.cs # HttpClient to Python
│
├── frontend/                       # React 18 + Vite + TypeScript
│   └── src/
│       ├── components/             # ChatWindow, Sidebar, LoginPage, ...
│       ├── context/AuthContext.tsx
│       ├── hooks/                  # useConversations, useSignalR
│       └── services/api.ts
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DOCKER.md
│   └── TUTORIAL.md
└── docker-compose.yml              # frontend, backend, ai-service, qdrant, mongodb
```

---

## API Quick Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account → JWT |
| POST | `/api/auth/login` | Sign in → JWT |

### Memory (JWT required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/facts` | Save a fact |
| GET | `/api/memory/facts` | List all facts |
| DELETE | `/api/memory/facts?keyword=` | Delete facts by keyword |
| POST | `/api/memory/wipe` | Delete all memory (irreversible) |
| POST | `/api/memory/clear-buffer` | Clear short-term buffer only |

### Documents (JWT required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload PDF / TXT / DOCX (multipart) |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{name}` | Remove a document |

### Chat (SignalR WebSocket)

```
Connect:  /hubs/chat?access_token=<jwt>
Send:     SendMessage(query, mode)
Receive:  ReceiveEvent(json)
```

**Modes:** `default` (library + memory), `strict` (library only), `chat` (memory only), `web` (DuckDuckGo)

**Event types:** `debug` (rephrased query), `status`, `token` (streamed answer), `sources`, `done`

---

## Features

| Feature | Description |
|---------|-------------|
| **Streaming Chat** | Token-by-token streaming via SignalR |
| **RAG Pipeline** | Qdrant retrieval over uploaded documents, per-user isolated |
| **Personal Memory** | Persistent facts + chat history stored in Qdrant |
| **Academic Research** | Concurrent search across Semantic Scholar, arXiv, OpenAlex, PubMed, CORE, CrossRef + LLM synthesis |
| **Document Library** | PDF, TXT, DOCX — cleaned, chunked (512/64), embedded (1024-dim) |
| **Fine-tuning** | SFT (QLoRA) + DPO preference alignment on your documents |
| **Evaluation** | RAGAS (RAG quality) + ROUGE + BERTScore (generation quality) |
| **Opik Tracing** | Per-request trace with retrieval + generation spans (optional) |
| **Inference Modes** | Ollama (default, zero startup) or local HF pipeline (fine-tuned adapter) |
| **JWT Auth** | Register/login; all data is scoped to the authenticated user |

---

## Security Notes

- **Change `Jwt__Key`** in `backend/appsettings.json` before exposing to any network
- All Qdrant queries are filtered by `user_id` — users cannot access each other's data
- File uploads are UUID-prefixed to prevent path traversal
- Temporary upload files are deleted after ingestion
- The Python AI service should not be publicly exposed — route all traffic through the .NET backend

---

## License

Provided for educational and personal use.


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
