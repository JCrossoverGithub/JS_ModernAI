# AI Service — Python FastAPI

The AI service is the intelligence layer of LibraryAI. It wraps the core `LibraryAI` class (originally `js_ai.py`) in a FastAPI web server, exposing the RAG pipeline, memory management, and document ingestion over HTTP.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Core Classes](#core-classes)
- [Streaming Protocol](#streaming-protocol)
- [Running Locally](#running-locally)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)

---

## Overview

| Component          | Technology                                  |
| ------------------ | ------------------------------------------- |
| Web Framework      | FastAPI 0.115+                              |
| LLM                | Ollama (`qwen3-coder:30b`) via LangChain    |
| Embeddings         | HuggingFace `BAAI/bge-large-en-v1.5` (CUDA) |
| Vector Database    | ChromaDB (local persistent)                 |
| Web Search         | DuckDuckGo (via LangChain community tools)  |
| Document Loaders   | PDF (PyPDF), TXT, DOCX (docx2txt)           |

The service is **stateful** — it holds a singleton `LibraryAI` instance in memory that maintains per-user short-term chat buffers and connections to ChromaDB.

---

## Architecture

```
                    ┌─────────────────────────────┐
 .NET Backend ────▶ │  FastAPI (main.py)           │
   (HTTP/SSE)       │  ├─ POST /chat/stream        │──── SSE stream
                    │  ├─ POST /memory/save         │
                    │  ├─ GET  /memory/facts/{uid}  │
                    │  ├─ DELETE /memory/facts/{uid} │
                    │  ├─ POST /memory/wipe          │
                    │  ├─ POST /memory/clear-buffer  │
                    │  ├─ POST /documents/upload      │
                    │  ├─ GET  /documents             │
                    │  └─ DELETE /documents/{name}    │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  LibraryAI (library_ai.py)       │
                    │  ├─ Rephrase chain (LangChain)   │
                    │  ├─ QA chain (LangChain)         │
                    │  ├─ ChromaDB (vector_db)         │
                    │  ├─ ChromaDB (memory_db)         │
                    │  ├─ DuckDuckGo web search        │
                    │  └─ Per-user chat buffers         │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         Ollama LLM      ChromaDB dir     HuggingFace
        (port 11434)    (./chroma_db)     Embeddings
```

---

## Configuration

All configuration is via **environment variables** with sensible defaults:

| Variable          | Default                    | Description                                                  |
| ----------------- | -------------------------- | ------------------------------------------------------------ |
| `CHROMA_DIR`      | `./chroma_db`              | Filesystem path for ChromaDB persistence                     |
| `OLLAMA_BASE_URL` | `http://localhost:11434`   | URL of the Ollama server                                     |
| `CORS_ORIGIN`     | `http://localhost:5000`    | Allowed CORS origin (the .NET backend URL)                   |
| `UPLOAD_DIR`      | `./uploads`                | Temporary directory for uploaded files before ingestion       |

---

## API Reference

### `GET /health`

Health check. Returns `{"status": "ok"}` when the server is running. The AI model is not loaded until the first request that requires it.

### `POST /chat/stream`

Streams a RAG response as **Server-Sent Events (SSE)**.

**Request body:**
```json
{
  "query": "What is the capital of France?",
  "mode": "default",
  "user_id": "user-abc-123"
}
```

| Field     | Type   | Required | Description                                            |
| --------- | ------ | -------- | ------------------------------------------------------ |
| `query`   | string | Yes      | The user's question                                    |
| `mode`    | string | No       | One of `default`, `strict`, `chat`, `web` (see below)  |
| `user_id` | string | Yes      | Unique user identifier (passed by .NET from JWT claims) |

**Modes:**

| Mode      | Library | Memory | Web   | Use Case                                   |
| --------- | ------- | ------ | ----- | ------------------------------------------ |
| `default` | Yes     | Yes    | No    | Normal — searches library and memory        |
| `strict`  | Yes     | No     | No    | Library documents only, ignores memory      |
| `chat`    | No      | Yes    | No    | Past conversations only, ignores library    |
| `web`     | No      | No     | Yes   | Live DuckDuckGo internet search             |

**Response:** `text/event-stream` — see [Streaming Protocol](#streaming-protocol).

### `POST /memory/save`

Saves an explicit user fact to the vector memory database.

```json
{ "fact": "My favorite color is blue", "user_id": "user-abc-123" }
```

### `GET /memory/facts/{user_id}`

Returns all explicit facts saved by this user.

```json
{ "facts": ["My favorite color is blue", "I work at Acme Corp"] }
```

### `DELETE /memory/facts/{user_id}?keyword=blue`

Deletes all facts containing the keyword (case-insensitive match).

```json
{ "deleted": 1 }
```

### `POST /memory/wipe`

Deletes **all** memory entries (facts + conversation history) for a user. Irreversible.

```json
{ "user_id": "user-abc-123" }
```

### `POST /memory/clear-buffer`

Clears only the short-term in-memory chat buffer (last 6 exchanges). Does not affect persisted memory.

```json
{ "user_id": "user-abc-123" }
```

### `POST /documents/upload`

Upload a document file to be chunked, embedded, and stored in ChromaDB.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — the document (`.pdf`, `.txt`, `.docx`, `.doc`)

```json
{ "success": true, "chunks": 42, "filename": "research-paper.pdf" }
```

### `GET /documents`

Lists all unique document filenames currently in the library.

```json
{ "documents": ["research-paper.pdf", "notes.txt"] }
```

### `DELETE /documents/{filename}`

Removes all chunks belonging to a document. Returns `404` if not found.

```json
{ "message": "'research-paper.pdf' removed." }
```

---

## Core Classes

### `LibraryAI` (library_ai.py)

The main AI engine. Refactored from the original `js_ai.py` CLI tool to be multi-user and return data instead of printing to stdout.

**Constructor:**

```python
LibraryAI(chroma_dir: str, ollama_base_url: str = "http://localhost:11434")
```

- Initializes HuggingFace `BAAI/bge-large-en-v1.5` embeddings on CUDA
- Connects to two ChromaDB collections: `vector_db` (documents) and `memory_db` (chat memory)
- Initializes the Ollama LLM client
- Sets up LangChain prompt template chains

**Key Methods:**

| Method                    | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| `save_fact(uid, fact)`    | Persist an explicit user fact to vector memory                     |
| `list_facts(uid)`         | Retrieve all explicit facts for a user                             |
| `delete_fact(uid, kw)`    | Delete facts matching a keyword                                    |
| `wipe_memory(uid)`        | Nuclear delete of all user memory                                  |
| `clear_buffer(uid)`       | Clear in-memory short-term buffer only                             |
| `ingest_document(path, name)` | Load, chunk, embed, and store a document                       |
| `remove_document(name)`   | Delete all chunks for a document                                   |
| `list_documents()`        | List all unique document names                                     |
| `parse_mode(mode)`        | Convert mode string to `(use_library, use_memory, force_web)` tuple |
| `execute_query_stream(...)` | The full RAG pipeline as a generator (see below)                 |

### RAG Pipeline (`execute_query_stream`)

The pipeline executes these steps as a Python generator that yields event dicts:

1. **Rephrase** — The user's query is rewritten as a standalone search query, resolving pronouns and references using the short-term buffer.
2. **Memory Retrieval** — If enabled, performs a similarity search against the memory ChromaDB collection (top 2 results).
3. **Library/Web Retrieval** — If library mode, searches the document ChromaDB (top 3). If web mode, queries DuckDuckGo.
4. **Answer Generation** — Streams the LLM response token-by-token.
5. **Fallback** — If the answer contains "I cannot find the answer" and web wasn't already used, automatically retries with a live web search.
6. **Source Attribution** — Yields the sources used (document names + pages, or "DuckDuckGo").
7. **Memory Persistence** — In default mode, saves the Q&A pair to the memory database.

---

## Streaming Protocol

The `/chat/stream` endpoint returns an SSE stream. Each line follows the format:

```
data: {"type": "...", ...}\n\n
```

**Event types:**

| Type     | Fields                   | Description                                 |
| -------- | ------------------------ | ------------------------------------------- |
| `debug`  | `standalone_query`       | The rephrased standalone search query        |
| `status` | `message`                | Human-readable status update (e.g., "Searching the web...") |
| `token`  | `content`                | A single streamed token from the LLM         |
| `sources`| `sources` (string array) | List of document sources or "DuckDuckGo"     |
| `done`   | *(none)*                 | End-of-stream signal                         |

**Example stream:**
```
data: {"type": "debug", "standalone_query": "What is the user's favorite programming language?"}

data: {"type": "token", "content": "Based"}

data: {"type": "token", "content": " on"}

data: {"type": "token", "content": " your"}

data: {"type": "token", "content": " documents"}

data: {"type": "token", "content": "..."}

data: {"type": "sources", "sources": ["notes.pdf (Page 3)"]}

data: {"type": "done"}

```

---

## Running Locally

**Prerequisites:** Python 3.11+, Ollama running with `qwen3-coder:30b`, CUDA GPU (optional but recommended for embeddings).

```bash
cd ai-service

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

The `--reload` flag enables auto-restart on file changes during development. Remove it for production.

**Verify:**
```bash
curl http://localhost:8100/health
# {"status":"ok"}
```

---

## Docker

```bash
docker build -t libraryai-service ./ai-service
docker run -p 8100:8100 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e CHROMA_DIR=/data/chroma_db \
  -v libraryai-data:/data \
  libraryai-service
```

> **Note:** The default Docker image does **not** include CUDA support. For GPU acceleration, use the NVIDIA Container Toolkit and uncomment the GPU section in `docker-compose.yml`.

---

## Troubleshooting

| Problem | Solution |
| ------- | -------- |
| `Connection refused` to Ollama | Ensure `ollama serve` is running. If using Docker, start Ollama with `$env:OLLAMA_HOST="0.0.0.0"` so it listens on all interfaces. |
| Slow first request | The HuggingFace embedding model downloads on first load (~1.3 GB). Subsequent starts use the cache. |
| `CUDA out of memory` | Reduce the Ollama model size or switch embeddings to CPU: change `model_kwargs={'device': 'cpu'}` in `library_ai.py` |
| Upload fails with "Unsupported file type" | Only `.pdf`, `.txt`, `.docx`, and `.doc` are supported |
| ChromaDB errors after upgrade | Delete the `chroma_db/` directory to rebuild (you will lose stored documents) |
