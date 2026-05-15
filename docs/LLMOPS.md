# LLMOps Guide — LLM Engineer's Handbook Rebuild

This document covers everything added in the Phase 1 restructure: new tools,
how to install them, how to run the pipelines, and what each part does.

---

## Table of Contents

1. [What Changed at a Glance](#what-changed-at-a-glance)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Environment Variables](#environment-variables)
5. [Starting the New Services](#starting-the-new-services)
6. [Running the Feature Pipeline](#running-the-feature-pipeline)
7. [Project Structure](#project-structure)
8. [Tool Reference](#tool-reference)
9. [Phase Roadmap](#phase-roadmap)

---

## What Changed at a Glance

| Concern | Before | After (Phase 1+) |
|---|---|---|
| Package manager | `pip` + `requirements.txt` | **Poetry** + `pyproject.toml` |
| Vector DB | ChromaDB (local file) | **Qdrant** (Docker container) |
| Raw data storage | None | **MongoDB** (Docker container) |
| Pipeline orchestration | None | **ZenML** |
| Prompt templates | Hardcoded strings in `library_ai.py` | **Jinja2** `.j2` files in `templates/` |
| Configuration | Scattered `os.getenv()` calls | **pydantic-settings** `config.py` |
| Domain models | None | **Pydantic** models in `models/` |

The existing app (`library_ai.py`, `main.py`, React, .NET) is **unchanged** —
it still runs exactly as before while the new architecture is built alongside it.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/) |
| Poetry | 1.8+ | `pip install poetry` |
| Docker Desktop | Latest | [docker.com](https://www.docker.com/) |
| NVIDIA GPU drivers + CUDA 12.x | For GPU embeddings | [nvidia.com](https://www.nvidia.com/Download/index.aspx) |
| Node.js 20+ | Frontend (unchanged) | [nodejs.org](https://nodejs.org/) |
| .NET SDK 8 | Backend (unchanged) | [microsoft.com](https://dotnet.microsoft.com/) |
| Ollama | LLM inference (Phase 1) | [ollama.ai](https://ollama.ai/) |

---

## Installation

### 1 — Install torch with CUDA first

Poetry pulls torch from PyPI (CPU-only) by default. Install the CUDA build
manually first so Poetry sees it as already satisfied:

```powershell
cd ai-service
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2 — Install everything else via Poetry

```powershell
poetry install
```

This reads `pyproject.toml` and installs all dependencies into your active
virtual environment (or creates one if none is active).

> **Using your existing `.venv`?**  
> Activate it first, then run `poetry install` — Poetry will install into it.
> ```powershell
> .venv\Scripts\Activate.ps1
> poetry install
> ```

### 3 — Dev tools (optional)

```powershell
poetry install --with dev
```

Installs `ruff` (linter), `mypy` (type checker), and `pytest`.

---

## Environment Variables

Create `ai-service/.env` for local development. All values have working
defaults — only override what you need.

```env
# --- Embeddings ---
EMBED_DEVICE=cuda           # or "cpu" if no GPU

# --- Ollama (Phase 1 default LLM) ---
OLLAMA_BASE_URL=http://localhost:11434

# --- Qdrant (Phase 3+ — run via Docker below) ---
QDRANT_HOST=localhost
QDRANT_PORT=6333

# --- MongoDB (Phase 2+ — run via Docker below) ---
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=llm_twin

# --- Inference mode ---
# "ollama" = use Ollama (current)
# "huggingface" = fine-tuned HF model (Phase 7)
INFERENCE_MODE=ollama

# --- HuggingFace (needed in Phase 4 for fine-tuning) ---
HUGGINGFACE_ACCESS_TOKEN=

# --- Comet ML / Opik (Phase 11) ---
COMET_API_KEY=
OPIK_API_KEY=
```

All variables are declared and typed in [ai-service/config.py](../ai-service/config.py).
Import them anywhere with:

```python
from config import get_settings
settings = get_settings()
print(settings.qdrant_host)
```

---

## Starting the New Services

### Option A — Docker Compose (recommended)

Start just the new backing services (Qdrant + MongoDB) without rebuilding
the full stack:

```powershell
docker compose up qdrant mongodb -d
```

Qdrant dashboard: http://localhost:6333/dashboard  
MongoDB (no dashboard by default — connect with MongoDB Compass or `mongosh`).

### Option B — Full stack

```powershell
docker compose up --build
```

Starts all 5 services: qdrant, mongodb, ai-service, backend, frontend.
GPU pass-through is enabled by default for the ai-service container.

### Stopping

```powershell
docker compose down          # stop but keep volumes
docker compose down -v       # stop AND delete all data
```

---

## Running the Feature Pipeline

The feature pipeline (Chapter 3-4 of the book) ingests documents and loads
them into Qdrant. It runs as a ZenML pipeline.

### Initialize ZenML (first time only)

```powershell
cd ai-service
zenml init
zenml up          # start local ZenML server (optional — shows pipeline UI)
```

### Run the pipeline

```powershell
python -m pipelines.feature_pipeline --author-id <your-user-id>
```

**What it does, step by step:**

```
load_raw_documents      ← reads documents from MongoDB for the user
        ↓
clean_documents         ← strips HTML, normalises whitespace, deduplicates
        ↓
chunk_documents         ← splits into 512-token overlapping chunks
        ↓
embed_chunks            ← generates bge-large-en-v1.5 embeddings on CUDA
        ↓
load_to_qdrant          ← batch upserts vectors into Qdrant
```

> **Phase 1 note:** `load_raw_documents` currently returns an empty list
> because MongoDB isn't wired up yet (Phase 2). Once Phase 2 is done,
> documents uploaded through the web UI will appear in MongoDB and flow
> through the pipeline automatically.

### Inspect pipeline runs (ZenML dashboard)

After `zenml up`, open http://localhost:8237 to see pipeline run history,
step artefacts, and caching behaviour.

---

## Project Structure

```
ai-service/
│
├── config.py                   # All env vars — pydantic-settings
├── main.py                     # FastAPI app (unchanged in Phase 1)
├── library_ai.py               # AI engine (unchanged in Phase 1)
│
├── pyproject.toml              # Poetry dependency manifest
│
├── models/                     # Pydantic domain models
│   ├── documents.py            #   RawDocument, CleanedDocument, DocumentChunk
│   └── conversations.py        #   ChatMessage, RetrievedContext
│
├── pipelines/                  # ZenML pipeline definitions
│   ├── feature_pipeline.py     #   Ch 3-4: ingest → clean → chunk → embed → store
│   └── training_pipeline.py    #   Ch 5-6: SFT and DPO (stubs)
│
├── steps/                      # ZenML step implementations
│   ├── data_collection/
│   │   └── loaders.py          #   load_raw_documents (MongoDB) + file helper
│   ├── feature_engineering/
│   │   ├── cleaning.py         #   HTML strip, normalise, dedup
│   │   ├── chunking.py         #   RecursiveCharacterTextSplitter
│   │   ├── embedding.py        #   sentence-transformers batched on CUDA
│   │   └── qdrant_loader.py    #   batch upsert to Qdrant
│   ├── training/               #   Stubs: sft.py, dpo.py, hub.py (Phase 4-5)
│   └── inference/              #   Stubs (Phase 8)
│
└── templates/                  # Jinja2 prompt templates
    ├── qa.j2                   #   Main QA prompt
    ├── rephrase.j2             #   Query rephrase / standalone question
    ├── research_query.j2       #   Academic search query extraction
    └── research_synthesis.j2   #   Research paper synthesis
```

---

## Tool Reference

### Poetry — dependency management

```powershell
poetry add <package>            # add a runtime dependency
poetry add --group dev <pkg>    # add a dev-only dependency
poetry update                   # update all packages to latest allowed versions
poetry show --tree              # print dependency tree
poetry env info                 # show active virtualenv path
```

### ZenML — pipeline orchestration

```powershell
zenml init                      # initialise ZenML in the current directory
zenml up                        # start local dashboard at http://localhost:8237
zenml pipeline list             # list registered pipelines
zenml pipeline runs list        # list all past runs
zenml stack list                # list configured stacks (local, cloud, etc.)
```

### Qdrant — vector database

```powershell
# REST API (after docker compose up qdrant)
curl http://localhost:6333/collections          # list collections
curl http://localhost:6333/collections/library_ai   # inspect collection
```

Or open http://localhost:6333/dashboard in a browser.

### Ruff — linting (dev only)

```powershell
cd ai-service
ruff check .                    # lint
ruff check . --fix              # auto-fix safe issues
ruff format .                   # format (like black)
```

### Pytest — tests (dev only)

```powershell
cd ai-service
pytest                          # run all tests in tests/
pytest -k "test_cleaning"       # run a specific test
```

---

## Phase Roadmap

| Phase | Chapter(s) | What gets built | Status |
|---|---|---|---|
| **1 — Foundation** | 1-2 | Poetry, config, domain models, pipeline structure, Qdrant + MongoDB in Docker | ✅ Done |
| **2 — Data Engineering** | 3 | MongoDB ingest, web scraper, raw document storage | ⬜ Next |
| **3 — RAG Feature Pipeline** | 4 | Replace ChromaDB with Qdrant, full feature pipeline live | ⬜ |
| **4 — Supervised Fine-Tuning** | 5 | Instruction dataset, SFT with PEFT/LoRA/bitsandbytes on GPU | ⬜ |
| **5 — Preference Alignment** | 6 | DPO preference dataset + DPOTrainer | ⬜ |
| **6 — Evaluation** | 7 | RAG eval (recall@k), LLM eval (BERTScore), Comet ML logging | ⬜ |
| **7 — Inference Optimization** | 8 | FlashAttention-2, Unsloth, replace Ollama with HF pipeline | ⬜ |
| **8 — Inference Pipeline** | 9 | Qdrant retrieval, Jinja2 templates, Opik monitoring | ⬜ |
| **9 — Deployment** | 10 | *(cloud — deferred)* | ⬜ |
| **10 — MLOps/CI** | 11-12 | *(cloud — deferred)* | ⬜ |
