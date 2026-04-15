# Docker Deployment Guide

This guide covers running LibraryAI with Docker Compose and configuring it for different environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Service Overview](#service-overview)
- [Environment Variables](#environment-variables)
- [Volumes & Data Persistence](#volumes--data-persistence)
- [GPU Support](#gpu-support)
- [Production Checklist](#production-checklist)
- [Common Operations](#common-operations)

---

## Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose v2 on Linux)
- **Ollama** running on the host machine (not in Docker — it needs GPU access)

> **Important:** Ollama must listen on all interfaces so Docker containers can reach it. Set `OLLAMA_HOST=0.0.0.0` before starting:

```powershell
$env:OLLAMA_HOST="0.0.0.0"
ollama serve
ollama pull qwen3-coder:30b
```

Without this, you'll get `Connection refused` errors — Ollama defaults to `127.0.0.1`, which is unreachable from inside Docker containers.

---

## Quick Start

```powershell
# From the project root
docker compose up --build

# Or detached (background)
docker compose up --build -d
```

| Service | URL |
| ------- | --- |
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:5000 |
| AI Service | http://localhost:8100 |
| Ollama | http://localhost:11434 (host) |

---

## Service Overview

### docker-compose.yml

```
services:
  frontend      → React app served by nginx (port 3000)
  backend       → .NET 8 API + SignalR (port 5000)
  ai-service    → Python FastAPI + LangChain (port 8100)
```

**Dependency chain:** `frontend` → `backend` → `ai-service` → `host Ollama`

### Network Flow (inside Docker)

```
Browser :3000 → nginx → backend:5000 → ai-service:8100 → host.docker.internal:11434
```

The nginx config in the frontend container proxies `/api/*` and `/hubs/*` to the `backend` Docker service by name.

---

## Environment Variables

### Backend Service

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `ConnectionStrings__DefaultConnection` | `Data Source=/data/app.db` | SQLite file path inside the container |
| `Jwt__Key` | *Q7mX2pL9vNc4Rk8sT1dF6jHb3WyZu5AeG0qVrM2nCx7Kp9LtSd4Jf8hBw1YzE6uN* | **CHANGE THIS** to a random 64+ character string |
| `Jwt__Issuer` | `LibraryAI` | JWT issuer claim |
| `Jwt__Audience` | `LibraryAI-Client` | JWT audience claim |
| `Jwt__ExpirationMinutes` | `1440` | Token lifetime (24h) |
| `PythonAI__BaseUrl` | `http://ai-service:8100` | Internal Docker URL of the Python service |
| `AllowedOrigins` | `http://localhost:3000,...` | Comma-separated CORS origins |

### AI Service

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `CHROMA_DIR` | `/data/chroma_db` | ChromaDB persistence inside the container |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Connects to Ollama on the Docker host |
| `CORS_ORIGIN` | `http://backend:5000` | Internal CORS origin |
| `UPLOAD_DIR` | `/data/uploads` | Temp directory for file processing |

---

## Volumes & Data Persistence

Two named volumes are used for persistent data:

| Volume | Mount Path | Contains |
| ------ | ---------- | -------- |
| `backend-data` | `/data` (backend) | `app.db` (SQLite user database) |
| `ai-data` | `/data` (ai-service) | `chroma_db/` (vector embeddings), `uploads/` (temp files) |

**Data survives `docker compose down`** but is destroyed by `docker compose down -v`.

### Backup

```powershell
# Backup ChromaDB and SQLite
docker compose cp ai-service:/data/chroma_db ./backup/chroma_db
docker compose cp backend:/data/app.db ./backup/app.db
```

### Fresh Start

```powershell
# Stop and destroy all data
docker compose down -v
```

---

## GPU Support

By default, the AI service runs embeddings on CPU inside Docker. To enable GPU:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the `deploy` section in `docker-compose.yml`:

```yaml
ai-service:
  # ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

3. Rebuild: `docker compose up --build`

> **Note:** You also need to change `model_kwargs={"device": "cpu"}` back to `"cuda"` in `library_ai.py` if you previously changed it.

---

## Production Checklist

Before deploying to a server:

- [ ] **Change the JWT key** — `Jwt__Key` must be a random string, not the placeholder
- [ ] **Set up HTTPS** — put nginx/Caddy in front with TLS certificates
- [ ] **Restrict CORS** — set `AllowedOrigins` to your actual domain
- [ ] **Don't expose the AI service** — only the frontend (port 3000) should be public
- [ ] **Set resource limits** — add `mem_limit` and `cpus` to each service in compose
- [ ] **Use a proper database** — consider PostgreSQL for the backend if expecting many users
- [ ] **Run Ollama externally** — on a dedicated GPU server if needed

---

## Common Operations

### View Logs

```powershell
# All services
docker compose logs -f

# Specific service
docker compose logs -f ai-service
```

### Restart a Single Service

```powershell
docker compose restart backend
```

### Rebuild After Code Changes

```powershell
docker compose up --build
```

### Shell into a Container

```powershell
# Python service
docker compose exec ai-service bash

# .NET backend
docker compose exec backend bash
```

### Check Service Health

```powershell
curl http://localhost:8100/health   # AI service
curl http://localhost:5000/api/auth/login  # Backend (should return 400, not 404)
curl http://localhost:3000          # Frontend (should return HTML)
```
