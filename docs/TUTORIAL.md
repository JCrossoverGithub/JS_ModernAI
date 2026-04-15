# Tutorial: Running LibraryAI from Scratch

This step-by-step guide walks you through setting up the entire LibraryAI stack on a fresh Windows machine, from installing prerequisites to chatting with your first uploaded document.

---

## Prerequisites Checklist

Before starting, install the following:

| Tool | Version | Download |
| ---- | ------- | -------- |
| **Node.js** | 20+ | https://nodejs.org/ |
| **Python** | 3.11+ | https://www.python.org/downloads/ |
| **.NET SDK** | 8.0+ | https://dotnet.microsoft.com/download/dotnet/8.0 |
| **Ollama** | Latest | https://ollama.ai/download |
| **Git** | Any | https://git-scm.com/downloads |
| **Docker Desktop** | Latest (optional) | https://www.docker.com/products/docker-desktop/ |

Verify installations:
```
node --version       # v20.x.x or higher
python --version     # 3.11.x or higher
dotnet --version     # 8.0.x
ollama --version     # 0.x.x
```

### GPU Note

The AI service uses HuggingFace embeddings which default to **CUDA (NVIDIA GPU)**. If you don't have an NVIDIA GPU, you'll need to change one line in `ai-service/library_ai.py`:

```python
# Change this:
model_kwargs={"device": "cuda"},
# To this:
model_kwargs={"device": "cpu"},
```

---

## Step 1: Pull the Ollama Model

Ollama runs the LLM locally. Start the server and download the model:

```powershell
# Start Ollama (runs in background)
# If using Docker, set OLLAMA_HOST so containers can reach it:
$env:OLLAMA_HOST="0.0.0.0"
ollama serve

# In a new terminal, pull the model (~18 GB download)
ollama pull qwen3-coder:30b
```

> **Tip:** If your machine has limited RAM (~16 GB), consider using a smaller model like `qwen3-coder:8b` and updating the model name in `ai-service/library_ai.py`.

Verify Ollama is running:
```powershell
curl http://localhost:11434/api/tags
```

You should see your model listed in the response.

---

## Step 2: Start the Python AI Service

```powershell
cd ai-service

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies (first time takes a few minutes)
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8100
```

**What happens on first start:**
1. The server starts on port 8100
2. The HuggingFace embedding model (`BAAI/bge-large-en-v1.5`, ~1.3 GB) downloads automatically on the first request
3. ChromaDB creates its data directory at `./chroma_db/`

**Test it:**
```powershell
curl http://localhost:8100/health
# {"status":"ok"}
```

Leave this terminal running and open a new one.

---

## Step 3: Start the .NET Backend

```powershell
cd backend

# Restore packages (first time)
dotnet restore

# Run the server
dotnet run --urls http://localhost:5000
```

**What happens on first start:**
1. EF Core creates the SQLite database (`app.db`) with Identity tables
2. The server starts on port 5000
3. The typed HttpClient connects to the Python AI service at `http://localhost:8100`

**Test it:**
```powershell
# Register a test user
curl -X POST http://localhost:5000/api/auth/register `
  -H "Content-Type: application/json" `
  -d '{"username":"testuser","email":"test@example.com","password":"test123"}'
```

You should get back a JSON response with a JWT token.

Leave this terminal running and open a new one.

---

## Step 4: Start the React Frontend

```powershell
cd frontend

# Install dependencies (first time)
npm install

# Start development server
npm run dev
```

The Vite dev server starts on `http://localhost:5173`.

Open your browser and go to **http://localhost:5173**.

---

## Step 5: Create an Account

1. You'll see the **LibraryAI** login page
2. Click **"Register"** at the bottom
3. Fill in:
   - **Username:** your display name
   - **Email:** your email
   - **Password:** at least 6 characters
4. Click **"Create Account"**

You're now logged in and see the main chat interface.

---

## Step 6: Upload a Document

1. In the left sidebar, click the **"Library"** tab
2. Click **"Upload PDF, TXT, or DOCX"**
3. Select a document from your computer
4. Wait for the upload and processing to complete

The document appears in the list. Its contents are now chunked, embedded as vectors, and stored in ChromaDB — ready for semantic search.

---

## Step 7: Chat with Your Documents

1. Make sure the mode is set to **"Default"** (sidebar → Mode tab)
2. Type a question about your uploaded document in the chat box
3. Press **Enter** or click **Send**

**What happens behind the scenes:**
1. Your message goes through SignalR to .NET
2. .NET forwards it to the Python AI service
3. Python rephrases your question into a search query
4. ChromaDB performs a similarity search against your documents
5. The top 3 matching chunks are combined with your question
6. The LLM generates an answer, streaming tokens back to your browser
7. Sources are displayed below the answer

---

## Step 8: Try Different Modes

### Strict Mode (Library Only)
Select **"Strict"** in the sidebar. The AI only searches uploaded documents and ignores any chat history.

### Chat Mode (Memory Only)
Select **"Chat"** in the sidebar. The AI only uses your conversation history and saved facts — no documents.

### Web Mode (Internet Search)
Select **"Web"** in the sidebar. The AI searches DuckDuckGo live and uses those results to answer.

### Default Mode
Uses both documents and memory. If neither has the answer, automatically falls back to a web search.

---

## Step 9: Save Personal Facts

1. Go to the **"Facts"** tab in the sidebar
2. Type a fact about yourself: `My name is John and I work at Acme Corp`
3. Click **+** or press Enter

Now try asking in the chat: *"What company do I work for?"*

The AI retrieves your saved fact and answers correctly.

---

## Step 10: Manage Your Data

| Action | How |
| ------ | --- |
| Delete a specific fact | Click ✕ next to it in the Facts tab |
| Remove a document | Click ✕ next to it in the Library tab |
| Clear short-term buffer | Mode tab → "Clear Buffer" button |
| Wipe ALL memory | Mode tab → "Wipe All Memory" button (irreversible) |

---

## Alternative: Docker Deployment

Instead of running each service manually, use Docker Compose:

```powershell
# From the project root
docker compose up --build
```

This starts all three services:
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:5000
- **AI Service:** http://localhost:8100

> **Important:** Ollama must still be running on your host machine. Docker connects to it via `host.docker.internal:11434`.

To stop everything:
```powershell
docker compose down
```

To stop and remove all data (ChromaDB, SQLite, uploads):
```powershell
docker compose down -v
```

---

## Common Issues

### "Connection refused" when chatting
Make sure all three services are running:
1. Ollama — must be started with `$env:OLLAMA_HOST="0.0.0.0"` if using Docker (so containers can reach it)
2. Python AI service (port 8100)
3. .NET backend (port 5000)

### Very slow first question
The first question triggers:
- HuggingFace model download (~1.3 GB, one-time)
- Ollama model loading into RAM/VRAM
Subsequent questions are much faster.

### "CUDA not available" warning
If you don't have an NVIDIA GPU, change `device: "cuda"` to `device: "cpu"` in `ai-service/library_ai.py`.

### SignalR connection drops
Check the browser console. If you see 401 errors, your JWT may have expired (default: 24 hours). Log out and log back in.

### Upload fails
Check the supported file types: `.pdf`, `.txt`, `.docx`, `.doc`. Maximum file size depends on your server configuration (FastAPI default: ~100 MB).

---

## Next Steps

- **Change the JWT secret** in `backend/appsettings.json` before any non-local deployment
- **Try a different LLM** — edit `model="qwen3-coder:30b"` in `library_ai.py` to any Ollama model
- **Add HTTPS** — put everything behind a reverse proxy (nginx/Caddy) with TLS certificates
- **Scale the AI service** — if response times are slow, run multiple FastAPI workers: `uvicorn main:app --workers 4`
