"""
main.py — FastAPI application for the LibraryAI service.

Exposes the LibraryAI engine over HTTP with:
  - POST /chat/stream   — SSE-streamed RAG chat responses
  - POST /memory/save   — Save an explicit user fact
  - GET  /memory/facts/{user_id} — List saved facts
  - DELETE /memory/facts/{user_id} — Delete facts by keyword
  - POST /memory/wipe    — Wipe all user memory
  - POST /memory/clear-buffer — Clear short-term buffer
  - POST /documents/upload — Ingest a document
  - GET  /documents       — List library documents
  - DELETE /documents/{filename} — Remove a document

This server is intended to be called by the .NET backend, not directly
by the frontend. CORS is configured to allow only the backend origin.
"""

import os
import json
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from library_ai import LibraryAI


app = FastAPI(title="LibraryAI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:5000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singleton AI instance ---
# Lazy-initialized on first request to avoid slow startup.
# The LibraryAI constructor downloads the embedding model and connects to ChromaDB.
_ai: Optional[LibraryAI] = None


def get_ai() -> LibraryAI:
    """FastAPI dependency that returns the singleton LibraryAI instance."""
    global _ai
    if _ai is None:
        chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _ai = LibraryAI(chroma_dir=chroma_dir, ollama_base_url=ollama_url)
    return _ai


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Request models ---
class ChatRequest(BaseModel):
    query: str
    mode: str = "default"  # default | strict | chat | web
    user_id: str


class FactRequest(BaseModel):
    fact: str
    user_id: str


class DeleteFactRequest(BaseModel):
    keyword: str
    user_id: str


class UserIdRequest(BaseModel):
    user_id: str


# --- Health check ---
@app.get("/health")
def health():
    """Simple health check. Returns OK without loading the AI model."""
    return {"status": "ok"}


# --- Chat (SSE streaming) ---
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, ai: LibraryAI = Depends(get_ai)):
    """Stream a RAG response as Server-Sent Events.

    Each event is a JSON object with a 'type' field.
    See library_ai.execute_query_stream() for event types.
    """
    use_library, use_memory, force_web = ai.parse_mode(req.mode)

    def event_generator():
        for event in ai.execute_query_stream(
            user_id=req.user_id,
            query=req.query,
            use_library=use_library,
            use_memory=use_memory,
            force_web=force_web,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Research (SSE streaming) ---
class ResearchRequest(BaseModel):
    query: str
    user_id: str


@app.post("/research/stream")
async def research_stream(req: ResearchRequest, ai: LibraryAI = Depends(get_ai)):
    """Stream an academic research response as Server-Sent Events.

    Searches Semantic Scholar and arXiv for papers, synthesizes a summary,
    and returns paper metadata with download links.
    """
    def event_generator():
        for event in ai.execute_research_stream(
            user_id=req.user_id,
            query=req.query,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Memory endpoints ---
@app.post("/memory/save")
def save_fact(req: FactRequest, ai: LibraryAI = Depends(get_ai)):
    msg = ai.save_fact(req.user_id, req.fact)
    return {"message": msg}


@app.get("/memory/facts/{user_id}")
def list_facts(user_id: str, ai: LibraryAI = Depends(get_ai)):
    return {"facts": ai.list_facts(user_id)}


@app.delete("/memory/facts/{user_id}")
def delete_fact(user_id: str, keyword: str, ai: LibraryAI = Depends(get_ai)):
    count = ai.delete_fact(user_id, keyword)
    return {"deleted": count}


@app.post("/memory/wipe")
def wipe_memory(req: UserIdRequest, ai: LibraryAI = Depends(get_ai)):
    ai.wipe_memory(req.user_id)
    return {"message": "All memory wiped."}


@app.post("/memory/clear-buffer")
def clear_buffer(req: UserIdRequest, ai: LibraryAI = Depends(get_ai)):
    ai.clear_buffer(req.user_id)
    return {"message": "Short-term buffer cleared."}


# --- Document endpoints ---
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), ai: LibraryAI = Depends(get_ai)):
    """Upload a document file (PDF/TXT/DOCX) to be chunked, embedded, and stored.

    The file is saved to a temp directory with a UUID prefix to prevent collisions,
    processed by the AI engine, then the temp file is deleted.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ai.ingest_document(save_path, file.filename)

    # Clean up temp file
    try:
        os.remove(save_path)
    except OSError:
        pass

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/documents")
def list_documents(ai: LibraryAI = Depends(get_ai)):
    return {"documents": ai.list_documents()}


@app.delete("/documents/{filename}")
def remove_document(filename: str, ai: LibraryAI = Depends(get_ai)):
    removed = ai.remove_document(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in library.")
    return {"message": f"'{filename}' removed."}
