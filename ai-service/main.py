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

import asyncio
import logging
import os
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from library_ai import LibraryAI
from config import get_settings
from db.mongo import ping as mongo_ping, raw_documents_collection, ensure_indexes
from steps.data_collection.loaders import load_file_as_raw_document

logger = logging.getLogger(__name__)


app = FastAPI(title="LibraryAI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:5000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singleton AI instance ---
# Eagerly initialized at startup so the first request isn't delayed
# by embedding model download / load.
_ai: Optional[LibraryAI] = None


@app.on_event("startup")
def _init_ai():
    """Pre-load the AI engine (embedding model + ChromaDB) at server start."""
    global _ai
    settings = get_settings()
    _ai = LibraryAI(chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"), ollama_base_url=settings.ollama_base_url)
    # Ensure MongoDB indexes (non-fatal if MongoDB is down)
    try:
        ensure_indexes()
    except Exception as exc:
        logger.warning("MongoDB startup check failed (non-fatal): %s", exc)


def get_ai() -> LibraryAI:
    """FastAPI dependency that returns the singleton LibraryAI instance."""
    if _ai is None:
        raise RuntimeError("AI engine not initialized")
    return _ai


UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread pool for background feature pipeline runs (one at a time avoids GPU contention)
_pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="feature-pipeline")


def _run_feature_pipeline(author_id: str) -> None:
    """Execute the full feature pipeline synchronously in a background thread.

    Reads all raw documents for *author_id* from MongoDB, cleans, chunks,
    embeds (GPU), and upserts into Qdrant. Safe to re-run — chunk IDs are
    deterministic so Qdrant upsert is idempotent.
    """
    # Lazy imports keep startup fast; these are only needed when a doc is uploaded
    from steps.data_collection import load_raw_documents
    from steps.feature_engineering import (
        clean_documents, chunk_documents, embed_chunks, load_to_qdrant
    )
    try:
        raw = load_raw_documents(author_id=author_id)
        if not raw:
            logger.info("Feature pipeline: no documents found for user '%s'.", author_id)
            return
        cleaned = clean_documents(documents=raw)
        chunks = chunk_documents(documents=cleaned)
        embedded = embed_chunks(chunks=chunks)
        load_to_qdrant(chunks=embedded)
        logger.info(
            "Feature pipeline complete for user '%s': %d chunks in Qdrant.",
            author_id, len(embedded),
        )
    except Exception as exc:
        logger.error("Feature pipeline failed for user '%s': %s", author_id, exc)


# --- Request models ---
class ChatRequest(BaseModel):
    query: str
    mode: str = "default"  # default | strict | chat | web
    user_id: str
    folder_context: str = ""


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
            folder_context=req.folder_context,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Research (SSE streaming) ---
class ResearchRequest(BaseModel):
    query: str
    user_id: str
    sources: Optional[list[str]] = None
    folder_context: str = ""


@app.post("/research/stream")
async def research_stream(req: ResearchRequest, ai: LibraryAI = Depends(get_ai)):
    """Stream an academic research response as Server-Sent Events.

    Searches selected academic APIs for papers, synthesizes a summary,
    and returns paper metadata with download links.
    """
    def event_generator():
        for event in ai.execute_research_stream(
            user_id=req.user_id,
            query=req.query,
            sources=req.sources,
            folder_context=req.folder_context,
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
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(default="global"),
    ai: LibraryAI = Depends(get_ai),
):
    """Upload a document file (PDF/TXT/DOCX) to be chunked, embedded, and stored.

    Streams progress events as SSE so large documents don't timeout.
    Also saves a RawDocument to MongoDB and triggers the Qdrant feature
    pipeline as a background task (GPU embedding runs concurrently).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # --- Save raw document to MongoDB (data warehouse) ---
    mongo_ok = mongo_ping()
    if mongo_ok:
        try:
            raw_doc = load_file_as_raw_document(save_path, file.filename, user_id)
            raw_documents_collection().replace_one(
                {"user_id": user_id, "source": file.filename},
                raw_doc.model_dump(),
                upsert=True,
            )
        except Exception as exc:
            logger.warning("MongoDB write failed (continuing with ChromaDB ingest): %s", exc)
            mongo_ok = False

    # --- Kick off Qdrant feature pipeline in background (non-blocking) ---
    loop = asyncio.get_running_loop()
    if mongo_ok:
        loop.run_in_executor(_pipeline_executor, _run_feature_pipeline, user_id)

    # --- Stream ChromaDB ingest progress to caller (existing behaviour) ---
    def event_generator():
        try:
            for event in ai.ingest_document_stream(save_path, file.filename):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            try:
                os.remove(save_path)
            except OSError:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/documents")
def list_documents(ai: LibraryAI = Depends(get_ai)):
    return {"documents": ai.list_documents()}


@app.delete("/documents/{filename}")
def remove_document(filename: str, ai: LibraryAI = Depends(get_ai)):
    removed = ai.remove_document(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in library.")
    return {"message": f"'{filename}' removed."}
