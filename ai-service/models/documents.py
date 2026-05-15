"""
models/documents.py — Pydantic domain models for the document pipeline.

Follows the LLM Engineer's Handbook data model pattern:
  RawDocument      → ingested from disk/web, stored in MongoDB
  CleanedDocument  → after cleaning step (HTML stripped, whitespace normalised)
  DocumentChunk    → after chunking + embedding, stored in Qdrant
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RawDocument(BaseModel):
    """A document as first ingested — no cleaning or chunking applied."""

    id: str = Field(default_factory=_new_id)
    content: str
    source: str                     # filename or URL
    source_type: str                # "pdf" | "txt" | "docx" | "url"
    user_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict = Field(default_factory=dict)


class CleanedDocument(BaseModel):
    """A document after the cleaning step: HTML stripped, normalised text."""

    id: str = Field(default_factory=_new_id)
    raw_document_id: str
    content: str
    source: str
    user_id: str
    metadata: dict = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A single chunk after splitting + embedding, ready for Qdrant upsert."""

    id: str = Field(default_factory=_new_id)
    document_id: str                # parent CleanedDocument.id
    content: str
    chunk_index: int
    source: str
    user_id: str
    embedding: Optional[list[float]] = None
    metadata: dict = Field(default_factory=dict)
