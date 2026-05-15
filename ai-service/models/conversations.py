"""
models/conversations.py — Pydantic domain models for inference context.

ChatMessage       → a single turn stored in the memory vector DB
RetrievedContext  → assembled prompt context for one inference request
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    """One conversation turn persisted to the memory vector DB."""

    id: str = Field(default_factory=_new_id)
    user_id: str
    role: str               # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """All context assembled before calling the LLM for a single query."""

    user_id: str
    query: str
    rephrased_query: str = ""
    chat_history: str = ""          # formatted recent turns
    document_context: str = ""      # formatted retrieved chunks
    source_references: list[str] = Field(default_factory=list)  # chunk ids / filenames
