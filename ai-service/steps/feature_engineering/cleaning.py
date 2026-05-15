"""
steps/feature_engineering/cleaning.py — ZenML step: clean raw documents.

Chapter 4 (RAG Feature Pipeline):
  - Strip HTML tags and decode entities
  - Normalise Unicode whitespace
  - Remove boilerplate (headers, footers, page numbers)
  - Deduplicate identical paragraphs within a document

Phase 3 will flesh out the body; this stub establishes the contract.
"""

import re
import unicodedata

from loguru import logger
from zenml import step

from models.documents import CleanedDocument, RawDocument


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    import html
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _deduplicate_paragraphs(text: str) -> str:
    """Remove consecutive duplicate paragraphs (common in PDF parsing)."""
    paragraphs = text.split("\n")
    seen: set[str] = set()
    unique: list[str] = []
    for p in paragraphs:
        stripped = p.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(p)
    return "\n".join(unique)


def clean_text(text: str) -> str:
    """Apply all cleaning operations to a single text string."""
    text = _strip_html(text)
    text = _deduplicate_paragraphs(text)
    text = _normalise_whitespace(text)
    return text


@step
def clean_documents(documents: list[RawDocument]) -> list[CleanedDocument]:
    """Clean a batch of RawDocuments and return CleanedDocuments."""
    cleaned: list[CleanedDocument] = []
    for doc in documents:
        try:
            clean_content = clean_text(doc.content)
            if not clean_content:
                logger.warning("Document %s is empty after cleaning; skipping.", doc.id)
                continue
            cleaned.append(
                CleanedDocument(
                    id=doc.id,  # inherit parent ID so downstream chunk IDs are stable
                    raw_document_id=doc.id,
                    content=clean_content,
                    source=doc.source,
                    user_id=doc.user_id,
                    metadata={**doc.metadata, "source_type": doc.source_type},
                )
            )
        except Exception as exc:
            logger.error("Failed to clean document %s: %s", doc.id, exc)

    logger.info("Cleaned %d / %d documents.", len(cleaned), len(documents))
    return cleaned
