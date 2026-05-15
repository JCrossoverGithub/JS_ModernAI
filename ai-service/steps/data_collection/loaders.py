"""
steps/data_collection/loaders.py — ZenML step: load raw documents.

Chapter 3 (Data Engineering):
  Reads documents from MongoDB (data warehouse) for the given author/user.
  Falls back to accepting uploaded files for Phase 1 compatibility.

Phase 2 implementation will replace the stub body with real MongoDB queries.
"""

from loguru import logger
from zenml import step

from models.documents import RawDocument


@step
def load_raw_documents(author_id: str) -> list[RawDocument]:
    """Load all raw documents for *author_id* from the MongoDB data warehouse."""
    from db.mongo import raw_documents_collection, ping

    if not ping():
        logger.error("load_raw_documents: MongoDB unreachable — returning empty list.")
        return []

    col = raw_documents_collection()
    cursor = col.find({"user_id": author_id}, {"_id": 0})
    docs: list[RawDocument] = []
    for record in cursor:
        try:
            docs.append(RawDocument(**record))
        except Exception as exc:
            logger.warning("Skipping malformed record %s: %s", record.get("id"), exc)

    logger.info("Loaded %d raw documents for author_id='%s'.", len(docs), author_id)
    return docs


# ---------------------------------------------------------------------------
# Helpers used by the FastAPI upload endpoint (Phase 1 compat, not ZenML steps)
# ---------------------------------------------------------------------------

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_file_as_raw_document(
    file_path: str,
    original_filename: str,
    user_id: str,
) -> RawDocument:
    """Load a single file from disk and return a RawDocument.

    Called by main.py's upload endpoint; will feed into the feature pipeline.
    """
    ext = os.path.splitext(original_filename)[1].lower()
    loader_map = {
        ".pdf": lambda: PyPDFLoader(file_path),
        ".txt": lambda: TextLoader(file_path, autodetect_encoding=True),
        ".docx": lambda: Docx2txtLoader(file_path),
        ".doc": lambda: Docx2txtLoader(file_path),
    }
    loader_factory = loader_map.get(ext)
    if loader_factory is None:
        raise ValueError(f"Unsupported file type: {ext}")

    pages = loader_factory().load()
    content = "\n\n".join(p.page_content for p in pages)

    source_type_map = {".pdf": "pdf", ".txt": "txt", ".docx": "docx", ".doc": "docx"}
    return RawDocument(
        content=content,
        source=original_filename,
        source_type=source_type_map.get(ext, "unknown"),
        user_id=user_id,
        metadata={"file_path": file_path, "pages": len(pages)},
    )
