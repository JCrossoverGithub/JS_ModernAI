"""
steps/feature_engineering/chunking.py — ZenML step: chunk cleaned documents.

Chapter 4 (RAG Feature Pipeline):
  Uses LangChain's RecursiveCharacterTextSplitter with sentence-aware
  boundaries. Chunk size and overlap are configurable via ZenML step config.

Phase 3 will add semantic / sentence-window chunking variants.
"""

from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenml import step

from models.documents import CleanedDocument, DocumentChunk

# Default chunking parameters — tune per your embedding model's context window
DEFAULT_CHUNK_SIZE = 512        # tokens ≈ characters / 4; bge-large supports 512
DEFAULT_CHUNK_OVERLAP = 64


@step
def chunk_documents(
    documents: list[CleanedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split CleanedDocuments into fixed-size overlapping DocumentChunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[DocumentChunk] = []

    for doc in documents:
        texts = splitter.split_text(doc.content)
        for idx, text in enumerate(texts):
            chunks.append(
                DocumentChunk(
                    document_id=doc.id,
                    content=text,
                    chunk_index=idx,
                    source=doc.source,
                    user_id=doc.user_id,
                    metadata={
                        **doc.metadata,
                        "total_chunks": len(texts),
                    },
                )
            )

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
