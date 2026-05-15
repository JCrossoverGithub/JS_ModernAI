"""
steps/feature_engineering/qdrant_loader.py — ZenML step: upsert to Qdrant.

Chapter 4 (RAG Feature Pipeline):
  Creates the Qdrant collection if it does not exist, then upserts all
  DocumentChunks in batches. Uses cosine distance to match the normalised
  embeddings produced by embed_chunks.

Phase 3 will wire this up to a live Qdrant container.
"""

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from zenml import step

from config import get_settings
from models.documents import DocumentChunk

UPSERT_BATCH_SIZE = 128
VECTOR_DIM = 1024       # BAAI/bge-large-en-v1.5 output dimension


@step
def load_to_qdrant(
    chunks: list[DocumentChunk],
    collection_name: str = "",     # defaults to settings.qdrant_collection_name
) -> None:
    """Upsert embedded DocumentChunks into Qdrant."""
    if not chunks:
        logger.warning("load_to_qdrant: received 0 chunks — nothing to upsert.")
        return

    # Validate all chunks have embeddings
    missing = [c.id for c in chunks if not c.embedding]
    if missing:
        raise ValueError(f"{len(missing)} chunks are missing embeddings. Run embed_chunks first.")

    settings = get_settings()
    col = collection_name or settings.qdrant_collection_name

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    _ensure_collection(client, col)

    # Upsert in batches
    total = 0
    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[i : i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    "source": chunk.source,
                    "user_id": chunk.user_id,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    **chunk.metadata,
                },
            )
            for chunk in batch
        ]
        client.upsert(collection_name=col, points=points, wait=True)
        total += len(batch)
        logger.info("Upserted %d / %d chunks to Qdrant collection '%s'.", total, len(chunks), col)

    logger.info("load_to_qdrant complete — %d chunks stored.", total)


def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the Qdrant collection if it does not exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", collection_name)
