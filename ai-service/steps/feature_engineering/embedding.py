"""
steps/feature_engineering/embedding.py — ZenML step: embed document chunks.

Chapter 4 (RAG Feature Pipeline):
  Generates dense embeddings for each DocumentChunk using the same
  sentence-transformers model used at query time (BAAI/bge-large-en-v1.5).
  Batched to avoid OOM on the RTX 3070 Ti (8 GB VRAM).
"""

import os

from loguru import logger
from sentence_transformers import SentenceTransformer
from zenml import step

from models.documents import DocumentChunk

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_BATCH_SIZE = 64       # safe for 8 GB VRAM with bge-large


@step
def embed_chunks(
    chunks: list[DocumentChunk],
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[DocumentChunk]:
    """Add dense embeddings to each DocumentChunk in place."""
    if not chunks:
        return []

    device = os.getenv("EMBED_DEVICE", "cuda")
    logger.info("Loading embedding model '%s' on device '%s'.", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    texts = [chunk.content for chunk in chunks]
    logger.info("Embedding %d chunks in batches of %d.", len(texts), batch_size)

    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk.embedding = embedding.tolist()

    logger.info("Embedding complete. Vector dim = %d.", len(all_embeddings[0]))
    return chunks
