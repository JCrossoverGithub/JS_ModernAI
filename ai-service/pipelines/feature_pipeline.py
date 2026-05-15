"""
pipelines/feature_pipeline.py — ZenML feature pipeline.

Implements Chapter 3-4 of the LLM Engineer's Handbook:
  1. collect_data       → load raw documents from MongoDB (Ch 3)
  2. clean_documents    → strip HTML, normalise text          (Ch 4)
  3. chunk_documents    → semantic / recursive splitting      (Ch 4)
  4. embed_chunks       → dense embeddings via sentence-transformers (Ch 4)
  5. load_to_qdrant     → upsert vectors into Qdrant          (Ch 4)

Run from the project root:
    python -m pipelines.feature_pipeline --author-id <user_id>
"""

from zenml import pipeline

from steps.data_collection import load_raw_documents
from steps.feature_engineering import (
    clean_documents,
    chunk_documents,
    embed_chunks,
    load_to_qdrant,
)


@pipeline(name="feature_pipeline", enable_cache=True)
def feature_pipeline(author_id: str) -> None:
    """End-to-end feature pipeline: ingest → clean → chunk → embed → store."""
    raw_docs = load_raw_documents(author_id=author_id)
    cleaned_docs = clean_documents(documents=raw_docs)
    chunks = chunk_documents(documents=cleaned_docs)
    embedded_chunks = embed_chunks(chunks=chunks)
    load_to_qdrant(chunks=embedded_chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the feature pipeline")
    parser.add_argument("--author-id", required=True, help="User / author ID to process")
    args = parser.parse_args()

    feature_pipeline(author_id=args.author_id)
