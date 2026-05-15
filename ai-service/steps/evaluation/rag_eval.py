"""
steps/evaluation/rag_eval.py — ZenML step: RAG quality evaluation with RAGAS.

Phase 6 (Chapter 7 — Evaluation):
  Samples stored documents from MongoDB, generates questions + ground-truth
  answers via Ollama, retrieves context from Qdrant, generates answers with
  the same LLM, then scores with four RAGAS metrics:

  - context_precision   : retrieved chunks are relevant to the question
  - context_recall      : retrieved chunks cover the ground-truth answer
  - faithfulness        : generated answer is grounded in retrieved context
  - answer_relevancy    : generated answer actually addresses the question

Returns a dict of metric_name → mean score (float, 0-1).
"""

import json
import os
from loguru import logger
from zenml import step


@step
def evaluate_rag(sample_size: int = 0) -> dict:
    """Run RAGAS evaluation over a sample of indexed documents.

    Args:
        sample_size: number of QA pairs to evaluate.
                     0 → uses settings.eval_sample_size.

    Returns:
        dict with keys: context_precision, context_recall,
                        faithfulness, answer_relevancy  (all 0-1).
    """
    from datasets import Dataset as HFDataset
    from langchain_ollama import OllamaLLM
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from config import get_settings
    from db.mongo import get_database

    settings = get_settings()
    n = sample_size or settings.eval_sample_size

    # ── 1. Sample documents from MongoDB ──────────────────────────────────────
    db = get_database()
    cursor = db["raw_documents"].find({}, {"content": 1, "_id": 0}).limit(n)
    docs = [d["content"] for d in cursor if d.get("content")]

    if not docs:
        raise ValueError("No documents in MongoDB. Upload documents first.")

    logger.info("Evaluating RAG over %d documents...", len(docs))

    # ── 2. Build LLM / embeddings (reuse Ollama already running) ──────────────
    llm = OllamaLLM(model=settings.eval_llm, base_url=settings.ollama_base_url)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.eval_embeddings_model,
        model_kwargs={"device": settings.embed_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # ── 3. Generate QA pairs & retrieve context ────────────────────────────────
    questions, ground_truths, contexts, answers = [], [], [], []

    for content in docs:
        snippet = content[:600].strip()
        if len(snippet) < 50:
            continue

        # Question
        q_prompt = (
            "Based on the following text, write a single clear factual question "
            "whose answer can be found within the text. Return ONLY the question.\n\n"
            f"TEXT:\n{snippet}"
        )
        try:
            question = llm.invoke(q_prompt).strip()
        except Exception as exc:
            logger.warning("Q generation failed: %s", exc)
            continue

        # Ground-truth answer
        gt_prompt = (
            f"Context:\n{snippet}\n\nQuestion: {question}\n\n"
            "Answer the question using only the context above. Be concise."
        )
        try:
            ground_truth = llm.invoke(gt_prompt).strip()
        except Exception as exc:
            logger.warning("GT generation failed: %s", exc)
            continue

        # Retrieve top-5 chunks from Qdrant
        try:
            q_vec = embeddings.embed_query(question)
            hits = qdrant.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=q_vec,
                limit=5,
            )
            retrieved = [h.payload.get("content", "") for h in hits if h.payload]
        except Exception as exc:
            logger.warning("Qdrant retrieval failed: %s", exc)
            retrieved = []

        if not retrieved:
            continue

        # Generate answer from retrieved context
        context_text = "\n\n---\n\n".join(retrieved)
        ans_prompt = (
            f"Context:\n{context_text}\n\nQuestion: {question}\n\n"
            "Answer using the context above. Be thorough and accurate."
        )
        try:
            answer = llm.invoke(ans_prompt).strip()
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            answer = ""

        questions.append(question)
        ground_truths.append(ground_truth)
        contexts.append(retrieved)
        answers.append(answer)

    if not questions:
        raise ValueError("No valid QA pairs could be generated for evaluation.")

    logger.info("Generated %d evaluation pairs.", len(questions))

    # ── 4. Run RAGAS ──────────────────────────────────────────────────────────
    eval_dataset = HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    result = evaluate(eval_dataset, metrics=metrics)
    scores = {k: float(v) for k, v in result.items()}

    # ── 5. Persist results ────────────────────────────────────────────────────
    os.makedirs(settings.eval_output_dir, exist_ok=True)
    out_path = os.path.join(settings.eval_output_dir, "rag_eval.json")
    with open(out_path, "w") as f:
        json.dump({"n_pairs": len(questions), "scores": scores}, f, indent=2)

    logger.info("RAGAS results → %s", out_path)
    for k, v in scores.items():
        logger.info("  %-25s %.4f", k, v)

    return scores
