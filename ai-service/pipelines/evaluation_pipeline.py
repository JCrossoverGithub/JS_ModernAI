"""
pipelines/evaluation_pipeline.py — ZenML evaluation pipeline.

Phase 6 (Chapter 7 — Evaluation):
  Runs RAG quality (RAGAS) and/or generation quality (ROUGE + BERTScore)
  evaluation and writes JSON results to settings.eval_output_dir.

Run:
    # ZenML-tracked run (both evaluations):
    python -m pipelines.evaluation_pipeline

    # RAG evaluation only:
    python -m pipelines.evaluation_pipeline --eval-type rag

    # Generation quality (fine-tuned adapter):
    python -m pipelines.evaluation_pipeline --eval-type gen \
        --model-id ./models/dpo/final_adapter

    # No ZenML tracking:
    python -m pipelines.evaluation_pipeline --no-zenml
"""

from zenml import pipeline

from steps.evaluation.rag_eval import evaluate_rag
from steps.evaluation.gen_eval import evaluate_generation


@pipeline(name="evaluation_pipeline_rag", enable_cache=False)
def rag_eval_pipeline(sample_size: int = 0) -> None:
    """RAG evaluation using RAGAS metrics (Chapter 7)."""
    evaluate_rag(sample_size=sample_size)


@pipeline(name="evaluation_pipeline_gen", enable_cache=False)
def gen_eval_pipeline(model_id: str = "", sample_size: int = 0) -> None:
    """Generation quality evaluation using ROUGE + BERTScore (Chapter 7)."""
    evaluate_generation(model_id=model_id, sample_size=sample_size)


@pipeline(name="evaluation_pipeline_full", enable_cache=False)
def full_eval_pipeline(model_id: str = "", sample_size: int = 0) -> None:
    """Combined RAG + generation quality evaluation."""
    evaluate_rag(sample_size=sample_size)
    evaluate_generation(model_id=model_id, sample_size=sample_size)


def _run_rag_local(sample_size: int) -> None:
    from steps.evaluation.rag_eval import evaluate_rag as _rag
    _rag.entrypoint(sample_size=sample_size)


def _run_gen_local(model_id: str, sample_size: int) -> None:
    from steps.evaluation.gen_eval import evaluate_generation as _gen
    _gen.entrypoint(model_id=model_id, sample_size=sample_size)


if __name__ == "__main__":
    import argparse
    from config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run the evaluation pipeline")
    parser.add_argument(
        "--eval-type",
        choices=["rag", "gen", "all"],
        default="all",
        help=(
            "rag = RAGAS retrieval metrics  |  "
            "gen = ROUGE + BERTScore  |  "
            "all = both (default)"
        ),
    )
    parser.add_argument(
        "--model-id",
        default="",
        help=(
            "HuggingFace model ID or local adapter path for generation eval. "
            "Defaults to Ollama (settings.eval_llm) if empty."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.eval_sample_size,
        help=f"Number of QA pairs to evaluate (default: {settings.eval_sample_size})",
    )
    parser.add_argument(
        "--no-zenml",
        action="store_true",
        help="Run steps directly without ZenML tracking",
    )
    args = parser.parse_args()

    if args.no_zenml:
        if args.eval_type in ("rag", "all"):
            _run_rag_local(args.sample_size)
        if args.eval_type in ("gen", "all"):
            _run_gen_local(args.model_id, args.sample_size)
    elif args.eval_type == "rag":
        rag_eval_pipeline(sample_size=args.sample_size)
    elif args.eval_type == "gen":
        gen_eval_pipeline(model_id=args.model_id, sample_size=args.sample_size)
    else:
        full_eval_pipeline(model_id=args.model_id, sample_size=args.sample_size)
