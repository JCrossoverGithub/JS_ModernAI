"""
pipelines/training_pipeline.py — ZenML training pipeline.

Implements Chapters 5-6 of the LLM Engineer's Handbook:
  Phase 4 (Ch 5): Supervised Fine-Tuning (SFT) with PEFT/LoRA + TRL
  Phase 5 (Ch 6): Preference Alignment (DPO)

Run:
    # ZenML-tracked run:
    python -m pipelines.training_pipeline --finetuning-type sft
    python -m pipelines.training_pipeline --finetuning-type dpo --model-id ./models/sft/final_adapter

    # Lightweight local run (no ZenML tracking, useful for quick tests):
    python -m pipelines.training_pipeline --finetuning-type sft --no-zenml
    python -m pipelines.training_pipeline --finetuning-type dpo --model-id ./models/sft/final_adapter --no-zenml
"""

from zenml import pipeline

from steps.training import (
    build_instruction_dataset,
    run_sft,
    build_preference_dataset,
    run_dpo,
    push_to_huggingface,
)


@pipeline(name="training_pipeline_sft", enable_cache=False)
def sft_pipeline(model_id: str, dataset_size: int = 1000) -> None:
    """Supervised fine-tuning pipeline (Chapter 5)."""
    dataset = build_instruction_dataset(model_id=model_id, max_samples=dataset_size)
    adapter_path = run_sft(base_model_id=model_id, dataset=dataset)
    push_to_huggingface(adapter_path=adapter_path)


@pipeline(name="training_pipeline_dpo", enable_cache=False)
def dpo_pipeline(sft_model_id: str, dataset_size: int = 500) -> None:
    """Preference alignment pipeline via DPO (Chapter 6)."""
    dataset = build_preference_dataset(model_id=sft_model_id, max_samples=dataset_size)
    adapter_path = run_dpo(base_model_id=sft_model_id, dataset=dataset)
    push_to_huggingface(adapter_path=adapter_path)


def _run_sft_local(model_id: str, dataset_size: int) -> None:
    """Run the SFT pipeline directly without ZenML tracking (for quick iteration)."""
    from steps.training.sft import build_instruction_dataset as _build, run_sft as _sft
    from steps.training.hub import push_to_huggingface as _push

    # Call the underlying functions directly (bypassing ZenML @step decorator)
    dataset = _build.entrypoint(model_id=model_id, max_samples=dataset_size)
    adapter_path = _sft.entrypoint(base_model_id=model_id, dataset=dataset)
    _push.entrypoint(adapter_path=adapter_path)


def _run_dpo_local(model_id: str, dataset_size: int) -> None:
    """Run the DPO pipeline directly without ZenML tracking."""
    from steps.training.dpo import build_preference_dataset as _build, run_dpo as _dpo
    from steps.training.hub import push_to_huggingface as _push

    dataset = _build.entrypoint(model_id=model_id, max_samples=dataset_size)
    adapter_path = _dpo.entrypoint(base_model_id=model_id, dataset=dataset)
    _push.entrypoint(adapter_path=adapter_path)


if __name__ == "__main__":
    import argparse
    from config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run the training pipeline")
    parser.add_argument(
        "--finetuning-type",
        choices=["sft", "dpo"],
        default="sft",
        help="sft = supervised fine-tuning  |  dpo = preference alignment",
    )
    parser.add_argument(
        "--model-id",
        default=settings.base_model_id,
        help="HuggingFace model ID to fine-tune",
    )
    parser.add_argument("--dataset-size", type=int, default=settings.sft_max_samples)
    parser.add_argument(
        "--no-zenml",
        action="store_true",
        help="Skip ZenML tracking and run steps directly (faster for local testing)",
    )
    args = parser.parse_args()

    if args.no_zenml:
        if args.finetuning_type == "sft":
            _run_sft_local(args.model_id, args.dataset_size)
        else:
            _run_dpo_local(args.model_id, args.dataset_size)
    elif args.finetuning_type == "sft":
        sft_pipeline(model_id=args.model_id, dataset_size=args.dataset_size)
    else:
        dpo_pipeline(sft_model_id=args.model_id, dataset_size=args.dataset_size)
