"""
steps/training/dpo.py — ZenML steps: preference dataset + DPO trainer.

Phase 5 (Chapter 6 — Preference Alignment).
Stubs only — bodies implemented in Phase 5.
"""

from zenml import step


@step
def build_preference_dataset(model_id: str, max_samples: int = 500):
    """Build a chosen/rejected preference Dataset for DPO.

    Phase 5 TODO:
      - Sample prompts from MongoDB documents
      - Generate chosen (SFT model) and rejected (base model) responses
      - Return a datasets.Dataset with columns: prompt, chosen, rejected
    """
    raise NotImplementedError("Implemented in Phase 5 (Chapter 6)")


@step
def run_dpo(base_model_id: str, dataset) -> str:
    """Run DPO training on top of the SFT adapter.

    Phase 5 TODO:
      - Load SFT adapter in 4-bit
      - Train with TRL DPOTrainer
      - Save updated adapter, return local path
    """
    raise NotImplementedError("Implemented in Phase 5 (Chapter 6)")
