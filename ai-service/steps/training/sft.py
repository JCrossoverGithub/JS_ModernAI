"""
steps/training/sft.py — ZenML steps: dataset construction + SFT trainer.

Phase 4 (Chapter 5 — Supervised Fine-Tuning).
Stubs only — bodies implemented in Phase 4.
"""

from zenml import step


@step
def build_instruction_dataset(model_id: str, max_samples: int = 1000):
    """Build an instruction-response HuggingFace Dataset from MongoDB documents.

    Phase 4 TODO:
      - Query MongoDB raw_documents for the given model/user
      - Generate Q/A pairs using a teacher model
      - Return a datasets.Dataset object
    """
    raise NotImplementedError("Implemented in Phase 4 (Chapter 5)")


@step
def run_sft(base_model_id: str, dataset) -> str:
    """Fine-tune *base_model_id* with SFTTrainer (PEFT LoRA + bitsandbytes 4-bit).

    Phase 4 TODO:
      - Load model in 4-bit via BitsAndBytesConfig
      - Apply LoRA via get_peft_model
      - Train with TRL SFTTrainer
      - Save adapter weights, return local path
    """
    raise NotImplementedError("Implemented in Phase 4 (Chapter 5)")
