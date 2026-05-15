"""
steps/training/hub.py — ZenML step: push fine-tuned adapter to HuggingFace Hub.

Phase 4+ (Chapters 5-6).
Stub only — body implemented in Phase 4.
"""

from zenml import step


@step
def push_to_huggingface(adapter_path: str) -> None:
    """Upload the LoRA adapter at *adapter_path* to the HuggingFace Hub.

    Phase 4 TODO:
      - Authenticate with settings.huggingface_access_token
      - Push adapter via huggingface_hub.upload_folder
      - Log run URL to Comet ML experiment
    """
    raise NotImplementedError("Implemented in Phase 4 (Chapter 5)")
