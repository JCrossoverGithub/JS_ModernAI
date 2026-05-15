"""
steps/training/hub.py — ZenML step: save adapter locally; optionally push to HF Hub.

Phase 4 (Chapter 5).
  - Always writes a training_manifest.json next to the adapter.
  - Pushes to HuggingFace Hub only when huggingface_access_token AND
    finetuned_model_id are both configured (cloud deployment is otherwise deferred).
"""

import json
import os
from loguru import logger
from zenml import step


@step
def push_to_huggingface(adapter_path: str) -> None:
    """Save adapter manifest locally; push to HF Hub when credentials are configured.

    Args:
        adapter_path: Local filesystem path to the saved LoRA adapter directory.
    """
    from config import get_settings

    settings = get_settings()

    manifest = {
        "adapter_path": adapter_path,
        "base_model_id": settings.base_model_id,
        "pushed_to_hub": False,
        "hub_repo_id": None,
    }

    if settings.huggingface_access_token and settings.finetuned_model_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=settings.huggingface_access_token)
            api.upload_folder(
                folder_path=adapter_path,
                repo_id=settings.finetuned_model_id,
                repo_type="model",
            )
            manifest["pushed_to_hub"] = True
            manifest["hub_repo_id"] = settings.finetuned_model_id
            logger.info("Adapter pushed to HF Hub: %s", settings.finetuned_model_id)
        except Exception as exc:
            logger.error("HF Hub push failed (adapter still saved locally): %s", exc)
    else:
        logger.info(
            "HF token / finetuned_model_id not set — adapter stays local: %s",
            adapter_path,
        )

    manifest_path = os.path.join(adapter_path, "training_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Training manifest written → %s", manifest_path)
