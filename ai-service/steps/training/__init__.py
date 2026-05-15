"""
steps/training/__init__.py — Stub exports for the training pipeline.

Implemented in Phase 4 (SFT) and Phase 5 (DPO).
"""

from .sft import build_instruction_dataset, run_sft
from .dpo import build_preference_dataset, run_dpo
from .hub import push_to_huggingface

__all__ = [
    "build_instruction_dataset",
    "run_sft",
    "build_preference_dataset",
    "run_dpo",
    "push_to_huggingface",
]
