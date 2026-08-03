"""Direct Preference Optimization (DPO) alignment for LLM post-training.

Experimental alignment path, separate from the RL quantization policy trainers.
Requires optional deps: ``pip install -e ".[alignment]"``.
"""

from __future__ import annotations

import importlib
from typing import Any

from seiso.distill_rl.dpo.config import DPOSettings  # noqa: F401

_EAGER_EXPORTS = ("DPOSettings",)

_LAZY: dict[str, tuple[str, str]] = {
    "DPODataCollator": (
        "seiso.distill_rl.dpo.data_collator",
        "DPODataCollator",
    ),
    "DPOMetrics": ("seiso.distill_rl.dpo.dpo_loss", "DPOMetrics"),
    "DPOTrainer": ("seiso.distill_rl.dpo.dpo_trainer", "DPOTrainer"),
    "clone_reference_from_policy": (
        "seiso.distill_rl.dpo.model_loading",
        "clone_reference_from_policy",
    ),
    "compute_dpo_loss": (
        "seiso.distill_rl.dpo.dpo_loss",
        "compute_dpo_loss",
    ),
    "get_batch_logps": (
        "seiso.distill_rl.dpo.dpo_loss",
        "get_batch_logps",
    ),
    "load_policy_and_reference": (
        "seiso.distill_rl.dpo.model_loading",
        "load_policy_and_reference",
    ),
    "load_preference_dataset": (
        "seiso.distill_rl.dpo.preference_data",
        "load_preference_dataset",
    ),
}

__all__ = sorted((*_EAGER_EXPORTS, *_LAZY))


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    return getattr(importlib.import_module(module_name), attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
