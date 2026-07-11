"""Training device map and dataset helpers."""

from __future__ import annotations

import os
from pathlib import Path

from seiso.memory.protection.constants import _MAX_JSONL_LOAD_MB


def jsonl_load_safe(path: Path) -> bool:
    """True when JSONL should use the datasets mmap loader.

    Always prefers mmap for local files (``load_training_dataset`` no longer
    size-gates). Kept for API compatibility with older call sites and tests.
    """
    _ = path, _MAX_JSONL_LOAD_MB
    return True


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        return {"": f"cuda:{local_rank}"}

    if device == "mps":
        return {"": "mps"}
    try:
        import torch

        if device == "cuda" or (device is None and torch.cuda.is_available()):
            return "auto"
    except ImportError:
        pass
    return None
