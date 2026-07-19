"""Training device map helpers."""

from __future__ import annotations

import os


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
