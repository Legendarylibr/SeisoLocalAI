"""Training device map helpers."""

from __future__ import annotations

import os


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            device_count = int(torch.cuda.device_count())
            # Stale WORLD_SIZE from a prior Accelerate job must not pin device_map.
            if world_size > device_count or local_rank >= device_count or world_size < 1:
                world_size = 1
                local_rank = 0
    except ImportError:
        cuda_ok = False

    if world_size > 1:
        return {"": f"cuda:{local_rank}"}

    if device == "mps":
        return {"": "mps"}
    if device == "cuda" or (device is None and cuda_ok):
        return "auto"
    return None
