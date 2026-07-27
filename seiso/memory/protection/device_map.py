"""Training device map helpers."""

from __future__ import annotations


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    from seiso.training.multi_gpu import resolve_distributed_env

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        device_count = int(torch.cuda.device_count()) if cuda_ok else 0
    except ImportError:
        cuda_ok = False
        device_count = 0

    dist_env = resolve_distributed_env(device_count)
    if dist_env.enabled:
        return {"": f"cuda:{dist_env.local_rank}"}

    if device == "mps":
        return {"": "mps"}
    if device == "cuda" or (device is None and cuda_ok):
        return "auto"
    return None
