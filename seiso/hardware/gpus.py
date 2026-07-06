"""Canonical local GPU enumeration — single source for profile, kernels, and caps."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from seiso.hardware.probes.apple import probe_apple_mlx_gpu
from seiso.hardware.probes.common import sanitize_hardware_label
from seiso.hardware.probes.nvidia import probe_nvidia_gpus
from seiso.hardware.probes.torch_cuda import probe_torch_gpus

# Back-compat aliases for tests and monkeypatching.
_torch_gpus = probe_torch_gpus
_nvidia_smi_gpus = probe_nvidia_gpus
_mlx_apple_gpu = probe_apple_mlx_gpu

__all__ = [
    "clear_gpu_enumeration_cache",
    "enumerate_compute_gpus",
    "enumerate_gpus",
    "gpu_count",
    "primary_gpu_name",
    "sanitize_hardware_label",
]


@lru_cache(maxsize=1)
def enumerate_gpus(*, include_mlx: bool = True) -> tuple[dict[str, Any], ...]:
    """Return GPUs as an immutable tuple (cached per process).

    Order: PyTorch CUDA → nvidia-smi (Linux/Windows) → MLX (Darwin, optional).
    """
    gpus = _torch_gpus()
    if not gpus:
        gpus = _nvidia_smi_gpus()
    if not gpus and include_mlx:
        gpus = _mlx_apple_gpu()
    return tuple(gpus)


def clear_gpu_enumeration_cache() -> None:
    enumerate_gpus.cache_clear()


def enumerate_compute_gpus() -> tuple[dict[str, Any], ...]:
    """CUDA-capable GPUs only (no MLX) — for training kernel selection."""
    return enumerate_gpus(include_mlx=False)


def gpu_count(*, include_mlx: bool = True) -> int:
    return len(enumerate_gpus(include_mlx=include_mlx))


def primary_gpu_name(*, include_mlx: bool = True) -> str | None:
    gpus = enumerate_gpus(include_mlx=include_mlx)
    if not gpus:
        return None
    name = gpus[0].get("name")
    return str(name) if name else None
