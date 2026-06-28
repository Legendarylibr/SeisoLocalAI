"""Canonical local GPU enumeration — single source for profile, kernels, and caps."""

from __future__ import annotations

import os
import platform
import re
from functools import lru_cache
from typing import Any

_SERIAL_RE = re.compile(r"\b(serial|s/n|uuid)[:\s#-]*[\w-]+", re.I)
_HOST_RE = re.compile(r"@[\w.-]+")


def sanitize_hardware_label(raw: str, *, max_len: int = 64) -> str:
    text = _SERIAL_RE.sub("", raw)
    text = _HOST_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text or "Unknown"


def _torch_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = sanitize_hardware_label(props.name)
            total_mb = int(props.total_memory / (1024**2))
            used_mb: int | None = None
            try:
                free, total = torch.cuda.mem_get_info(i)
                used_mb = int((total - free) / (1024**2))
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "vram_total_mb": total_mb,
                    "vram_used_mb": used_mb,
                    "utilization_pct": None,
                    "temperature_c": None,
                }
            )
    except ImportError:
        pass
    return gpus


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    """Enumerate GPUs via nvidia-smi when PyTorch CUDA is unavailable (e.g. CPU-only wheel)."""
    if platform.system().lower() not in {"linux", "windows"}:
        return []
    from seiso.security.nvidia_boundary import query_nvidia_gpus

    gpus: list[dict[str, Any]] = []
    for item in query_nvidia_gpus():
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        total_raw = item.get("memory_total_mb")
        total_mb = (
            int(total_raw)
            if isinstance(total_raw, (int, float)) and total_raw > 0
            else None
        )
        gpus.append(
            {
                "index": int(item.get("index", len(gpus))),
                "name": sanitize_hardware_label(name),
                "vram_total_mb": total_mb,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        )
    return gpus


def _mlx_apple_gpu() -> list[dict[str, Any]]:
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return []
    try:
        import mlx.core as mx  # noqa: F401

        return [
            {
                "index": 0,
                "name": "Apple GPU (MLX)",
                "vram_total_mb": None,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        ]
    except ImportError:
        return []


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
