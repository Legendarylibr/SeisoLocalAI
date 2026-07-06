"""OOM detection and cache release."""

from __future__ import annotations

import gc
import os

from seiso.memory.protection.constants import *  # noqa: F403

class MemoryLoadBlockedError(RuntimeError):
    """Raised when a model load would exceed available memory."""


def allow_memory_overcommit() -> bool:
    """When true, log warnings instead of blocking oversized loads."""
    return env_bool("SEISO_ALLOW_MEMORY_OVERCOMMIT", False)


def is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA/MPS/CPU out-of-memory failures across backends."""
    if exc is None:
        return False
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "AcceleratorError"}:
        return True
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "mps out of memory",
        "insufficient memory",
        "failed to allocate",
        "cannot allocate memory",
    )
    return any(n in msg for n in needles)


def release_cached_memory(*, sync: bool = False) -> None:
    """Best-effort GPU/RAM cache release."""
    gc.collect()
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            import mlx.core as mx  # pylint: disable=import-error,no-name-in-module

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if sync:
                torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


