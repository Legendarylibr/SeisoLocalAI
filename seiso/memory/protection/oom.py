"""OOM detection and cache release."""

from __future__ import annotations

import gc
import sys

from seiso.env import env_bool


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
        "failed to create llama_context",
        "failed to create llama context",
        "cannot allocate memory",
    )
    return any(n in msg for n in needles)


def release_cached_memory(*, sync: bool = False) -> None:
    """Best-effort GPU/RAM cache release.

    Only touches MLX/torch when already imported so Free memory does not pull
    native runtimes into a lean idle process. ``SEISO_SKIP_MLX_PROBE`` must not
    block Metal reclaim when ``mlx.core`` is already loaded.
    """
    gc.collect()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            metal = getattr(mx, "metal", None)
            clear_cache = getattr(metal, "clear_cache", None) if metal is not None else None
            if callable(clear_cache):
                clear_cache()
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if sync:
                torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            if sync and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        pass


