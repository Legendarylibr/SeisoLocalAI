"""One-time CLI runtime setup — mirrors Forge startup for inference/training."""

from __future__ import annotations

_bootstrap_done = False


def bootstrap_runtime() -> None:
    """Load CUDA libs and apply platform memory defaults (idempotent)."""
    global _bootstrap_done
    if _bootstrap_done:
        return
    try:
        from seiso.platform import ensure_cuda_library_path

        ensure_cuda_library_path()
    except ImportError:
        pass
    try:
        from seiso.memory.platform_profile import apply_platform_memory_profile

        apply_platform_memory_profile()
    except ImportError:
        pass
    _bootstrap_done = True