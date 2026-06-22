"""Low-VRAM training kernel mode — in-place fused ops and tuning."""

from __future__ import annotations

import os

_LOW_VRAM_HEADROOM_MB = 8192


def kernel_low_vram_enabled() -> bool:
    """True when fused kernels should prefer lowest VRAM paths."""
    raw = os.environ.get("SEISO_KERNEL_LOW_VRAM", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    try:
        from seiso.memory.protection import headroom_mb

        headroom = headroom_mb()
        return headroom > 0 and headroom < _LOW_VRAM_HEADROOM_MB
    except ImportError:
        return False


def apply_low_vram_kernel_tuning() -> None:
    """Pick launch configs that trade a little speed for lower peak shared memory."""
    try:
        from seiso.kernels.tuning import apply_kernel_profile

        apply_kernel_profile(3)  # narrow_opt
    except ImportError:
        try:
            from seiso.kernels.cuda_ops import set_kernel_tuning

            set_kernel_tuning(0, 4, 16)
        except ImportError:
            pass
