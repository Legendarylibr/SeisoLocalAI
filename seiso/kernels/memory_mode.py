"""Low-VRAM training kernel mode — in-place fused ops and tuning."""

from __future__ import annotations

from seiso.env import env_bool


def kernel_low_vram_enabled() -> bool:
    """True when fused kernels should prefer lowest VRAM paths by explicit opt-in."""
    return env_bool("SEISO_KERNEL_LOW_VRAM", False)


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
