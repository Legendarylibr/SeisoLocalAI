"""Apple MLX GPU probing."""

from __future__ import annotations

import os
from typing import Any

from seiso.hardware.probes.common import sanitize_hardware_label


def _mlx_device_info() -> dict[str, Any]:
    """Return MLX device_info dict (new API preferred, metal fallback)."""
    import mlx.core as mx

    info_fn = getattr(mx, "device_info", None)
    if callable(info_fn):
        info = info_fn()
        if isinstance(info, dict):
            return info
    metal = getattr(mx, "metal", None)
    metal_info = getattr(metal, "device_info", None) if metal is not None else None
    if callable(metal_info):
        info = metal_info()
        if isinstance(info, dict):
            return info
    return {}


def _mlx_active_memory_bytes() -> int | None:
    import mlx.core as mx

    for owner in (mx, getattr(mx, "metal", None)):
        if owner is None:
            continue
        fn = getattr(owner, "get_active_memory", None)
        if callable(fn):
            try:
                return int(fn())
            except (TypeError, ValueError, RuntimeError):
                return None
    return None


def probe_apple_mlx_gpu() -> list[dict[str, Any]]:
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return []
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return []

    info = _mlx_device_info()
    device_name = sanitize_hardware_label(str(info.get("device_name") or "Apple GPU"))
    # Prefer full unified memory_size — NOT max_recommended_working_set_size,
    # which is a Metal soft budget (~70% of RAM) and under-reports capacity.
    memory_size = info.get("memory_size")
    vram_total_mb: float | None = None
    try:
        if memory_size is not None and int(memory_size) > 0:
            vram_total_mb = round(int(memory_size) / (1024**2), 1)
    except (TypeError, ValueError):
        vram_total_mb = None

    vram_used_mb: float | None = None
    active = _mlx_active_memory_bytes()
    if active is not None and active >= 0:
        vram_used_mb = round(active / (1024**2), 1)

    name = device_name if "mlx" in device_name.lower() else f"{device_name} (MLX)"
    return [
        {
            "index": 0,
            "name": name,
            "vram_total_mb": vram_total_mb,
            "vram_used_mb": vram_used_mb,
            "utilization_pct": None,
            "temperature_c": None,
            # Keep tier/headroom logic on unified RAM, not discrete-GPU paths.
            "unified_memory": True,
        }
    ]
