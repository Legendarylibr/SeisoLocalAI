"""Apple MLX GPU probing."""

from __future__ import annotations

import os
from typing import Any


def probe_apple_mlx_gpu() -> list[dict[str, Any]]:
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
