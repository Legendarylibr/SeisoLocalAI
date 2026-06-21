"""OS/tier memory defaults applied at Forge startup — setdefault only."""

from __future__ import annotations

import os
import platform
from typing import Any

from seiso.hardware.tiers import HardwareTier, classify_tier, vram_headroom_mb
from seiso.training.platform_caps import training_capabilities


def memory_profile_label(profile: dict[str, Any]) -> str:
    """Derive low vs balanced from live headroom."""
    headroom = vram_headroom_mb(profile)
    ram_gb = float(profile.get("ram_gb") or 0)
    if headroom < 6144 or (ram_gb > 0 and ram_gb <= 16):
        return "low"
    if headroom < 12288 or (ram_gb > 0 and ram_gb <= 24):
        return "balanced"
    return "balanced"


def apply_platform_memory_profile(*, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Apply lean RAM defaults for this OS/tier.

    Uses os.environ.setdefault — never overrides explicit user configuration.
    """
    if profile is None:
        from seiso.hardware.profile import hardware_profile

        profile = hardware_profile()

    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    ram_gb = float(profile.get("ram_gb") or 0)
    caps = training_capabilities()
    system = platform.system()
    low = os.environ.get("SEISO_MEMORY_PROFILE", "").strip().lower() == "low" or memory_profile_label(
        profile
    ) == "low"

    os.environ.setdefault("SEISO_LLAMA_USE_MMAP", "true")
    os.environ.setdefault("SEISO_LLAMA_USE_MLOCK", "false")
    os.environ.setdefault("SEISO_LLAMA_NO_PERF", "true")

    if low or headroom < 8192:
        os.environ.setdefault("SEISO_LLAMA_PROMPT_CACHE", "false")
        os.environ.setdefault("SEISO_LLAMA_CACHE_MB", "0")
    else:
        os.environ.setdefault("SEISO_LLAMA_PROMPT_CACHE", "true")
        cache_mb = "256" if headroom < 12288 else "512"
        os.environ.setdefault("SEISO_LLAMA_CACHE_MB", cache_mb)

    if system == "Darwin":
        if tier == HardwareTier.CPU_ONLY:
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
            os.environ.setdefault("SEISO_LLAMA_THREADS", str(min(max((os.cpu_count() or 4) - 2, 2), 8)))
        elif tier == HardwareTier.APPLE_UNIFIED and (ram_gb <= 24 or headroom < 12288):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512" if headroom >= 8192 else "256")
        if tier == HardwareTier.APPLE_UNIFIED and ram_gb <= 24 and not caps.get("supports_mlx_inference"):
            os.environ.setdefault("SEISO_SKIP_MLX_PROBE", "1")

    elif system == "Windows":
        if caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
        elif headroom < 8192:
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")

    elif system == "Linux":
        if caps.get("wsl2"):
            data_dir = os.environ.get("SEISO_DATA_DIR", "")
            if data_dir.startswith("/mnt/"):
                import logging

                logging.getLogger(__name__).warning(
                    "SEISO_DATA_DIR on /mnt/ (Windows filesystem) — move to ~/ for better mmap performance"
                )
        if caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")

    if low:
        os.environ.setdefault("SEISO_LLAMA_BATCH", "256")
        os.environ.setdefault("SEISO_LLAMA_UBATCH", "128")

    return {
        "memory_profile": memory_profile_label(profile),
        "tier": tier.value,
        "headroom_mb": headroom,
        "ram_gb": ram_gb,
        "os": system,
    }
