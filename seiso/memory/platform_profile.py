"""OS/tier memory defaults applied at Forge startup — setdefault only."""

from __future__ import annotations

import os
import platform
from typing import Any

from seiso.env import env_bool
from seiso.hardware.tiers import HardwareTier, classify_tier, vram_headroom_mb
from seiso.training.platform_caps import training_capabilities


def memory_profile_label(profile: dict[str, Any]) -> str:
    """Derive low vs balanced from live headroom."""
    headroom = vram_headroom_mb(profile)
    ram_gb = float(profile.get("ram_gb") or 0)
    if headroom < 4096 or (ram_gb > 0 and ram_gb <= 12):
        return "low"
    if headroom < 12288 or (ram_gb > 0 and ram_gb <= 24):
        return "balanced"
    return "balanced"


def apply_platform_memory_profile(
    *, profile: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    memory_caps_disabled = env_bool("SEISO_DISABLE_MEMORY_CAPS", False)
    low = not memory_caps_disabled and (
        os.environ.get("SEISO_MEMORY_PROFILE", "").strip().lower() == "low"
        or memory_profile_label(profile) == "low"
    )

    os.environ.setdefault("SEISO_LLAMA_USE_MMAP", "true")
    os.environ.setdefault("SEISO_LLAMA_USE_MLOCK", "false")
    os.environ.setdefault("SEISO_LLAMA_NO_PERF", "true")

    os.environ.setdefault("SEISO_LLAMA_PROMPT_CACHE", "true")
    cache_mb = "2048" if tier == HardwareTier.WORKSTATION and ram_gb >= 32 else "1024"
    if system == "Linux":
        try:
            from seiso.platform import is_native_linux_nvidia

            if is_native_linux_nvidia(profile=profile):
                cache_cap = 256 if low else 512
                cache_mb = str(min(int(cache_mb), cache_cap))
        except ImportError:
            pass
    os.environ.setdefault("SEISO_LLAMA_CACHE_MB", cache_mb)

    if system == "Darwin":
        if tier == HardwareTier.CPU_ONLY:
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
            os.environ.setdefault(
                "SEISO_LLAMA_THREADS", str(min(max((os.cpu_count() or 4) - 2, 2), 8))
            )
        elif tier == HardwareTier.APPLE_UNIFIED:
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
        if (
            tier == HardwareTier.APPLE_UNIFIED
            and ram_gb <= 24
            and not caps.get("supports_mlx_inference")
        ):
            os.environ.setdefault("SEISO_SKIP_MLX_PROBE", "1")

    elif system == "Windows":
        if caps.get("nvidia_hardware") or (
            caps.get("gpu_count", 0) > 0 and caps.get("vendor") == "nvidia"
        ):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
        elif caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")

    elif system == "Linux":
        if caps.get("wsl2"):
            data_dir = os.environ.get("SEISO_DATA_DIR", "")
            if data_dir.startswith("/mnt/"):
                import logging

                logging.getLogger(__name__).warning(
                    "SEISO_DATA_DIR on /mnt/ (Windows filesystem) — move to ~/ for better mmap performance"
                )
        if caps.get("nvidia_hardware") or (
            caps.get("gpu_count", 0) > 0 and caps.get("vendor") == "nvidia"
        ):
            # Only request GPU offload when the installed llama-cpp-python
            # wheel actually supports it; a CPU-only wheel would crash.
            _llama_gpu_ok = True
            try:
                from seiso.inference.model_pool import _llama_gpu_offload_ok

                _llama_gpu_ok = _llama_gpu_offload_ok()
            except ImportError:
                pass
            if _llama_gpu_ok:
                os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
            else:
                os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            if tier in (
                HardwareTier.WORKSTATION,
                HardwareTier.CAPABLE,
                HardwareTier.MODEST,
                HardwareTier.EDGE,
            ):
                from seiso.memory.protection import llama_batch_limits_for_headroom
                from seiso.platform import is_native_linux_nvidia

                batch, ubatch = llama_batch_limits_for_headroom(headroom)
                if is_native_linux_nvidia(profile=profile):
                    batch = min(batch, 512)
                    ubatch = min(ubatch, 128)
                    if low:
                        batch = min(batch, 256)
                        ubatch = min(ubatch, 128)
                    os.environ.setdefault("SEISO_LLAMA_SPEED_SCALE", "false")
                    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "false")
                os.environ.setdefault("SEISO_LLAMA_BATCH", str(batch))
                os.environ.setdefault("SEISO_LLAMA_UBATCH", str(ubatch))
            if not low and tier in (HardwareTier.WORKSTATION, HardwareTier.CAPABLE):
                from seiso.platform import is_native_linux_nvidia

                if not is_native_linux_nvidia(profile=profile):
                    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "true")
                os.environ.setdefault("SEISO_LLAMA_OP_OFFLOAD", "true")
                os.environ.setdefault("SEISO_LLAMA_OFFLOAD_KQV", "true")
                if tier == HardwareTier.WORKSTATION:
                    os.environ.setdefault("SEISO_STREAM_BATCH_CHARS", "16")
        elif caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")

    return {
        "memory_profile": memory_profile_label(profile),
        "tier": tier.value,
        "headroom_mb": headroom,
        "ram_gb": ram_gb,
        "os": system,
    }
