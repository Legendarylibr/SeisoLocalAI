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


def _discrete_nvidia_visible(caps: dict[str, Any]) -> bool:
    return bool(
        caps.get("nvidia_hardware")
        or (caps.get("gpu_count", 0) > 0 and caps.get("vendor") == "nvidia")
    )


def _llama_nvidia_offload_ok() -> bool:
    try:
        from seiso.inference.model_pool import _llama_gpu_offload_ok

        return _llama_gpu_offload_ok()
    except ImportError:
        return False


def _apply_llama_throughput_defaults(
    *, low: bool, headroom: int, ram_gb: float, mlx: bool = False
) -> None:
    """Throughput-oriented llama.cpp defaults when GPU offload has headroom."""
    if low:
        return
    if mlx:
        if headroom < 5120 and ram_gb < 20:
            return
        roomy = headroom >= 12288 or ram_gb >= 32
    else:
        # Discrete NVIDIA — size batches from free VRAM, not system RAM.
        if headroom < 5120:
            return
        roomy = headroom >= 12288
    if roomy:
        os.environ.setdefault("SEISO_LLAMA_BATCH", "8192")
        os.environ.setdefault("SEISO_LLAMA_UBATCH", "2048")
        stream_batch = "128"
    elif headroom >= 8192 or (mlx and ram_gb >= 20):
        os.environ.setdefault("SEISO_LLAMA_BATCH", "4096")
        os.environ.setdefault("SEISO_LLAMA_UBATCH", "1536")
        stream_batch = "96"
    elif not mlx and headroom >= 5120:
        os.environ.setdefault("SEISO_LLAMA_BATCH", "4096")
        os.environ.setdefault("SEISO_LLAMA_UBATCH", "1536")
        stream_batch = "96"
    else:
        return
    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "true")
    os.environ.setdefault("SEISO_LLAMA_OP_OFFLOAD", "true")
    os.environ.setdefault("SEISO_LLAMA_OFFLOAD_KQV", "true")
    os.environ.setdefault("SEISO_STREAM_BATCH_CHARS", stream_batch)
    if mlx:
        prefill = "8192" if ram_gb >= 20 else "4096"
        os.environ.setdefault("SEISO_MLX_PREFILL_STEP", prefill)
    cpus = os.cpu_count() or 4
    os.environ.setdefault("SEISO_LLAMA_THREADS_BATCH", str(min(cpus, 32)))


def _configure_discrete_nvidia_llama_defaults(
    *,
    low: bool,
    headroom: int,
    ram_gb: float,
    tier: HardwareTier,
    caps: dict[str, Any],
) -> None:
    """Shared Linux/Windows llama.cpp defaults for NVIDIA GPU offload."""
    if not _discrete_nvidia_visible(caps):
        if caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
        return
    if _llama_nvidia_offload_ok():
        os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
    else:
        os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
        return
    if not low and tier in (
        HardwareTier.WORKSTATION,
        HardwareTier.CAPABLE,
        HardwareTier.MODEST,
    ):
        _apply_llama_throughput_defaults(
            low=low, headroom=headroom, ram_gb=ram_gb, mlx=False
        )
    elif tier == HardwareTier.EDGE and headroom >= 2560:
        os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
        os.environ.setdefault("SEISO_LLAMA_UBATCH", "256")


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
    mlock_user_set = "SEISO_LLAMA_USE_MLOCK" in os.environ
    memory_caps_disabled = env_bool("SEISO_DISABLE_MEMORY_CAPS", False)
    low = not memory_caps_disabled and (
        os.environ.get("SEISO_MEMORY_PROFILE", "").strip().lower() == "low"
        or memory_profile_label(profile) == "low"
    )

    os.environ.setdefault("SEISO_LLAMA_USE_MMAP", "true")
    os.environ.setdefault("SEISO_LLAMA_USE_MLOCK", "false")
    os.environ.setdefault("SEISO_LLAMA_NO_PERF", "true")

    os.environ.setdefault("SEISO_LLAMA_PROMPT_CACHE", "true")
    if tier == HardwareTier.WORKSTATION and (ram_gb >= 32 or headroom >= 12288):
        cache_mb = "2048"
    elif tier == HardwareTier.APPLE_UNIFIED and ram_gb >= 24:
        cache_mb = "2048"
    else:
        cache_mb = "1024"
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
            _apply_llama_throughput_defaults(
                low=low, headroom=headroom, ram_gb=ram_gb, mlx=True
            )
            if (
                not low
                and ram_gb >= 32
                and headroom >= 8192
                and not mlock_user_set
            ):
                os.environ["SEISO_LLAMA_USE_MLOCK"] = "true"
        if (
            tier == HardwareTier.APPLE_UNIFIED
            and ram_gb <= 24
            and not caps.get("supports_mlx_inference")
        ):
            os.environ.setdefault("SEISO_SKIP_MLX_PROBE", "1")

    elif system == "Windows":
        _configure_discrete_nvidia_llama_defaults(
            low=low,
            headroom=headroom,
            ram_gb=ram_gb,
            tier=tier,
            caps=caps,
        )

    elif system == "Linux":
        if caps.get("wsl2"):
            data_dir = os.environ.get("SEISO_DATA_DIR", "")
            if data_dir.startswith("/mnt/"):
                import logging

                logging.getLogger(__name__).warning(
                    "SEISO_DATA_DIR on /mnt/ (Windows filesystem) — move to ~/ for better mmap performance"
                )
        _configure_discrete_nvidia_llama_defaults(
            low=low,
            headroom=headroom,
            ram_gb=ram_gb,
            tier=tier,
            caps=caps,
        )

    return {
        "memory_profile": memory_profile_label(profile),
        "tier": tier.value,
        "headroom_mb": headroom,
        "ram_gb": ram_gb,
        "os": system,
    }
