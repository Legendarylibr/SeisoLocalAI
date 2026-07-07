"""OS/tier memory defaults applied at Forge startup — setdefault only."""

from __future__ import annotations

import os
import platform
from typing import Any

from seiso.env import env_bool
from seiso.hardware.tiers import (
    HardwareTier,
    classify_tier,
    performance_headroom_mb,
    vram_headroom_mb,
)
from seiso.memory.protection import (
    discrete_gpu_total_mb,
    gpu_batch_tier_caps,
    llama_batch_limits_for_headroom,
)
from seiso.memory.protection.constants import _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
from seiso.training.platform_caps import training_capabilities


def native_linux_nvidia_llama_batch_caps(
    *,
    tier: HardwareTier,
    headroom_mb: int,
    low: bool,
    gpu_total_mb: int = 0,
) -> tuple[int, int, int]:
    """VRAM-derived llama.cpp batch/ubatch/cache caps for native Linux NVIDIA."""
    total = gpu_total_mb or discrete_gpu_total_mb()
    if total > 0:
        batch, ubatch = gpu_batch_tier_caps(total, "normal")
    else:
        batch, ubatch = _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
    if low:
        if total > 0:
            low_batch, low_ubatch = gpu_batch_tier_caps(total, "compact")
        else:
            low_batch, low_ubatch = _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
        batch = min(batch, low_batch)
        ubatch = min(ubatch, low_ubatch, batch)
    cache_cap = min(2048, max(256, batch * 2))
    return batch, ubatch, cache_cap


def _refresh_native_linux_llama_env(
    *, batch_cap: int, ubatch_cap: int, cache_cap: int
) -> None:
    """Pin native Linux llama.cpp batch env to VRAM-derived caps on every Forge start."""
    if env_bool("SEISO_DISABLE_MEMORY_CAPS", False):
        return

    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "false")
    os.environ.setdefault("SEISO_LLAMA_KV_QUANT", "false")
    os.environ.setdefault("SEISO_CHAT_CONTEXT_CHARS", "12000")
    os.environ["SEISO_LLAMA_BATCH"] = str(batch_cap)
    os.environ["SEISO_LLAMA_UBATCH"] = str(min(ubatch_cap, batch_cap))
    os.environ["SEISO_LLAMA_CACHE_MB"] = str(cache_cap)
    if env_bool("SEISO_LLAMA_CONSERVATIVE", False):
        os.environ["SEISO_LLAMA_FLASH_ATTN"] = "false"
        os.environ["SEISO_LLAMA_KV_QUANT"] = "false"


def memory_profile_label(profile: dict[str, Any]) -> str:
    """Derive low vs balanced from hardware budget (not transient free VRAM)."""
    headroom = performance_headroom_mb(profile)
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
    free_headroom = vram_headroom_mb(profile)
    headroom = performance_headroom_mb(profile)
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
    native_linux_nvidia = False
    if system == "Linux":
        try:
            from seiso.platform import is_native_linux_nvidia

            native_linux_nvidia = is_native_linux_nvidia(profile=profile)
        except ImportError:
            pass
    # Native Linux sets cache from batch_caps once below; avoid a second, looser cap.
    if not native_linux_nvidia:
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
                os.environ.setdefault(
                    "SEISO_LLAMA_THREADS",
                    str(
                        min(
                            max((os.cpu_count() or 4) - 2, 4),
                            14 if tier == HardwareTier.WORKSTATION else 10,
                        )
                    ),
                )
                if native_linux_nvidia:
                    gpu_total = discrete_gpu_total_mb(profile)
                    batch, ubatch, cache_cap = native_linux_nvidia_llama_batch_caps(
                        tier=tier,
                        headroom_mb=headroom,
                        low=low,
                        gpu_total_mb=gpu_total,
                    )
                    _refresh_native_linux_llama_env(
                        batch_cap=batch,
                        ubatch_cap=ubatch,
                        cache_cap=min(int(cache_mb), cache_cap),
                    )
                else:
                    batch, ubatch = llama_batch_limits_for_headroom(headroom)
                    os.environ.setdefault("SEISO_LLAMA_BATCH", str(batch))
                    os.environ.setdefault("SEISO_LLAMA_UBATCH", str(ubatch))
            elif native_linux_nvidia:
                gpu_total = discrete_gpu_total_mb(profile)
                batch, ubatch, cache_cap = native_linux_nvidia_llama_batch_caps(
                    tier=tier,
                    headroom_mb=headroom,
                    low=low,
                    gpu_total_mb=gpu_total,
                )
                _refresh_native_linux_llama_env(
                    batch_cap=batch,
                    ubatch_cap=ubatch,
                    cache_cap=cache_cap,
                )
            if not low and tier in (HardwareTier.WORKSTATION, HardwareTier.CAPABLE):
                if not native_linux_nvidia:
                    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "true")
                os.environ.setdefault("SEISO_LLAMA_OP_OFFLOAD", "true")
                os.environ.setdefault("SEISO_LLAMA_OFFLOAD_KQV", "true")
                if tier == HardwareTier.WORKSTATION:
                    os.environ.setdefault("SEISO_STREAM_BATCH_CHARS", "16")
        elif caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            if native_linux_nvidia:
                os.environ.setdefault("SEISO_LLAMA_CACHE_MB", "256")

    result = {
        "memory_profile": memory_profile_label(profile),
        "tier": tier.value,
        "headroom_mb": headroom,
        "free_headroom_mb": free_headroom,
        "ram_gb": ram_gb,
        "os": system,
    }
    _log_platform_profile_applied(
        profile=profile,
        system=system,
        tier=tier,
        headroom=headroom,
        low=low,
    )
    return result


def _log_platform_profile_applied(
    *,
    profile: dict[str, Any],
    system: str,
    tier: HardwareTier,
    headroom: int,
    low: bool,
) -> None:
    _ = (profile, system, tier, headroom, low)
