"""llama.cpp batch tier caps and clamping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from seiso.hardware import hardware_profile
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import (
    _LOAD_TIER_BATCH_CAPS,
    _MAX_LLAMA_BATCH,
    _MIN_LLAMA_BATCH,
    LlamaLoadTier,
)


def discrete_gpu_total_mb(profile: dict[str, Any] | None = None) -> int:
    """Largest discrete NVIDIA GPU VRAM total in MB, or 0 when unknown."""
    try:
        from seiso.hardware.tiers import discrete_vram_total_mb

        if profile is None:
            profile = hardware_profile()
        return max(int(discrete_vram_total_mb(profile)), 0)
    except Exception:
        return 0


def comfortable_vram_slack_ratio(*, gpu_total_mb: int | None = None) -> float:
    """Free/pre-load VRAM multiple of (weight+KV) that counts as a roomy fit."""
    total = gpu_total_mb if gpu_total_mb is not None else protection().discrete_gpu_total_mb()
    if total <= 0:
        return 1.75
    gpu_gb = total / 1024
    return max(1.35, min(2.25, 1.25 + gpu_gb / 48.0))


def gpu_batch_tier_caps(gpu_total_mb: int, load_tier: LlamaLoadTier) -> tuple[int, int]:
    """Scale llama.cpp batch ceilings with GPU VRAM instead of fixed tier tables."""
    if gpu_total_mb <= 0:
        return _LOAD_TIER_BATCH_CAPS.get(load_tier, _LOAD_TIER_BATCH_CAPS["normal"])
    gpu_gb = max(1.0, gpu_total_mb / 1024)
    scaled_batch = int(gpu_gb * 22)
    rounded_batch = (scaled_batch // _MIN_LLAMA_BATCH) * _MIN_LLAMA_BATCH
    normal_batch = min(
        _MAX_LLAMA_BATCH,
        max(_MIN_LLAMA_BATCH, rounded_batch),
    )
    normal_ubatch = min(512, max(_MIN_LLAMA_BATCH, normal_batch // 4))
    if load_tier == "compact":
        return min(normal_batch, max(256, normal_batch // 2)), min(normal_ubatch, 128)
    if load_tier == "minimal":
        return min(normal_batch, 256), min(normal_ubatch, 128)
    return normal_batch, normal_ubatch


def tight_batch_caps(gpu_total_mb: int) -> tuple[int, int]:
    """Conservative batch pair for tight VRAM fits on any GPU size."""
    batch, ubatch = gpu_batch_tier_caps(gpu_total_mb, "compact")
    return min(batch, 256), min(ubatch, 128)


def roomy_native_linux_batch_floor(
    *,
    model_path: str | Path,
    free_mb: int,
    gpu_total_mb: int,
    n_gpu_layers: int,
    load_tier: LlamaLoadTier = "normal",
    tight: bool = False,
) -> tuple[int, int] | None:
    """July-3-style GPU tier batch floor for tiny models on mostly-empty VRAM.

    Only applies when the model is small relative to *free* VRAM and most of the
    GPU capacity is still available (other processes have not consumed the card).
    """
    if (
        load_tier != "normal"
        or tight
        or n_gpu_layers == 0
        or gpu_total_mb <= 0
        or free_mb < int(gpu_total_mb * 0.65)
    ):
        return None
    weight_mb = int(protection().estimate_path_vram_mb(Path(model_path)))
    if weight_mb <= 0 or weight_mb > free_mb // 8:
        return None
    return gpu_batch_tier_caps(gpu_total_mb, "normal")


def clamp_llama_batch_pair(
    batch: int,
    ubatch: int,
    *,
    native_linux_nvidia: bool = False,
    load_tier: LlamaLoadTier = "normal",
    tight: bool = False,
    gpu_total_mb: int | None = None,
) -> tuple[int, int]:
    """Normalize a llama.cpp batch/ubatch pair (single source of ceilings)."""
    batch = max(_MIN_LLAMA_BATCH, int(batch))
    ubatch = max(_MIN_LLAMA_BATCH, min(int(ubatch), batch))
    gpu_total = (
        int(gpu_total_mb)
        if gpu_total_mb is not None and gpu_total_mb > 0
        else (protection().discrete_gpu_total_mb() if native_linux_nvidia else 0)
    )
    if native_linux_nvidia:
        if gpu_total > 0:
            tier_batch, tier_ubatch = gpu_batch_tier_caps(gpu_total, load_tier)
            if tight:
                tight_batch, tight_ubatch = tight_batch_caps(gpu_total)
                tier_batch = min(tier_batch, tight_batch)
                tier_ubatch = min(tier_ubatch, tight_ubatch)
        else:
            tier_batch, tier_ubatch = {
                "normal": (512, 128),
                "compact": (256, 128),
                "minimal": (256, 128),
            }.get(load_tier, (512, 128))
    else:
        tier_batch, tier_ubatch = _LOAD_TIER_BATCH_CAPS.get(
            load_tier, _LOAD_TIER_BATCH_CAPS["normal"]
        )
    batch = min(batch, tier_batch)
    ubatch = min(ubatch, tier_ubatch, batch)
    return batch, ubatch


def llama_batch_limits_for_headroom(headroom_mb_value: int) -> tuple[int, int]:
    """Largest llama.cpp batch/ubatch pair for available headroom (no platform cap)."""
    if headroom_mb_value < 2048:
        return 256, 128
    if headroom_mb_value < 4096:
        return 512, 128
    if headroom_mb_value < 8192:
        return 512, 256
    if headroom_mb_value < 16384:
        return 1024, 256
    if headroom_mb_value < 32768:
        return 2048, 512
    return 4096, 1024


def resolve_llama_batch_limits(
    headroom_mb_value: int,
    *,
    native_linux_nvidia: bool = False,
    load_tier: LlamaLoadTier = "normal",
    tight: bool = False,
) -> tuple[int, int]:
    """Headroom table plus platform/tier ceilings for a batch/ubatch pair."""
    batch, ubatch = llama_batch_limits_for_headroom(headroom_mb_value)
    return clamp_llama_batch_pair(
        batch,
        ubatch,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=load_tier,
        tight=tight,
    )


def llama_oom_recovery_batch(
    *,
    safe_batch: int,
    safe_ubatch: int,
    loaded_batch: int,
    next_tier: LlamaLoadTier,
) -> tuple[int, int]:
    """Next batch/ubatch after an inference OOM, clipped to the recovery tier."""
    # Always use the tighter native tier table — we already blew up once.
    if safe_batch > 0 and safe_ubatch > 0:
        return clamp_llama_batch_pair(
            min(safe_batch, loaded_batch or safe_batch) // 2,
            safe_ubatch // 2,
            native_linux_nvidia=True,
            load_tier=next_tier,
        )
    tier_batch, tier_ubatch = gpu_batch_tier_caps(
        protection().discrete_gpu_total_mb(), next_tier
    )
    return clamp_llama_batch_pair(
        tier_batch,
        tier_ubatch,
        native_linux_nvidia=True,
        load_tier=next_tier,
    )


