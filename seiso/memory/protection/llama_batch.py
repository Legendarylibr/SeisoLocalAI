"""llama.cpp batch tier caps and clamping."""

from __future__ import annotations

from typing import Any

from seiso.hardware import hardware_profile
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import (
    _LOAD_TIER_BATCH_CAPS,
    _MAX_LLAMA_BATCH,
    _MIN_LLAMA_BATCH,
    _NATIVE_LINUX_BATCH_TOKENS_PER_GB,
    _NATIVE_LINUX_COMPACT_BATCH_FLOOR,
    _NATIVE_LINUX_COMPACT_UBATCH_FLOOR,
    _NATIVE_LINUX_MAX_NORMAL_BATCH,
    _NATIVE_LINUX_MINIMAL_BATCH_FLOOR,
    _NATIVE_LINUX_MINIMAL_UBATCH_FLOOR,
    _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS,
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
        if load_tier == "compact":
            return _NATIVE_LINUX_COMPACT_BATCH_FLOOR, _NATIVE_LINUX_COMPACT_UBATCH_FLOOR
        if load_tier == "minimal":
            return _NATIVE_LINUX_MINIMAL_BATCH_FLOOR, _NATIVE_LINUX_MINIMAL_UBATCH_FLOOR
        return _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
    gpu_gb = max(1.0, gpu_total_mb / 1024)
    scaled_batch = int(gpu_gb * _NATIVE_LINUX_BATCH_TOKENS_PER_GB)
    rounded_batch = (scaled_batch // _MIN_LLAMA_BATCH) * _MIN_LLAMA_BATCH
    normal_batch = min(
        _NATIVE_LINUX_MAX_NORMAL_BATCH,
        _MAX_LLAMA_BATCH,
        max(_MIN_LLAMA_BATCH, rounded_batch),
    )
    normal_ubatch = min(64, max(_NATIVE_LINUX_COMPACT_UBATCH_FLOOR, normal_batch // 4))
    if load_tier == "compact":
        compact_batch = max(_NATIVE_LINUX_COMPACT_BATCH_FLOOR, normal_batch // 2)
        compact_ubatch = max(
            _NATIVE_LINUX_COMPACT_UBATCH_FLOOR,
            min(32, normal_ubatch, compact_batch // 2),
        )
        return compact_batch, compact_ubatch
    if load_tier == "minimal":
        compact_batch, compact_ubatch = gpu_batch_tier_caps(gpu_total_mb, "compact")
        minimal_batch = max(_NATIVE_LINUX_MINIMAL_BATCH_FLOOR, compact_batch // 2)
        minimal_ubatch = max(
            _NATIVE_LINUX_MINIMAL_UBATCH_FLOOR,
            min(compact_ubatch, minimal_batch // 2),
        )
        return minimal_batch, minimal_ubatch
    return normal_batch, normal_ubatch


def cap_llama_batch_for_context(batch: int, ubatch: int, n_ctx: int) -> tuple[int, int]:
    """Reduce native Linux prefill/decode batches as KV context gets large."""
    ctx = max(0, int(n_ctx))
    if ctx >= 32768:
        batch = min(batch, 32)
        ubatch = min(ubatch, 16)
    elif ctx >= 16384:
        batch = min(batch, 64)
        ubatch = min(ubatch, 32)
    elif ctx >= 8192:
        batch = min(batch, 128)
        ubatch = min(ubatch, 64)
    return max(1, batch), max(1, min(ubatch, batch))


def native_linux_batch_defaults(gpu_total_mb: int | None = None) -> tuple[int, int]:
    """VRAM-tier batch pair for native Linux when no model-specific headroom is available."""
    total = (
        int(gpu_total_mb)
        if gpu_total_mb is not None and gpu_total_mb > 0
        else protection().discrete_gpu_total_mb()
    )
    if total > 0:
        return gpu_batch_tier_caps(total, "normal")
    return _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS


def tight_batch_caps(gpu_total_mb: int) -> tuple[int, int]:
    """Conservative batch pair for tight VRAM fits on any GPU size."""
    batch, ubatch = gpu_batch_tier_caps(gpu_total_mb, "compact")
    return min(batch, 256), min(ubatch, 128)


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
    min_batch = _MIN_LLAMA_BATCH
    min_ubatch = _MIN_LLAMA_BATCH
    if native_linux_nvidia:
        min_batch = _NATIVE_LINUX_COMPACT_BATCH_FLOOR
        min_ubatch = _NATIVE_LINUX_COMPACT_UBATCH_FLOOR
        if load_tier == "compact":
            min_batch = _NATIVE_LINUX_COMPACT_BATCH_FLOOR
            min_ubatch = _NATIVE_LINUX_COMPACT_UBATCH_FLOOR
        elif load_tier == "minimal":
            min_batch = _NATIVE_LINUX_MINIMAL_BATCH_FLOOR
            min_ubatch = _NATIVE_LINUX_MINIMAL_UBATCH_FLOOR
    batch = max(min_batch, int(batch))
    ubatch = max(min_ubatch, min(int(ubatch), batch))
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
            tier_batch, tier_ubatch = _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
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
    loaded_ubatch: int = 0,
    next_tier: LlamaLoadTier,
) -> tuple[int, int]:
    """Next batch/ubatch after an inference OOM, clipped to the recovery tier."""

    def cap_to_loaded(batch: int, ubatch: int) -> tuple[int, int]:
        if loaded_batch > 0:
            batch = min(batch, loaded_batch)
        if loaded_ubatch > 0:
            ubatch = min(ubatch, loaded_ubatch)
        return batch, min(ubatch, batch)

    # Always use the tighter native tier table — we already blew up once.
    if safe_batch > 0 and safe_ubatch > 0:
        batch, ubatch = clamp_llama_batch_pair(
            min(safe_batch, loaded_batch or safe_batch) // 2,
            min(safe_ubatch, loaded_ubatch or safe_ubatch) // 2,
            native_linux_nvidia=True,
            load_tier=next_tier,
        )
        return cap_to_loaded(batch, ubatch)
    tier_batch, tier_ubatch = gpu_batch_tier_caps(protection().discrete_gpu_total_mb(), next_tier)
    batch, ubatch = clamp_llama_batch_pair(
        tier_batch,
        tier_ubatch,
        native_linux_nvidia=True,
        load_tier=next_tier,
    )
    return cap_to_loaded(batch, ubatch)
