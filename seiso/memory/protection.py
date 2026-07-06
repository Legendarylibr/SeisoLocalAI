"""Cross-cutting OOM prevention — headroom probes, cache release, and fallbacks."""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import platform
import re
from pathlib import Path
from typing import Any, Literal

from seiso import platform as seiso_platform
from seiso.env import env_bool
from seiso.hardware import (
    assess_hardware_fit,
    hardware_profile,
    training_defaults,
    vram_headroom_mb,
)
from seiso.hardware.tiers import fit_headroom_mb
from seiso.inference.backends import gguf_total_layers
from seiso.io.files import iter_matching_files
from seiso.memory.estimates import (
    estimate_chat_vram_gb,
    estimate_training_vram_gb,
    guess_params_from_name,
)

logger = logging.getLogger(__name__)

# Reserve a slice of free memory for OS / display / other processes.
_DEFAULT_RESERVE_RATIO = 0.03
# Generation + activations overhead on top of weight estimate.
_INFERENCE_OVERHEAD_MB = 256
_TRAINING_OVERHEAD_RATIO = 2.0
# Absolute ceilings — never exceed even on large machines.
_MAX_INFERENCE_TOKENS = 8192
_MAX_LLAMA_CTX = 131072
_MIN_LLAMA_CTX = 2048
_MAX_LLAMA_BATCH = 4096
_MIN_LLAMA_BATCH = 128
# Coarse n_ctx buckets — avoid reloading the model every few chat turns.
_LLAMA_CTX_BUCKETS = (
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)
_NATIVE_LINUX_CTX_BUCKETS = (
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    65536,
    131072,
)
# Post-weight headroom below this on native Linux → clamp batches (prefill crash zone).
_NATIVE_LINUX_PREFILL_CLAMP_MB = 6144
_NATIVE_LINUX_PREFILL_HEADROOM_DROP_RATIO = 0.85
_NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO = 0.92
_NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB = 256
_TIGHT_VRAM_FIT_RATIO = 0.65
_NATIVE_LINUX_TIGHT_VRAM_FIT_RATIO = 0.60
_MAX_JSONL_LOAD_MB = 512
_MODEL_WEIGHT_VRAM_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin"})

_VRAM_ESTIMATE_CACHE_MAX = 256
_vram_estimate_cache: dict[tuple, int] = {}

LlamaLoadTier = Literal["normal", "compact", "minimal"]

# Load-tier recovery ceilings — absolute fallbacks when GPU total is unknown.
_LOAD_TIER_BATCH_CAPS: dict[LlamaLoadTier, tuple[int, int]] = {
    "normal": (_MAX_LLAMA_BATCH, 1024),
    "compact": (512, 128),
    "minimal": (256, 128),
}


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
    total = gpu_total_mb if gpu_total_mb is not None else discrete_gpu_total_mb()
    if total <= 0:
        return 1.75
    gpu_gb = total / 1024
    return max(1.35, min(2.25, 1.25 + gpu_gb / 48.0))


def gpu_batch_tier_caps(gpu_total_mb: int, load_tier: LlamaLoadTier) -> tuple[int, int]:
    """Scale llama.cpp batch ceilings with GPU VRAM instead of fixed tier tables."""
    if gpu_total_mb <= 0:
        return _LOAD_TIER_BATCH_CAPS.get(load_tier, _LOAD_TIER_BATCH_CAPS["normal"])
    gpu_gb = max(1.0, gpu_total_mb / 1024)
    normal_batch = min(
        _MAX_LLAMA_BATCH,
        max(_MIN_LLAMA_BATCH, int(gpu_gb * 43)),
    )
    normal_ubatch = min(1024, max(_MIN_LLAMA_BATCH, normal_batch // 4))
    if load_tier == "compact":
        return min(normal_batch, max(256, normal_batch // 2)), min(normal_ubatch, 128)
    if load_tier == "minimal":
        return min(normal_batch, 256), min(normal_ubatch, 128)
    return normal_batch, normal_ubatch


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
) -> tuple[int, int]:
    """Normalize a llama.cpp batch/ubatch pair (single source of ceilings)."""
    batch = max(_MIN_LLAMA_BATCH, int(batch))
    ubatch = max(_MIN_LLAMA_BATCH, min(int(ubatch), batch))
    gpu_total = discrete_gpu_total_mb() if native_linux_nvidia else 0
    if native_linux_nvidia and gpu_total > 0:
        tier_batch, tier_ubatch = gpu_batch_tier_caps(gpu_total, load_tier)
        if tight:
            tight_batch, tight_ubatch = tight_batch_caps(gpu_total)
            tier_batch = min(tier_batch, tight_batch)
            tier_ubatch = min(tier_ubatch, tight_ubatch)
    else:
        tier_batch, tier_ubatch = _LOAD_TIER_BATCH_CAPS.get(
            load_tier, _LOAD_TIER_BATCH_CAPS["normal"]
        )
    batch = min(batch, tier_batch)
    ubatch = min(ubatch, tier_ubatch, batch)
    return batch, ubatch


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
        discrete_gpu_total_mb(), next_tier
    )
    return clamp_llama_batch_pair(
        tier_batch,
        tier_ubatch,
        native_linux_nvidia=True,
        load_tier=next_tier,
    )


def _path_stat_key(p: Path) -> tuple | None:
    try:
        stat = p.stat()
        resolved = str(p.resolve())
        if p.is_file():
            return ("file", resolved, stat.st_mtime, stat.st_size)
        if p.is_dir():
            return ("dir", resolved, stat.st_mtime)
    except OSError:
        return None
    return None


def estimate_path_vram_mb(path: str | Path, *, mode: str = "chat") -> int:
    """Conservative runtime memory estimate from path, size, or name."""
    p = Path(path).expanduser()
    cache_key = _path_stat_key(p)
    if cache_key is not None:
        cached = _vram_estimate_cache.get((cache_key, mode))
        if cached is not None:
            return cached

    est = _estimate_path_vram_mb_uncached(p, mode=mode)

    if cache_key is not None:
        if len(_vram_estimate_cache) >= _VRAM_ESTIMATE_CACHE_MAX:
            _vram_estimate_cache.pop(next(iter(_vram_estimate_cache)))
        _vram_estimate_cache[(cache_key, mode)] = est
    return est


def _hub_model_vram_mb(path_str: str, *, mode: str) -> int | None:
    """VRAM estimate for HuggingFace repo ids (not local paths)."""
    from seiso.models.hub_quant import (
        infer_active_params_b,
        is_hub_model_id,
        is_native_hub_quant_model,
        peek_hub_config,
    )

    if not is_hub_model_id(path_str):
        return None

    config = peek_hub_config(path_str)
    params_b = infer_active_params_b(path_str, config=config)
    label = f"{params_b:g}B"
    native = is_native_hub_quant_model(path_str, config=config, peek=False)
    quant = "mxfp4" if native and mode == "train" else ("Q8_0" if native else "4bit")
    est_gb = (
        estimate_training_vram_gb(label, quant=quant, repo_id=path_str)
        if mode == "train"
        else estimate_chat_vram_gb(label, quant=quant, repo_id=path_str)
    )
    return int(est_gb * 1024)


def _estimate_path_vram_mb_uncached(p: Path, *, mode: str = "chat") -> int:
    name = p.name.lower()
    path_str = str(p)
    from_hub = False

    if not p.exists():
        hub_est = _hub_model_vram_mb(path_str, mode=mode)
        if hub_est is not None:
            return hub_est

    if p.is_file() and p.suffix.lower() in {
        ".gguf",
        ".bin",
        ".safetensors",
        ".pt",
        ".pth",
    }:
        file_mb = max(p.stat().st_size / (1024**2), 1)
        if p.suffix.lower() == ".gguf":
            # GGUF weights map ~1:1 to VRAM when fully offloaded; add KV/activation headroom.
            est = int(file_mb + _INFERENCE_OVERHEAD_MB)
        else:
            est = int(file_mb * 1.15 + _INFERENCE_OVERHEAD_MB)
    elif p.is_dir():
        weight_bytes = 0
        has_gguf = False
        for f in iter_matching_files(p, suffixes=_MODEL_WEIGHT_VRAM_SUFFIXES):
            suffix = f.suffix.lower()
            has_gguf = has_gguf or suffix == ".gguf"
            weight_bytes += f.stat().st_size
        if weight_bytes > 0:
            weight_mb = weight_bytes / (1024**2)
            if has_gguf:
                est = int(weight_mb + _INFERENCE_OVERHEAD_MB)
            else:
                est = int(weight_mb * 1.15 + _INFERENCE_OVERHEAD_MB)
        else:
            guessed = guess_params_from_name(name) or 7.0
            est = int(estimate_chat_vram_gb(f"{guessed}B") * 1024)
    else:
        hub_est = _hub_model_vram_mb(path_str, mode=mode)
        if hub_est is not None:
            est = hub_est
            from_hub = True
        else:
            guessed = guess_params_from_name(name) or guess_params_from_name(path_str) or 7.0
            est = int(estimate_chat_vram_gb(f"{guessed}B", repo_id=path_str) * 1024)
            if mode == "train":
                est = int(
                    estimate_training_vram_gb(
                        f"{guessed}B",
                        quant="4bit",
                        repo_id=path_str,
                    )
                    * 1024
                )

    if mode == "train" and not from_hub and p.exists():
        est = int(est * _TRAINING_OVERHEAD_RATIO)
    return max(est, 256)


class MemoryLoadBlockedError(RuntimeError):
    """Raised when a model load would exceed available memory."""


def allow_memory_overcommit() -> bool:
    """When true, log warnings instead of blocking oversized loads."""
    return env_bool("SEISO_ALLOW_MEMORY_OVERCOMMIT", False)


def is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA/MPS/CPU out-of-memory failures across backends."""
    if exc is None:
        return False
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "AcceleratorError"}:
        return True
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "mps out of memory",
        "insufficient memory",
        "failed to allocate",
        "cannot allocate memory",
    )
    return any(n in msg for n in needles)


def release_cached_memory(*, sync: bool = False) -> None:
    """Best-effort GPU/RAM cache release."""
    gc.collect()
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            import mlx.core as mx  # pylint: disable=import-error,no-name-in-module

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if sync:
                torch.cuda.synchronize()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _host_os_reserve_mb(ram_mb: int) -> int:
    return max(512, int(ram_mb * 0.08))


def _gpu_layer_fraction(n_gpu_layers: int, total_layers: int) -> float:
    if n_gpu_layers == -1:
        return 1.0
    return max(0.0, min(float(n_gpu_layers) / float(total_layers or 64), 1.0))


def _llama_model_likely_resident(
    free_mb: int,
    total_need_mb: int,
    *,
    weights_resident: bool = False,
) -> bool:
    """True when ``free_mb`` is post-load residual VRAM, not a pre-load budget."""
    _ = (free_mb, total_need_mb)
    return weights_resident


def llama_batch_headroom_mb(
    free_mb: int,
    *,
    model_path: str | Path | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    weights_resident: bool = False,
) -> int:
    """VRAM left for llama.cpp batch/KV after estimated weight offload."""
    if not model_path or n_gpu_layers == 0:
        return free_mb
    path = Path(model_path)
    try:
        weight_mb = int(estimate_path_vram_mb(path))
        total_layers = gguf_total_layers(path) if path.is_file() else 64
        if n_gpu_layers == -1:
            gpu_weight_mb = weight_mb
        else:
            gpu_weight_mb = int(weight_mb * _gpu_layer_fraction(n_gpu_layers, total_layers)) + 256
        kv_mb = llama_kv_cache_reserve_mb(
            path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            total_layers=total_layers,
            weight_mb=weight_mb,
            free_mb=free_mb,
        )
        total_need = gpu_weight_mb + kv_mb
        if _llama_model_likely_resident(
            free_mb, total_need, weights_resident=weights_resident
        ):
            return max(_MIN_LLAMA_BATCH * 2, free_mb)
        return max(_MIN_LLAMA_BATCH * 2, free_mb - total_need)
    except Exception:
        return free_mb


def _estimate_gguf_params_b(path: Path, weight_mb: int) -> float:
    guessed = guess_params_from_name(path.name) or guess_params_from_name(str(path))
    if guessed:
        return float(guessed)
    # Most local chat GGUFs are Q4/Q5. Inferring params from file size is
    # intentionally conservative because underestimating KV cache causes OOM.
    return max(1.0, float(weight_mb) / 1024.0 / 0.55)


def _gguf_exact_kv_per_token_mb(path: Path) -> float | None:
    """Exact fp16 KV MB/token from GGUF attention metadata, or None."""
    if not path.is_file():
        return None
    try:
        from seiso.inference.backends import gguf_kv_bytes_per_token

        kv_bytes = gguf_kv_bytes_per_token(str(path))
    except Exception:
        return None
    if not kv_bytes:
        return None
    return kv_bytes / (1024**2)


def _llama_effective_kv_ctx(path: Path, n_ctx: int) -> int:
    """Context tokens used for KV sizing (SWA models cap at the sliding window)."""
    ctx = max(_MIN_LLAMA_CTX, min(int(n_ctx), _MAX_LLAMA_CTX))
    if env_bool("SEISO_LLAMA_SWA_FULL", False):
        return ctx
    try:
        from seiso.inference.backends import (
            gguf_sliding_window,
            gguf_swa_layer_fraction,
            gguf_uses_sliding_window_attention,
        )

        if gguf_uses_sliding_window_attention(str(path)):
            sw = gguf_sliding_window(str(path))
            local_ctx = min(ctx, int(sw)) if sw and sw > 0 else min(ctx, 4096)
            swa_frac = gguf_swa_layer_fraction(str(path))
            swa_frac = 0.85 if swa_frac is None else max(0.0, min(float(swa_frac), 1.0))
            global_frac = 1.0 - swa_frac
            return max(
                local_ctx,
                int(swa_frac * local_ctx + global_frac * ctx),
            )
    except Exception:
        pass
    return ctx


def llama_kv_cache_reserve_mb(
    model_path: str | Path,
    *,
    n_ctx: int,
    n_gpu_layers: int,
    total_layers: int | None = None,
    weight_mb: int | None = None,
    free_mb: int = 0,
) -> int:
    """VRAM reserve for llama.cpp KV cache at the requested context.

    Prefers exact GGUF attention geometry (GQA-aware, correct on every NVIDIA
    card); falls back to a conservative parameter-count heuristic when the
    metadata is unavailable.
    """
    if n_gpu_layers == 0:
        return 0
    path = Path(model_path)
    if weight_mb is None:
        weight_mb = int(estimate_path_vram_mb(path))
    if total_layers is None:
        total_layers = gguf_total_layers(path)

    layer_fraction = _gpu_layer_fraction(n_gpu_layers, total_layers)
    ctx = _llama_effective_kv_ctx(path, n_ctx)

    exact_per_token_mb = _gguf_exact_kv_per_token_mb(path)
    if exact_per_token_mb is not None:
        # 10% covers KV padding and per-sequence bookkeeping.
        estimated = int(ctx * exact_per_token_mb * 1.10 * layer_fraction)
        return max(256, estimated)

    params_b = _estimate_gguf_params_b(path, int(weight_mb))
    # Approximate fp16 K+V cache per token. The coefficient tracks observed
    # llama-family/GQA memory by parameter scale while keeping small models fast.
    per_token_mb = max(0.16, min(params_b * 0.045, 3.5))
    estimated = int(ctx * per_token_mb * layer_fraction)
    legacy_floor = max(256, min(int(max(free_mb, 0) * 0.08), 1024))
    return max(legacy_floor, estimated)


def llama_offload_fits_headroom(
    model_path: str | Path,
    *,
    headroom_mb: int,
    n_gpu_layers: int,
    n_ctx: int = 2048,
    weight_mb: int | None = None,
    total_layers: int | None = None,
) -> bool:
    """True when estimated GPU weight + KV for ``n_gpu_layers`` fits within headroom."""
    if n_gpu_layers == 0:
        return True
    if headroom_mb <= 0:
        return False

    path = Path(model_path)
    if weight_mb is None:
        weight_mb = int(estimate_path_vram_mb(path))
    if total_layers is None:
        total_layers = gguf_total_layers(path)

    if n_gpu_layers == -1:
        gpu_weight_mb = weight_mb
    else:
        gpu_weight_mb = int(weight_mb * _gpu_layer_fraction(n_gpu_layers, total_layers)) + 256

    kv_mb = llama_kv_cache_reserve_mb(
        path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        total_layers=total_layers,
        weight_mb=weight_mb,
        free_mb=headroom_mb,
    )
    return gpu_weight_mb + kv_mb <= headroom_mb


def llama_host_batch_headroom_mb(
    *,
    model_path: str | Path,
    n_gpu_layers: int,
    free_vram_mb: int,
) -> int | None:
    """Host RAM budget for mmap pages, prompt cache, and CPU-side KV on Linux NVIDIA."""
    if not seiso_platform.use_linux_nvidia_inference_guards():
        return None
    ram_mb = available_ram_mb()
    if ram_mb <= 0:
        return None

    path = Path(model_path)
    weight_mb = max(int(estimate_path_vram_mb(path)), 0)
    total_layers = max(gguf_total_layers(path), 1)

    if n_gpu_layers == 0:
        host_weight_mb = weight_mb
    elif n_gpu_layers == -1:
        # Fully offloaded weights stay mostly in VRAM; reserve modest mmap pages.
        host_weight_mb = max(256, int(weight_mb * 0.12))
    else:
        cpu_fraction = 1.0 - _gpu_layer_fraction(n_gpu_layers, total_layers)
        host_weight_mb = max(256, int(weight_mb * cpu_fraction) + 256)

    spill_mb = max(256, min(int(max(free_vram_mb, 0) * 0.05), 512))
    reserve_mb = _host_os_reserve_mb(ram_mb)
    # When host weight exceeds free RAM, force the minimum batch budget so
    # clamp_llama_load_kwargs still reduces n_batch instead of over-allocating.
    remaining = ram_mb - host_weight_mb - reserve_mb - spill_mb
    return max(_MIN_LLAMA_BATCH * 2, remaining)


def llama_model_is_tight_vram_fit(
    *,
    model_path: str | Path,
    free_mb: int,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    weights_resident: bool = False,
) -> bool:
    """True when a model consumes most of the available GPU budget."""
    path = Path(model_path)
    weight_mb = int(estimate_path_vram_mb(path))
    kv_mb = llama_kv_cache_reserve_mb(
        path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        free_mb=free_mb,
    )
    total_need = weight_mb + kv_mb
    ratio = _TIGHT_VRAM_FIT_RATIO
    with contextlib.suppress(Exception):
        if seiso_platform.use_linux_nvidia_inference_guards():
            ratio = _NATIVE_LINUX_TIGHT_VRAM_FIT_RATIO
            from seiso.inference.family_policy import policy_for_gguf

            ratio = ratio / max(policy_for_gguf(str(path)).prefill_tightness, 1.0)

    if _llama_model_likely_resident(
        free_mb, total_need, weights_resident=weights_resident
    ):
        required = max(_MIN_LLAMA_BATCH * 4, int(total_need * 0.15))
        if seiso_platform.use_linux_nvidia_inference_guards():
            required = max(required, int(total_need * 0.20))
        return free_mb < required

    if free_mb >= total_need:
        slack_ratio = free_mb / max(total_need, 1)
        if slack_ratio >= comfortable_vram_slack_ratio():
            return False
    return total_need >= int(free_mb * ratio)


def llama_effective_batch_headroom_mb(
    free_mb: int,
    *,
    model_path: str | Path | None = None,
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    weights_resident: bool = False,
) -> int:
    """Conservative batch/KV budget — minimum of GPU post-weight and host RAM headroom."""
    gpu_headroom = llama_batch_headroom_mb(
        free_mb,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        weights_resident=weights_resident,
    )
    if not model_path:
        return gpu_headroom
    host_headroom = llama_host_batch_headroom_mb(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        free_vram_mb=free_mb,
    )
    if host_headroom is None:
        return gpu_headroom
    effective = min(gpu_headroom, host_headroom)
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        if use_linux_nvidia_inference_guards() and llama_model_is_tight_vram_fit(
            model_path=model_path,
            free_mb=free_mb,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            weights_resident=weights_resident,
        ):
            # Reserve headroom for prefill activations on near-capacity models only.
            effective = max(_MIN_LLAMA_BATCH * 2, int(effective * 0.85) - 256)
    except ImportError:
        pass
    return effective


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


def resolve_llama_model_batches(
    *,
    model_path: str | Path,
    free_mb: int,
    n_ctx: int,
    n_gpu_layers: int,
    load_tier: LlamaLoadTier = "normal",
    weights_resident: bool = False,
    load_budget_mb: int | None = None,
    prompt_tokens: int | None = None,
    vision_prefill: bool = False,
    has_mmproj_sibling: bool = False,
    native_linux_nvidia: bool | None = None,
) -> tuple[int, int, bool]:
    """Model-aware n_batch (prefill) and n_ubatch (decode chunk) for llama.cpp."""
    budget_mb = load_budget_mb if load_budget_mb is not None else free_mb
    tight = llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=budget_mb,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        weights_resident=False,
    )
    if native_linux_nvidia is None:
        try:
            native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
        except Exception:
            native_linux_nvidia = False

    effective = llama_effective_batch_headroom_mb(
        free_mb,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        weights_resident=weights_resident,
    )
    if weights_resident and load_budget_mb is not None:
        load_effective = llama_effective_batch_headroom_mb(
            load_budget_mb,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            weights_resident=False,
        )
        effective = min(effective, load_effective)

    if prompt_tokens is not None:
        prefill_tokens = max(prompt_tokens, _MIN_LLAMA_BATCH)
        reserve_steps = max(1, (prefill_tokens + 255) // 256)
        reserve_mb = reserve_steps * _NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB
        effective = max(_MIN_LLAMA_BATCH * 2, effective - reserve_mb)
    if vision_prefill:
        effective = max(_MIN_LLAMA_BATCH * 2, effective - 512)
    elif has_mmproj_sibling:
        effective = max(_MIN_LLAMA_BATCH * 2, effective - 256)

    batch, ubatch = resolve_llama_batch_limits(
        effective,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=load_tier,
        tight=tight,
    )
    return batch, ubatch, tight


def llama_prefill_needs_reload(
    *,
    model_path: str,
    messages: list[dict[str, Any]],
    n_ctx: int,
    loaded_n_batch: int,
    loaded_n_ubatch: int | None = None,
    loaded_n_gpu_layers: int,
    load_tier: LlamaLoadTier = "normal",
    loaded_headroom_mb: int | None = None,
) -> tuple[bool, int, int]:
    """True when a cached native-Linux llama handle should reload before prefill."""
    try:
        native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    if not native_linux_nvidia:
        batch, ubatch = clamp_llama_batch_pair(
            loaded_n_batch or _MAX_LLAMA_BATCH,
            loaded_n_batch or _MAX_LLAMA_BATCH,
        )
        return False, batch, ubatch

    with contextlib.suppress(Exception):
        hardware_profile(force_refresh=True)

    free_mb = headroom_mb()
    prompt_tokens = _estimate_prompt_tokens(messages)
    vision_prefill = _messages_have_vision_content(messages)
    load_budget_mb = loaded_headroom_mb if loaded_headroom_mb else free_mb
    safe_batch, safe_ubatch, tight_prefill = resolve_llama_model_batches(
        model_path=model_path,
        free_mb=free_mb,
        n_ctx=n_ctx,
        n_gpu_layers=loaded_n_gpu_layers,
        load_tier=load_tier,
        weights_resident=True,
        load_budget_mb=load_budget_mb,
        prompt_tokens=prompt_tokens,
        vision_prefill=vision_prefill,
        has_mmproj_sibling=_gguf_has_mmproj_sibling(model_path),
    )
    headroom_dropped = (
        loaded_headroom_mb is not None
        and loaded_headroom_mb > 0
        and free_mb < int(loaded_headroom_mb * _NATIVE_LINUX_PREFILL_HEADROOM_DROP_RATIO)
    )
    prefill_exceeds_safe = prompt_tokens > safe_batch
    loaded_batch = int(loaded_n_batch or 0)
    loaded_ubatch_explicit = loaded_n_ubatch is not None
    loaded_ubatch = int(loaded_n_ubatch if loaded_ubatch_explicit else loaded_batch or 0)
    headroom_shrank = (
        loaded_headroom_mb is not None
        and loaded_headroom_mb > 0
        and free_mb < int(loaded_headroom_mb * _NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO)
    )
    # Reload only when the loaded batch is unsafe for this prefill. Do not
    # thrash on "long prompt" alone — that caused mid-conversation reloads
    # (and OOM risk) as chat history grew on native Linux.
    batch_unsafe = loaded_batch > safe_batch
    batch_far_over = loaded_batch > max(safe_batch * 2, safe_batch + 256)
    ubatch_far_over = loaded_ubatch > max(safe_ubatch * 2, safe_ubatch + 128)
    needs_reload = batch_unsafe and (
        prefill_exceeds_safe
        or headroom_dropped
        or headroom_shrank
        or vision_prefill
        or batch_far_over
    )
    if (
        loaded_ubatch_explicit
        and loaded_ubatch > safe_ubatch
        and (
            headroom_dropped
            or headroom_shrank
            or vision_prefill
            or prefill_exceeds_safe
            or ubatch_far_over
            or loaded_batch <= safe_batch
        )
    ):
        needs_reload = True
    if not needs_reload and tight_prefill:
        if loaded_batch > 0:
            safe_batch = min(safe_batch, loaded_batch)
        if loaded_ubatch > 0:
            safe_ubatch = min(safe_ubatch, loaded_ubatch)
    return needs_reload, safe_batch, safe_ubatch


def llama_load_profile_ladder(
    *,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    free_mb: int,
    base_batch: int,
    base_ubatch: int,
    tier: LlamaLoadTier = "normal",
) -> list[dict[str, Any]]:
    """Ordered llama.cpp memory profiles from fastest safe settings to compact fallbacks."""
    tight = llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=free_mb,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )
    effective = llama_effective_batch_headroom_mb(
        free_mb, model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx
    )
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        native_linux_nvidia = use_linux_nvidia_inference_guards()
    except ImportError:
        native_linux_nvidia = False

    top_batch, top_ubatch = resolve_llama_batch_limits(
        effective,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=tier,
        tight=tight,
    )
    apply_headroom_cap = native_linux_nvidia or tight or effective < _NATIVE_LINUX_PREFILL_CLAMP_MB
    if apply_headroom_cap:
        base_batch = min(int(base_batch), top_batch)
        base_ubatch = min(int(base_ubatch), top_ubatch)
    base_batch, base_ubatch = clamp_llama_batch_pair(
        base_batch,
        base_ubatch,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=tier,
        tight=tight,
    )

    steps: list[tuple[int, int, int | None, bool]] = []
    speed_scale = env_bool("SEISO_LLAMA_SPEED_SCALE", not native_linux_nvidia)
    native_flash_ok = not native_linux_nvidia or env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False)
    primary_flash = (
        n_gpu_layers != 0
        and not tight
        and native_flash_ok
        and env_bool("SEISO_LLAMA_FLASH_ATTN", True)
    )

    if tier == "normal":
        if tight:
            tight_batch, tight_ubatch = tight_batch_caps(discrete_gpu_total_mb())
            steps.append(
                (
                    min(base_batch, tight_batch),
                    min(base_ubatch, tight_ubatch),
                    min(n_ctx, 2048),
                    False,
                )
            )
        if (
            speed_scale
            and not native_linux_nvidia
            and n_gpu_layers != 0
            and (top_batch > base_batch or top_ubatch > base_ubatch)
        ):
            steps.append((top_batch, top_ubatch, None, True))
        steps.append((base_batch, base_ubatch, None, primary_flash))
        for batch, ubatch, ctx_cap in (
            (512, 256, min(n_ctx, 4096)),
            (512, 128, min(n_ctx, 4096)),
            (256, 128, min(n_ctx, 2048)),
        ):
            steps.append(
                (
                    min(base_batch, batch),
                    min(base_ubatch, ubatch),
                    ctx_cap,
                    False,
                )
            )
    else:
        steps.append(
            (
                base_batch,
                base_ubatch,
                min(n_ctx, 4096 if tier == "compact" else 2048),
                False,
            )
        )

    profiles: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for batch, ubatch, ctx_cap, flash in steps:
        profile: dict[str, Any] = {"n_batch": batch, "n_ubatch": ubatch}
        if ctx_cap is not None:
            profile["n_ctx"] = ctx_cap
        if not flash:
            profile["flash_attn"] = False
        if tier != "normal":
            profile["_seiso_prompt_cache"] = False
        key = tuple(sorted(profile.items()))
        if key in seen:
            continue
        seen.add(key)
        profiles.append(profile)
    return profiles


def llama_next_recovery_tier(current: LlamaLoadTier) -> LlamaLoadTier | None:
    """Next load tier after an inference OOM, or None when exhausted."""
    if current == "normal":
        return "compact"
    if current == "compact":
        return "minimal"
    return None


def headroom_mb() -> int:
    """Free memory headroom in MB for fit labels and status reporting."""
    profile = hardware_profile()
    try:
        return int(vram_headroom_mb(profile))
    except Exception:
        gpus = profile.get("gpus") or []
        if gpus:
            best = 0
            for gpu in gpus:
                total = int(gpu.get("vram_total_mb") or 0)
                used = int(gpu.get("vram_used_mb") or 0)
                if total > 0:
                    best = max(best, max(total - used, 0))
            if best > 0:
                return best
        ram = float(profile.get("ram_gb") or 8)
        avail = available_ram_mb()
        if avail > 0:
            return int(min(avail * 0.72, ram * 1024 * 0.45))
        return int(ram * 1024 * 0.35)


def available_ram_mb() -> int:
    """Cross-platform available RAM in MB (Linux, macOS, Windows)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available / (1024**2))
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys / (1024**2))
        except Exception:
            pass
    return int(float(hardware_profile().get("ram_gb") or 8) * 1024 * 0.5)


def build_hf_max_memory(*, reserve_ratio: float = _DEFAULT_RESERVE_RATIO) -> dict[int, str] | None:
    """Build HuggingFace ``max_memory`` unless caps are explicitly disabled."""
    if env_bool("SEISO_DISABLE_MEMORY_CAPS", False):
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    max_memory: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(i)
        except Exception:
            props = torch.cuda.get_device_properties(i)
            free_bytes = int(props.total_memory * (1.0 - reserve_ratio))
        usable = max(int(free_bytes * (1.0 - reserve_ratio)), 256 * 1024**2)
        max_memory[i] = f"{usable // (1024**2)}MiB"
    return max_memory or None


def assess_path_memory_fit(path: str | Path, *, mode: str = "chat") -> dict[str, Any]:
    """Return fit metadata compatible with Forge hardware assessments."""
    p = Path(path).expanduser()
    est_mb = estimate_path_vram_mb(p, mode=mode)
    if p.is_file() and p.suffix.lower() == ".gguf":
        try:
            from seiso.inference.llama_vision import resolve_mmproj_path

            mmproj = resolve_mmproj_path(p)
            if mmproj:
                est_mb += estimate_path_vram_mb(mmproj, mode=mode)
        except ImportError:
            pass
    est_gb = round(est_mb / 1024, 2)
    profile = hardware_profile()
    try:
        return assess_hardware_fit(est_gb, profile, mode=mode)
    except Exception:
        capacity = int(fit_headroom_mb(profile))
        free = int(vram_headroom_mb(profile))
        raw_budget = free if free > 0 else capacity
        reserve = max(256, int(raw_budget * 0.02)) if raw_budget > 0 else 0
        budget = max(0, raw_budget - reserve)
        blocked = budget > 0 and est_mb > budget
        budget_gb = round(budget / 1024, 1)
        return {
            "hardware_fit": "unlikely" if blocked else "good",
            "est_vram_mb": est_mb,
            "memory_load_blocked": blocked,
            "memory_load_blocked_reason": (
                f"Needs ~{est_gb:.1f} GB at runtime but only ~{budget_gb} GB is safely available right now."
                if blocked
                else None
            ),
        }


_LLAMACPP_DEFER_WARNINGS: dict[str, str] = {
    "apple_unified": (
        "Low free unified memory — trying llama.cpp with mmap plus Mac CPU "
        "offload fallback. Close apps if loading still fails."
    ),
    "linux_nvidia": (
        "Low free VRAM — trying full GPU offload with conservative batch limits. "
        "Close other GPU apps if loading still fails."
    ),
}


def assess_path_memory_fit_for_load(
    path: str | Path,
    *,
    mode: str = "chat",
    pool: Any | None = None,
    backend: str | None = None,
    unload_if_needed: bool = True,
) -> dict[str, Any]:
    """Assess fit after unloading any active Seiso model that would be replaced."""
    from seiso.inference.model_pool import get_model_pool

    active_pool = pool or get_model_pool()
    if unload_if_needed:
        active_pool.prepare_for_load(str(path), backend)
    fit = assess_path_memory_fit(path, mode=mode)
    profile = hardware_profile()
    defer = _llamacpp_deferred_preflight_platform(fit, backend=backend, mode=mode, profile=profile)
    if defer:
        fit = dict(fit)
        fit["memory_load_blocked"] = False
        fit["memory_load_blocked_reason"] = None
        fit["memory_load_warning"] = _LLAMACPP_DEFER_WARNINGS.get(
            defer,
            "Low free memory — trying llama.cpp with conservative fallbacks.",
        )
    return fit


def _llamacpp_deferred_preflight_platform(
    fit: dict[str, Any],
    *,
    backend: str | None,
    mode: str,
    profile: dict[str, Any] | None = None,
) -> str | None:
    """Return platform id when llama.cpp should try load despite preflight block."""
    if mode != "chat":
        return None
    if str(backend or "").lower() not in {"llamacpp", "llama"}:
        return None

    blocked = bool(fit.get("memory_load_blocked"))
    low_free = bool(fit.get("memory_load_budget_exceeded")) and not blocked
    if not blocked and not low_free:
        return None

    defer = seiso_platform.llamacpp_deferred_preflight_platform(profile=profile)
    if not defer:
        return None

    if defer == "linux_nvidia":
        est_mb = int(fit.get("est_vram_mb") or 0)
        try:
            capacity_mb = fit_headroom_mb(profile or hardware_profile())
        except Exception:
            return None
        if est_mb <= 0 or capacity_mb <= 0 or est_mb > capacity_mb:
            return None
        try:
            from seiso.inference.model_pool import _llama_gpu_offload_ok

            if not _llama_gpu_offload_ok():
                return None
        except ImportError:
            pass
    return defer


def ensure_load_fits(
    path: str | Path,
    *,
    mode: str = "chat",
    backend: str | None = None,
) -> dict[str, Any]:
    """Block model loads that exceed measured memory headroom."""
    fit = assess_path_memory_fit_for_load(path, mode=mode, backend=backend)
    backend_key = str(backend or "").lower()
    llamacpp_backend = backend_key in {"llamacpp", "llama"}
    if fit.get("memory_load_blocked"):
        reason = fit.get("memory_load_blocked_reason") or "Model exceeds available memory"
        if allow_memory_overcommit():
            logger.warning("Memory overcommit allowed: %s", reason)
        else:
            raise MemoryLoadBlockedError(reason)
    if (
        mode == "chat"
        and fit.get("memory_load_budget_exceeded")
        and not llamacpp_backend
    ):
        est_gb = round(int(fit.get("est_vram_mb") or 0) / 1024, 1)
        reason = (
            f"Needs ~{est_gb:.1f} GB at runtime but free memory is low right now. "
            "Free memory or use llama.cpp for tiered GPU load fallbacks."
        )
        if allow_memory_overcommit():
            logger.warning("Memory overcommit allowed: %s", reason)
        else:
            raise MemoryLoadBlockedError(reason)
    return fit


_VISION_TOKENS_PER_IMAGE = 1024
_DATA_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,", re.I)
_VISION_CONTENT_MARKERS = (
    "image_url",
    '"type":"image"',
    '"type": "image"',
    "data:image/",
)
_CONTEXT_TRIM_MARKER = "[...older content omitted...]\n"


def _text_chars_to_tokens(chars: int) -> int:
    return max(0, int(chars / 3.2))


def _count_images_in_content(content: Any) -> int:
    if isinstance(content, list):
        count = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type in {"image", "image_url"}:
                count += 1
        return count
    if not isinstance(content, str):
        return 0
    stripped = content.lstrip()
    if stripped.startswith("["):
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return _count_images_in_content(parsed)
    lower = content.lower()
    embedded = len(_DATA_IMAGE_RE.findall(content))
    if embedded:
        return embedded
    if any(marker in lower for marker in _VISION_CONTENT_MARKERS):
        return max(1, lower.count("image_url"))
    return 0


def _text_chars_from_content(content: Any) -> int:
    if isinstance(content, list):
        chars = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "text")).lower()
            if part_type in {"text", "input_text"}:
                chars += len(str(part.get("text") or part.get("content") or ""))
        return chars
    if isinstance(content, str):
        stripped = content.lstrip()
        if stripped.startswith("["):
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return _text_chars_from_content(parsed)
        if _count_images_in_content(content):
            # OpenAI-style JSON with embedded base64 — avoid treating payload as text.
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return _text_chars_from_content(parsed)
            return min(len(content), 512)
        return len(content)
    return len(str(content))


def _message_content_token_estimate(content: Any) -> int:
    images = _count_images_in_content(content)
    text_tokens = _text_chars_to_tokens(_text_chars_from_content(content))
    if images:
        return text_tokens + images * _VISION_TOKENS_PER_IMAGE
    return text_tokens


def _messages_have_vision_content(messages: list[dict[str, Any]]) -> bool:
    return any(_count_images_in_content(m.get("content")) > 0 for m in messages)


def _gguf_has_mmproj_sibling(model_path: str | Path) -> bool:
    """True when a colocated mmproj GGUF is present for a vision-capable chat model."""
    path = Path(model_path)
    if not path.is_file():
        return False
    from seiso.inference.llama_vision import model_suggests_vision, resolve_mmproj_path

    if not model_suggests_vision(path):
        return False
    return resolve_mmproj_path(path) is not None


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total = sum(_message_content_token_estimate(m.get("content", "")) for m in messages)
    return max(64, total)


def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    char_budget = max(1, int(token_budget * 3.2))
    if len(text) <= char_budget:
        return text
    if char_budget <= len(_CONTEXT_TRIM_MARKER):
        return text[-char_budget:]
    keep = char_budget - len(_CONTEXT_TRIM_MARKER)
    return f"{_CONTEXT_TRIM_MARKER}{text[-keep:]}"


def _trim_message_content_to_token_budget(content: Any, token_budget: int) -> Any:
    if isinstance(content, str):
        return _trim_text_to_token_budget(content, token_budget)
    if isinstance(content, list):
        remaining = max(0, token_budget)
        out: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                out.append(part)
                continue
            part_type = str(part.get("type", "text")).lower()
            if part_type not in {"text", "input_text"}:
                out.append(part)
                continue
            text = str(part.get("text") or part.get("content") or "")
            trimmed = _trim_text_to_token_budget(text, remaining)
            remaining = max(0, remaining - _text_chars_to_tokens(len(trimmed)))
            key = "text" if "text" in part else "content"
            out.append({**part, key: trimmed})
        return out
    return content


def trim_llama_messages_to_context(
    messages: list[dict[str, Any]],
    *,
    n_ctx: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Trim prompt content so llama.cpp prefill stays within the loaded context."""
    if not messages:
        return []

    prompt_budget = max(256, int(n_ctx) - max(1, int(max_tokens)) - 128)
    if _estimate_prompt_tokens(messages) <= prompt_budget:
        return messages

    trimmed = [dict(message) for message in messages]
    latest_idx = len(trimmed) - 1

    # Drop oldest conversational turns first; keep system/knowledge instructions
    # until content trimming is required.
    idx = 0
    while _estimate_prompt_tokens(trimmed) > prompt_budget and idx < latest_idx:
        role = str(trimmed[idx].get("role", "")).lower()
        if role in {"user", "assistant"}:
            trimmed.pop(idx)
            latest_idx -= 1
            continue
        idx += 1

    # Then trim oversized message bodies, newest user last.
    order = sorted(
        range(len(trimmed)),
        key=lambda i: (
            i == len(trimmed) - 1,
            str(trimmed[i].get("role", "")).lower() == "system" and i == 0,
            -_message_content_token_estimate(trimmed[i].get("content", "")),
        ),
    )
    for idx in order:
        current = _estimate_prompt_tokens(trimmed)
        if current <= prompt_budget:
            break
        content = trimmed[idx].get("content", "")
        content_tokens = _message_content_token_estimate(content)
        if content_tokens <= 0:
            continue
        target = max(32, content_tokens - (current - prompt_budget))
        trimmed[idx]["content"] = _trim_message_content_to_token_budget(content, target)

    return trimmed


def sanitize_inference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Clamp generation limits to available memory without changing intent."""
    out = dict(payload)
    messages = out.get("messages") or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    headroom = headroom_mb()

    max_tokens = int(out.get("max_tokens") or 2048)
    max_tokens = max(1, min(max_tokens, _MAX_INFERENCE_TOKENS))

    if headroom > _INFERENCE_OVERHEAD_MB:
        kv_budget_tokens = max(
            512,
            int((headroom - _INFERENCE_OVERHEAD_MB) * 128 / 1.15),
        )
        max_tokens = min(max_tokens, max(128, kv_budget_tokens - prompt_tokens - 32))
    out["max_tokens"] = max_tokens

    if out.get("n_ctx") is not None:
        out["n_ctx"] = clamp_llama_n_ctx(
            int(out["n_ctx"]),
            messages=messages,
            max_tokens=max_tokens,
            model_path=out.get("model_path"),
            model_format=out.get("model_format"),
        )
    return out


def bucket_llama_n_ctx(needed: int, *, ceiling: int | None = None) -> int:
    """Snap context to coarse buckets so multi-turn chat reuses one loaded KV size."""
    try:
        native_linux = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux = False
    buckets = _NATIVE_LINUX_CTX_BUCKETS if native_linux else _LLAMA_CTX_BUCKETS
    floor = buckets[0]
    cap = _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))
    need = max(min(floor, cap), int(needed))
    for bucket in buckets:
        if need <= bucket:
            return min(bucket, cap)
    return min(cap, max(min(floor, cap), need))


def clamp_llama_n_ctx(
    n_ctx: int,
    *,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 512,
    model_path: str | None = None,
    model_format: str | None = None,
    model_name: str | None = None,
) -> int:
    """Bound llama.cpp context to prompt + generation + headroom + model capability."""
    from seiso.inference.context_limits import effective_context_ceiling

    messages = messages or []
    prompt_tokens = _estimate_prompt_tokens(messages)
    needed = prompt_tokens + max_tokens + 128

    ctx_cap = effective_context_ceiling(
        model_path,
        model_format=model_format,
        model_name=model_name,
    )
    sized = bucket_llama_n_ctx(needed, ceiling=ctx_cap)
    return min(max(int(n_ctx), sized), ctx_cap)


def clamp_llama_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize llama.cpp load kwargs and trim oversized batches near VRAM limits."""
    out = dict(kwargs)
    model_path = out.pop("_model_path", None)
    native_linux_hint = out.pop("_native_linux_nvidia", None)
    n_ctx = int(out.get("n_ctx") or _MIN_LLAMA_CTX)
    out["n_batch"] = max(_MIN_LLAMA_BATCH, int(out.get("n_batch") or _MAX_LLAMA_BATCH))
    out["n_ubatch"] = max(
        _MIN_LLAMA_BATCH,
        min(int(out.get("n_ubatch") or out["n_batch"]), out["n_batch"]),
    )

    n_gpu_layers = int(out.get("n_gpu_layers") or 0)
    native_linux_nvidia = False
    if model_path:
        free_mb = headroom_mb()
        tight = llama_model_is_tight_vram_fit(
            model_path=model_path,
            free_mb=free_mb,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
        if native_linux_hint is not None:
            native_linux_nvidia = bool(native_linux_hint)
        else:
            with contextlib.suppress(Exception):
                native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
        if native_linux_nvidia and n_gpu_layers == 0:
            batch_headroom = llama_host_batch_headroom_mb(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                free_vram_mb=free_mb,
            )
            if batch_headroom is None:
                batch_headroom = free_mb
            max_batch, max_ubatch, _tight = resolve_llama_model_batches(
                model_path=model_path,
                free_mb=free_mb,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                load_tier="normal",
                weights_resident=False,
                native_linux_nvidia=native_linux_nvidia,
            )
            max_batch = min(max_batch, batch_headroom)
            max_ubatch = min(max_ubatch, max_batch)
        else:
            max_batch, max_ubatch, _tight = resolve_llama_model_batches(
                model_path=model_path,
                free_mb=free_mb,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                load_tier="normal",
                weights_resident=False,
                native_linux_nvidia=native_linux_nvidia,
            )
            batch_headroom = max_batch
        if native_linux_nvidia and _gguf_has_mmproj_sibling(model_path):
            batch_headroom = max(_MIN_LLAMA_BATCH * 2, batch_headroom - 512)
            max_batch = min(max_batch, batch_headroom)
            max_ubatch = min(max_ubatch, max_batch)
        elif not (native_linux_nvidia or tight):
            max_batch, max_ubatch = clamp_llama_batch_pair(_MAX_LLAMA_BATCH, 1024)
        out["n_batch"], out["n_ubatch"] = clamp_llama_batch_pair(
            min(out["n_batch"], max_batch),
            min(out["n_ubatch"], max_ubatch),
            native_linux_nvidia=native_linux_nvidia,
            tight=tight,
        )
        if (
            native_linux_nvidia
            and out.get("flash_attn")
            and not env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False)
        ):
            try:
                from seiso.inference.family_policy import policy_for_gguf

                if model_path and not policy_for_gguf(str(model_path)).allow_flash_attn:
                    out.pop("flash_attn", None)
            except (ImportError, OSError, ValueError):
                pass
        if native_linux_nvidia and tight and n_gpu_layers != 0:
            if not env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False):
                out.pop("flash_attn", None)
            if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False):
                out["op_offload"] = False
            total_layers = gguf_total_layers(model_path)
            if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False) and (
                n_gpu_layers == -1 or n_gpu_layers >= total_layers
            ):
                out["offload_kqv"] = False

    ctx_cap = clamp_llama_n_ctx(
        n_ctx,
        max_tokens=512,
        model_path=str(model_path) if model_path else None,
        model_format="gguf" if model_path else None,
    )
    if n_ctx > ctx_cap:
        out["n_ctx"] = ctx_cap
    # #region agent log
    if model_path:
        from seiso.agent_debug_log import agent_debug_enabled, agent_debug_log

        if agent_debug_enabled():
            log_tight = n_gpu_layers != 0 and llama_model_is_tight_vram_fit(
                model_path=model_path,
                free_mb=headroom_mb(),
                n_gpu_layers=n_gpu_layers,
                n_ctx=int(out.get("n_ctx") or n_ctx),
            )
            agent_debug_log(
                hypothesis_id="B",
                location="protection.py:clamp_llama_load_kwargs",
                message="clamped llama load kwargs",
                data={
                    "model": Path(model_path).name,
                    "tight_fit": log_tight,
                    "native_linux_nvidia": native_linux_nvidia
                    if model_path and n_gpu_layers != 0
                    else False,
                    "free_mb": headroom_mb(),
                    "n_gpu_layers": n_gpu_layers,
                    "n_ctx": out.get("n_ctx"),
                    "n_batch": out.get("n_batch"),
                    "n_ubatch": out.get("n_ubatch"),
                    "flash_attn": out.get("flash_attn"),
                },
            )
    # #endregion
    return out


def clamp_llama_cache_mb(
    default_mb: int,
    *,
    model_path: str | Path | None = None,
) -> int:
    """Cap llama.cpp RAM prompt cache using host memory and model mmap footprint."""
    default_mb = max(0, int(default_mb))
    if default_mb <= 0:
        return 0
    if platform.system() != "Linux":
        return default_mb

    ram_mb = available_ram_mb()
    if ram_mb <= 0:
        return min(default_mb, 512)

    cap = min(default_mb, max(128, ram_mb // 24))
    if model_path and seiso_platform.use_linux_nvidia_inference_guards():
        weight_mb = int(estimate_path_vram_mb(model_path))
        mmap_reserve = max(512, int(weight_mb * 0.12))
        host_budget = max(128, ram_mb - mmap_reserve - _host_os_reserve_mb(ram_mb))
        cap = min(cap, max(0, host_budget // 8))
    return max(0, cap)


def training_pin_memory() -> bool:
    """Pin memory only when CUDA training is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_training_memory_guards(config: Any) -> Any:
    """Keep compatibility fixes while leaving user RAM/VRAM sizing untouched."""
    from seiso.training.config import TrainConfig

    if not isinstance(config, TrainConfig):
        return config

    profile = hardware_profile()
    try:
        defaults = training_defaults(profile)
    except Exception:
        defaults = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
        }

    updates: dict[str, Any] = {}

    # Downgrade quant to the platform-recommended value when the requested mode
    # is unavailable (e.g. QLoRA/4-bit on macOS where bitsandbytes is absent,
    # or 4-bit on a CPU-only box). Without this, torch_loader silently loads a
    # 16-bit model while the trainer still requests paged_adamw_8bit, crashing
    # at optimizer creation with ImportError: bitsandbytes.
    recommended_quant = defaults.get("quant")
    if recommended_quant and str(config.quant) != str(recommended_quant):
        target: Any = None
        try:
            from seiso.training.config import QuantMode

            target = QuantMode(recommended_quant)
        except (ValueError, ImportError):
            target = None
        if target is not None and target != config.quant:
            # Only downgrade — never upgrade beyond what the user asked for.
            rank = {QuantMode.NONE: 0, QuantMode.INT16: 1, QuantMode.INT8: 2, QuantMode.INT4: 3}  # type: ignore[name-defined]
            if rank.get(target, 0) < rank.get(config.quant, 0):
                updates["quant"] = target
                logger.info(
                    "Training memory guards: quant %s -> %s (platform recommendation)",
                    config.quant.value,
                    target.value,
                )

    if not updates:
        return config

    logger.info("Training memory guards applied: %s", updates)
    return config.model_copy(update=updates)


def apply_training_oom_fallback(config: Any) -> Any:
    """Halve batch / seq after an OOM during training."""
    batch = max(1, int(config.batch_size) // 2)
    accum = int(config.gradient_accumulation_steps) * 2
    max_seq = max(128, int(config.max_seq_length) // 2)
    logger.warning(
        "OOM recovery: batch_size=%d accum=%d max_seq_length=%d",
        batch,
        accum,
        max_seq,
    )
    return config.model_copy(
        update={
            "batch_size": batch,
            "gradient_accumulation_steps": accum,
            "max_seq_length": max_seq,
        }
    )


def apply_rl_memory_guards(flat: dict[str, Any]) -> dict[str, Any]:
    """Leave RL quant sizing untouched; OOM fallback handles real failures."""
    out = dict(flat)
    ctx = int(out.get("llama_cpp_context") or 0)
    if ctx > 0:
        out["llama_cpp_context"] = min(ctx, clamp_llama_n_ctx(ctx, max_tokens=512))

    return out


def jsonl_load_safe(path: Path) -> bool:
    """True when JSONL should use datasets loader instead of in-memory list."""
    try:
        return path.stat().st_size > _MAX_JSONL_LOAD_MB * 1024**2
    except OSError:
        return False


def resolve_training_device_map(
    device: str | None = None,
) -> str | dict[str, str] | None:
    """Single-device placement for DDP; auto only for single-process CUDA."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        return {"": f"cuda:{local_rank}"}

    if device == "mps":
        return {"": "mps"}
    try:
        import torch

        if device == "cuda" or (device is None and torch.cuda.is_available()):
            return "auto"
    except ImportError:
        pass
    return None
