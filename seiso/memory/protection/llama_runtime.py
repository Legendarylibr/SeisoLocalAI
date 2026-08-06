"""llama.cpp runtime profiles, prefill, and native Linux guards."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso import platform as seiso_platform
from seiso.env import env_bool
from seiso.inference.backends import gguf_total_layers
from seiso.inference.family_policy import policy_for_gguf
from seiso.memory.protection._facade import protection
from seiso.memory.protection.constants import (
    _MAX_LLAMA_BATCH,
    _MAX_LLAMA_CTX,
    _MIN_LLAMA_BATCH,
    _MIN_LLAMA_CTX,
    _NATIVE_LINUX_CTX_BUCKETS,
    _NATIVE_LINUX_MMPROJ_RESERVE_MB,
    _NATIVE_LINUX_PREFILL_CLAMP_MB,
    _NATIVE_LINUX_PREFILL_HEADROOM_DROP_RATIO,
    _NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO,
    _NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB,
    _NATIVE_LINUX_TIGHT_VRAM_FIT_RATIO,
    _TIGHT_VRAM_FIT_RATIO,
    LlamaLoadTier,
)
from seiso.memory.protection.llama_batch import (
    cap_llama_batch_for_context,
    clamp_llama_batch_pair,
    gpu_batch_tier_caps,
    resolve_llama_batch_limits,
    tight_batch_caps,
)
from seiso.memory.protection.llama_kv import (
    _gpu_layer_fraction,
    _host_os_reserve_mb,
    llama_batch_headroom_mb,
    llama_offload_fits_headroom,
)


@dataclass(frozen=True, slots=True)
class LlamaDecodeBudget:
    """Safe llama.cpp request shape for the current native Linux GPU headroom."""

    n_ctx: int
    max_tokens: int
    n_batch: int
    n_ubatch: int
    reserve_mb: int
    tight: bool


def llama_decode_reserve_mb(
    *,
    gpu_total_mb: int,
    free_mb: int,
    max_tokens: int,
    prompt_tokens: int = 0,
    model_path: str | Path | None = None,
) -> int:
    """Native Linux allocator/decode workspace reserve beyond static weight+KV."""
    total = max(int(gpu_total_mb or 0), int(free_mb or 0), 1024)
    token_steps = max(1, (max(1, int(max_tokens)) + 255) // 256)
    prompt_steps = max(0, int(prompt_tokens) // 1024)
    base = max(384, min(int(total * 0.025), 1536))
    reserve = base + token_steps * 128 + min(prompt_steps * 64, 512)
    if model_path:
        with contextlib.suppress(Exception):
            reserve = int(reserve * max(policy_for_gguf(str(model_path)).prefill_tightness, 1.0))
    return max(512, reserve)


def native_linux_llama_context_cap(
    model_path: str | Path | None,
    *,
    free_mb: int,
    n_gpu_layers: int = -1,
    ceiling: int | None = None,
    max_tokens: int = 512,
) -> int:
    """Largest native Linux llama.cpp context that leaves VRAM for prefill."""
    if not model_path or free_mb <= 0:
        return _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))
    try:
        if not seiso_platform.use_linux_nvidia_inference_guards():
            return _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))
    except Exception:
        return _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))

    cap = _MAX_LLAMA_CTX if ceiling is None else max(1, int(ceiling))
    # Keep allocator/prefill slack outside the KV estimate; long prompts were
    # hitting OOM even when the static weight+KV estimate barely fit.
    gpu_total = protection().discrete_gpu_total_mb()
    decode_reserve = llama_decode_reserve_mb(
        gpu_total_mb=gpu_total,
        free_mb=free_mb,
        max_tokens=max_tokens,
        model_path=model_path,
    )
    budget = max(0, int(free_mb * 0.88) - decode_reserve)
    candidates = sorted(
        {2048, *[bucket for bucket in _NATIVE_LINUX_CTX_BUCKETS if bucket <= cap]},
        reverse=True,
    )
    for candidate in candidates:
        if llama_offload_fits_headroom(
            model_path,
            headroom_mb=budget,
            n_gpu_layers=n_gpu_layers,
            n_ctx=candidate,
        ):
            return min(candidate, cap)
    return min(2048, cap)


def resolve_llama_decode_budget(
    *,
    model_path: str | Path,
    free_mb: int,
    n_ctx: int,
    max_tokens: int,
    n_gpu_layers: int,
    load_tier: LlamaLoadTier = "normal",
    prompt_tokens: int = 0,
    weights_resident: bool = True,
    stream: bool = False,
) -> LlamaDecodeBudget:
    """Clamp decode shape from GPU-size-aware headroom before generation starts."""
    try:
        native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    requested_tokens = max(1, int(max_tokens))
    ctx = max(_MIN_LLAMA_CTX, int(n_ctx))
    prompt = max(0, int(prompt_tokens))
    context_limited_tokens = max(1, ctx - prompt - 128)
    safe_tokens = min(requested_tokens, context_limited_tokens)
    if not native_linux_nvidia:
        batch, ubatch = clamp_llama_batch_pair(_MAX_LLAMA_BATCH, 1024)
        return LlamaDecodeBudget(ctx, safe_tokens, batch, ubatch, 0, False)

    gpu_total = protection().discrete_gpu_total_mb()
    if gpu_total <= 0:
        gpu_total = max(int(free_mb), 0)
    if not env_bool("SEISO_LLAMA_UNSAFE_LONG_COMPLETIONS", False):
        from seiso.memory.protection.constants import (
            _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS,
            _NATIVE_LINUX_MAX_COMPLETION_TOKENS,
        )

        native_cap = _NATIVE_LINUX_MAX_COMPLETION_TOKENS
        if free_mb < _NATIVE_LINUX_PREFILL_CLAMP_MB:
            native_cap = min(native_cap, _NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS)
        safe_tokens = min(safe_tokens, native_cap)

    reserve = llama_decode_reserve_mb(
        gpu_total_mb=gpu_total,
        free_mb=free_mb,
        max_tokens=safe_tokens,
        prompt_tokens=prompt,
        model_path=model_path,
    )
    if stream:
        reserve = int(reserve * 1.15)
    budget_free = max(_MIN_LLAMA_BATCH * 2, int(free_mb) - reserve)
    batch, ubatch, tight = resolve_llama_model_batches(
        model_path=model_path,
        free_mb=budget_free,
        n_ctx=ctx,
        n_gpu_layers=n_gpu_layers,
        load_tier=load_tier,
        weights_resident=weights_resident,
        prompt_tokens=prompt,
        native_linux_nvidia=True,
    )
    return LlamaDecodeBudget(ctx, safe_tokens, batch, ubatch, reserve, tight)


def llama_host_batch_headroom_mb(
    *,
    model_path: str | Path,
    n_gpu_layers: int,
    free_vram_mb: int,
) -> int | None:
    """Host RAM budget for mmap pages, prompt cache, and CPU-side KV on Linux NVIDIA."""
    if not seiso_platform.use_linux_nvidia_inference_guards():
        return None
    ram_mb = protection().available_ram_mb()
    if ram_mb <= 0:
        return None

    path = Path(model_path)
    weight_mb = max(int(protection().estimate_path_vram_mb(path)), 0)
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
    weight_mb = int(protection().estimate_path_vram_mb(path))
    kv_mb = protection().llama_kv_cache_reserve_mb(
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
            ratio = ratio / max(policy_for_gguf(str(path)).prefill_tightness, 1.0)

    if weights_resident:
        required = max(_MIN_LLAMA_BATCH * 4, int(total_need * 0.15))
        if seiso_platform.use_linux_nvidia_inference_guards():
            required = max(required, int(total_need * 0.20))
        return free_mb < required

    if free_mb >= total_need:
        slack_ratio = free_mb / max(total_need, 1)
        if slack_ratio >= protection().comfortable_vram_slack_ratio():
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

        if use_linux_nvidia_inference_guards() and protection().llama_model_is_tight_vram_fit(
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
    tight = protection().llama_model_is_tight_vram_fit(
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

    effective = protection().llama_effective_batch_headroom_mb(
        free_mb,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        weights_resident=weights_resident,
    )
    if weights_resident and load_budget_mb is not None:
        load_effective = protection().llama_effective_batch_headroom_mb(
            load_budget_mb,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            weights_resident=False,
        )
        if tight:
            effective = min(effective, load_effective)

    if prompt_tokens is not None:
        prefill_tokens = max(prompt_tokens, _MIN_LLAMA_BATCH)
        reserve_steps = max(1, (prefill_tokens + 255) // 256)
        reserve_mb = reserve_steps * _NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB
        effective = max(_MIN_LLAMA_BATCH * 2, effective - reserve_mb)
    if vision_prefill or has_mmproj_sibling:
        effective = max(_MIN_LLAMA_BATCH * 2, effective - _NATIVE_LINUX_MMPROJ_RESERVE_MB)

    batch, ubatch = resolve_llama_batch_limits(
        effective,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=load_tier,
        tight=tight,
    )
    if native_linux_nvidia:
        batch, ubatch = cap_llama_batch_for_context(batch, ubatch, n_ctx)
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
    prompt_tokens: int | None = None,
) -> tuple[bool, int, int]:
    """True when a cached native-Linux llama handle should reload before prefill."""
    try:
        native_linux_nvidia = seiso_platform.use_linux_nvidia_inference_guards()
    except Exception:
        native_linux_nvidia = False
    if not native_linux_nvidia:
        batch, ubatch = clamp_llama_batch_pair(
            loaded_n_batch or _MAX_LLAMA_BATCH,
            loaded_n_ubatch if loaded_n_ubatch is not None else loaded_n_batch or _MAX_LLAMA_BATCH,
        )
        return False, batch, ubatch

    with contextlib.suppress(Exception):
        protection().hardware_profile(force_refresh=True)

    free_mb = protection().headroom_mb()
    from seiso.memory.protection.chat_guards import (
        _estimate_prompt_tokens,
        _gguf_has_mmproj_sibling,
        _messages_have_vision_content,
    )

    prompt_tokens = (
        max(0, int(prompt_tokens))
        if prompt_tokens is not None
        else _estimate_prompt_tokens(messages)
    )
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
    gpu_total = protection().discrete_gpu_total_mb()
    tier_batch, tier_ubatch = gpu_batch_tier_caps(gpu_total or free_mb, load_tier)
    handle_within_tier = loaded_batch <= tier_batch and (
        not loaded_ubatch_explicit or loaded_ubatch <= tier_ubatch
    )
    trust_loaded_handle = (
        tight_prefill
        and not prefill_exceeds_safe
        and not vision_prefill
        and handle_within_tier
        and loaded_batch <= safe_batch
        and loaded_ubatch <= safe_ubatch
    )
    headroom_shrank = (
        loaded_headroom_mb is not None
        and loaded_headroom_mb > 0
        and free_mb < int(loaded_headroom_mb * _NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO)
    )
    # Reload whenever the cached handle's batch is above the current safe cap.
    # Native Linux llama.cpp allocates prefill buffers from the loaded batch
    # settings, so even short prompts can OOM when a stale handle was loaded
    # under roomier headroom.
    batch_unsafe = loaded_batch > safe_batch
    ubatch_far_over = loaded_ubatch > max(safe_ubatch * 2, safe_ubatch + 128)
    needs_reload = batch_unsafe and not trust_loaded_handle
    if (
        not trust_loaded_handle
        and (headroom_dropped or headroom_shrank)
        and (tight_prefill or prefill_exceeds_safe or loaded_batch > safe_batch)
    ):
        needs_reload = True
    if (
        loaded_ubatch_explicit
        and loaded_ubatch > safe_ubatch
        and (
            batch_unsafe
            or headroom_dropped
            or headroom_shrank
            or vision_prefill
            or prefill_exceeds_safe
            or ubatch_far_over
            or loaded_batch <= safe_batch
        )
        and not trust_loaded_handle
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
    tight = protection().llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=free_mb,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )
    effective = protection().llama_effective_batch_headroom_mb(
        free_mb, model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx
    )
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        native_linux_nvidia = use_linux_nvidia_inference_guards()
    except ImportError:
        native_linux_nvidia = False

    gpu_total = protection().discrete_gpu_total_mb()
    if native_linux_nvidia and gpu_total <= 0 and free_mb > 0:
        gpu_total = int(free_mb)

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
    requested_batch = int(base_batch)
    requested_ubatch = int(base_ubatch)
    base_batch, base_ubatch = clamp_llama_batch_pair(
        base_batch,
        base_ubatch,
        native_linux_nvidia=native_linux_nvidia,
        load_tier=tier,
        tight=tight,
        gpu_total_mb=gpu_total if native_linux_nvidia else None,
    )
    if native_linux_nvidia and tier != "normal":
        base_batch = min(base_batch, requested_batch)
        base_ubatch = min(base_ubatch, requested_ubatch, base_batch)
    if native_linux_nvidia:
        base_batch, base_ubatch = cap_llama_batch_for_context(base_batch, base_ubatch, n_ctx)

    steps: list[tuple[int, int, int | None, bool]] = []
    native_flash_ok = not native_linux_nvidia or env_bool("SEISO_LLAMA_UNSAFE_FLASH_ATTN", False)
    primary_flash = (
        n_gpu_layers != 0
        and not tight
        and native_flash_ok
        and env_bool("SEISO_LLAMA_FLASH_ATTN", True)
    )

    if tier == "normal":
        if tight:
            tight_batch, tight_ubatch = tight_batch_caps(protection().discrete_gpu_total_mb())
            steps.append(
                (
                    min(base_batch, tight_batch),
                    min(base_ubatch, tight_ubatch),
                    min(n_ctx, 2048),
                    False,
                )
            )
        steps.append((base_batch, base_ubatch, None, primary_flash))
        if native_linux_nvidia:
            compact_batch, compact_ubatch = gpu_batch_tier_caps(gpu_total, "compact")
            minimal_batch, minimal_ubatch = gpu_batch_tier_caps(gpu_total, "minimal")
            fallback_steps = (
                (
                    *cap_llama_batch_for_context(compact_batch, compact_ubatch, 8192),
                    min(n_ctx, 4096),
                ),
                (
                    *cap_llama_batch_for_context(minimal_batch, minimal_ubatch, 16384),
                    min(n_ctx, 2048),
                ),
            )
        else:
            fallback_steps = (
                (512, 256, min(n_ctx, 4096)),
                (512, 128, min(n_ctx, 4096)),
                (256, 128, min(n_ctx, 2048)),
            )
        for batch, ubatch, ctx_cap in fallback_steps:
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
