"""VRAM-managed model pool — unloads previous model when switching."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seiso.compat import StrEnum
from seiso.env import env_bool, env_int
from seiso.inference.backends import gguf_total_layers

logger = logging.getLogger(__name__)


def _default_llama_threads() -> int:
    cpus = _available_cpu_count()
    # Decode is latency-sensitive; leave one core free for OS, driver, and
    # Python streaming while still giving llama.cpp enough workers.
    return max(2, min(cpus - 1 if cpus > 4 else cpus, 16))


def _default_llama_threads_batch(n_threads: int) -> int:
    """Use wider CPU parallelism for prompt prefill without changing decode threads."""
    if "SEISO_LLAMA_THREADS" in os.environ:
        return n_threads
    cpus = _available_cpu_count()
    return max(n_threads, min(cpus, 32))


def _available_cpu_count() -> int:
    """CPU count honoring Linux affinity masks when present."""
    try:
        affinity = getattr(os, "sched_getaffinity", None)
        if callable(affinity):
            # pylint cannot infer the callable guard for this Linux-only attr.
            # pylint: disable-next=not-callable
            return max(1, len(affinity(0)))
    except Exception:
        pass
    return os.cpu_count() or 4


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _nvidia_hardware_visible() -> bool:
    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except ImportError:
        return False


def _apple_silicon_metal() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _mac_cpu_offload_enabled() -> bool:
    return env_bool("SEISO_LLAMA_MAC_CPU_OFFLOAD", True)


def _default_llama_gpu_layers() -> int:
    if _apple_silicon_metal():
        return -1
    if (_cuda_available() or _nvidia_hardware_visible()) and _llama_gpu_offload_ok():
        # Only request full GPU offload when the installed llama-cpp-python
        # wheel actually supports it. A CPU-only wheel on an NVIDIA box will
        # crash if n_gpu_layers != 0.
        return -1
    return 0


_llama_offload_checked = False
_llama_offload_supported = False


def reset_llama_gpu_offload_cache() -> None:
    """Allow GPU offload probe to retry after CUDA libs become available."""
    global _llama_offload_checked, _llama_offload_supported
    _llama_offload_checked = False
    _llama_offload_supported = False


def _llama_gpu_offload_ok() -> bool:
    """True when the installed llama-cpp-python can offload to GPU."""
    global _llama_offload_checked, _llama_offload_supported
    if _llama_offload_checked:
        return _llama_offload_supported
    try:
        from seiso.platform import ensure_cuda_library_path

        ensure_cuda_library_path()
    except ImportError:
        pass
    try:
        import llama_cpp

        for candidate in (
            getattr(llama_cpp, "llama_supports_gpu_offload", None),
            getattr(
                getattr(llama_cpp, "llama_cpp", None),
                "llama_supports_gpu_offload",
                None,
            ),
        ):
            if callable(candidate):
                _llama_offload_supported = bool(candidate())
                _llama_offload_checked = True
                return _llama_offload_supported
    except Exception:
        # Do not cache failure — CUDA preload may succeed on a later call.
        return False
    _llama_offload_checked = True
    return False


def _native_linux_nvidia() -> bool:
    """Linux NVIDIA inference guards — bare metal, or WSL when acknowledged."""
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        return use_linux_nvidia_inference_guards()
    except ImportError:
        return False


def _llama_skip_partial_offload(model_path: str) -> bool:
    """Block partial GPU offload for SWA models (Gemma) on native Linux NVIDIA."""
    if not _native_linux_nvidia():
        return False
    if env_bool("SEISO_LLAMA_UNSAFE_PARTIAL_OFFLOAD", False):
        return False
    if env_bool("SEISO_LLAMA_SKIP_PARTIAL_OFFLOAD", False):
        return True
    try:
        from seiso.inference.family_policy import policy_for_gguf

        return not policy_for_gguf(model_path).allow_partial_offload
    except Exception:
        return False


def _llama_speed_scale_enabled() -> bool:
    # Upscaled batches OOM during prefill after weights land on GPU.
    if _native_linux_nvidia() and not env_bool("SEISO_LLAMA_UNSAFE_SPEED_SCALE", False):
        return False
    default = not _native_linux_nvidia()
    return env_bool("SEISO_LLAMA_SPEED_SCALE", default)


def _default_llama_flash_attn(model_path: str | None = None) -> bool:
    """flash_attn policy on Linux NVIDIA; defaults off, opt in via ``SEISO_LLAMA_FLASH_ATTN=true``."""
    if not _native_linux_nvidia():
        return env_bool("SEISO_LLAMA_FLASH_ATTN", True)
    if not env_bool("SEISO_LLAMA_FLASH_ATTN", False):
        return False
    if not model_path:
        return True
    try:
        from seiso.inference.family_policy import policy_for_gguf

        return policy_for_gguf(model_path).allow_flash_attn
    except Exception:
        return False


def _llama_batch_defaults(model_path: str | None = None) -> tuple[int, int]:
    """Speed-first llama.cpp prompt/decode batch defaults (tight-fit models clamp at load)."""
    if _native_linux_nvidia():
        try:
            from seiso.memory.protection import discrete_gpu_total_mb, gpu_batch_tier_caps

            total = discrete_gpu_total_mb()
            if total > 0:
                return gpu_batch_tier_caps(total, "normal")
        except Exception:
            pass
    return 4096, 1024


def fit_llama_gpu_layers(
    model_path: str,
    requested: int,
    headroom_mb: int,
    *,
    n_ctx: int = 2048,
) -> int:
    """Estimate a fallback GPU layer count after full offload fails."""
    if requested == 0 or headroom_mb <= 0 or not _llama_gpu_offload_ok():
        return 0

    from seiso.memory.protection import (
        estimate_path_vram_mb,
        llama_kv_cache_reserve_mb,
        llama_model_is_tight_vram_fit,
        llama_offload_fits_headroom,
    )

    weight_mb = max(int(estimate_path_vram_mb(model_path)), 256)
    total_layers = gguf_total_layers(model_path)

    try:
        from seiso.memory.protection import discrete_gpu_total_mb

        capacity_mb = discrete_gpu_total_mb() or 0
    except Exception:
        capacity_mb = 0

    def _fits(layers: int, budget_mb: int) -> bool:
        return llama_offload_fits_headroom(
            model_path,
            headroom_mb=budget_mb,
            n_gpu_layers=layers,
            n_ctx=n_ctx,
            weight_mb=weight_mb,
            total_layers=total_layers,
        )

    # Prefer full GPU offload whenever weight+KV fits free VRAM or total GPU capacity.
    if requested == -1:
        if _fits(-1, headroom_mb):
            return -1
        if capacity_mb > headroom_mb and _fits(-1, capacity_mb):
            return -1

    if requested > 0:
        capped = min(requested, total_layers)
        if capped >= total_layers and (
            _fits(-1, headroom_mb) or (capacity_mb > headroom_mb and _fits(-1, capacity_mb))
        ):
            return capped
        if _fits(capped, headroom_mb):
            return capped
        if capacity_mb > headroom_mb and _fits(capped, capacity_mb):
            return capped

    partial_budget = headroom_mb
    if _native_linux_nvidia() and llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=headroom_mb,
        n_gpu_layers=-1 if requested == -1 else max(requested, 1),
        n_ctx=n_ctx,
    ):
        try:
            from seiso.inference.family_policy import policy_for_gguf

            policy = policy_for_gguf(model_path)
            reserve_ratio = min(0.25, 0.12 + (policy.prefill_tightness - 1.0) * 0.20)
        except Exception:
            reserve_ratio = 0.15
        reserve_mb = max(2048, int(headroom_mb * reserve_ratio))
        partial_budget = max(0, headroom_mb - reserve_mb)

    if _llama_skip_partial_offload(model_path):
        try:
            from seiso.memory.protection import discrete_gpu_total_mb

            capacity_mb = discrete_gpu_total_mb() or headroom_mb
        except Exception:
            capacity_mb = headroom_mb
        if capacity_mb > 0 and llama_offload_fits_headroom(
            model_path,
            headroom_mb=capacity_mb,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            weight_mb=weight_mb,
            total_layers=total_layers,
        ):
            return -1
        logger.warning(
            "Partial GPU offload is unsafe for SWA model %s — using CPU",
            Path(model_path).name,
        )
        return 0

    kv_reserve_mb = llama_kv_cache_reserve_mb(
        model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1 if requested == -1 else min(max(requested, 1), total_layers),
        total_layers=total_layers,
        weight_mb=weight_mb,
        free_mb=partial_budget,
    )
    avail_mb = partial_budget - kv_reserve_mb

    if avail_mb < 256:
        logger.warning(
            "VRAM too tight for GPU offload (~%.1f GB free) — falling back to CPU",
            partial_budget / 1024,
        )
        return 0

    fraction = max(0.05, min(avail_mb / weight_mb, 1.0))
    partial = max(1, int(total_layers * fraction))
    if requested not in (-1, 0) and requested > 0:
        partial = min(partial, requested)

    while partial > 0 and not _fits(partial, partial_budget):
        partial -= 1

    if partial <= 0:
        logger.warning(
            "VRAM too tight for GPU offload (~%.1f GB free) — falling back to CPU",
            headroom_mb / 1024,
        )
        return 0
    if partial < total_layers and _llama_skip_partial_offload(model_path):
        logger.warning(
            "Partial GPU offload is unsafe for SWA model %s — using CPU",
            Path(model_path).name,
        )
        return 0
    return partial


def _llama_layer_attempts(
    model_path: str,
    requested: int,
    free_mb: int,
    *,
    n_ctx: int = 2048,
    fitted: int | None = None,
) -> list[int]:
    """Layer counts for partial offload — full GPU is handled separately."""
    if requested == 0 or not _llama_gpu_offload_ok():
        return [0]

    if _llama_skip_partial_offload(model_path):
        return [0]

    total_layers = gguf_total_layers(model_path)

    if not _native_linux_nvidia() and _apple_silicon_metal() and _mac_cpu_offload_enabled():
        max_layers = total_layers if requested == -1 else min(requested, total_layers)
        candidates = [
            max_layers - 1,
            int(max_layers * 0.875),
            int(max_layers * 0.75),
            int(max_layers * 0.5),
            int(max_layers * 0.25),
        ]
        attempts: list[int] = []
        for layers in candidates:
            if 0 < layers < max_layers and layers not in attempts:
                attempts.append(layers)
        if 0 not in attempts:
            attempts.append(0)
        return attempts

    if fitted is None:
        fitted = fit_llama_gpu_layers(model_path, requested, free_mb, n_ctx=n_ctx)
    if fitted in (-1, 0):
        return [0]

    fallback_attempts: list[int] = []
    high = fitted if _native_linux_nvidia() else min(total_layers - 1, fitted + 6)
    step = 4 if high - fitted > 12 else 2
    for layers in range(high, fitted - 1, -step):
        if layers > 0 and layers not in fallback_attempts:
            fallback_attempts.append(layers)
    if fitted not in fallback_attempts:
        fallback_attempts.append(fitted)
    if fitted > 8 and (fitted // 2) not in fallback_attempts:
        fallback_attempts.append(fitted // 2)
    if 0 not in fallback_attempts:
        fallback_attempts.append(0)
    return fallback_attempts


def _llama_partial_memory_profiles(
    memory_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Profiles to try for partial GPU offload after full offload fails."""
    if _apple_silicon_metal() and _mac_cpu_offload_enabled():
        return memory_profiles or [{}]
    return [memory_profiles[-1] if memory_profiles else {}]


def _llama_partial_kqv_options() -> list[dict[str, Any]]:
    """Mac can run larger models by keeping some KV/KQV work off Metal."""
    if not (_apple_silicon_metal() and _mac_cpu_offload_enabled()):
        return [{}]
    return [{}, {"offload_kqv": False}]


def _llama_full_gpu_targets(requested: int) -> list[int]:
    """Layer counts that mean 'all layers on GPU'."""
    if requested == 0 or not _llama_gpu_offload_ok():
        return []
    if requested == -1:
        return [-1]
    if requested > 0:
        return [requested]
    return []


def _llama_speed_extras(model_path: str) -> dict[str, Any]:
    """GGUF-metadata-driven llama.cpp knobs for throughput and VRAM headroom."""
    extras: dict[str, Any] = {}
    try:
        from seiso.inference.family_policy import policy_for_gguf

        if not policy_for_gguf(model_path).swa_full_default and not env_bool(
            "SEISO_LLAMA_SWA_FULL", False
        ):
            extras["swa_full"] = False
    except Exception:
        pass
    return extras


def _llama_kv_quant_options(model_path: str) -> list[dict[str, Any]]:
    """KV-cache quant tiers to try after the unquantized cache fails."""
    _ = model_path
    try:
        from llama_cpp import llama_cpp as lc
    except (ImportError, Exception):
        return [{}]

    options: list[dict[str, Any]] = [{}]
    q8 = {"type_k": lc.GGML_TYPE_Q8_0, "type_v": lc.GGML_TYPE_Q8_0}
    q4 = {"type_k": lc.GGML_TYPE_Q4_K, "type_v": lc.GGML_TYPE_Q4_K}

    if _native_linux_nvidia():
        unsafe = env_bool("SEISO_LLAMA_UNSAFE_KV_QUANT", False)
        if not unsafe and model_path:
            try:
                from seiso.inference.family_policy import policy_for_gguf

                if not policy_for_gguf(model_path).allow_kv_quant:
                    return [{}]
            except Exception:
                pass
        if unsafe:
            options.extend((q8, q4))
        elif env_bool("SEISO_LLAMA_KV_QUANT", False):
            options.append(q8)
    elif env_bool("SEISO_LLAMA_KV_QUANT", True):
        options.extend((q8, q4))

    unique: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for option in options:
        key = tuple(sorted(option.items()))
        if key not in seen:
            seen.add(key)
            unique.append(option)
    return unique


_optimal_layers_cache: dict[tuple[str, int, int, int], tuple[int, float]] = {}
_optimal_layers_lock = threading.Lock()
_OPTIMAL_LAYERS_TTL_S = 8.0


def _refresh_headroom_stats(*, force: bool = False) -> None:
    """Refresh GPU/RAM stats — force only after unload/load, not per chat token."""
    try:
        from seiso.hardware.profile import hardware_profile

        hardware_profile(force_refresh=force)
    except ImportError:
        pass


def _clear_optimal_layers_cache() -> None:
    with _optimal_layers_lock:
        _optimal_layers_cache.clear()


def _llama_gpu_layers_optimal(model_path: str, requested: int, *, n_ctx: int = 2048) -> int:
    """Best layer count for current free VRAM — used to decide cache reload."""
    now = time.time()
    from seiso.memory.protection import headroom_mb

    free_mb = headroom_mb()
    cache_key = (model_path, requested, free_mb // 512, max(int(n_ctx), 2048) // 512)
    with _optimal_layers_lock:
        cached = _optimal_layers_cache.get(cache_key)
        if cached and now - cached[1] < _OPTIMAL_LAYERS_TTL_S:
            return cached[0]

    layers = fit_llama_gpu_layers(model_path, requested, free_mb, n_ctx=n_ctx)
    with _optimal_layers_lock:
        _optimal_layers_cache[cache_key] = (layers, now)
    return layers


def _llama_cache_is_optimal(
    model_path: str, cached_layers: int, requested: int, *, n_ctx: int = 2048
) -> bool:
    """True when a cached llama handle already uses the best GPU offload available."""
    if requested == 0:
        return cached_layers == 0
    if cached_layers == -1:
        if not _native_linux_nvidia():
            return True
        return _llama_gpu_layers_optimal(model_path, requested, n_ctx=n_ctx) == -1
    optimal = _llama_gpu_layers_optimal(model_path, requested, n_ctx=n_ctx)
    if optimal == -1:
        return False
    return cached_layers >= optimal


def _llama_cache_headroom_ok(handle: Any) -> bool:
    """Native Linux cache hit guard for handles loaded before VRAM changed."""
    if not _native_linux_nvidia():
        return True
    loaded_headroom = getattr(handle, "_seiso_load_headroom_mb", None)
    if not loaded_headroom:
        return True
    _refresh_headroom_stats(force=True)
    from seiso.memory.protection import headroom_mb

    return headroom_mb() >= int(int(loaded_headroom) * 0.85)


def llama_load_kwargs(n_ctx: int, *, model_path: str | None = None) -> dict[str, Any]:
    """Tuned llama.cpp defaults for faster preload/first token, overrideable by env."""
    from seiso.memory.protection import clamp_llama_load_kwargs

    n_threads = env_int("SEISO_LLAMA_THREADS", _default_llama_threads())
    n_gpu_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
    # Safety net: if the user or platform_profile set n_gpu_layers != 0 but the
    # installed llama-cpp-python wheel can't actually offload (e.g. CPU-only
    # wheel on an NVIDIA Linux box), force 0 to avoid a crash at Llama init.
    if n_gpu_layers != 0 and not _llama_gpu_offload_ok():
        logger.debug("llama-cpp-python wheel lacks GPU offload support — forcing n_gpu_layers=0")
        n_gpu_layers = 0

    batch_default, ubatch_default = _llama_batch_defaults(model_path)

    n_batch = env_int("SEISO_LLAMA_BATCH", batch_default)
    n_ubatch = min(env_int("SEISO_LLAMA_UBATCH", min(n_batch, ubatch_default)), n_batch)
    kwargs: dict[str, Any] = {
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_threads_batch": env_int(
            "SEISO_LLAMA_THREADS_BATCH",
            _default_llama_threads_batch(n_threads),
        ),
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "n_gpu_layers": n_gpu_layers,
        "use_mmap": env_bool("SEISO_LLAMA_USE_MMAP", True),
        "use_mlock": env_bool("SEISO_LLAMA_USE_MLOCK", False),
        "verbose": env_bool("SEISO_LLAMA_VERBOSE", False),
        "offload_kqv": env_bool("SEISO_LLAMA_OFFLOAD_KQV", n_gpu_layers != 0),
        "no_perf": env_bool("SEISO_LLAMA_NO_PERF", True),
    }
    if n_gpu_layers != 0:
        kwargs["op_offload"] = env_bool("SEISO_LLAMA_OP_OFFLOAD", True)
    if n_gpu_layers != 0 and _default_llama_flash_attn(model_path):
        kwargs["flash_attn"] = True
    if model_path:
        kwargs["_model_path"] = model_path
    return clamp_llama_load_kwargs(kwargs)


def _llama_load_retryable(exc: BaseException) -> bool:
    """True when llama.cpp init failed due to VRAM pressure and a smaller offload may work."""
    msg = str(exc)
    if "Failed to load model from file" in msg or "Failed to create llama_context" in msg:
        return True
    try:
        from seiso.memory.protection import is_oom_error

        return is_oom_error(exc)
    except Exception:
        return False


def _load_llama_model(
    path: str,
    n_ctx: int,
    *,
    tier: str = "normal",
    batch_override: tuple[int, int] | None = None,
) -> Any:
    """Load a GGUF with VRAM-aware layer offload and clear OOM errors."""
    from seiso.memory.protection import (
        LlamaLoadTier,
        clamp_llama_batch_pair,
        llama_load_profile_ladder,
    )

    load_tier: LlamaLoadTier = (
        tier if tier in {"normal", "compact", "minimal"} else "normal"  # type: ignore[assignment]
    )
    try:
        from seiso.platform import ensure_cuda_library_path

        ensure_cuda_library_path()
    except ImportError:
        pass
    from llama_cpp import Llama

    from seiso.inference.tuning import attach_llama_prompt_cache
    from seiso.memory.protection import (
        estimate_path_vram_mb,
        headroom_mb,
        release_cached_memory,
    )

    release_cached_memory(sync=True)
    _clear_optimal_layers_cache()
    _refresh_headroom_stats(force=True)

    est_mb = int(estimate_path_vram_mb(path))
    if est_mb >= 6000:
        try:
            from seiso.hardware.vram_processes import warn_before_large_model_load

            warn_before_large_model_load(model_path=path, est_mb=est_mb)
        except ImportError:
            pass

    kwargs = llama_load_kwargs(n_ctx, model_path=path)
    if batch_override is not None:
        override_batch, override_ubatch = batch_override
        kwargs["n_batch"], kwargs["n_ubatch"] = clamp_llama_batch_pair(
            override_batch,
            override_ubatch,
            native_linux_nvidia=_native_linux_nvidia(),
            load_tier=load_tier,
        )
    speed_extras = _llama_speed_extras(path)
    requested = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
    if requested != 0 and not _llama_gpu_offload_ok():
        requested = 0

    free_mb = headroom_mb()
    n_gpu_layers = int(kwargs.get("n_gpu_layers") or 0)
    effective_n_ctx = int(kwargs.get("n_ctx") or n_ctx)
    memory_profiles = llama_load_profile_ladder(
        model_path=path,
        n_ctx=effective_n_ctx,
        n_gpu_layers=n_gpu_layers,
        free_mb=free_mb,
        base_batch=int(kwargs.get("n_batch") or 512),
        base_ubatch=int(kwargs.get("n_ubatch") or 256),
        tier=load_tier,
    )
    full_gpu_profiles = [
        *memory_profiles,
        {"n_batch": 128, "n_ubatch": 128, "n_ctx": min(effective_n_ctx, 2048)},
    ]
    kv_options = _llama_kv_quant_options(path)
    fitted_layers = fit_llama_gpu_layers(path, requested, free_mb, n_ctx=effective_n_ctx)
    full_targets = _llama_full_gpu_targets(requested) if requested != 0 else []
    partial_targets = (
        _llama_layer_attempts(
            path,
            requested,
            free_mb,
            n_ctx=effective_n_ctx,
            fitted=fitted_layers,
        )
        if requested != 0
        else [0]
    )

    last_exc: Exception | None = None
    seen: set[tuple[int, tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]] = set()

    def _try_load(
        layers: int,
        profile: dict[str, Any],
        kv_quant: dict[str, Any],
        *,
        log_retry: bool,
    ) -> Any | None:
        nonlocal last_exc
        key = (layers, tuple(sorted(profile.items())), tuple(sorted(kv_quant.items())))
        if key in seen:
            return None
        seen.add(key)

        load_kwargs = dict(kwargs)
        load_kwargs.update(speed_extras)
        profile_opts = dict(profile)
        use_prompt_cache = profile_opts.pop("_seiso_prompt_cache", load_tier == "normal")
        if profile_opts.get("flash_attn") is False:
            load_kwargs.pop("flash_attn", None)
            profile_opts.pop("flash_attn", None)
        load_kwargs.update(profile_opts)
        load_kwargs.update(kv_quant)
        load_kwargs["n_gpu_layers"] = layers
        if layers == 0:
            load_kwargs["offload_kqv"] = False
        else:
            load_kwargs["offload_kqv"] = bool(load_kwargs.get("offload_kqv", layers != 0))
        total_layers = gguf_total_layers(path)
        if layers > 0 and layers < total_layers:
            load_kwargs.pop("flash_attn", None)
            if _native_linux_nvidia():
                if not env_bool("SEISO_LLAMA_UNSAFE_PARTIAL_KQV", False):
                    load_kwargs["offload_kqv"] = False
                if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False):
                    load_kwargs["op_offload"] = False
        _refresh_headroom_stats(force=True)
        from seiso.memory.protection import clamp_llama_load_kwargs

        load_kwargs["_model_path"] = path
        load_kwargs = clamp_llama_load_kwargs(load_kwargs)
        load_kwargs.pop("_model_path", None)
        from seiso.inference.llama_vision import apply_llama_vision_load_kwargs

        load_kwargs = apply_llama_vision_load_kwargs(load_kwargs, path)
        # #region agent log
        from seiso.agent_debug_log import agent_debug_enabled, agent_debug_log
        from seiso.memory.protection import llama_model_is_tight_vram_fit

        if agent_debug_enabled():
            agent_debug_log(
                hypothesis_id="D",
                location="model_pool.py:_try_load:before_llama_init",
                message="attempting llama.cpp load",
                data={
                    "model": Path(path).name,
                    "layers": layers,
                    "total_layers": total_layers,
                    "partial_offload": layers > 0 and layers < total_layers,
                    "load_tier": load_tier,
                    "tight_fit": llama_model_is_tight_vram_fit(
                        model_path=path,
                        free_mb=headroom_mb(),
                        n_gpu_layers=layers,
                        n_ctx=int(load_kwargs.get("n_ctx") or effective_n_ctx),
                    ),
                    "n_ctx": load_kwargs.get("n_ctx"),
                    "n_batch": load_kwargs.get("n_batch"),
                    "n_ubatch": load_kwargs.get("n_ubatch"),
                    "flash_attn": load_kwargs.get("flash_attn"),
                    "offload_kqv": load_kwargs.get("offload_kqv"),
                    "op_offload": load_kwargs.get("op_offload"),
                },
            )
        # #endregion
        try:
            llm = Llama(model_path=path, **load_kwargs)
            llm._seiso_n_gpu_layers = layers  # noqa: SLF001
            llm._seiso_load_tier = load_tier  # noqa: SLF001
            llm._seiso_n_batch = int(load_kwargs.get("n_batch") or 0)  # noqa: SLF001
            llm._seiso_n_ubatch = int(load_kwargs.get("n_ubatch") or 0)  # noqa: SLF001
            llm._seiso_n_ctx = int(load_kwargs.get("n_ctx") or effective_n_ctx)  # noqa: SLF001
            llm._seiso_model_path = path  # noqa: SLF001
            llm._seiso_load_headroom_mb = headroom_mb()  # noqa: SLF001
            if batch_override is not None:
                llm._seiso_last_safe_batch = int(load_kwargs.get("n_batch") or 0)  # noqa: SLF001
                llm._seiso_last_safe_ubatch = int(load_kwargs.get("n_ubatch") or 0)  # noqa: SLF001
            if layers > 0:
                total_layers = gguf_total_layers(path)
                if layers < total_layers:
                    logger.warning(
                        "Partial GPU offload for %s: %d/%d layers (~%.1f GB free) — "
                        "close other GPU apps for full GPU offload and ~3× faster generation",
                        Path(path).name,
                        layers,
                        total_layers,
                        headroom_mb() / 1024,
                    )
            if use_prompt_cache:
                attach_llama_prompt_cache(llm, model_path=path)
            # #region agent log
            if agent_debug_enabled():
                agent_debug_log(
                    hypothesis_id="E",
                    location="model_pool.py:_try_load:load_success",
                    message="llama.cpp load succeeded",
                    data={
                        "model": Path(path).name,
                        "layers": layers,
                        "load_tier": load_tier,
                        "n_ctx": load_kwargs.get("n_ctx"),
                        "n_batch": load_kwargs.get("n_batch"),
                    },
                )
            # #endregion
            return llm
        except Exception as exc:
            if not _llama_load_retryable(exc):
                raise
            last_exc = exc
            release_cached_memory(sync=True)
            if log_retry:
                logger.warning("llama.cpp load failed at n_gpu_layers=%s — retrying", layers)
            return None

    for layers in full_targets:
        for profile_idx, profile in enumerate(full_gpu_profiles):
            for kv_idx, kv_quant in enumerate(kv_options):
                log_retry = (
                    profile_idx == len(full_gpu_profiles) - 1 and kv_idx == len(kv_options) - 1
                )
                llm = _try_load(layers, profile, kv_quant, log_retry=log_retry)
                if llm is not None:
                    return llm

    partial_profiles = _llama_partial_memory_profiles(memory_profiles)
    partial_kqv_options = _llama_partial_kqv_options()
    skip_partial = _llama_skip_partial_offload(path)
    for layer_idx, layers in enumerate(partial_targets):
        if layers in full_targets:
            continue
        if skip_partial and layers not in (0, -1):
            continue
        for profile_idx, profile in enumerate(partial_profiles):
            for kqv_idx, kqv_option in enumerate(partial_kqv_options):
                # KV-quantized caches can crash llama.cpp on partial offload (Qwen/MoE).
                first_partial_attempt = layer_idx == 0 and profile_idx == 0 and kqv_idx == 0
                final_partial_attempt = (
                    layer_idx == len(partial_targets) - 1
                    and profile_idx == len(partial_profiles) - 1
                    and kqv_idx == len(partial_kqv_options) - 1
                )
                log_retry = first_partial_attempt or final_partial_attempt
                llm = _try_load(layers, profile, kqv_option, log_retry=log_retry)
                if llm is not None:
                    return llm

    free_gb = round(headroom_mb() / 1024, 1)
    need_gb = round(estimate_path_vram_mb(path) / 1024, 1)
    raise RuntimeError(
        f"Could not load model — needs ~{need_gb} GB GPU/RAM headroom but only ~{free_gb} GB is free. "
        "Close other GPU apps (browser, games, other llama.cpp sessions), unload the previous model, "
        "or pick a smaller quant."
    ) from last_exc


_POOL_BACKEND_BY_API: dict[str, str] = {
    "llamacpp": "llamacpp",
    "llama": "llamacpp",
    "llamaswap": "llamaswap",
    "mlx": "mlx",
    "torch": "torch",
}


class BackendKind(StrEnum):
    LLAMACPP = "llamacpp"
    LLAMASWAP = "llamaswap"
    MLX = "mlx"
    TORCH = "torch"

    # Backward-compatible alias for older call sites.
    LLAMA = LLAMACPP


_switch_load_lock = threading.Lock()
_dflash_key_locks: dict[str, threading.Lock] = {}
_dflash_key_locks_guard = threading.Lock()


@dataclass
class LoadedModel:
    key: str
    backend: BackendKind
    handle: Any
    meta: dict = field(default_factory=dict)


class ModelPool:
    """
    Singleton pool holding at most one active inference model.
    Switching models unloads the previous one from VRAM/RAM.
    """

    _instance: ModelPool | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._active: LoadedModel | None = None
        self._generation = 0
        self._inference_refs = 0
        self._unload_pending = False

    @classmethod
    def get(cls) -> ModelPool:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def active_key(self) -> str | None:
        with self._lock:
            return self._active.key if self._active else None

    @staticmethod
    def normalize_path(model_path: str) -> str:
        return str(Path(model_path).expanduser().resolve())

    def bump_generation(self) -> int:
        """Invalidate in-flight streams (e.g. before loading another model)."""
        with self._lock:
            self._generation += 1
            return self._generation

    def is_generation_active(self, generation_id: int) -> bool:
        with self._lock:
            return generation_id == self._generation

    def begin_inference(self) -> None:
        with self._lock:
            self._inference_refs += 1

    def end_inference(self) -> None:
        should_unload = False
        with self._lock:
            self._inference_refs = max(0, self._inference_refs - 1)
            if self._inference_refs == 0 and self._unload_pending:
                should_unload = True
        if should_unload:
            self.unload_all()

    def cancel_and_unload(self) -> None:
        """Stop lagging streams and release VRAM/RAM."""
        self.bump_generation()
        with self._lock:
            self._unload_pending = True
            if self._inference_refs > 0:
                return
        self.unload_all()
        clear_dflash_draft_cache()

    def _wait_for_inference_idle(self, timeout_s: float = 30.0) -> bool:
        """Wait for active completions/streams to release their pool refs.

        Returns True when idle. On timeout, does not force-clear refs.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._inference_refs == 0:
                    return True
            time.sleep(0.05)
        with self._lock:
            still_busy = self._inference_refs > 0
        if still_busy:
            logger.warning(
                "Inference still active after %.1fs — proceeding with forced unload",
                timeout_s,
            )
        return not still_busy

    def _release_handle(self, active: LoadedModel) -> None:
        """Close one pool handle and free GPU caches."""
        backend = active.backend
        key = active.key
        handle = active.handle
        logger.info("Unloading model from VRAM: %s", key)

        if backend == BackendKind.LLAMA:
            llm = handle
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                logger.debug("Failed to close llama handle for %s", key, exc_info=True)
            del llm

        elif backend == BackendKind.LLAMASWAP:
            del handle

        elif backend == BackendKind.TORCH:
            try:
                from seiso.inference.speculative import TorchSpeculativeBundle
                from seiso.kernels.lifecycle import release_training_memory

                if isinstance(handle, TorchSpeculativeBundle):
                    release_training_memory(handle.target_model, sync=False)
                    release_training_memory(handle.draft_model, sync=False)
                else:
                    model = handle[0] if isinstance(handle, tuple) and handle else handle
                    release_training_memory(model, sync=False)
            except Exception:
                logger.debug("Failed to release torch handle for %s", key, exc_info=True)
            del handle

        elif backend == BackendKind.MLX:
            del handle

        self._free_memory(sync=True)
        clear_dflash_draft_cache()

    def _unload_active_immediate(self) -> None:
        """Drop the active handle without canceling the in-flight request.

        Used by OOM/prefill recovery: the caller already holds an inference ref
        and will replace the handle, so we must not wait on our own ref or
        invalidate generation_id.
        """
        with self._lock:
            active = self._active
            self._active = None
            self._unload_pending = False
        if active is not None:
            self._release_handle(active)
        else:
            clear_dflash_draft_cache()

    def _is_same_model_reload(
        self,
        key: str,
        backend: BackendKind,
    ) -> bool:
        """True when switch() is reloading the same llama.cpp pool entry."""
        if backend != BackendKind.LLAMA:
            return False
        with self._lock:
            active = self._active
        return bool(active and active.backend == backend and active.key == key)

    def _clear_active_for_switch(self) -> None:
        """Stop streams, wait for idle, then unload so a new model can load."""
        self.bump_generation()
        idle = self._wait_for_inference_idle()
        with self._lock:
            self._unload_pending = False
            if not idle and self._inference_refs > 0:
                # Streams are already cancelled via generation bump; force the
                # unload path so a new model can load. Native work may still
                # be finishing, but holding the old model blocks all GPU tasks.
                self._inference_refs = 0
            active = self._active
            self._active = None
        if active is not None:
            self._release_handle(active)
        else:
            clear_dflash_draft_cache()

    def would_switch_model(
        self, target_path: str, backend: str | BackendKind | None = None
    ) -> bool:
        """True when loading target_path would replace the active inference model."""
        with self._lock:
            active = self._active
        if not active:
            return False
        if backend is not None:
            raw = backend.value if isinstance(backend, BackendKind) else str(backend).lower()
            expected = _POOL_BACKEND_BY_API.get(raw, raw)
            if active.backend.value != expected:
                return True
        active_path = active.meta.get("path") or active.meta.get("norm_path")
        if not active_path:
            return False
        norm_target = self.normalize_path(target_path)
        norm_active = self.normalize_path(str(active_path))
        if norm_active != norm_target:
            return True
        # Speculative bundles (spec:target:draft) are not interchangeable with
        # single-model pool handles that share the same target path.
        return active.key.startswith("spec:")

    def prepare_for_load(
        self,
        target_path: str | None = None,
        backend: str | BackendKind | None = None,
    ) -> bool:
        """Unload the active model when switching and refresh GPU memory stats."""
        should_unload = target_path is None or self.would_switch_model(target_path, backend)
        unloaded = False
        if should_unload and self.active_key:
            # Wait for in-flight inference so VRAM is actually freed before load.
            self._clear_active_for_switch()
            unloaded = True
        if unloaded:
            _clear_optimal_layers_cache()
            _refresh_headroom_stats(force=True)
        return unloaded

    def _switch_cache_hit(
        self,
        key: str,
        backend: BackendKind,
        load_path: str,
        meta: dict[str, Any],
    ) -> Any | None:
        with self._lock:
            if not self._active or self._active.key != key:
                return None
            needed_ctx = int(meta.get("n_ctx") or 0)
            cached_ctx = int(self._active.meta.get("n_ctx") or 0)
            if needed_ctx > 0 and cached_ctx < needed_ctx:
                return None
            if backend != BackendKind.LLAMA:
                return self._active.handle
            cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
            requested_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
            if _llama_cache_is_optimal(
                load_path,
                cached_layers,
                requested_layers,
                n_ctx=needed_ctx or cached_ctx or 2048,
            ) and _llama_cache_headroom_ok(self._active.handle):
                return self._active.handle
            return None

    def switch(
        self,
        model_path: str,
        backend: BackendKind,
        loader_fn,
        *,
        cache_key: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        """Load model_path, unloading any previously active model first."""
        norm = self.normalize_path(model_path)
        # Only resolve to an absolute filesystem path when the input is an
        # existing local file/dir.  HuggingFace repo IDs like "org/model-name"
        # must be passed through unchanged so transformers can download them.
        raw = Path(model_path).expanduser()
        load_path = str(raw.absolute()) if raw.exists() else str(model_path)
        key = cache_key or f"{backend.value}:{norm}"
        meta = meta or {}
        cached = self._switch_cache_hit(key, backend, load_path, meta)
        if cached is not None:
            return cached

        with _switch_load_lock:
            cached = self._switch_cache_hit(key, backend, load_path, meta)
            if cached is not None:
                return cached

            if self._active:
                if self._is_same_model_reload(key, backend):
                    # Preload/chat often reload the warmed handle (larger n_ctx,
                    # safer batch, layer fit). Preserve generation so the active
                    # chat request is not discarded after reload completes.
                    self._unload_active_immediate()
                else:
                    self._clear_active_for_switch()
                _clear_optimal_layers_cache()
                _refresh_headroom_stats(force=True)
            from seiso.memory.protection import (
                ensure_load_fits,
                estimate_path_vram_mb,
                headroom_mb,
                release_cached_memory,
            )

            if backend == BackendKind.LLAMA:
                from seiso.inference.llama_vision import resolve_mmproj_path

                est_mb = int(estimate_path_vram_mb(load_path))
                mmproj = resolve_mmproj_path(load_path)
                if mmproj:
                    est_mb += int(estimate_path_vram_mb(mmproj))
            else:
                est_mb = int(estimate_path_vram_mb(load_path))
            if est_mb >= 8000 and headroom_mb() < int(est_mb * 0.98):
                if self._active:
                    self._clear_active_for_switch()
                self._free_memory()
                release_cached_memory(sync=True)
                _clear_optimal_layers_cache()
                _refresh_headroom_stats(force=True)

            logger.info("Loading model: %s (%s)", norm, backend.value)
            ensure_load_fits(load_path, mode="chat", backend=backend.value)
            try:
                handle = loader_fn(load_path)
            except Exception:
                self._free_memory()
                raise
            layer_meta: dict[str, Any] = {}
            if backend == BackendKind.LLAMA:
                layer_meta["n_gpu_layers"] = int(getattr(handle, "_seiso_n_gpu_layers", -1))
                requested_ctx = int((meta or {}).get("n_ctx") or 0)
                layer_meta["n_ctx"] = int(
                    getattr(handle, "_seiso_n_ctx", requested_ctx) or requested_ctx or 0
                )
            with self._lock:
                if (
                    self._active
                    and self._active.key == key
                    and self._switch_cache_hit(key, backend, load_path, meta) is not None
                ):
                    try:
                        if hasattr(handle, "close"):
                            handle.close()
                    except Exception:
                        logger.debug(
                            "Failed to close duplicate load handle",
                            exc_info=True,
                        )
                    return self._active.handle
                if self._active:
                    # Force-clear any stale active handle before install.
                    stale = self._active
                    self._active = None
                    self._unload_pending = False
                else:
                    stale = None
                    self._unload_pending = False
                self._active = LoadedModel(
                    key=key,
                    backend=backend,
                    handle=handle,
                    meta={
                        "path": load_path,
                        "norm_path": norm,
                        **(meta or {}),
                        **layer_meta,
                    },
                )
            if stale is not None:
                try:
                    if hasattr(stale.handle, "close"):
                        stale.handle.close()
                except Exception:
                    logger.debug("Failed to close stale pool handle", exc_info=True)
            return handle

    def get_llama(
        self,
        model_path: str,
        n_ctx: int = 4096,
        *,
        tier: str = "normal",
    ) -> Any:
        def loader(path: str):
            return _load_llama_model(path, n_ctx, tier=tier)

        norm = self.normalize_path(model_path)
        requested_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
        with self._lock:
            if (
                tier == "normal"
                and self._active
                and self._active.backend == BackendKind.LLAMA
                and self._active.meta.get("norm_path") == norm
                and self._active.meta.get("load_tier", "normal") == "normal"
            ):
                cached_ctx = int(self._active.meta.get("n_ctx") or 0)
                cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
                if (
                    cached_ctx >= n_ctx
                    and _llama_cache_is_optimal(
                        str(self._active.meta.get("path") or model_path),
                        cached_layers,
                        requested_layers,
                        n_ctx=n_ctx,
                    )
                    and _llama_cache_headroom_ok(self._active.handle)
                ):
                    return self._active.handle

        key = f"llama:{norm}" if tier == "normal" else f"llama:{norm}:{tier}"
        return self.switch(
            model_path,
            BackendKind.LLAMA,
            loader,
            cache_key=key,
            meta={"n_ctx": n_ctx, "load_tier": tier},
        )

    def reload_llama(
        self,
        model_path: str,
        n_ctx: int,
        *,
        tier: str,
        batch_override: tuple[int, int] | None = None,
    ) -> Any:
        """Unload and reload llama.cpp at a lower memory tier after inference OOM.

        Preserves the active generation id and does not wait on the caller's
        own inference ref (recovery always runs under begin_inference).
        """
        self._unload_active_immediate()
        _clear_optimal_layers_cache()
        _refresh_headroom_stats(force=True)
        if batch_override is None:
            return self.get_llama(model_path, n_ctx=n_ctx, tier=tier)

        def loader(path: str):
            return _load_llama_model(path, n_ctx, tier=tier, batch_override=batch_override)

        norm = self.normalize_path(model_path)
        key = f"llama:{norm}:{tier}:batch:{batch_override[0]}:{batch_override[1]}"
        return self.switch(
            model_path,
            BackendKind.LLAMA,
            loader,
            cache_key=key,
            meta={"n_ctx": n_ctx, "load_tier": tier},
        )

    def reload_llama_compact(self, model_path: str, n_ctx: int) -> Any:
        """Backward-compatible alias for compact-tier reload."""
        return self.reload_llama(model_path, n_ctx, tier="compact")

    def get_llamaswap(self, model_path: str) -> Any:
        def loader(_path: str):
            from seiso.inference.llamaswap import LlamaSwapClient

            return LlamaSwapClient()

        norm = self.normalize_path(model_path)
        key = f"llamaswap:{norm}"
        return self.switch(
            model_path,
            BackendKind.LLAMASWAP,
            loader,
            cache_key=key,
            meta={"sidecar": True},
        )

    def get_mlx(self, model_path: str) -> tuple[Any, Any]:
        def loader(path: str):
            from seiso.models.loader import LoadOptions, ModelKind
            from seiso.models.mlx_loader import load_mlx

            return load_mlx(LoadOptions(model_id=path, kind=ModelKind.TEXT))

        return self.switch(model_path, BackendKind.MLX, loader)

    def get_torch(self, model_path: str, *, load_in_4bit: bool = True) -> tuple[Any, Any]:
        def loader(path: str):
            return self._load_torch_pair(path, load_in_4bit=load_in_4bit)

        return self.switch(model_path, BackendKind.TORCH, loader)

    def _load_torch_pair(self, model_path: str, *, load_in_4bit: bool = True) -> tuple[Any, Any]:
        from seiso.inference.tuning import apply_inference_kernels, prepare_torch_model
        from seiso.models.loader import LoadOptions, ModelKind, load_model

        model, tokenizer = load_model(
            LoadOptions(
                model_id=model_path,
                kind=ModelKind.TEXT,
                load_in_4bit=load_in_4bit,
                device_map="auto",
            )
        )
        prepare_torch_model(model)
        apply_inference_kernels(model)
        return model, tokenizer

    def get_torch_speculative(
        self, target_path: str, draft_path: str, *, load_in_4bit: bool = True
    ) -> Any:
        """Load target + draft models for speculative decoding (torch draft)."""
        from seiso.inference.speculative import TorchSpeculativeBundle

        target_norm = self.normalize_path(target_path)
        draft_norm = self.normalize_path(draft_path)
        key = f"spec:{target_norm}:{draft_norm}"

        def loader(_path: str) -> TorchSpeculativeBundle:
            from seiso.memory.protection import (
                MemoryLoadBlockedError,
                allow_memory_overcommit,
                ensure_load_fits,
                estimate_path_vram_mb,
                headroom_mb,
            )

            target_mb = int(estimate_path_vram_mb(target_path, mode="chat"))
            draft_mb = int(estimate_path_vram_mb(draft_path, mode="chat"))
            needed_mb = target_mb + draft_mb
            free_mb = headroom_mb()
            if needed_mb > free_mb and not allow_memory_overcommit():
                raise MemoryLoadBlockedError(
                    f"Speculative pair needs ~{needed_mb}MB "
                    f"(target={target_mb}MB + draft={draft_mb}MB) but only {free_mb}MB free"
                )
            ensure_load_fits(target_path, mode="chat", backend=BackendKind.TORCH.value)
            ensure_load_fits(draft_path, mode="chat", backend=BackendKind.TORCH.value)
            target_model, target_tokenizer = self._load_torch_pair(
                target_path, load_in_4bit=load_in_4bit
            )
            draft_model, draft_tokenizer = self._load_torch_pair(
                draft_path, load_in_4bit=load_in_4bit
            )
            return TorchSpeculativeBundle(
                target_model=target_model,
                target_tokenizer=target_tokenizer,
                draft_model=draft_model,
                draft_tokenizer=draft_tokenizer,
            )

        return self.switch(
            target_path,
            BackendKind.TORCH,
            loader,
            cache_key=key,
            meta={
                "path": target_path,
                "norm_path": target_norm,
                "draft_path": draft_path,
                "draft_norm_path": draft_norm,
            },
        )

    # Note: dflash drafts are loaded directly in the runner using llama_cpp.Llama
    # to avoid interfering with the primary target model's active handle in the pool.

    def unload_all(self) -> None:
        """Release all loaded models and clear GPU memory."""
        with self._lock:
            if self._inference_refs > 0:
                self._unload_pending = True
                return
            if not self._active:
                self._unload_pending = False
                clear_dflash_draft_cache()
                return
            active = self._active
            self._active = None
            self._unload_pending = False

        self._release_handle(active)

    def _free_memory(self, *, sync: bool = False) -> None:
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=sync)
        _clear_optimal_layers_cache()

    def status(self) -> dict:
        with self._lock:
            active = self._active
            return {
                "active_model": active.key if active else None,
                "backend": active.backend.value if active else None,
                "path": active.meta.get("path") if active else None,
                "draft_path": active.meta.get("draft_path") if active else None,
            }


class DflashDraftHandle:
    """Thread-safe wrapper around a cached llama.cpp dflash/draft model."""

    __slots__ = ("llm", "n_ctx", "_infer_lock")

    def __init__(self, llm: Any, n_ctx: int = 0) -> None:
        self.llm = llm
        self.n_ctx = n_ctx
        self._infer_lock = threading.Lock()

    def dispose(self) -> None:
        """Close the native handle only after in-flight infer finishes."""
        with self._infer_lock:
            llm = self.llm
            self.llm = None
            if llm is None:
                return
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                logger.debug("Failed to close dflash draft handle", exc_info=True)


_dflash_draft_cache: dict[str, DflashDraftHandle] = {}
_dflash_draft_lock = threading.Lock()


def _load_dflash_llm(resolved_path: str, n_ctx: int) -> Any:
    return _load_llama_model(resolved_path, n_ctx)


def _dflash_lock_for(norm: str) -> threading.Lock:
    with _dflash_key_locks_guard:
        return _dflash_key_locks.setdefault(norm, threading.Lock())


def get_dflash_draft(model_path: str, *, n_ctx: int = 4096) -> DflashDraftHandle:
    """Return a cached, thread-safe llama.cpp handle for dflash/draft GGUF models."""
    from seiso.inference.backends import BACKEND_LLAMACPP, prepare_model_path

    resolved = prepare_model_path(model_path, BACKEND_LLAMACPP)
    norm = str(Path(resolved).resolve())
    with _dflash_lock_for(norm):
        with _dflash_draft_lock:
            cached = _dflash_draft_cache.get(norm)
            if cached is not None and cached.n_ctx >= n_ctx and cached.llm is not None:
                return cached

        llm = _load_dflash_llm(resolved, n_ctx)

        old_cached: DflashDraftHandle | None = None
        with _dflash_draft_lock:
            cached = _dflash_draft_cache.get(norm)
            if cached is not None and cached.n_ctx >= n_ctx and cached.llm is not None:
                try:
                    if hasattr(llm, "close"):
                        llm.close()
                except Exception:
                    logger.debug("Failed to close duplicate dflash draft", exc_info=True)
                return cached
            if cached is not None:
                # Drop from cache first so new callers do not receive a disposed handle.
                _dflash_draft_cache.pop(norm, None)
                old_cached = cached
            handle = DflashDraftHandle(llm, n_ctx=n_ctx)
            _dflash_draft_cache[norm] = handle
        if old_cached is not None:
            old_cached.dispose()
        return handle


def dflash_draft_infer(
    draft: Any,
    current_text: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Run a single dflash draft completion under the per-handle inference lock."""
    if isinstance(draft, DflashDraftHandle):
        llm = draft.llm
        infer_lock = draft._infer_lock
    else:
        llm = draft
        infer_lock = None

    gen_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "echo": False,
        "temperature": max(temperature, 0.0) if temperature and temperature > 0 else 0.0,
    }
    if not temperature or temperature <= 0:
        gen_kwargs["temperature"] = 0.0

    if infer_lock is not None:
        with infer_lock:
            llm = draft.llm
            if llm is None:
                return ""
            out = llm(current_text, **gen_kwargs)
    else:
        out = llm(current_text, **gen_kwargs)

    return out["choices"][0]["text"] if out.get("choices") else ""


def clear_dflash_draft_cache() -> None:
    """Release cached dflash draft models."""
    with _dflash_draft_lock:
        handles = list(_dflash_draft_cache.values())
        _dflash_draft_cache.clear()
    for handle in handles:
        handle.dispose()


def get_model_pool() -> ModelPool:
    return ModelPool.get()
