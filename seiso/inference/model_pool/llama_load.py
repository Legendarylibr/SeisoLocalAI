"""llama.cpp load heuristics, kwargs, and model loading."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

from seiso.env import env_bool, env_int
from seiso.inference.family_policy import policy_for_gguf
from seiso.inference.model_pool._facade import model_pool as _mp
from seiso.memory.protection._facade import protection as _prot
from seiso.memory.protection.constants import (
    _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS,
    LlamaLoadTier,
)

logger = logging.getLogger(__name__)


def _default_llama_threads() -> int:
    cpus = _mp()._available_cpu_count()
    # Decode is latency-sensitive; leave one core free for OS, driver, and
    # Python streaming while still giving llama.cpp enough workers.
    return max(2, min(cpus - 1 if cpus > 4 else cpus, 16))


def _default_llama_threads_batch(n_threads: int) -> int:
    """Use wider CPU parallelism for prompt prefill without changing decode threads."""
    if "SEISO_LLAMA_THREADS" in os.environ:
        return n_threads
    cpus = _mp()._available_cpu_count()
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
    if _mp()._apple_silicon_metal():
        return -1
    if (
        _mp()._cuda_available() or _mp()._nvidia_hardware_visible()
    ) and _mp()._llama_gpu_offload_ok():
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
    if not _mp()._native_linux_nvidia():
        return False
    if env_bool("SEISO_LLAMA_UNSAFE_PARTIAL_OFFLOAD", False):
        return False
    try:
        return not policy_for_gguf(model_path).allow_partial_offload
    except Exception:
        return False


def _default_llama_flash_attn(model_path: str | None = None) -> bool:
    """flash_attn policy on Linux NVIDIA; defaults off, opt in via ``SEISO_LLAMA_FLASH_ATTN=true``."""
    if not _mp()._native_linux_nvidia():
        return env_bool("SEISO_LLAMA_FLASH_ATTN", True)
    if not env_bool("SEISO_LLAMA_FLASH_ATTN", False):
        return False
    if not model_path:
        return True
    try:
        return policy_for_gguf(model_path).allow_flash_attn
    except Exception:
        return False


def _llama_batch_defaults(model_path: str | None = None) -> tuple[int, int]:
    """llama.cpp prompt/decode batch defaults before model-aware load clamps."""
    if _mp()._native_linux_nvidia():
        try:
            total = _prot().discrete_gpu_total_mb()
            gpu_batch_tier_caps = _prot().gpu_batch_tier_caps
            if total > 0:
                return gpu_batch_tier_caps(total, "normal")
        except Exception:
            pass
        return _NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS
    return 4096, 1024


def fit_llama_gpu_layers(
    model_path: str,
    requested: int,
    headroom_mb: int,
    *,
    n_ctx: int = 2048,
) -> int:
    """Estimate a fallback GPU layer count after full offload fails."""
    if requested == 0 or headroom_mb <= 0 or not _mp()._llama_gpu_offload_ok():
        return 0

    weight_mb = max(int(_prot().estimate_path_vram_mb(model_path)), 256)
    total_layers = _mp().gguf_total_layers(model_path)
    decode_reserve_mb = (
        _prot().llama_decode_reserve_mb(
            gpu_total_mb=_prot().discrete_gpu_total_mb(),
            free_mb=headroom_mb,
            max_tokens=512,
            model_path=model_path,
        )
        if _mp()._native_linux_nvidia()
        else 0
    )
    fit_budget_mb = max(0, headroom_mb - decode_reserve_mb)

    def _fits(layers: int, budget_mb: int) -> bool:
        return _prot().llama_offload_fits_headroom(
            model_path,
            headroom_mb=budget_mb,
            n_gpu_layers=layers,
            n_ctx=n_ctx,
            weight_mb=weight_mb,
            total_layers=total_layers,
        )

    # Prefer full GPU offload only when weight+KV fits currently free VRAM.
    if requested == -1 and _fits(-1, fit_budget_mb):
        return -1

    if requested > 0:
        capped = min(requested, total_layers)
        if capped >= total_layers and _fits(-1, fit_budget_mb):
            return capped
        if _fits(capped, fit_budget_mb):
            return capped

    partial_budget = fit_budget_mb
    if _mp()._native_linux_nvidia() and _prot().llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=headroom_mb,
        n_gpu_layers=-1 if requested == -1 else max(requested, 1),
        n_ctx=n_ctx,
    ):
        try:
            policy = policy_for_gguf(model_path)
            reserve_ratio = min(0.25, 0.12 + (policy.prefill_tightness - 1.0) * 0.20)
        except Exception:
            reserve_ratio = 0.15
        reserve_mb = max(2048, int(headroom_mb * reserve_ratio))
        partial_budget = max(0, headroom_mb - reserve_mb)

    if _llama_skip_partial_offload(model_path):
        try:
            capacity_mb = _prot().discrete_gpu_total_mb() or headroom_mb
        except Exception:
            capacity_mb = headroom_mb
        capacity_budget_mb = max(0, capacity_mb - decode_reserve_mb)
        if capacity_mb > 0 and _prot().llama_offload_fits_headroom(
            model_path,
            headroom_mb=capacity_budget_mb,
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

    kv_reserve_mb = _prot().llama_kv_cache_reserve_mb(
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
    if requested == 0 or not _mp()._llama_gpu_offload_ok():
        return [0]

    if _llama_skip_partial_offload(model_path):
        return [0]

    total_layers = _mp().gguf_total_layers(model_path)

    if (
        not _mp()._native_linux_nvidia()
        and _mp()._apple_silicon_metal()
        and _mp()._mac_cpu_offload_enabled()
    ):
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
        fitted = _mp().fit_llama_gpu_layers(model_path, requested, free_mb, n_ctx=n_ctx)
    if fitted in (-1, 0):
        return [0]

    fallback_attempts: list[int] = []
    high = fitted if _mp()._native_linux_nvidia() else min(total_layers - 1, fitted + 6)
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
    if _mp()._apple_silicon_metal() and _mp()._mac_cpu_offload_enabled():
        return memory_profiles or [{}]
    return [memory_profiles[-1] if memory_profiles else {}]


def _llama_partial_kqv_options() -> list[dict[str, Any]]:
    """Mac can run larger models by keeping some KV/KQV work off Metal."""
    if not (_mp()._apple_silicon_metal() and _mp()._mac_cpu_offload_enabled()):
        return [{}]
    return [{}, {"offload_kqv": False}]


def _llama_full_gpu_targets(requested: int) -> list[int]:
    """Layer counts that mean 'all layers on GPU'."""
    if requested == 0 or not _mp()._llama_gpu_offload_ok():
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
        if not policy_for_gguf(model_path).swa_full_default and not env_bool(
            "SEISO_LLAMA_SWA_FULL", False
        ):
            extras["swa_full"] = False
    except Exception:
        pass
    return extras


def _llama_kv_quant_options(model_path: str) -> list[dict[str, Any]]:
    """KV-cache quant tiers to try after the unquantized cache fails."""
    try:
        from llama_cpp import llama_cpp as lc
    except (ImportError, Exception):
        return [{}]

    options: list[dict[str, Any]] = [{}]
    q8 = {"type_k": lc.GGML_TYPE_Q8_0, "type_v": lc.GGML_TYPE_Q8_0}
    q4 = {"type_k": lc.GGML_TYPE_Q4_K, "type_v": lc.GGML_TYPE_Q4_K}

    if _mp()._native_linux_nvidia():
        unsafe = env_bool("SEISO_LLAMA_UNSAFE_KV_QUANT", False)
        if not unsafe and model_path:
            try:
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
    free_mb = _prot().headroom_mb()
    cache_key = (model_path, requested, free_mb // 512, max(int(n_ctx), 2048) // 512)
    with _optimal_layers_lock:
        cached = _optimal_layers_cache.get(cache_key)
        if cached and now - cached[1] < _OPTIMAL_LAYERS_TTL_S:
            return cached[0]

    layers = _mp().fit_llama_gpu_layers(model_path, requested, free_mb, n_ctx=n_ctx)
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
        if not _mp()._native_linux_nvidia():
            return True
        return _mp()._llama_gpu_layers_optimal(model_path, requested, n_ctx=n_ctx) == -1
    optimal = _mp()._llama_gpu_layers_optimal(model_path, requested, n_ctx=n_ctx)
    if optimal == -1:
        return False
    return cached_layers >= optimal


def _llama_cache_headroom_ok(handle: Any) -> bool:
    """Native Linux cache hit guard for handles loaded before VRAM changed."""
    if not _mp()._native_linux_nvidia():
        return True
    loaded_headroom = getattr(handle, "_seiso_load_headroom_mb", None)
    if not loaded_headroom:
        return True
    _mp()._refresh_headroom_stats(force=True)
    current = _prot().headroom_mb()
    if current < int(int(loaded_headroom) * 0.85):
        return False
    if _mp()._native_linux_nvidia():
        reserve = _prot().llama_decode_reserve_mb(
            gpu_total_mb=_prot().discrete_gpu_total_mb(),
            free_mb=current,
            max_tokens=512,
            model_path=getattr(handle, "_seiso_model_path", None),
        )
        return current > reserve
    return True


def llama_load_kwargs(n_ctx: int, *, model_path: str | None = None) -> dict[str, Any]:
    """Tuned llama.cpp defaults for faster preload/first token, overrideable by env."""
    n_threads = env_int("SEISO_LLAMA_THREADS", _default_llama_threads())
    n_gpu_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _mp()._default_llama_gpu_layers())
    # Safety net: if the user or platform_profile set n_gpu_layers != 0 but the
    # installed llama-cpp-python wheel can't actually offload (e.g. CPU-only
    # wheel on an NVIDIA Linux box), force 0 to avoid a crash at Llama init.
    if n_gpu_layers != 0 and not _mp()._llama_gpu_offload_ok():
        logger.debug("llama-cpp-python wheel lacks GPU offload support — forcing n_gpu_layers=0")
        n_gpu_layers = 0

    batch_default, ubatch_default = _llama_batch_defaults(model_path)
    if _mp()._native_linux_nvidia():
        n_batch = batch_default
        n_ubatch = min(batch_default, ubatch_default)
    else:
        n_batch = env_int("SEISO_LLAMA_BATCH", batch_default)
        n_ubatch = min(env_int("SEISO_LLAMA_UBATCH", min(n_batch, ubatch_default)), n_batch)
    native_linux_nvidia = _mp()._native_linux_nvidia()
    native_offload_kqv = n_gpu_layers != 0 and env_bool("SEISO_LLAMA_UNSAFE_KQV_OFFLOAD", False)
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
        "offload_kqv": (
            native_offload_kqv
            if native_linux_nvidia
            else env_bool("SEISO_LLAMA_OFFLOAD_KQV", n_gpu_layers != 0)
        ),
        "no_perf": env_bool("SEISO_LLAMA_NO_PERF", True),
    }
    if n_gpu_layers != 0 and (
        not native_linux_nvidia or env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False)
    ):
        kwargs["op_offload"] = env_bool("SEISO_LLAMA_OP_OFFLOAD", True)
    if n_gpu_layers != 0 and _default_llama_flash_attn(model_path):
        kwargs["flash_attn"] = True
    if model_path:
        kwargs["_model_path"] = model_path
        kwargs["_native_linux_nvidia"] = native_linux_nvidia
    return _prot().clamp_llama_load_kwargs(kwargs)


def _llama_load_retryable(exc: BaseException) -> bool:
    """True when llama.cpp init failed due to VRAM pressure and a smaller offload may work."""
    msg = str(exc)
    if "Failed to load model from file" in msg or "Failed to create llama_context" in msg:
        return True
    try:
        return _prot().is_oom_error(exc)
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

    _prot().release_cached_memory(sync=True)
    _mp()._clear_optimal_layers_cache()
    _mp()._refresh_headroom_stats(force=True)

    est_mb = int(_prot().estimate_path_vram_mb(path))
    try:
        from seiso.hardware.vram_processes import warn_before_model_load

        warn_before_model_load(model_path=path, est_mb=est_mb)
    except Exception:
        pass

    kwargs = _mp().llama_load_kwargs(n_ctx, model_path=path)
    if batch_override is not None:
        override_batch, override_ubatch = batch_override
        clamped_batch, clamped_ubatch = _prot().clamp_llama_batch_pair(
            override_batch,
            override_ubatch,
            native_linux_nvidia=_mp()._native_linux_nvidia(),
            load_tier=load_tier,
        )
        kwargs["n_batch"] = min(clamped_batch, override_batch)
        kwargs["n_ubatch"] = min(clamped_ubatch, override_ubatch, kwargs["n_batch"])
    speed_extras = _mp()._llama_speed_extras(path)
    requested = env_int("SEISO_LLAMA_GPU_LAYERS", _mp()._default_llama_gpu_layers())
    if requested != 0 and not _mp()._llama_gpu_offload_ok():
        requested = 0

    free_mb = _prot().headroom_mb()
    n_gpu_layers = int(kwargs.get("n_gpu_layers") or 0)
    effective_n_ctx = int(kwargs.get("n_ctx") or n_ctx)
    ladder_batch, ladder_ubatch = _mp()._llama_batch_defaults(path)
    memory_profiles = _prot().llama_load_profile_ladder(
        model_path=path,
        n_ctx=effective_n_ctx,
        n_gpu_layers=n_gpu_layers,
        free_mb=free_mb,
        base_batch=int(kwargs.get("n_batch") or ladder_batch),
        base_ubatch=int(kwargs.get("n_ubatch") or ladder_ubatch),
        tier=load_tier,
    )
    full_gpu_profiles = [
        *memory_profiles,
        {"n_batch": 128, "n_ubatch": 128, "n_ctx": min(effective_n_ctx, 2048)},
    ]
    kv_options = _mp()._llama_kv_quant_options(path)
    fitted_layers = _mp().fit_llama_gpu_layers(path, requested, free_mb, n_ctx=effective_n_ctx)
    full_targets = _mp()._llama_full_gpu_targets(requested) if requested != 0 else []
    partial_targets = (
        _mp()._llama_layer_attempts(
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
        total_layers = _mp().gguf_total_layers(path)
        if layers > 0 and layers < total_layers:
            load_kwargs.pop("flash_attn", None)
            if _mp()._native_linux_nvidia():
                if not env_bool("SEISO_LLAMA_UNSAFE_PARTIAL_KQV", False):
                    load_kwargs["offload_kqv"] = False
                if not env_bool("SEISO_LLAMA_UNSAFE_OP_OFFLOAD", False):
                    load_kwargs.pop("op_offload", None)
        _mp()._refresh_headroom_stats(force=True)
        load_kwargs["_model_path"] = path
        load_kwargs = _prot().clamp_llama_load_kwargs(load_kwargs)
        load_kwargs.pop("_model_path", None)
        from seiso.inference.llama_vision import apply_llama_vision_load_kwargs

        load_kwargs = apply_llama_vision_load_kwargs(load_kwargs, path)
        try:
            llm = Llama(model_path=path, **load_kwargs)
            llm._seiso_n_gpu_layers = layers  # noqa: SLF001
            llm._seiso_load_tier = load_tier  # noqa: SLF001
            llm._seiso_n_batch = int(load_kwargs.get("n_batch") or 0)  # noqa: SLF001
            llm._seiso_n_ubatch = int(load_kwargs.get("n_ubatch") or 0)  # noqa: SLF001
            llm._seiso_n_ctx = int(load_kwargs.get("n_ctx") or effective_n_ctx)  # noqa: SLF001
            llm._seiso_model_path = path  # noqa: SLF001
            llm._seiso_load_headroom_mb = _prot().headroom_mb()  # noqa: SLF001
            if batch_override is not None:
                llm._seiso_last_safe_batch = int(load_kwargs.get("n_batch") or 0)  # noqa: SLF001
                llm._seiso_last_safe_ubatch = int(load_kwargs.get("n_ubatch") or 0)  # noqa: SLF001
            if layers > 0:
                total_layers = _mp().gguf_total_layers(path)
                if layers < total_layers:
                    logger.warning(
                        "Partial GPU offload for %s: %d/%d layers (~%.1f GB free) — "
                        "close other GPU apps for full GPU offload and ~3× faster generation",
                        Path(path).name,
                        layers,
                        total_layers,
                        _prot().headroom_mb() / 1024,
                    )
            if use_prompt_cache:
                attach_llama_prompt_cache(llm, model_path=path)
            return llm
        except Exception as exc:
            if not _llama_load_retryable(exc):
                raise
            last_exc = exc
            _prot().release_cached_memory(sync=True)
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

    free_gb = round(_prot().headroom_mb() / 1024, 1)
    need_gb = round(_prot().estimate_path_vram_mb(path) / 1024, 1)
    raise RuntimeError(
        f"Could not load model — needs ~{need_gb} GB GPU/RAM headroom but only ~{free_gb} GB is free. "
        "Close other GPU apps (browser, games, other llama.cpp sessions), unload the previous model, "
        "or pick a smaller quant."
    ) from last_exc
