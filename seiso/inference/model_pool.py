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


def _llama_gpu_offload_ok() -> bool:
    """True when the installed llama-cpp-python can offload to GPU."""
    global _llama_offload_checked, _llama_offload_supported
    if _llama_offload_checked:
        return _llama_offload_supported
    _llama_offload_checked = True
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
                return _llama_offload_supported
    except Exception:
        pass
    return False


def _native_linux_nvidia() -> bool:
    """Bare-metal Linux with discrete NVIDIA — inference uses llama.cpp GGUF."""
    try:
        from seiso.platform import is_native_linux_nvidia

        return is_native_linux_nvidia()
    except ImportError:
        return False


def _llama_speed_scale_enabled() -> bool:
    # Upscaled batches OOM during prefill after weights land on GPU.
    if _native_linux_nvidia() and not env_bool(
        "SEISO_LLAMA_UNSAFE_SPEED_SCALE", False
    ):
        return False
    default = not _native_linux_nvidia()
    return env_bool("SEISO_LLAMA_SPEED_SCALE", default)


def _default_llama_flash_attn() -> bool:
    return env_bool("SEISO_LLAMA_FLASH_ATTN", True)


def _llama_batch_defaults() -> tuple[int, int]:
    """Speed-first llama.cpp prompt/decode batch defaults (tight-fit models clamp at load)."""
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
    if _native_linux_nvidia() and llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=headroom_mb,
        n_gpu_layers=-1 if requested == -1 else max(requested, 1),
        n_ctx=n_ctx,
    ):
        # Leave real VRAM slack for near-capacity models during first-message prefill.
        reserve_mb = max(2048, int(headroom_mb * 0.15))
        headroom_mb = max(0, headroom_mb - reserve_mb)

    total_layers = gguf_total_layers(model_path)

    def _fits(layers: int) -> bool:
        return llama_offload_fits_headroom(
            model_path,
            headroom_mb=headroom_mb,
            n_gpu_layers=layers,
            n_ctx=n_ctx,
            weight_mb=weight_mb,
            total_layers=total_layers,
        )

    if requested == -1 and _fits(-1):
        return -1

    if requested > 0:
        capped = min(requested, total_layers)
        if capped >= total_layers and _fits(-1):
            return capped
        if _fits(capped):
            return capped

    kv_reserve_mb = llama_kv_cache_reserve_mb(
        model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1 if requested == -1 else min(max(requested, 1), total_layers),
        total_layers=total_layers,
        weight_mb=weight_mb,
        free_mb=headroom_mb,
    )
    avail_mb = headroom_mb - kv_reserve_mb

    if avail_mb < 256:
        logger.warning(
            "VRAM too tight for GPU offload (~%.1f GB free) — falling back to CPU",
            headroom_mb / 1024,
        )
        return 0

    fraction = max(0.05, min(avail_mb / weight_mb, 1.0))
    partial = max(1, int(total_layers * fraction))
    if requested not in (-1, 0) and requested > 0:
        partial = min(partial, requested)

    while partial > 0 and not _fits(partial):
        partial -= 1

    if partial <= 0:
        logger.warning(
            "VRAM too tight for GPU offload (~%.1f GB free) — falling back to CPU",
            headroom_mb / 1024,
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

    total_layers = gguf_total_layers(model_path)

    if _apple_silicon_metal() and _mac_cpu_offload_enabled():
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
    high = min(total_layers - 1, fitted + 6)
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
        from seiso.inference.backends import gguf_uses_sliding_window_attention

        if gguf_uses_sliding_window_attention(model_path) and not env_bool(
            "SEISO_LLAMA_SWA_FULL", False
        ):
            extras["swa_full"] = False
    except Exception:
        pass
    return extras


def _llama_kv_quant_options(model_path: str) -> list[dict[str, Any]]:
    """KV-cache quant tiers to try after the unquantized cache fails."""
    if not env_bool("SEISO_LLAMA_KV_QUANT", True):
        return [{}]
    if _native_linux_nvidia() and not env_bool("SEISO_LLAMA_UNSAFE_KV_QUANT", False):
        return [{}]
    try:
        from llama_cpp import llama_cpp as lc

    except (ImportError, Exception):
        return [{}]

    options: list[dict[str, Any]] = [{}]
    _ = model_path
    options.append(
        {
            "type_k": lc.GGML_TYPE_Q8_0,
            "type_v": lc.GGML_TYPE_Q8_0,
        }
    )
    options.append(
        {
            "type_k": lc.GGML_TYPE_Q4_K,
            "type_v": lc.GGML_TYPE_Q4_K,
        }
    )

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
    _optimal_layers_cache.clear()


def _llama_gpu_layers_optimal(
    model_path: str, requested: int, *, n_ctx: int = 2048
) -> int:
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
        return True
    optimal = _llama_gpu_layers_optimal(model_path, requested, n_ctx=n_ctx)
    if optimal == -1:
        return False
    return cached_layers >= optimal


def llama_load_kwargs(n_ctx: int, *, model_path: str | None = None) -> dict[str, Any]:
    """Tuned llama.cpp defaults for faster preload/first token, overrideable by env."""
    from seiso.memory.protection import clamp_llama_load_kwargs

    n_threads = env_int("SEISO_LLAMA_THREADS", _default_llama_threads())
    n_gpu_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
    # Safety net: if the user or platform_profile set n_gpu_layers != 0 but the
    # installed llama-cpp-python wheel can't actually offload (e.g. CPU-only
    # wheel on an NVIDIA Linux box), force 0 to avoid a crash at Llama init.
    if n_gpu_layers != 0 and not _llama_gpu_offload_ok():
        logger.debug(
            "llama-cpp-python wheel lacks GPU offload support — forcing n_gpu_layers=0"
        )
        n_gpu_layers = 0

    batch_default, ubatch_default = _llama_batch_defaults()

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
    if n_gpu_layers != 0 and _default_llama_flash_attn():
        kwargs["flash_attn"] = True
    if model_path:
        kwargs["_model_path"] = model_path
    return clamp_llama_load_kwargs(kwargs)


def _llama_load_retryable(exc: BaseException) -> bool:
    """True when llama.cpp init failed due to VRAM pressure and a smaller offload may work."""
    msg = str(exc)
    if (
        "Failed to load model from file" in msg
        or "Failed to create llama_context" in msg
    ):
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
) -> Any:
    """Load a GGUF with VRAM-aware layer offload and clear OOM errors."""
    from seiso.memory.protection import LlamaLoadTier, llama_load_profile_ladder

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
    kv_options = _llama_kv_quant_options(path)
    fitted_layers = fit_llama_gpu_layers(
        path, requested, free_mb, n_ctx=effective_n_ctx
    )
    full_targets = _llama_full_gpu_targets(requested) if fitted_layers == -1 else []
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
    seen: set[tuple[int, tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]] = (
        set()
    )

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
            load_kwargs["offload_kqv"] = bool(
                load_kwargs.get("offload_kqv", layers != 0)
            )
        total_layers = gguf_total_layers(path)
        if layers > 0 and layers < total_layers:
            load_kwargs.pop("flash_attn", None)
            if _native_linux_nvidia():
                load_kwargs["offload_kqv"] = False
                load_kwargs["op_offload"] = False
        from seiso.memory.protection import clamp_llama_load_kwargs

        load_kwargs["_model_path"] = path
        load_kwargs = clamp_llama_load_kwargs(load_kwargs)
        load_kwargs.pop("_model_path", None)
        try:
            llm = Llama(model_path=path, **load_kwargs)
            llm._seiso_n_gpu_layers = layers  # noqa: SLF001
            llm._seiso_load_tier = load_tier  # noqa: SLF001
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
            return llm
        except Exception as exc:
            if not _llama_load_retryable(exc):
                raise
            last_exc = exc
            release_cached_memory(sync=True)
            if log_retry:
                logger.warning(
                    "llama.cpp load failed at n_gpu_layers=%s — retrying", layers
                )
            return None

    for layers in full_targets:
        for profile_idx, profile in enumerate(memory_profiles):
            for kv_idx, kv_quant in enumerate(kv_options):
                log_retry = (
                    profile_idx == len(memory_profiles) - 1
                    and kv_idx == len(kv_options) - 1
                )
                llm = _try_load(layers, profile, kv_quant, log_retry=log_retry)
                if llm is not None:
                    return llm

    partial_profiles = _llama_partial_memory_profiles(memory_profiles)
    partial_kqv_options = _llama_partial_kqv_options()
    for layer_idx, layers in enumerate(partial_targets):
        if layers in full_targets:
            continue
        for profile_idx, profile in enumerate(partial_profiles):
            for kqv_idx, kqv_option in enumerate(partial_kqv_options):
                # KV-quantized caches can crash llama.cpp on partial offload (Qwen/MoE).
                first_partial_attempt = (
                    layer_idx == 0 and profile_idx == 0 and kqv_idx == 0
                )
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

    def cancel_and_unload(self) -> None:
        """Stop lagging streams and release VRAM/RAM."""
        self.bump_generation()
        self.unload_all()
        clear_dflash_draft_cache()

    def would_switch_model(
        self, target_path: str, backend: str | BackendKind | None = None
    ) -> bool:
        """True when loading target_path would replace the active inference model."""
        status = self.status()
        if not status.get("active_model"):
            return False
        if backend is not None:
            raw = (
                backend.value
                if isinstance(backend, BackendKind)
                else str(backend).lower()
            )
            expected = _POOL_BACKEND_BY_API.get(raw, raw)
            if status.get("backend") != expected:
                return True
        active_path = status.get("path")
        if not active_path:
            return False
        return self.normalize_path(active_path) != self.normalize_path(target_path)

    def prepare_for_load(
        self,
        target_path: str | None = None,
        backend: str | BackendKind | None = None,
    ) -> bool:
        """Unload the active model when switching and refresh GPU memory stats."""
        should_unload = target_path is None or self.would_switch_model(
            target_path, backend
        )
        unloaded = False
        if should_unload and self.active_key:
            self.cancel_and_unload()
            unloaded = True
        if unloaded:
            _clear_optimal_layers_cache()
            _refresh_headroom_stats(force=True)
        return unloaded

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
        with self._lock:
            if self._active and self._active.key == key:
                needed_ctx = int(meta.get("n_ctx") or 0)
                cached_ctx = int(self._active.meta.get("n_ctx") or 0)
                if needed_ctx <= 0 or cached_ctx >= needed_ctx:
                    if backend != BackendKind.LLAMA:
                        return self._active.handle
                    cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
                    requested_layers = env_int(
                        "SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers()
                    )
                    if _llama_cache_is_optimal(
                        load_path,
                        cached_layers,
                        requested_layers,
                        n_ctx=needed_ctx or cached_ctx or 2048,
                    ):
                        return self._active.handle

            self.prepare_for_load(load_path, backend)
            from seiso.memory.protection import (
                ensure_load_fits,
                estimate_path_vram_mb,
                headroom_mb,
                release_cached_memory,
            )

            est_mb = int(estimate_path_vram_mb(load_path))
            if est_mb >= 8000 and headroom_mb() < int(est_mb * 0.98):
                if self._active:
                    self.cancel_and_unload()
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
                layer_meta["n_gpu_layers"] = int(
                    getattr(handle, "_seiso_n_gpu_layers", -1)
                )
            self._active = LoadedModel(
                key=key,
                backend=backend,
                handle=handle,
                meta={
                    "path": load_path,
                    "norm_path": norm,
                    **layer_meta,
                    **(meta or {}),
                },
            )
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
        requested_layers = env_int(
            "SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers()
        )
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
                if cached_ctx >= n_ctx and _llama_cache_is_optimal(
                    str(self._active.meta.get("path") or model_path),
                    cached_layers,
                    requested_layers,
                    n_ctx=n_ctx,
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

    def reload_llama(self, model_path: str, n_ctx: int, *, tier: str) -> Any:
        """Unload and reload llama.cpp at a lower memory tier after inference OOM."""
        self.cancel_and_unload()
        return self.get_llama(model_path, n_ctx=n_ctx, tier=tier)

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

    def get_torch(
        self, model_path: str, *, load_in_4bit: bool = True
    ) -> tuple[Any, Any]:
        def loader(path: str):
            return self._load_torch_pair(path, load_in_4bit=load_in_4bit)

        return self.switch(model_path, BackendKind.TORCH, loader)

    def _load_torch_pair(
        self, model_path: str, *, load_in_4bit: bool = True
    ) -> tuple[Any, Any]:
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
            from seiso.memory.protection import ensure_load_fits

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
            if not self._active:
                clear_dflash_draft_cache()
                return
            backend = self._active.backend
            key = self._active.key
            handle = self._active.handle
            self._active = None

        logger.info("Unloading model from VRAM: %s", key)

        if backend == BackendKind.LLAMA:
            llm = handle
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
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
                    model = (
                        handle[0] if isinstance(handle, tuple) and handle else handle
                    )
                    release_training_memory(model, sync=False)
            except Exception:
                pass
            del handle

        elif backend == BackendKind.MLX:
            del handle

        self._free_memory(sync=True)
        clear_dflash_draft_cache()

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


_dflash_draft_cache: dict[str, DflashDraftHandle] = {}
_dflash_draft_lock = threading.Lock()


def _load_dflash_llm(resolved_path: str, n_ctx: int) -> Any:
    return _load_llama_model(resolved_path, n_ctx)


def get_dflash_draft(model_path: str, *, n_ctx: int = 4096) -> DflashDraftHandle:
    """Return a cached, thread-safe llama.cpp handle for dflash/draft GGUF models."""
    from seiso.inference.backends import BACKEND_LLAMACPP, prepare_model_path

    resolved = prepare_model_path(model_path, BACKEND_LLAMACPP)
    norm = str(Path(resolved).resolve())
    with _dflash_draft_lock:
        cached = _dflash_draft_cache.get(norm)
        if cached is not None and cached.n_ctx >= n_ctx:
            return cached

    llm = _load_dflash_llm(resolved, n_ctx)

    with _dflash_draft_lock:
        cached = _dflash_draft_cache.get(norm)
        if cached is not None and cached.n_ctx >= n_ctx:
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            return cached
        if cached is not None:
            try:
                if hasattr(cached.llm, "close"):
                    cached.llm.close()
            except Exception:
                pass
        handle = DflashDraftHandle(llm, n_ctx=n_ctx)
        _dflash_draft_cache[norm] = handle
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
        "temperature": max(temperature, 0.0) if temperature > 0 else 0.0,
    }
    if temperature <= 0:
        gen_kwargs["temperature"] = 0.0

    if infer_lock is not None:
        with infer_lock:
            out = llm(current_text, **gen_kwargs)
    else:
        out = llm(current_text, **gen_kwargs)

    return out["choices"][0]["text"] if out.get("choices") else ""


def clear_dflash_draft_cache() -> None:
    """Release cached dflash draft models."""
    with _dflash_draft_lock:
        for handle in _dflash_draft_cache.values():
            try:
                if hasattr(handle.llm, "close"):
                    handle.llm.close()
            except Exception:
                pass
        _dflash_draft_cache.clear()


def get_model_pool() -> ModelPool:
    return ModelPool.get()
