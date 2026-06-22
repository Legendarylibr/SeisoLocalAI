"""VRAM-managed model pool — unloads previous model when switching."""

from __future__ import annotations

import logging
import os
import platform
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seiso.compat import StrEnum
from seiso.env import env_bool, env_int

logger = logging.getLogger(__name__)


def _default_llama_threads() -> int:
    cpus = os.cpu_count() or 4
    if _default_llama_gpu_layers() != 0:
        return max(2, min(cpus // 2, 12))
    return max(2, min(cpus - 2 if cpus > 4 else cpus, 12))


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


def _default_llama_gpu_layers() -> int:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        return -1
    if _cuda_available() or _nvidia_hardware_visible():
        # Only request full GPU offload when the installed llama-cpp-python
        # wheel actually supports it. A CPU-only wheel on an NVIDIA box will
        # crash if n_gpu_layers != 0.
        if _llama_gpu_offload_ok():
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
            getattr(getattr(llama_cpp, "llama_cpp", None), "llama_supports_gpu_offload", None),
        ):
            if callable(candidate):
                _llama_offload_supported = bool(candidate())
                return _llama_offload_supported
    except Exception:
        pass
    return False


def _reset_llama_offload_cache() -> None:
    """Clear the GPU offload probe cache (for tests / post-install)."""
    global _llama_offload_checked, _llama_offload_supported
    _llama_offload_checked = False
    _llama_offload_supported = False


def _default_llama_batch() -> int:
    # Conservative default — clamp_llama_load_kwargs scales up on roomy hardware.
    return 512


def _default_llama_ubatch(n_batch: int) -> int:
    # Smaller micro-batches cap peak VRAM during prompt prefill.
    return min(n_batch, 256 if _default_llama_gpu_layers() != 0 else 512)


def _headroom_llama_batch_caps(headroom: int) -> tuple[int, int]:
    """Return (n_batch, n_ubatch) ceilings from free VRAM/RAM."""
    if headroom < 4096:
        return 256, 128
    if headroom < 8192:
        return 512, 256
    if headroom < 16384:
        return 1024, 512
    return 2048, 512


def _speed_llama_batch_defaults(headroom: int) -> tuple[int, int] | None:
    """Workstation-tier defaults when env vars are unset."""
    if "SEISO_LLAMA_BATCH" in os.environ or "SEISO_LLAMA_UBATCH" in os.environ:
        return None
    try:
        from seiso.hardware.profile import hardware_profile
        from seiso.hardware.tiers import HardwareTier, classify_tier

        tier = classify_tier(hardware_profile())
    except Exception:
        return None
    if tier == HardwareTier.WORKSTATION and headroom >= 8192:
        return 2048, 512
    if tier == HardwareTier.CAPABLE and headroom >= 8192:
        return 1024, 512
    return None


def fit_llama_gpu_layers(model_path: str, requested: int, headroom_mb: int) -> int:
    """Estimate a GPU layer count that fits current free VRAM (conservative)."""
    if requested == 0 or headroom_mb <= 0 or not _llama_gpu_offload_ok():
        return 0

    from seiso.inference.backends import gguf_block_count
    from seiso.memory.protection import estimate_path_vram_mb

    weight_mb = max(int(estimate_path_vram_mb(model_path) - 512), 256)
    total_layers = gguf_block_count(model_path) or 64
    kv_reserve_mb = max(768, min(int(headroom_mb * 0.15), 2048))
    avail_mb = headroom_mb - kv_reserve_mb

    if avail_mb >= int(weight_mb * 0.92):
        if requested == -1:
            return -1
        return max(0, min(requested, total_layers))

    if avail_mb < 384:
        logger.warning(
            "VRAM too tight for GPU offload (~%.1f GB free) — running GGUF on CPU",
            headroom_mb / 1024,
        )
        return 0

    fraction = max(0.05, min(avail_mb / weight_mb, 1.0))
    partial = max(1, int(total_layers * fraction))
    if requested not in (-1, 0) and requested > 0:
        partial = min(partial, requested)
    return partial


def _llama_layer_attempts(model_path: str, requested: int, free_mb: int) -> list[int]:
    """Try full GPU first, then fall back to smaller partial layer counts."""
    if requested == 0 or not _llama_gpu_offload_ok():
        return [0]

    attempts: list[int] = []
    if requested == -1:
        attempts.append(-1)
    elif requested > 0:
        attempts.append(requested)

    fitted = fit_llama_gpu_layers(model_path, requested, free_mb)
    for layers in (fitted, max(fitted // 2, 8) if fitted > 8 else None, 0):
        if layers is not None and layers not in attempts:
            attempts.append(layers)
    return attempts


def _llama_speed_extras(model_path: str) -> dict[str, Any]:
    """Model-specific llama.cpp knobs for throughput and VRAM headroom."""
    extras: dict[str, Any] = {}
    try:
        from seiso.inference.backends import gguf_architecture
        from seiso.memory.protection import estimate_path_vram_mb

        arch = (gguf_architecture(model_path) or "").lower()
        weight_mb = int(estimate_path_vram_mb(model_path))
    except Exception:
        return extras

    if "qwen" in arch and not env_bool("SEISO_LLAMA_SWA_FULL", False):
        extras["swa_full"] = False

    if env_bool("SEISO_LLAMA_KV_QUANT", True) and weight_mb >= 12000:
        try:
            from llama_cpp import llama_cpp as lc

            extras["type_k"] = lc.GGML_TYPE_Q8_0
            extras["type_v"] = lc.GGML_TYPE_Q8_0
        except ImportError:
            pass
    return extras


def _llama_gpu_layers_optimal(model_path: str, requested: int) -> int:
    """Best layer count for current free VRAM — used to decide cache reload."""
    from seiso.memory.protection import headroom_mb, release_cached_memory

    release_cached_memory(sync=False)
    try:
        from seiso.hardware.profile import hardware_profile

        hardware_profile(force_refresh=True)
    except ImportError:
        pass
    return fit_llama_gpu_layers(model_path, requested, headroom_mb())


def _llama_cache_is_optimal(model_path: str, cached_layers: int, requested: int) -> bool:
    """True when a cached llama handle already uses the best GPU offload available."""
    if requested == 0:
        return cached_layers == 0
    if cached_layers == -1:
        return True
    optimal = _llama_gpu_layers_optimal(model_path, requested)
    if optimal == -1:
        return False
    return cached_layers >= optimal


def llama_load_kwargs(n_ctx: int, *, model_path: str | None = None) -> dict[str, Any]:
    """Tuned llama.cpp defaults for faster preload/first token, overrideable by env."""
    from seiso.memory.protection import clamp_llama_load_kwargs, headroom_mb

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
    from seiso.memory.protection import llama_batch_headroom_mb

    free_mb = headroom_mb()
    layer_hint = n_gpu_layers
    if n_gpu_layers != 0 and model_path:
        from seiso.inference.backends import gguf_block_count

        layer_hint = (
            n_gpu_layers
            if n_gpu_layers > 0
            else (gguf_block_count(model_path) or 64)
        )
    batch_headroom = llama_batch_headroom_mb(
        free_mb, model_path=model_path, n_gpu_layers=layer_hint
    )
    batch_cap, ubatch_cap = _headroom_llama_batch_caps(batch_headroom)
    speed_defaults = _speed_llama_batch_defaults(batch_headroom)
    if speed_defaults is not None:
        batch_cap, ubatch_cap = speed_defaults
    n_batch = env_int("SEISO_LLAMA_BATCH", batch_cap)
    n_ubatch = env_int("SEISO_LLAMA_UBATCH", min(n_batch, ubatch_cap))
    n_batch = min(n_batch, batch_cap)
    n_ubatch = min(n_ubatch, ubatch_cap, n_batch)
    kwargs: dict[str, Any] = {
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "n_threads_batch": env_int("SEISO_LLAMA_THREADS_BATCH", n_threads),
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
    if n_gpu_layers != 0 and env_bool("SEISO_LLAMA_FLASH_ATTN", True):
        kwargs["flash_attn"] = True
    if model_path:
        kwargs["_model_path"] = model_path
    return clamp_llama_load_kwargs(kwargs)


def _llama_load_retryable(exc: ValueError) -> bool:
    """True when llama.cpp init failed due to VRAM pressure and a smaller offload may work."""
    msg = str(exc)
    return (
        "Failed to load model from file" in msg
        or "Failed to create llama_context" in msg
    )


def _load_llama_model(path: str, n_ctx: int) -> Any:
    """Load a GGUF with VRAM-aware layer offload and clear OOM errors."""
    from llama_cpp import Llama

    from seiso.inference.tuning import attach_llama_prompt_cache
    from seiso.memory.protection import estimate_path_vram_mb, headroom_mb, release_cached_memory

    release_cached_memory(sync=True)
    try:
        from seiso.hardware.profile import hardware_profile

        hardware_profile(force_refresh=True)
    except ImportError:
        pass

    kwargs = llama_load_kwargs(n_ctx, model_path=path)
    kwargs.update(_llama_speed_extras(path))
    requested = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
    if requested != 0 and not _llama_gpu_offload_ok():
        requested = 0
    attempts = _llama_layer_attempts(path, requested, headroom_mb())

    last_exc: Exception | None = None
    seen: set[int] = set()
    for attempt in attempts:
        if attempt in seen:
            continue
        seen.add(attempt)
        load_kwargs = dict(kwargs)
        load_kwargs["n_gpu_layers"] = attempt
        load_kwargs["offload_kqv"] = attempt != 0
        load_kwargs.pop("_model_path", None)
        try:
            llm = Llama(model_path=path, **load_kwargs)
            llm._seiso_n_gpu_layers = attempt  # noqa: SLF001
            if attempt > 0:
                from seiso.inference.backends import gguf_block_count

                total_layers = gguf_block_count(path) or 64
                if attempt < total_layers:
                    logger.warning(
                        "Partial GPU offload for %s: %d/%d layers (~%.1f GB free) — "
                        "close other GPU apps for ~3× faster generation",
                        Path(path).name,
                        attempt,
                        total_layers,
                        headroom_mb() / 1024,
                    )
            attach_llama_prompt_cache(llm)
            return llm
        except ValueError as exc:
            if not _llama_load_retryable(exc):
                raise
            last_exc = exc
            release_cached_memory(sync=True)
            logger.warning("llama.cpp load failed at n_gpu_layers=%s — retrying", attempt)

    free_gb = round(headroom_mb() / 1024, 1)
    need_gb = round(estimate_path_vram_mb(path) / 1024, 1)
    raise RuntimeError(
        f"Could not load model — needs ~{need_gb} GB VRAM but only ~{free_gb} GB is free. "
        "Close other GPU apps (browser, games), unload the previous model, or pick a smaller quant."
    ) from last_exc


_POOL_BACKEND_BY_API: dict[str, str] = {
    "llamacpp": "llamacpp",
    "llama": "llamacpp",
    "mlx": "mlx",
    "torch": "torch",
}


class BackendKind(StrEnum):
    LLAMACPP = "llamacpp"
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

    def would_switch_model(self, target_path: str, backend: str | BackendKind | None = None) -> bool:
        """True when loading target_path would replace the active inference model."""
        status = self.status()
        if not status.get("active_model"):
            return False
        if backend is not None:
            raw = backend.value if isinstance(backend, BackendKind) else str(backend).lower()
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
        should_unload = target_path is None or self.would_switch_model(target_path, backend)
        unloaded = False
        if should_unload and self.active_key:
            self.cancel_and_unload()
            unloaded = True
        else:
            self._free_memory()
        try:
            from seiso.hardware.profile import hardware_profile

            hardware_profile(force_refresh=True)
        except ImportError:
            pass
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
        if raw.exists():
            load_path = str(raw.absolute())
        else:
            load_path = str(model_path)
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
                    if _llama_cache_is_optimal(load_path, cached_layers, requested_layers):
                        return self._active.handle

            self.prepare_for_load(load_path, backend)
            from seiso.memory.protection import ensure_load_fits, estimate_path_vram_mb, headroom_mb

            if int(estimate_path_vram_mb(load_path)) >= 12000 and headroom_mb() < int(
                estimate_path_vram_mb(load_path) * 0.95
            ):
                if self._active:
                    self.cancel_and_unload()
                self._free_memory()
                try:
                    from seiso.hardware.profile import hardware_profile

                    hardware_profile(force_refresh=True)
                except ImportError:
                    pass

            logger.info("Loading model: %s (%s)", norm, backend.value)
            ensure_load_fits(load_path, mode="chat")
            try:
                handle = loader_fn(load_path)
            except Exception:
                self._free_memory()
                raise
            layer_meta: dict[str, Any] = {}
            if backend == BackendKind.LLAMA:
                layer_meta["n_gpu_layers"] = int(getattr(handle, "_seiso_n_gpu_layers", -1))
            self._active = LoadedModel(
                key=key,
                backend=backend,
                handle=handle,
                meta={"path": load_path, "norm_path": norm, **layer_meta, **(meta or {})},
            )
            return handle

    def get_llama(self, model_path: str, n_ctx: int = 4096) -> Any:
        def loader(path: str):
            return _load_llama_model(path, n_ctx)

        norm = self.normalize_path(model_path)
        requested_layers = env_int("SEISO_LLAMA_GPU_LAYERS", _default_llama_gpu_layers())
        with self._lock:
            if (
                self._active
                and self._active.backend == BackendKind.LLAMA
                and self._active.meta.get("norm_path") == norm
            ):
                cached_ctx = int(self._active.meta.get("n_ctx") or 0)
                cached_layers = int(self._active.meta.get("n_gpu_layers", -1))
                if cached_ctx >= n_ctx and _llama_cache_is_optimal(
                    str(self._active.meta.get("path") or model_path),
                    cached_layers,
                    requested_layers,
                ):
                    return self._active.handle

        key = f"llama:{norm}"
        return self.switch(
            model_path, BackendKind.LLAMA, loader, cache_key=key, meta={"n_ctx": n_ctx}
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
        """Load target + draft models for speculative decoding."""
        from seiso.inference.speculative import TorchSpeculativeBundle

        target_norm = self.normalize_path(target_path)
        draft_norm = self.normalize_path(draft_path)
        key = f"spec:{target_norm}:{draft_norm}"

        def loader(_path: str) -> TorchSpeculativeBundle:
            from seiso.memory.protection import ensure_load_fits

            ensure_load_fits(target_path, mode="chat")
            ensure_load_fits(draft_path, mode="chat")
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

    def unload_all(self) -> None:
        """Release all loaded models and clear GPU memory."""
        with self._lock:
            if not self._active:
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
                pass
            del handle

        elif backend == BackendKind.MLX:
            del handle

        self._free_memory()

    def _free_memory(self) -> None:
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=True)

    def status(self) -> dict:
        with self._lock:
            active = self._active
            return {
                "active_model": active.key if active else None,
                "backend": active.backend.value if active else None,
                "path": active.meta.get("path") if active else None,
                "draft_path": active.meta.get("draft_path") if active else None,
            }


def get_model_pool() -> ModelPool:
    return ModelPool.get()
