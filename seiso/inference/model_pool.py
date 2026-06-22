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
        return max(2, min(cpus // 2, 8))
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


def llama_load_kwargs(n_ctx: int) -> dict[str, Any]:
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
    headroom = headroom_mb()
    batch_cap, ubatch_cap = _headroom_llama_batch_caps(headroom)
    n_batch = env_int("SEISO_LLAMA_BATCH", batch_cap)
    n_ubatch = env_int("SEISO_LLAMA_UBATCH", min(n_batch, ubatch_cap))
    n_batch = min(n_batch, batch_cap)
    n_ubatch = min(n_ubatch, ubatch_cap, n_batch)
    if n_gpu_layers != 0 and headroom > 0:
        if headroom < 4096:
            n_gpu_layers = min(n_gpu_layers if n_gpu_layers > 0 else 24, 16)
        elif headroom < 6144:
            n_gpu_layers = min(n_gpu_layers if n_gpu_layers > 0 else 32, 24)
        elif headroom < 10240 and n_gpu_layers == -1:
            n_gpu_layers = -1
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
    return clamp_llama_load_kwargs(kwargs)


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
                    return self._active.handle

            self.unload_all()
            from seiso.memory.protection import ensure_load_fits, release_cached_memory

            release_cached_memory(sync=True)
            logger.info("Loading model: %s (%s)", norm, backend.value)
            ensure_load_fits(load_path, mode="chat")
            try:
                handle = loader_fn(load_path)
            except Exception:
                self._free_memory()
                raise
            self._active = LoadedModel(
                key=key,
                backend=backend,
                handle=handle,
                meta={"path": load_path, "norm_path": norm, **(meta or {})},
            )
            return handle

    def get_llama(self, model_path: str, n_ctx: int = 4096) -> Any:
        def loader(path: str):
            from llama_cpp import Llama

            from seiso.inference.tuning import attach_llama_prompt_cache

            llm = Llama(model_path=path, **llama_load_kwargs(n_ctx))
            attach_llama_prompt_cache(llm)
            return llm

        norm = self.normalize_path(model_path)
        with self._lock:
            if (
                self._active
                and self._active.backend == BackendKind.LLAMA
                and self._active.meta.get("norm_path") == norm
            ):
                cached_ctx = int(self._active.meta.get("n_ctx") or 0)
                if cached_ctx >= n_ctx:
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
        from seiso.inference.tuning import maybe_apply_fused_kernels, prepare_torch_model
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
        maybe_apply_fused_kernels(model)
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
            from seiso.memory.protection import ensure_load_fits, release_cached_memory

            release_cached_memory(sync=True)
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
