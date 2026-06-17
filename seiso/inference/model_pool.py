"""VRAM-managed model pool — unloads previous model when switching."""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BackendKind(StrEnum):
    LLAMA = "llama"
    MLX = "mlx"
    TORCH = "torch"


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
    _lock = threading.Lock()

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

    def switch(self, model_path: str, backend: BackendKind, loader_fn) -> Any:
        """Load model_path, unloading any previously active model first."""
        norm = self.normalize_path(model_path)
        key = f"{backend.value}:{norm}"
        if self._active and self._active.key == key:
            return self._active.handle

        self.unload_all()
        logger.info("Loading model: %s (%s)", norm, backend.value)
        handle = loader_fn(norm)
        self._active = LoadedModel(key=key, backend=backend, handle=handle, meta={"path": norm})
        return handle

    def get_llama(self, model_path: str, n_ctx: int = 4096) -> Any:
        def loader(path: str):
            from llama_cpp import Llama

            return Llama(model_path=path, n_ctx=n_ctx, verbose=False)

        return self.switch(model_path, BackendKind.LLAMA, loader)

    def get_mlx(self, model_path: str) -> tuple[Any, Any]:
        def loader(path: str):
            from seiso.models.loader import LoadOptions, ModelKind
            from seiso.models.mlx_loader import load_mlx

            return load_mlx(LoadOptions(model_id=path, kind=ModelKind.TEXT))

        return self.switch(model_path, BackendKind.MLX, loader)

    def get_torch(self, model_path: str, *, load_in_4bit: bool = True) -> tuple[Any, Any]:
        def loader(path: str):
            from seiso.models.loader import LoadOptions, ModelKind, load_model

            return load_model(
                LoadOptions(
                    model_id=path,
                    kind=ModelKind.TEXT,
                    load_in_4bit=load_in_4bit,
                    device_map="auto",
                )
            )

        return self.switch(model_path, BackendKind.TORCH, loader)

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

        elif backend in (BackendKind.MLX, BackendKind.TORCH):
            if isinstance(handle, tuple):
                del handle
            else:
                del handle

        self._free_memory()

    def _free_memory(self) -> None:
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def status(self) -> dict:
        return {
            "active_model": self._active.key if self._active else None,
            "backend": self._active.backend.value if self._active else None,
            "path": self._active.meta.get("path") if self._active else None,
        }


def get_model_pool() -> ModelPool:
    return ModelPool.get()
