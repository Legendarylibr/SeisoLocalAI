"""Leak-safe kernel patch lifecycle and GPU memory release."""

from __future__ import annotations

import gc
from contextvars import ContextVar, Token
from typing import Any

# model_id -> weakref to model (for validation) + list of modules patched
_PATCH_REGISTRY: dict[int, list[Any]] = {}
_ACTIVE_PATCH_SESSION: ContextVar[KernelPatchSession | None] = ContextVar(
    "seiso_kernel_patch_session", default=None
)


def register_patch(model: Any, module: Any) -> None:
    """Track a module whose forward was replaced."""
    model_id = id(model)
    if model_id not in _PATCH_REGISTRY:
        _PATCH_REGISTRY[model_id] = []
    _PATCH_REGISTRY[model_id].append(module)
    if session := _ACTIVE_PATCH_SESSION.get():
        session.record(module)


def _unregister_module(module: Any) -> None:
    for model_id, modules in list(_PATCH_REGISTRY.items()):
        _PATCH_REGISTRY[model_id] = [m for m in modules if m is not module]
        if not _PATCH_REGISTRY[model_id]:
            _PATCH_REGISTRY.pop(model_id, None)


class KernelPatchSession:
    """Transaction-like scope that restores patched forwards in LIFO order."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self._modules: list[Any] = []
        self._token: Token[KernelPatchSession | None] | None = None
        self._restored = False

    def __enter__(self) -> KernelPatchSession:
        self._token = _ACTIVE_PATCH_SESSION.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.restore()
        if self._token is not None:
            _ACTIVE_PATCH_SESSION.reset(self._token)
            self._token = None

    def record(self, module: Any) -> None:
        if module not in self._modules:
            self._modules.append(module)

    def restore(self) -> int:
        if self._restored:
            return 0
        restored = 0
        for module in reversed(self._modules):
            if hasattr(module, "_seiso_orig_forward"):
                module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                restored += 1
            _clear_patch_markers(module)
            _unregister_module(module)
        self._modules.clear()
        self._restored = True
        return restored


def _clear_patch_markers(module: Any) -> None:
    for attr in (
        "_seiso_orig_forward",
        "_seiso_residual_norm_forward",
        "_seiso_residual_decoder_forward",
        "_seiso_residual",
    ):
        if hasattr(module, attr):
            delattr(module, attr)


def restore_kernel_patches(model: Any | None = None) -> int:
    """
    Restore original ``forward`` methods and drop patch references.

    Returns the number of modules restored. Call after every training run so
    patched closures never pin model graphs in memory.

    When ``model`` is given, restores that id and any remaining registry entries
    (LoRA/PeftModel wrap changes ``id(model)`` after patches were registered
    on the base module).
    """
    restored = 0

    if model is not None:
        model_id = id(model)
        modules = _PATCH_REGISTRY.pop(model_id, [])
        for module in modules:
            if hasattr(module, "_seiso_orig_forward"):
                module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                restored += 1
            _clear_patch_markers(module)
        # PeftModel / compile wrappers use a different id than register_patch.
        if _PATCH_REGISTRY:
            restored += restore_kernel_patches()
        return restored

    for modules in list(_PATCH_REGISTRY.values()):
        for module in modules:
            if hasattr(module, "_seiso_orig_forward"):
                module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                restored += 1
            _clear_patch_markers(module)
    _PATCH_REGISTRY.clear()
    return restored


def release_training_memory(model: Any | None = None, *, sync: bool = True) -> None:
    """
    Full post-training cleanup: restore patches, delete model ref, GC, empty cache.

    Designed to avoid VRAM leaks from monkey-patched forwards and stale tensors.
    """
    restore_kernel_patches(model)

    if model is not None:
        del model

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
