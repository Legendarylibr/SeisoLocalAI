"""Leak-safe kernel patch lifecycle and GPU memory release."""

from __future__ import annotations

import gc
import weakref
from contextvars import ContextVar, Token
from typing import Any

# model_id -> (weakref to model for validation, list of modules patched).
# The weakref is validated on every keyed access so a garbage-collected model's
# id cannot be reused by a different live model and resurrect stale entries.
_PATCH_REGISTRY: dict[int, tuple[weakref.ReferenceType | None, list[Any]]] = {}
_ACTIVE_PATCH_SESSION: ContextVar[KernelPatchSession | None] = ContextVar(
    "seiso_kernel_patch_session", default=None
)


def _model_ref(model: Any) -> weakref.ReferenceType | None:
    """Weakref to ``model`` for registry validation; None when not weakref-able."""
    try:
        return weakref.ref(model)
    except TypeError:
        return None


def _entry_is_stale(entry: tuple[Any, list[Any]], model: Any | None = None) -> bool:
    """True when the entry's model was garbage-collected (or is another object)."""
    ref, _modules = entry
    if ref is None:
        return False  # Non-weakref-able model: id keying is all we have.
    live = ref()
    if live is None:
        return True
    return model is not None and live is not model


def register_patch(model: Any, module: Any) -> None:
    """Track a module whose forward was replaced."""
    model_id = id(model)
    entry = _PATCH_REGISTRY.get(model_id)
    if entry is not None and _entry_is_stale(entry, model):
        # Stale entry from a garbage-collected model whose id was reused:
        # release its modules before tracking the new model under this id.
        _restore_registry_key(model_id)
        entry = None
    if entry is None:
        entry = (_model_ref(model), [])
        _PATCH_REGISTRY[model_id] = entry
    entry[1].append(module)
    if session := _ACTIVE_PATCH_SESSION.get():
        session.record(module)


def _unregister_module(module: Any) -> None:
    for model_id, (ref, modules) in list(_PATCH_REGISTRY.items()):
        remaining = [m for m in modules if m is not module]
        if remaining:
            _PATCH_REGISTRY[model_id] = (ref, remaining)
        else:
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
        try:
            self.restore()
        finally:
            if self._token is not None:
                _ACTIVE_PATCH_SESSION.reset(self._token)
                self._token = None

    def record(self, module: Any) -> None:
        if module not in self._modules:
            self._modules.append(module)

    def commit(self) -> None:
        """Keep patches applied; end the active session without restoring.

        Used by long-lived inference loads where patched forwards must remain
        until an explicit ``restore_kernel_patches`` / unload path runs.
        """
        if self._token is not None:
            _ACTIVE_PATCH_SESSION.reset(self._token)
            self._token = None
        # Drop session-local tracking; modules stay in ``_PATCH_REGISTRY``.
        self._modules.clear()
        self._restored = True

    def restore(self) -> int:
        if self._restored:
            return 0
        restored = 0
        # LIFO restore; on failure keep remaining modules for retry (same contract
        # as ``_restore_registry_key``).
        pending = list(reversed(self._modules))
        for idx, module in enumerate(pending):
            try:
                if hasattr(module, "_seiso_orig_forward"):
                    module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                    restored += 1
                _clear_patch_markers(module)
                _unregister_module(module)
            except Exception:
                # ``pending`` is LIFO; convert remaining back to registration order.
                self._modules = list(reversed(pending[idx:]))
                raise
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
        entry = _PATCH_REGISTRY.get(model_id)
        if entry is not None and _entry_is_stale(entry, model):
            # id reused after the original model was garbage-collected — the
            # entry is not this model's; release it without touching this model.
            _restore_registry_key(model_id)
        restored += _restore_registry_key(model_id)
        # PeftModel / compile wrappers use a different id than register_patch —
        # restore only registry entries whose modules belong to this model tree.
        orphan_keys = [
            mid
            for mid, (_ref, modules) in list(_PATCH_REGISTRY.items())
            if any(_module_belongs_to_model(model, module) for module in modules)
        ]
        for mid in orphan_keys:
            restored += _restore_registry_key(mid)
        return restored

    for mid in list(_PATCH_REGISTRY.keys()):
        restored += _restore_registry_key(mid)
    _PATCH_REGISTRY.clear()
    return restored


def _module_belongs_to_model(model: Any, module: Any) -> bool:
    try:
        for child in model.modules():
            if child is module:
                return True
    except Exception:
        return False
    return False


def _restore_registry_key(model_id: int) -> int:
    """Restore modules for one registry key; pop only after success.

    If restoring module *k* fails, modules ``k…N`` stay registered so a later
    ``restore_kernel_patches`` can retry.
    """
    entry = _PATCH_REGISTRY.get(model_id)
    if entry is None:
        return 0
    ref, modules = entry
    restored = 0
    for idx, module in enumerate(modules):
        try:
            if hasattr(module, "_seiso_orig_forward"):
                module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                restored += 1
            _clear_patch_markers(module)
        except Exception:
            _PATCH_REGISTRY[model_id] = (ref, modules[idx:])
            raise
    _PATCH_REGISTRY.pop(model_id, None)
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
