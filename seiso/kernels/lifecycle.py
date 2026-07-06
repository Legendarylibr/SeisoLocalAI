"""Leak-safe kernel patch lifecycle and GPU memory release."""

from __future__ import annotations

import gc
from typing import Any

# model_id -> weakref to model (for validation) + list of modules patched
_PATCH_REGISTRY: dict[int, list[Any]] = {}


def register_patch(model: Any, module: Any) -> None:
    """Track a module whose forward was replaced."""
    model_id = id(model)
    if model_id not in _PATCH_REGISTRY:
        _PATCH_REGISTRY[model_id] = []
    _PATCH_REGISTRY[model_id].append(module)


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
                delattr(module, "_seiso_orig_forward")
                restored += 1
        # PeftModel / compile wrappers use a different id than register_patch.
        if _PATCH_REGISTRY:
            restored += restore_kernel_patches()
        return restored

    for modules in list(_PATCH_REGISTRY.values()):
        for module in modules:
            if hasattr(module, "_seiso_orig_forward"):
                module.forward = module._seiso_orig_forward  # type: ignore[method-assign]
                delattr(module, "_seiso_orig_forward")
                restored += 1
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
