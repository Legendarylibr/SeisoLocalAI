"""Workarounds for torch.compile + gradient-checkpointing incompatibilities.

PyTorch activation checkpointing assumes forward and recompute run the same
compiled graphs. With ``automatic_dynamic_shapes``, Dynamo's LRU cache can pick
a different graph on recompute and raise ``CheckpointError``. See
https://github.com/pytorch/pytorch/issues/166926
"""

from __future__ import annotations

import logging
from typing import Any

from seiso.env import env_bool

logger = logging.getLogger(__name__)

_LRU_CACHE_CONFIGURED = False


def needs_compile_checkpoint_workaround(
    *, torch_compile: bool, gradient_checkpointing: bool
) -> bool:
    """Return True when compile + activation checkpointing need Dynamo tweaks."""
    return bool(torch_compile and gradient_checkpointing)


def configure_compile_checkpoint_compat(
    *,
    torch_compile: bool,
    gradient_checkpointing: bool,
    enabled: bool | None = None,
) -> bool:
    """Disable Dynamo LRU cache reordering when compile and GC are both enabled.

    Returns True when the workaround was applied.
    """
    global _LRU_CACHE_CONFIGURED
    if _LRU_CACHE_CONFIGURED:
        return True
    if not needs_compile_checkpoint_workaround(
        torch_compile=torch_compile, gradient_checkpointing=gradient_checkpointing
    ):
        return False
    if enabled is None:
        enabled = env_bool("SEISO_TORCH_COMPILE_CHECKPOINT_FIX", True)
    if not enabled:
        logger.warning(
            "torch_compile with gradient_checkpointing is enabled but "
            "SEISO_TORCH_COMPILE_CHECKPOINT_FIX=0 — CheckpointError risk remains"
        )
        return False

    try:
        import torch

        set_lru_cache = getattr(
            getattr(getattr(torch, "_C", None), "_dynamo", None),
            "eval_frame",
            None,
        )
        set_lru_cache = getattr(set_lru_cache, "_set_lru_cache", None)
        if set_lru_cache is None:
            logger.warning(
                "torch._C._dynamo.eval_frame._set_lru_cache unavailable — "
                "compile+gradient_checkpointing may raise CheckpointError"
            )
            return False
        set_lru_cache(False)
    except Exception as exc:
        logger.warning(
            "Failed to disable Dynamo LRU cache for compile+checkpointing: %s",
            exc,
        )
        return False

    _LRU_CACHE_CONFIGURED = True
    logger.info(
        "Applied torch.compile + gradient_checkpointing workaround "
        "(disabled Dynamo LRU cache reordering)"
    )
    return True


def _mark_dynamic_training_tensor(tensor: Any) -> None:
    try:
        import torch
        import torch._dynamo as dynamo
    except ImportError:
        return
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
        return
    dynamo.mark_dynamic(tensor, 0)
    dynamo.mark_dynamic(tensor, 1)


def _mark_dynamic_batch_tensors(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    for value in args:
        if isinstance(value, dict):
            for inner in value.values():
                _mark_dynamic_training_tensor(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                _mark_dynamic_training_tensor(inner)
        else:
            _mark_dynamic_training_tensor(value)
    for value in kwargs.values():
        if isinstance(value, dict):
            for inner in value.values():
                _mark_dynamic_training_tensor(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                _mark_dynamic_training_tensor(inner)
        else:
            _mark_dynamic_training_tensor(value)


def wrap_model_forward_for_dynamic_shapes(model: Any) -> Any:
    """Mark batch/sequence dims dynamic before each forward (compile+GC helper)."""
    if getattr(model, "_seiso_dynamic_shape_forward", False):
        return model

    original_forward = model.forward

    def forward_with_dynamic_shapes(*args: Any, **kwargs: Any) -> Any:
        _mark_dynamic_batch_tensors(args, kwargs)
        return original_forward(*args, **kwargs)

    model.forward = forward_with_dynamic_shapes  # type: ignore[method-assign]
    model._seiso_dynamic_shape_forward = True
    return model


def apply_compile_checkpoint_workarounds(
    model: Any,
    *,
    torch_compile: bool,
    gradient_checkpointing: bool,
) -> Any:
    """Apply process-wide and per-model fixes for compile + checkpointing."""
    if not needs_compile_checkpoint_workaround(
        torch_compile=torch_compile, gradient_checkpointing=gradient_checkpointing
    ):
        return model
    configure_compile_checkpoint_compat(
        torch_compile=torch_compile,
        gradient_checkpointing=gradient_checkpointing,
    )
    if env_bool("SEISO_TORCH_COMPILE_MARK_DYNAMIC", True):
        model = wrap_model_forward_for_dynamic_shapes(model)
        logger.info(
            "Marked training batch/sequence dims dynamic for torch.compile "
            "(disable with SEISO_TORCH_COMPILE_MARK_DYNAMIC=0)"
        )
    return model
