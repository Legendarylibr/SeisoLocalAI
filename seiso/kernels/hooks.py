"""Patch model modules with Seiso fused kernels during training."""

from __future__ import annotations

import logging
from typing import Any

from seiso.kernels.triton_ops import estimate_vram_savings_pct, fused_rms_norm, is_triton_available

logger = logging.getLogger(__name__)

_PATCHED: set[int] = set()


def clear_kernel_patches() -> None:
    """Release patched-model tracking after training completes."""
    _PATCHED.clear()


def apply_training_kernels(model: Any, *, use_triton: bool = True) -> dict[str, Any]:
    """
    Replace compatible RMSNorm forwards with fused Triton path.
    Returns metadata about applied optimizations.
    """
    meta: dict[str, Any] = {
        "triton_available": is_triton_available(),
        "triton_applied": False,
        "modules_patched": 0,
    }

    if not use_triton or not is_triton_available():
        logger.info("Triton kernels skipped (unavailable or disabled)")
        return meta

    try:
        import torch
    except ImportError:
        return meta

    model_id = id(model)
    if model_id in _PATCHED:
        meta["triton_applied"] = True
        return meta

    patched = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls not in ("LlamaRMSNorm", "Qwen2RMSNorm", "GemmaRMSNorm", "RMSNorm"):
            continue
        if not hasattr(module, "weight"):
            continue

        orig_forward = module.forward
        weight = module.weight
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

        def make_forward(w, e, orig):
            def forward(hidden_states):
                if hidden_states.is_cuda:
                    return fused_rms_norm(hidden_states, w, eps=e)
                return orig(hidden_states)

            return forward

        module.forward = make_forward(weight, eps, orig_forward)
        patched += 1

    _PATCHED.add(model_id)
    meta["triton_applied"] = patched > 0
    meta["modules_patched"] = patched
    meta["estimated_vram_savings_pct"] = estimate_vram_savings_pct(True, False)
    logger.info("Patched %d RMSNorm modules with Triton kernels", patched)
    return meta
