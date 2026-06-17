"""Patch model modules with leak-safe fused kernels during training."""

from __future__ import annotations

import logging
import types
from typing import Any, Callable

from seiso.kernels.dispatch import estimate_vram_savings_pct, fused_rms_norm, fused_swiglu, kernel_metadata
from seiso.kernels.lifecycle import register_patch, restore_kernel_patches
from seiso.kernels.platform import detect_gpu

logger = logging.getLogger(__name__)

_RMSNORM_CLASSES = frozenset({"LlamaRMSNorm", "Qwen2RMSNorm", "GemmaRMSNorm", "RMSNorm"})
_MLP_CLASSES = frozenset(
    {
        "LlamaMLP",
        "MistralMLP",
        "Qwen2MLP",
        "Qwen3MLP",
        "Phi3MLP",
        "GemmaMLP",
        "Gemma2MLP",
        "MixtralMLP",
    }
)


def clear_kernel_patches(model: Any | None = None) -> None:
    """Restore original forwards and release patch registry."""
    restore_kernel_patches(model)


def _patch_forward(model: Any, module: Any, forward_fn: Callable) -> None:
    if hasattr(module, "_seiso_orig_forward"):
        return
    module._seiso_orig_forward = module.forward
    module.forward = types.MethodType(forward_fn, module)
    register_patch(model, module)


def _is_swiglu_mlp(module: Any) -> bool:
    if type(module).__name__ in _MLP_CLASSES:
        return True
    if not all(hasattr(module, a) for a in ("gate_proj", "up_proj", "down_proj")):
        return False
    cls = type(module).__name__
    if "Moe" in cls or "MoE" in cls or "Sparse" in cls:
        return False
    act = getattr(module, "act_fn", None)
    if act is None:
        return True
    act_name = type(act).__name__.lower()
    return "silu" in act_name or "swish" in act_name


def apply_training_kernels(
    model: Any,
    *,
    use_cuda: bool = True,
    use_triton: bool = True,
    patch_mlp: bool = True,
) -> dict[str, Any]:
    """
    Patch RMSNorm and SwiGLU MLP modules with fused GPU kernels.

    NVIDIA: native CUDA. AMD ROCm: Triton. Patches are always restored on cleanup.
    """
    platform = detect_gpu()
    meta = kernel_metadata()
    meta.update(
        {
            "cuda_applied": False,
            "triton_applied": False,
            "rmsnorm_patched": 0,
            "mlp_patched": 0,
            "modules_patched": 0,
            "fused_enabled": use_cuda or use_triton,
            "patch_mlp": patch_mlp,
        }
    )

    if not (use_cuda or use_triton):
        logger.info("Fused kernels disabled by config")
        return meta

    if platform.device_count == 0:
        logger.info("Fused kernels skipped (no GPU)")
        return meta

    if platform.vendor.value == "amd" and not use_triton:
        logger.info("AMD GPU detected — enable use_triton for fused kernels")
        return meta

    try:
        import torch  # noqa: F401
    except ImportError:
        return meta

    rms_patched = 0
    mlp_patched = 0
    backend = meta["kernel_backend"]

    for _name, module in model.named_modules():
        cls = type(module).__name__

        if cls in _RMSNORM_CLASSES and hasattr(module, "weight"):
            if hasattr(module, "_seiso_orig_forward"):
                continue
            eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

            def _rms_forward(self, hidden_states, _eps=eps):
                if hidden_states.is_cuda:
                    return fused_rms_norm(hidden_states, self.weight, eps=_eps)
                return self._seiso_orig_forward(hidden_states)

            _patch_forward(model, module, _rms_forward)
            rms_patched += 1

        if patch_mlp and _is_swiglu_mlp(module):
            if hasattr(module, "_seiso_orig_forward"):
                continue

            def _mlp_forward(self, hidden_states):
                if hidden_states.is_cuda:
                    gate = self.gate_proj(hidden_states)
                    up = self.up_proj(hidden_states)
                    return self.down_proj(fused_swiglu(gate, up))
                return self._seiso_orig_forward(hidden_states)

            _patch_forward(model, module, _mlp_forward)
            mlp_patched += 1

    total = rms_patched + mlp_patched
    meta["rmsnorm_patched"] = rms_patched
    meta["mlp_patched"] = mlp_patched
    meta["modules_patched"] = total
    meta["cuda_applied"] = backend == "cuda" and total > 0
    meta["triton_applied"] = backend == "triton" and total > 0
    meta["estimated_vram_savings_pct"] = estimate_vram_savings_pct(total > 0, False)

    if total:
        logger.info(
            "Fused kernels: %d RMSNorm + %d MLP | vendor=%s backend=%s device=%s",
            rms_patched,
            mlp_patched,
            platform.vendor.value,
            backend,
            platform.device_name,
        )
    return meta
