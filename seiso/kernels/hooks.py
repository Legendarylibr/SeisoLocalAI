"""Patch model modules with leak-safe fused kernels during training."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from typing import Any

from seiso.kernels.dispatch import (
    active_backend,
    estimate_vram_savings_pct,
    fused_lora_delta,
    fused_rms_norm,
    fused_swiglu,
    kernel_metadata,
)
from seiso.kernels.lifecycle import register_patch, restore_kernel_patches
from seiso.kernels.memory_mode import apply_low_vram_kernel_tuning, kernel_low_vram_enabled
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
    low_vram: bool | None = None,
) -> dict[str, Any]:
    """
    Patch RMSNorm and SwiGLU MLP modules with fused GPU kernels.

    NVIDIA: native CUDA. AMD ROCm: Triton. Patches are always restored on cleanup.
    """
    if low_vram is None:
        low_vram = kernel_low_vram_enabled()
    if low_vram:
        apply_low_vram_kernel_tuning()

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
            "kernel_low_vram": low_vram,
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
    meta["estimated_vram_savings_pct"] = estimate_vram_savings_pct(
        total > 0, False, low_vram=low_vram
    )

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


def _is_peft_lora_linear(module: Any) -> bool:
    return (
        hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "base_layer")
        and hasattr(module, "scaling")
    )


def apply_fused_lora_kernels(
    model: Any, *, max_rank: int = 64, low_vram: bool | None = None
) -> dict[str, Any]:
    """
    Patch PEFT LoRA linear layers with fused CUDA low-rank delta kernels.

    Active on NVIDIA CUDA and WSL2 paths when rank <= max_rank.
    In low-VRAM mode, writes the LoRA delta in-place into the base output.
    """
    if low_vram is None:
        low_vram = kernel_low_vram_enabled()

    meta = {
        "fused_lora_enabled": False,
        "lora_patched": 0,
        "lora_skipped": 0,
        "kernel_low_vram": low_vram,
    }
    platform = detect_gpu()
    if not platform.uses_optimized_cuda_kernels or active_backend() != "cuda":
        return meta

    try:
        import torch.nn.functional as F
    except ImportError:
        return meta

    patched = 0
    skipped = 0

    for _name, module in model.named_modules():
        if not _is_peft_lora_linear(module):
            continue
        if getattr(module, "use_dora", False):
            skipped += 1
            continue
        if hasattr(module, "_seiso_orig_forward"):
            continue

        def _fused_lora_forward(self, x, *args, **kwargs):
            if self.disable_adapters:
                return self.base_layer(x, *args, **kwargs)

            result = self.base_layer(x, *args, **kwargs)
            if not self.active_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                lora_a = self.lora_A[active_adapter]
                lora_b = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                rank = lora_a.weight.size(0)

                x_mod = x
                if hasattr(self, "_cast_input_dtype"):
                    x_mod = self._cast_input_dtype(x_mod, lora_a.weight.dtype)
                if dropout > 0 and self.training:
                    x_mod = F.dropout(x_mod, p=dropout)

                if (
                    x_mod.is_cuda
                    and rank <= max_rank
                    and x_mod.dim() >= 2
                    and active_backend() == "cuda"
                ):
                    flat_x = x_mod.reshape(-1, x_mod.shape[-1])
                    if low_vram:
                        flat_out = result.reshape(-1, result.shape[-1])
                        fused_lora_delta(
                            flat_x,
                            lora_a.weight,
                            lora_b.weight,
                            base=flat_out,
                            scale=scaling,
                            inplace=True,
                        )
                    else:
                        delta = fused_lora_delta(
                            flat_x, lora_a.weight, lora_b.weight, scale=scaling
                        )
                        delta = delta.reshape(*x_mod.shape[:-1], delta.shape[-1])
                        result = result + delta.to(result.dtype)
                else:
                    result = result + lora_b(lora_a(x_mod)) * scaling

            return result

        _patch_forward(model, module, _fused_lora_forward)
        patched += 1

    meta["fused_lora_enabled"] = patched > 0
    meta["lora_patched"] = patched
    meta["lora_skipped"] = skipped
    if patched:
        target = "WSL2 CUDA" if platform.is_wsl2 else "CUDA"
        logger.info("Fused LoRA: %d layers patched (%s path)", patched, target)
    return meta
