"""Unified fused-op dispatch — NVIDIA CUDA, AMD Triton, PyTorch fallback."""

from __future__ import annotations

import logging
from functools import lru_cache

from seiso.kernels.fallback_ops import pytorch_rms_norm as _pytorch_rms_norm
from seiso.kernels.platform import GpuPlatform, detect_gpu

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def active_backend() -> str:
    """Resolved kernel backend for this process."""
    platform = detect_gpu()
    backend = platform.preferred_kernel_backend

    if backend == "cuda":
        from seiso.kernels.cuda_ops import is_cuda_available

        if is_cuda_available():
            return "cuda"
        if platform.supports_triton:
            return "triton"
        return "pytorch"

    if backend == "triton":
        from seiso.kernels.triton_ops import is_triton_available

        if is_triton_available():
            return "triton"
        return "pytorch"

    return "pytorch"


def fused_rms_norm(x, weight, eps: float = 1e-6, residual=None):
    """``rms_norm(x + residual) * weight`` via best available backend."""
    import torch

    if not getattr(x, "is_cuda", False):
        return _pytorch_rms_norm(x, weight, eps, residual)

    backend = active_backend()
    if backend == "cuda":
        from seiso.kernels.cuda_ops import fused_rms_norm as cuda_rms

        return cuda_rms(x, weight, eps=eps, residual=residual)

    if backend == "triton":
        # Triton RMSNorm is inference-only (no autograd); keep training on PyTorch.
        if torch.is_grad_enabled() and (x.requires_grad or getattr(weight, "requires_grad", False)):
            return _pytorch_rms_norm(x, weight, eps, residual)
        from seiso.kernels.triton_ops import fused_rms_norm as triton_rms

        return triton_rms(x, weight, eps=eps, residual=residual)

    return _pytorch_rms_norm(x, weight, eps, residual)


def fused_swiglu(gate, up):
    """``silu(gate) * up`` via best available backend."""
    import torch

    if not getattr(gate, "is_cuda", False):
        return torch.nn.functional.silu(gate) * up

    backend = active_backend()
    if backend == "cuda":
        from seiso.kernels.cuda_ops import fused_swiglu as cuda_swiglu

        return cuda_swiglu(gate, up)

    if backend == "triton":
        from seiso.kernels.triton_ops import fused_swiglu as triton_swiglu

        return triton_swiglu(gate, up)

    return torch.nn.functional.silu(gate) * up


def fused_cross_entropy_loss(logits, labels, *, ignore_index: int = -100):
    from seiso.kernels.loss import fused_cross_entropy_loss as _fused_ce

    return _fused_ce(logits, labels, ignore_index=ignore_index)


def fused_lora_delta(
    x,
    lora_A,
    lora_B,
    base=None,
    scale: float = 1.0,
    *,
    inplace: bool = False,
):
    """Fused low-rank delta when native CUDA is available."""

    if not getattr(x, "is_cuda", False):
        hidden = x @ lora_A.t()
        delta = scale * (hidden @ lora_B.t())
        if base is None:
            return delta
        if inplace:
            base.add_(delta.to(base.dtype))
            return base
        return base + delta

    backend = active_backend()
    if backend == "cuda":
        from seiso.kernels.cuda_ops import fused_lora_delta as cuda_lora

        return cuda_lora(x, lora_A, lora_B, base=base, scale=scale, inplace=inplace)

    hidden = x @ lora_A.t()
    delta = scale * (hidden @ lora_B.t())
    if base is None:
        return delta
    if inplace:
        base.add_(delta.to(base.dtype))
        return base
    return base + delta


def kernel_metadata() -> dict:
    """Runtime kernel stack info for manifests and UI."""
    platform: GpuPlatform = detect_gpu()
    backend = active_backend()
    boundary: dict = {}
    try:
        from seiso.security.nvidia_boundary import nvidia_boundary_report

        boundary = nvidia_boundary_report()
    except ImportError:
        pass
    try:
        from seiso.kernels.memory_mode import kernel_low_vram_enabled

        low_vram = kernel_low_vram_enabled()
    except ImportError:
        low_vram = False
    return {
        "vendor": platform.vendor.value,
        "device_label": (
            "nvidia_gpu"
            if platform.vendor.value == "nvidia"
            else "amd_gpu"
            if platform.vendor.value == "amd"
            else "cpu"
        ),
        "device_count": platform.device_count,
        "kernel_backend": backend,
        "native_cuda": platform.supports_native_cuda and backend == "cuda",
        "optimized_cuda_path": platform.uses_optimized_cuda_kernels and backend == "cuda",
        "wsl2": platform.is_wsl2,
        "cuda_compute_capability": (
            list(platform.cuda_compute_capability) if platform.cuda_compute_capability else None
        ),
        "triton": platform.supports_triton,
        "nvidia_boundary": boundary,
        "kernel_low_vram": low_vram,
    }


def estimate_vram_savings_pct(use_fused: bool, use_4bit: bool, *, low_vram: bool = False) -> float:
    savings = 0.0
    if use_4bit:
        savings += 55.0
    if use_fused:
        backend = active_backend()
        savings += 28.0 if backend == "cuda" else 18.0 if backend == "triton" else 0.0
        if low_vram:
            savings += 6.0 if backend == "cuda" else 4.0 if backend == "triton" else 0.0
    return min(savings, 82.0)
