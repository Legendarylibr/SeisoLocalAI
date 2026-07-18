"""Unified fused-op dispatch — NVIDIA CUDA, AMD Triton, PyTorch fallback."""

from __future__ import annotations

import logging
from functools import lru_cache

from seiso.kernels.fallback_ops import pytorch_rms_norm as _pytorch_rms_norm
from seiso.kernels.platform import GpuPlatform, detect_gpu

logger = logging.getLogger(__name__)


def _needs_pytorch_autograd(*tensors) -> bool:
    """Native CUDA/Triton fused ops lack autograd — route training to PyTorch."""
    import torch

    if not torch.is_grad_enabled():
        return False
    return any(getattr(t, "requires_grad", False) for t in tensors if t is not None)


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

    if not getattr(x, "is_cuda", False):
        return _pytorch_rms_norm(x, weight, eps, residual)

    backend = active_backend()
    if backend == "cuda":
        from seiso.kernels.cuda_ops import fused_rms_norm as cuda_rms

        if _needs_pytorch_autograd(x, weight, residual):
            from seiso.kernels.autograd_ops import fused_rms_norm_autograd

            return fused_rms_norm_autograd(x, weight, eps, residual)
        return cuda_rms(x, weight, eps=eps, residual=residual)

    if backend == "triton":
        # Triton RMSNorm is inference-only (no autograd); keep training on PyTorch.
        if _needs_pytorch_autograd(x, weight, residual):
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

        if _needs_pytorch_autograd(gate, up):
            from seiso.kernels.autograd_ops import fused_swiglu_autograd

            return fused_swiglu_autograd(gate, up)
        return cuda_swiglu(gate, up)

    if _needs_pytorch_autograd(gate, up):
        return torch.nn.functional.silu(gate) * up

    if backend == "triton":
        from seiso.kernels.triton_ops import fused_swiglu as triton_swiglu

        return triton_swiglu(gate, up)

    return torch.nn.functional.silu(gate) * up


def fused_cross_entropy_loss(logits, labels, *, ignore_index: int = -100):
    from seiso.kernels.loss import fused_cross_entropy_loss as _fused_ce

    return _fused_ce(logits, labels, ignore_index=ignore_index)


def _mlp_gemm_swiglu(x, W_gate, W_up):
    """cuBLAS/torch GEMMs for gate/up + fused SwiGLU epilogue (production path).

    When gate/up weights share shape/dtype/device, use a single stacked GEMM
    (``W = cat(W_gate, W_up)``) then split — same FLOPs, fewer launches.
    """
    import torch

    if (
        W_gate.shape == W_up.shape
        and W_gate.dtype == W_up.dtype
        and W_gate.device == W_up.device
        and W_gate.is_floating_point()
    ):
        mid = int(W_gate.shape[0])
        stacked = torch.cat((W_gate, W_up), dim=0)
        hidden = x @ stacked.t()
        gate, up = hidden.split(mid, dim=-1)
        # split() views are non-contiguous; CUDA SwiGLU requires contiguous rows.
        return fused_swiglu(gate.contiguous(), up.contiguous())

    gate = x @ W_gate.t()
    up = x @ W_up.t()
    return fused_swiglu(gate, up)


def fused_mlp_swiglu(x, W_gate, W_up):
    """``silu(x @ W_gate^T) * (x @ W_up^T)``.

    Production path always uses torch/cuBLAS matmuls (prefer one stacked GEMM)
    plus the fused SwiGLU elementwise kernel. Naive scalar CUDA matmul is not
    used here — it cannot compete with Tensor Cores.
    """
    return _mlp_gemm_swiglu(x, W_gate, W_up)


def _fused_lora_qkv_delta_torch(
    x,
    out_q,
    out_k,
    out_v,
    lora_A_q,
    lora_B_q,
    lora_A_k,
    lora_B_k,
    lora_A_v,
    lora_B_v,
    *,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
) -> None:
    """cuBLAS-backed LoRA Q/K/V deltas — shared-x A GEMM; batched B when aligned."""
    import torch

    rank_q = lora_A_q.size(0)
    rank_k = lora_A_k.size(0)
    rank_v = lora_A_v.size(0)
    if rank_q == rank_k == rank_v:
        hidden_all = x @ torch.cat((lora_A_q, lora_A_k, lora_A_v), dim=0).t()
        h_q, h_k, h_v = hidden_all.split((rank_q, rank_k, rank_v), dim=-1)
    elif rank_k == rank_v:
        h_q = x @ lora_A_q.t()
        hidden_kv = x @ torch.cat((lora_A_k, lora_A_v), dim=0).t()
        h_k, h_v = hidden_kv.split((rank_k, rank_v), dim=-1)
    else:
        h_q = x @ lora_A_q.t()
        h_k = x @ lora_A_k.t()
        h_v = x @ lora_A_v.t()

    # Batched B@h when out dims, ranks, dtypes, and scales match (one bmm).
    if (
        lora_B_q.shape == lora_B_k.shape == lora_B_v.shape
        and lora_B_q.dtype == lora_B_k.dtype == lora_B_v.dtype
        and float(scale_q) == float(scale_k) == float(scale_v)
        and h_q.shape == h_k.shape == h_v.shape
    ):
        h = torch.stack((h_q, h_k, h_v), dim=0)  # [3, rows, rank]
        b = torch.stack((lora_B_q, lora_B_k, lora_B_v), dim=0)  # [3, out, rank]
        deltas = torch.bmm(h, b.transpose(1, 2)).mul_(float(scale_q))  # [3, rows, out]
        out_q.add_(deltas[0].to(dtype=out_q.dtype))
        out_k.add_(deltas[1].to(dtype=out_k.dtype))
        out_v.add_(deltas[2].to(dtype=out_v.dtype))
        return

    out_q.add_((scale_q * (h_q @ lora_B_q.t())).to(out_q.dtype))
    out_k.add_((scale_k * (h_k @ lora_B_k.t())).to(out_k.dtype))
    out_v.add_((scale_v * (h_v @ lora_B_v.t())).to(out_v.dtype))


def _prefer_cublas_lora(x) -> bool:
    """Prefer torch/cuBLAS LoRA GEMMs over the scalar custom CUDA path.

    The native LoRA kernels serialize ``A@x`` on a single thread per block, so
    they only win on tiny micro-shapes. Production always uses cuBLAS unless
    ``SEISO_KERNEL_ALLOW_NAIVE_LORA=1`` and the shape is tiny (for experiments).
    """
    import os

    import torch

    if torch.is_grad_enabled():
        return True
    allow_naive = os.environ.get("SEISO_KERNEL_ALLOW_NAIVE_LORA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_naive:
        return True
    rows = max(1, x.numel() // x.shape[-1])
    in_dim = int(x.shape[-1])
    # Only allow naive kernel for very small no-grad microbenches.
    return rows * in_dim > 32 * 128


# Backward-compatible alias
_prefer_cublas_lora_qkv = _prefer_cublas_lora


def fused_lora_qkv_delta(
    x,
    out_q,
    out_k,
    out_v,
    lora_A_q,
    lora_B_q,
    lora_A_k,
    lora_B_k,
    lora_A_v,
    lora_B_v,
    *,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
):
    """In-place fused LoRA deltas for Q/K/V projections (cuBLAS by default)."""
    if not getattr(x, "is_cuda", False):
        return _fused_lora_qkv_delta_torch(
            x,
            out_q,
            out_k,
            out_v,
            lora_A_q,
            lora_B_q,
            lora_A_k,
            lora_B_k,
            lora_A_v,
            lora_B_v,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
        )

    if _prefer_cublas_lora(x):
        return _fused_lora_qkv_delta_torch(
            x,
            out_q,
            out_k,
            out_v,
            lora_A_q,
            lora_B_q,
            lora_A_k,
            lora_B_k,
            lora_A_v,
            lora_B_v,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
        )

    backend = active_backend()
    if backend == "cuda":
        try:
            from seiso.kernels.cuda_ops import fused_lora_qkv_delta as cuda_qkv

            return cuda_qkv(
                x,
                out_q,
                out_k,
                out_v,
                lora_A_q,
                lora_B_q,
                lora_A_k,
                lora_B_k,
                lora_A_v,
                lora_B_v,
                scale_q=scale_q,
                scale_k=scale_k,
                scale_v=scale_v,
            )
        except (RuntimeError, ImportError):
            pass

    return _fused_lora_qkv_delta_torch(
        x,
        out_q,
        out_k,
        out_v,
        lora_A_q,
        lora_B_q,
        lora_A_k,
        lora_B_k,
        lora_A_v,
        lora_B_v,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )


def _lora_delta_torch(x, lora_A, lora_B, base=None, scale: float = 1.0, *, inplace: bool = False):
    hidden = x @ lora_A.t()
    delta = scale * (hidden @ lora_B.t())
    if base is None:
        return delta
    if inplace:
        base.add_(delta.to(base.dtype))
        return base
    return base + delta


def fused_lora_delta(
    x,
    lora_A,
    lora_B,
    base=None,
    scale: float = 1.0,
    *,
    inplace: bool = False,
):
    """Low-rank delta ``base + scale * B @ (A @ x)`` — cuBLAS/torch by default."""

    if not getattr(x, "is_cuda", False):
        return _lora_delta_torch(x, lora_A, lora_B, base=base, scale=scale, inplace=inplace)

    if _prefer_cublas_lora(x):
        return _lora_delta_torch(x, lora_A, lora_B, base=base, scale=scale, inplace=inplace)

    if active_backend() == "cuda":
        try:
            from seiso.kernels.cuda_ops import fused_lora_delta as cuda_lora

            return cuda_lora(x, lora_A, lora_B, base=base, scale=scale, inplace=inplace)
        except (RuntimeError, ImportError):
            pass

    return _lora_delta_torch(x, lora_A, lora_B, base=base, scale=scale, inplace=inplace)


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
    arch_meta: dict = {}
    try:
        from seiso.kernels.arch_tuning import detect_arch_tuning

        arch = detect_arch_tuning()
        arch_meta = {
            "arch_family": arch.family.value,
            "arch_sm": arch.sm,
            "use_wmma": arch.use_wmma,
            "use_persistent_kernels": arch.use_persistent_kernels,
            "prefer_flash_attn": arch.prefer_flash_attn,
        }
    except ImportError:
        pass
    attn_meta: dict = {}
    try:
        from seiso.kernels.attention import attention_metadata, enable_torch_sdpa_backends

        enable_torch_sdpa_backends()
        attn_meta = attention_metadata()
    except ImportError:
        pass

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
        **arch_meta,
        **attn_meta,
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
