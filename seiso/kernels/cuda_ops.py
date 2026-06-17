"""Native CUDA fused kernels — NVIDIA only; AMD uses Triton via dispatch."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from seiso.kernels.platform import GpuVendor, detect_gpu

logger = logging.getLogger(__name__)

_CUDA_DIR = Path(__file__).resolve().parent / "cuda"
_EXT: Any | None = None
_EXT_ERROR: str | None = None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """True when native CUDA extension loaded (NVIDIA GPUs only)."""
    if detect_gpu().vendor != GpuVendor.NVIDIA:
        return False
    return _load_extension() is not None


def _load_extension() -> Any | None:
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        return None
    if detect_gpu().vendor != GpuVendor.NVIDIA:
        _EXT_ERROR = "native CUDA kernels require NVIDIA GPU"
        return None
    if not _torch_cuda_available():
        _EXT_ERROR = "torch CUDA unavailable"
        return None

    try:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="seiso_cuda_kernels",
            sources=[
                str(_CUDA_DIR / "extension.cpp"),
                str(_CUDA_DIR / "rms_norm.cu"),
                str(_CUDA_DIR / "fused_swiglu.cu"),
                str(_CUDA_DIR / "fused_lora.cu"),
                str(_CUDA_DIR / "fused_cross_entropy.cu"),
            ],
            extra_include_paths=[str(_CUDA_DIR / "include"), str(_CUDA_DIR)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            ],
            verbose=False,
        )
        logger.info("Seiso native CUDA kernels loaded (stripe RMSNorm)")
        return _EXT
    except Exception as exc:  # noqa: BLE001
        _EXT_ERROR = str(exc)
        logger.debug("CUDA kernel load failed: %s", exc)
        return None


def fused_rms_norm(x, weight, eps: float = 1e-6, residual=None):
    """
    Stripe RMSNorm with optional fused residual — NVIDIA native path.

    Falls back to Triton then PyTorch when extension unavailable.
    """
    import torch

    if not x.is_cuda:
        return _pytorch_rms_norm(x, weight, eps, residual)

    ext = _load_extension()
    if ext is not None:
        return ext.fused_rmsnorm(x, weight, residual, eps)

    from seiso.kernels.triton_ops import fused_rms_norm as triton_rms

    return triton_rms(x, weight, eps=eps, residual=residual)


def fused_swiglu(gate, up):
    """``silu(gate) * up`` with vectorized CUDA kernel."""
    import torch

    if not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up

    ext = _load_extension()
    if ext is not None:
        return ext.fused_swiglu(gate, up)
    return torch.nn.functional.silu(gate) * up


def fused_lora_delta(x, lora_A, lora_B, base=None, scale: float = 1.0):
    """Fused low-rank delta: ``base + scale * B @ (A @ x)``."""
    import torch

    if not x.is_cuda:
        delta = scale * (lora_B @ (lora_A @ x))
        return base + delta if base is not None else delta

    ext = _load_extension()
    if ext is not None and x.dim() == 1 and lora_A.size(0) <= 64:
        return ext.fused_lora_delta(x, lora_A, lora_B, base, scale)

    hidden = lora_A @ x
    delta = scale * (lora_B @ hidden)
    return base + delta if base is not None else delta


def cross_entropy_forward(logits, labels, ignore_index: int = -100):
    """Returns (row_loss, row_max, row_lse) float tensors."""
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA cross_entropy_forward requires native extension")
    return ext.cross_entropy_forward(logits, labels, ignore_index)


def cross_entropy_backward(logits, labels, row_max, row_lse, ignore_index: int = -100, grad_scale: float = 1.0):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA cross_entropy_backward requires native extension")
    return ext.cross_entropy_backward(logits, labels, row_max, row_lse, ignore_index, grad_scale)


def _pytorch_rms_norm(x, weight, eps: float, residual):
    import torch

    if residual is not None:
        x = x + residual
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def estimate_vram_savings_pct(use_cuda: bool, use_4bit: bool) -> float:
    savings = 0.0
    if use_4bit:
        savings += 55.0
    if use_cuda:
        savings += 24.0
    return min(savings, 78.0)
