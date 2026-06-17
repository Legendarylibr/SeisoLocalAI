"""Fused RMSNorm — Triton when available, else PyTorch."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TRITON = False
try:
    import triton
    import triton.language as tl

    _TRITON = True
except ImportError:
    tl = None  # type: ignore[assignment]
    triton = None  # type: ignore[assignment]


def is_triton_available() -> bool:
    return _TRITON


if _TRITON:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        n_cols,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rms = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = x * rms * w
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def fused_rms_norm(x, weight, eps: float = 1e-6):
    """Row-wise RMS normalization. Input shape (rows, cols)."""
    import torch

    if not _TRITON or not x.is_cuda:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight

    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)  # type: ignore[union-attr]
    grid = (rows,)
    _rms_norm_kernel[grid](x, weight, out, cols, eps, BLOCK=BLOCK)  # type: ignore[misc]
    return out


def estimate_vram_savings_pct(use_triton: bool, use_4bit: bool) -> float:
    """Heuristic VRAM reduction estimate for manifest logging."""
    savings = 0.0
    if use_4bit:
        savings += 55.0
    if use_triton:
        savings += 15.0
    return min(savings, 75.0)
