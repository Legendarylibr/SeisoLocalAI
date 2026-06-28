"""Fused RMSNorm / SwiGLU — Triton for AMD ROCm and CUDA fallback."""

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
        r_ptr,
        w_ptr,
        out_ptr,
        n_cols,
        eps,
        FUSE_RESIDUAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        if FUSE_RESIDUAL:
            r = tl.load(r_ptr + row * n_cols + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x = x + r
        var = tl.sum(x * x, axis=0) / n_cols
        rms = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = x * rms * w
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def _flatten_rows(x):
    """Collapse leading dims so Triton kernels see (rows, cols)."""
    if x.dim() <= 2:
        return x, None
    cols = x.shape[-1]
    return x.reshape(-1, cols), x.shape


def _restore_rows(out, orig_shape):
    if orig_shape is None:
        return out
    return out.reshape(orig_shape)


def fused_rms_norm(x, weight, eps: float = 1e-6, residual=None):
    """Row-wise RMS normalization with optional fused residual add."""
    import torch

    on_gpu = getattr(x, "is_cuda", False)
    if not _TRITON or not on_gpu:
        if residual is not None:
            x = x + residual
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight

    x, orig_shape = _flatten_rows(x)
    if residual is not None:
        residual, _ = _flatten_rows(residual)

    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)  # type: ignore[union-attr]
    grid = (rows,)
    fuse = residual is not None
    r_ptr = residual if fuse else x  # unused when FUSE_RESIDUAL=False
    _rms_norm_kernel[grid](  # type: ignore[misc]
        x,
        r_ptr,
        weight,
        out,
        cols,
        eps,
        FUSE_RESIDUAL=fuse,
        BLOCK=BLOCK,
    )
    return _restore_rows(out, orig_shape)


if _TRITON:

    @triton.jit
    def _swiglu_kernel(g_ptr, u_ptr, o_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        g = tl.load(g_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = (g / (1.0 + tl.exp(-g))) * u
        tl.store(o_ptr + row * n_cols + cols, out, mask=mask)


def fused_swiglu(gate, up):
    """``silu(gate) * up`` row-wise."""
    import torch

    on_gpu = getattr(gate, "is_cuda", False)
    if not _TRITON or not on_gpu:
        return torch.nn.functional.silu(gate) * up

    gate, orig_shape = _flatten_rows(gate)
    up, _ = _flatten_rows(up)

    rows, cols = gate.shape
    out = torch.empty_like(gate)
    BLOCK = triton.next_power_of_2(cols)  # type: ignore[union-attr]
    _swiglu_kernel[(rows,)](gate, up, out, cols, BLOCK=BLOCK)  # type: ignore[misc]
    return _restore_rows(out, orig_shape)


def fused_cross_entropy_forward(logits, labels, ignore_index: int = -100):
    """Return (row_loss, row_max, row_lse) on GPU."""
    import torch

    rows, vocab = logits.shape
    row_loss = torch.zeros(rows, device=logits.device, dtype=torch.float32)
    row_max = torch.zeros(rows, device=logits.device, dtype=torch.float32)
    row_lse = torch.ones(rows, device=logits.device, dtype=torch.float32)

    if not _TRITON or not logits.is_cuda:
        for i in range(rows):
            label = int(labels[i].item())
            if label == ignore_index:
                continue
            row = logits[i].float()
            m = row.max()
            lse = torch.logsumexp(row - m, dim=0) + m
            row_max[i] = m
            row_lse[i] = lse
            row_loss[i] = lse - row[label]
        return row_loss, row_max, row_lse

    @triton.jit
    def _ce_fwd_kernel(
        logits_ptr,
        labels_ptr,
        loss_ptr,
        max_ptr,
        lse_ptr,
        vocab,
        ignore_idx,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        if label == ignore_idx:
            tl.store(loss_ptr + row, 0.0)
            tl.store(max_ptr + row, 0.0)
            tl.store(lse_ptr + row, 1.0)
            return
        cols = tl.arange(0, BLOCK)
        mask = cols < vocab
        vals = tl.load(
            logits_ptr + row * vocab + cols, mask=mask, other=-float("inf")
        ).to(tl.float32)
        row_max = tl.max(vals, axis=0)
        expv = tl.exp(vals - row_max)
        denom = tl.sum(expv, axis=0)
        lse = tl.log(denom) + row_max
        target = tl.load(logits_ptr + row * vocab + label).to(tl.float32)
        tl.store(loss_ptr + row, lse - target)
        tl.store(max_ptr + row, row_max)
        tl.store(lse_ptr + row, lse)

    BLOCK = triton.next_power_of_2(vocab)  # type: ignore[union-attr]
    _ce_fwd_kernel[(rows,)](  # type: ignore[misc]
        logits, labels, row_loss, row_max, row_lse, vocab, ignore_index, BLOCK=BLOCK
    )
    return row_loss, row_max, row_lse


def fused_cross_entropy_backward(
    logits, labels, row_max, row_lse, ignore_index: int = -100, grad_scale: float = 1.0
):
    import torch

    rows, vocab = logits.shape
    grad = torch.zeros_like(logits)
    if not _TRITON or not logits.is_cuda:
        for i in range(rows):
            label = int(labels[i].item())
            if label == ignore_index:
                continue
            row = logits[i].float()
            probs = torch.exp(row - row_lse[i])
            probs[label] -= 1.0
            grad[i] = (probs * grad_scale).to(logits.dtype)
        return grad

    @triton.jit
    def _ce_bwd_kernel(
        logits_ptr,
        labels_ptr,
        max_ptr,
        lse_ptr,
        grad_ptr,
        vocab,
        ignore_idx,
        scale,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        cols = tl.arange(0, BLOCK)
        mask = cols < vocab
        if label == ignore_idx:
            tl.store(grad_ptr + row * vocab + cols, 0.0, mask=mask)
            return
        lse = tl.load(lse_ptr + row)
        vals = tl.load(logits_ptr + row * vocab + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        probs = tl.exp(vals - lse)
        probs = tl.where(cols == label, probs - 1.0, probs)
        tl.store(grad_ptr + row * vocab + cols, probs * scale, mask=mask)

    BLOCK = triton.next_power_of_2(vocab)  # type: ignore[union-attr]
    _ce_bwd_kernel[(rows,)](  # type: ignore[misc]
        logits,
        labels,
        row_max,
        row_lse,
        grad,
        vocab,
        ignore_index,
        grad_scale,
        BLOCK=BLOCK,
    )
    return grad
