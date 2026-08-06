"""Fused cross-entropy loss with autograd — no full softmax buffer."""

from __future__ import annotations

import torch

from seiso.kernels.dispatch import active_backend


class _FusedCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int):
        backend = active_backend()
        if backend == "cuda":
            from seiso.kernels.cuda_ops import cross_entropy_forward

            row_loss, row_max, row_lse = cross_entropy_forward(logits, labels, ignore_index)
        else:
            from seiso.kernels.triton_ops import fused_cross_entropy_forward

            row_loss, row_max, row_lse = fused_cross_entropy_forward(logits, labels, ignore_index)

        mask = labels != ignore_index
        valid = mask.sum()
        loss = row_loss[mask].mean()
        ctx.save_for_backward(logits, labels, row_max, row_lse)
        ctx.ignore_index = ignore_index
        ctx.valid = valid
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, labels, row_max, row_lse = ctx.saved_tensors
        valid = int(ctx.valid.item()) if hasattr(ctx.valid, "item") else int(ctx.valid)
        inv = float(grad_output) / max(valid, 1)
        backend = active_backend()

        if backend == "cuda":
            from seiso.kernels.cuda_ops import cross_entropy_backward

            grad_logits = cross_entropy_backward(
                logits, labels, row_max, row_lse, ctx.ignore_index, inv
            )
        else:
            from seiso.kernels.triton_ops import fused_cross_entropy_backward

            grad_logits = fused_cross_entropy_backward(
                logits, labels, row_max, row_lse, ctx.ignore_index, inv
            )
        return grad_logits, None, None


def fused_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Mean cross-entropy without materializing ``[batch, vocab]`` softmax.

    Falls back to PyTorch on CPU.
    """
    if not logits.is_cuda:
        return torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=ignore_index)

    if logits.dim() != 2:
        raise ValueError("logits must be 2D")
    if labels.dim() != 1:
        raise ValueError("labels must be 1D")

    if not bool((labels != ignore_index).any()):
        return logits.sum() * 0.0

    return _FusedCrossEntropyFn.apply(logits, labels, ignore_index)


def shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM shift: predict token t+1 from position t."""
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)
    return shift_logits, shift_labels
