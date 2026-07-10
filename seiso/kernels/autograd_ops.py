"""Autograd-safe wrappers around fused CUDA forward kernels.

Native fused_rmsnorm / fused_swiglu lack CUDA backward kernels. These Functions
run the fused forward for speed, then apply analytically correct gradients in
backward — unlocking training-time fused forwards without sacrificing correctness.
"""

from __future__ import annotations

_FusedRMSNormFn = None
_FusedSwiGLUFn = None


def _ensure_autograd_fns():
    global _FusedRMSNormFn, _FusedSwiGLUFn
    if _FusedRMSNormFn is not None:
        return

    import torch

    class FusedRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, residual, eps: float):
            from seiso.kernels.cuda_ops import fused_rms_norm as cuda_rms

            x_in = x + residual if residual is not None else x
            with torch.no_grad():
                out = cuda_rms(x, weight, eps=float(eps), residual=residual)
            ctx.eps = float(eps)
            ctx.has_residual = residual is not None
            ctx.save_for_backward(x_in, weight)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x_in, weight = ctx.saved_tensors
            eps = ctx.eps
            var = x_in.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(var + eps)
            dx = grad_out * weight * inv_rms
            n = x_in.shape[-1]
            grad_var = (grad_out * weight * x_in * (-0.5) * inv_rms.pow(3)).sum(
                dim=-1, keepdim=True
            )
            dx = dx + (2.0 / n) * x_in * grad_var
            reduce_dims = tuple(range(grad_out.dim() - 1))
            dweight = (grad_out * x_in * inv_rms).sum(dim=reduce_dims)
            if ctx.has_residual:
                return dx, dweight, dx, None
            return dx, dweight, None, None

    class FusedSwiGLUFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            from seiso.kernels.cuda_ops import fused_swiglu as cuda_swiglu

            with torch.no_grad():
                out = cuda_swiglu(gate, up)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            gate, up = ctx.saved_tensors
            sig = torch.sigmoid(gate)
            silu = gate * sig
            dsilu = sig * (1.0 + gate * (1.0 - sig))
            dgate = grad_out * up * dsilu
            dup = grad_out * silu
            return dgate, dup

    _FusedRMSNormFn = FusedRMSNormFn
    _FusedSwiGLUFn = FusedSwiGLUFn


def fused_rms_norm_autograd(x, weight, eps: float, residual):
    """Fused RMSNorm forward + analytic backward."""
    _ensure_autograd_fns()
    return _FusedRMSNormFn.apply(x, weight, residual, eps)


def fused_swiglu_autograd(gate, up):
    """Fused SwiGLU forward + analytic backward."""
    _ensure_autograd_fns()
    return _FusedSwiGLUFn.apply(gate, up)
