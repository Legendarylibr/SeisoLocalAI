"""PyTorch fallback ops when fused CUDA/Triton kernels are unavailable."""


def pytorch_rms_norm(x, weight, eps: float, residual):
    import torch

    if residual is not None:
        x = x + residual
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight
