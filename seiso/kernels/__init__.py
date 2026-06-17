"""Optimized CUDA kernels — Triton with PyTorch fallback."""

from seiso.kernels.hooks import apply_training_kernels, clear_kernel_patches, is_triton_available
from seiso.kernels.triton_ops import fused_rms_norm

__all__ = ["apply_training_kernels", "clear_kernel_patches", "is_triton_available", "fused_rms_norm"]
