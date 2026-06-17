"""Fused GPU kernels — NVIDIA CUDA, AMD Triton, leak-safe lifecycle."""

from seiso.kernels.dispatch import (
    active_backend,
    estimate_vram_savings_pct,
    fused_cross_entropy_loss,
    fused_lora_delta,
    fused_rms_norm,
    fused_swiglu,
    kernel_metadata,
)
from seiso.kernels.hooks import (
    apply_fused_lora_kernels,
    apply_training_kernels,
    clear_kernel_patches,
)
from seiso.kernels.lifecycle import release_training_memory, restore_kernel_patches
from seiso.kernels.platform import GpuPlatform, GpuVendor, detect_gpu, is_amd, is_nvidia
from seiso.kernels.triton_ops import is_triton_available

__all__ = [
    "GpuPlatform",
    "GpuVendor",
    "active_backend",
    "apply_fused_lora_kernels",
    "apply_training_kernels",
    "clear_kernel_patches",
    "detect_gpu",
    "estimate_vram_savings_pct",
    "fused_cross_entropy_loss",
    "fused_lora_delta",
    "fused_rms_norm",
    "fused_swiglu",
    "is_amd",
    "is_nvidia",
    "is_triton_available",
    "kernel_metadata",
    "release_training_memory",
    "restore_kernel_patches",
]
