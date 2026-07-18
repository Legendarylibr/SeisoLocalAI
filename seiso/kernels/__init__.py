"""Fused GPU kernels — NVIDIA CUDA, AMD Triton, leak-safe lifecycle."""

from seiso.kernels.attention import (
    attention_doctor_lines,
    attention_metadata,
    enable_torch_sdpa_backends,
    resolve_attention_implementation,
)
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
from seiso.kernels.platform import GpuPlatform, GpuVendor, detect_gpu
from seiso.kernels.training_profile import (
    CudaTrainingMode,
    last_cuda_training_profile,
    prepare_cuda_training_profile,
)
from seiso.kernels.triton_ops import is_triton_available
from seiso.kernels.tuning import (
    KERNEL_PROFILES,
    apply_kernel_profile,
    benchmark_kernel_profile,
    kernel_metrics_dict,
    kernel_profile_count,
)

__all__ = [
    "GpuPlatform",
    "GpuVendor",
    "active_backend",
    "apply_fused_lora_kernels",
    "apply_training_kernels",
    "attention_doctor_lines",
    "attention_metadata",
    "clear_kernel_patches",
    "CudaTrainingMode",
    "detect_gpu",
    "enable_torch_sdpa_backends",
    "estimate_vram_savings_pct",
    "fused_cross_entropy_loss",
    "fused_lora_delta",
    "fused_rms_norm",
    "fused_swiglu",
    "is_triton_available",
    "KERNEL_PROFILES",
    "apply_kernel_profile",
    "benchmark_kernel_profile",
    "kernel_metrics_dict",
    "kernel_profile_count",
    "kernel_metadata",
    "last_cuda_training_profile",
    "prepare_cuda_training_profile",
    "release_training_memory",
    "resolve_attention_implementation",
    "restore_kernel_patches",
]
