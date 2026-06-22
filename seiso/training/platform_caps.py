"""Training capability matrix per OS/GPU — used by Forge UI and docs."""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from typing import Any

from seiso.kernels.platform import GpuVendor, detect_gpu


@lru_cache(maxsize=1)
def training_capabilities() -> dict[str, Any]:
    """
    What training features are available on this machine.

    Keys are stable for API/UI consumption.
    """
    system = platform.system()  # Linux, Darwin, Windows
    machine = platform.machine()
    gpu = detect_gpu()

    cuda_runtime = False
    try:
        import torch

        cuda_runtime = torch.cuda.is_available()
    except ImportError:
        pass

    supports_bnb = system != "Darwin"
    if supports_bnb:
        # bitsandbytes may be absent even on Linux (CPU-only box or skipped install).
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            supports_bnb = False
    has_nvidia_hardware = gpu.device_count > 0 and gpu.vendor == GpuVendor.NVIDIA
    has_cuda_gpu = has_nvidia_hardware and cuda_runtime
    has_rocm_gpu = gpu.device_count > 0 and gpu.vendor == GpuVendor.AMD and cuda_runtime

    triton_ok = False
    try:
        from seiso.kernels.triton_ops import is_triton_available

        triton_ok = is_triton_available()
    except ImportError:
        pass

    cuda_ext_ok = False
    if has_cuda_gpu:
        try:
            from seiso.kernels.cuda_ops import is_cuda_available

            cuda_ext_ok = is_cuda_available()
        except ImportError:
            pass

    fused_kernels = has_cuda_gpu or (has_rocm_gpu and triton_ok)
    kernel_backend = "none"
    if has_cuda_gpu:
        kernel_backend = "cuda" if cuda_ext_ok else ("triton" if triton_ok else "pytorch")

    elif has_rocm_gpu and triton_ok:
        kernel_backend = "triton"

    wsl2 = gpu.is_wsl2
    fused_lora_available = has_cuda_gpu and cuda_ext_ok

    mps_ok = False
    try:
        import torch

        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass

    mlx_ok = False
    if os.environ.get("SEISO_SKIP_MLX_PROBE", "").strip().lower() not in {"1", "true", "yes"}:
        try:
            import mlx.core  # noqa: F401

            mlx_ok = True
        except ImportError:
            pass

    if has_cuda_gpu or has_rocm_gpu:
        train_platform = "cuda" if has_cuda_gpu else "rocm"
    elif mps_ok and system == "Darwin":
        train_platform = "mps"
    else:
        train_platform = "cpu"

    return {
        "os": system,
        "arch": machine,
        "train_platform": train_platform,
        "cuda_runtime": cuda_runtime,
        "nvidia_hardware": has_nvidia_hardware,
        "supports_qlora": supports_bnb and (has_cuda_gpu or has_rocm_gpu),
        "supports_training": True,
        "supports_mps_training": mps_ok and system == "Darwin",
        "supports_mlx_inference": mlx_ok and system == "Darwin",
        "fused_kernels_available": fused_kernels,
        "fused_ce_available": fused_kernels,
        "fused_lora_available": fused_lora_available,
        "kernel_backend": kernel_backend,
        "wsl2": wsl2,
        "optimized_cuda_path": has_cuda_gpu and cuda_ext_ok,
        "multi_gpu_available": has_cuda_gpu and gpu.device_count > 1,
        "vendor": gpu.vendor.value,
        "gpu_count": gpu.device_count,
        "device_label": _gpu_label(gpu),
        "recommended_quant": "4bit" if supports_bnb and (has_cuda_gpu or has_rocm_gpu) else "16bit",
        "install_extra": _install_extra(
            system,
            has_nvidia_hardware,
            has_rocm_gpu,
            mlx_ok,
            cuda_runtime,
        ),
    }


def _gpu_label(gpu) -> str:
    if gpu.device_count <= 0:
        return "cpu"
    if gpu.vendor == GpuVendor.NVIDIA:
        return "nvidia_gpu"
    if gpu.vendor == GpuVendor.AMD:
        return "amd_gpu"
    return "gpu"


def _install_extra(
    system: str,
    nvidia_hw: bool,
    rocm: bool,
    mlx: bool,
    cuda_runtime: bool,
) -> str:
    if nvidia_hw and system == "Linux":
        if cuda_runtime:
            return 'pip install -e ".[forge,train,cuda,llamacpp]"'
        return (
            'pip install -e ".[forge,train,cuda,llamacpp]"'
            "  # NVIDIA GPU detected — install CUDA-enabled PyTorch if missing"
        )
    if nvidia_hw and system == "Windows":
        return (
            'pip install -e ".[forge,train,llamacpp]"'
            "  # NVIDIA GPU — use CUDA llama-cpp-python wheel for GGUF GPU chat"
        )
    if rocm:
        return 'pip install -e ".[forge,train]" && pip install triton  # ROCm PyTorch wheel first'
    if system == "Darwin" and mlx:
        return 'pip install -e ".[forge,train,llamacpp,mlx]"'
    if system == "Darwin":
        return 'pip install -e ".[forge,train,llamacpp]"  # optional extras: .[mlx] (macOS), .[cuda] (Linux NVIDIA)'
    return 'pip install -e ".[forge,train]"'
