"""GPU platform detection — NVIDIA CUDA, AMD ROCm, or CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from seiso.compat import StrEnum


class GpuVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


def detect_wsl2() -> bool:
    """True when running inside WSL2 (CUDA path uses same native kernels as Linux)."""
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            version = f.read().lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl2" in version


def _cuda_compute_capability() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except ImportError:
        pass
    return None


@dataclass(frozen=True)
class GpuPlatform:
    vendor: GpuVendor
    device_name: str
    device_count: int
    supports_native_cuda: bool
    supports_triton: bool
    is_wsl2: bool = False
    cuda_compute_capability: tuple[int, int] | None = None

    @property
    def preferred_kernel_backend(self) -> str:
        # CUDA and WSL2 both use highly optimized native .cu kernels when available.
        if self.vendor == GpuVendor.NVIDIA and self.supports_native_cuda:
            return "cuda"
        if self.supports_triton and self.device_count > 0:
            return "triton"
        return "pytorch"

    @property
    def uses_optimized_cuda_kernels(self) -> bool:
        """True when native CUDA fused kernels are the preferred training path."""
        return self.vendor == GpuVendor.NVIDIA and self.supports_native_cuda


@lru_cache(maxsize=1)
def detect_gpu() -> GpuPlatform:
    """Detect the active GPU stack once per process."""
    wsl2 = detect_wsl2()
    try:
        import torch
    except ImportError:
        return GpuPlatform(GpuVendor.CPU, "cpu", 0, False, False, wsl2, None)

    if not torch.cuda.is_available():
        return GpuPlatform(GpuVendor.CPU, "cpu", 0, False, False, wsl2, None)

    device_count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if device_count else "unknown"
    is_amd = bool(getattr(torch.version, "hip", None))
    cc = _cuda_compute_capability()

    triton_ok = False
    try:
        import triton  # noqa: F401

        triton_ok = True
    except ImportError:
        pass

    if is_amd:
        return GpuPlatform(
            vendor=GpuVendor.AMD,
            device_name=name,
            device_count=device_count,
            supports_native_cuda=False,
            supports_triton=triton_ok,
            is_wsl2=wsl2,
            cuda_compute_capability=cc,
        )

    return GpuPlatform(
        vendor=GpuVendor.NVIDIA,
        device_name=name,
        device_count=device_count,
        supports_native_cuda=True,
        supports_triton=triton_ok,
        is_wsl2=wsl2,
        cuda_compute_capability=cc,
    )


def is_nvidia() -> bool:
    return detect_gpu().vendor == GpuVendor.NVIDIA


def is_amd() -> bool:
    return detect_gpu().vendor == GpuVendor.AMD
