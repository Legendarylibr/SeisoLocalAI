"""GPU platform detection — NVIDIA CUDA, AMD ROCm, or CPU."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import lru_cache


class GpuVendor(enum.StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


@dataclass(frozen=True)
class GpuPlatform:
    vendor: GpuVendor
    device_name: str
    device_count: int
    supports_native_cuda: bool
    supports_triton: bool

    @property
    def preferred_kernel_backend(self) -> str:
        if self.vendor == GpuVendor.NVIDIA and self.supports_native_cuda:
            return "cuda"
        if self.supports_triton and self.device_count > 0:
            return "triton"
        return "pytorch"


@lru_cache(maxsize=1)
def detect_gpu() -> GpuPlatform:
    """Detect the active GPU stack once per process."""
    try:
        import torch
    except ImportError:
        return GpuPlatform(GpuVendor.CPU, "cpu", 0, False, False)

    if not torch.cuda.is_available():
        return GpuPlatform(GpuVendor.CPU, "cpu", 0, False, False)

    device_count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if device_count else "unknown"
    is_amd = bool(getattr(torch.version, "hip", None))

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
        )

    return GpuPlatform(
        vendor=GpuVendor.NVIDIA,
        device_name=name,
        device_count=device_count,
        supports_native_cuda=True,
        supports_triton=triton_ok,
    )


def is_nvidia() -> bool:
    return detect_gpu().vendor == GpuVendor.NVIDIA


def is_amd() -> bool:
    return detect_gpu().vendor == GpuVendor.AMD


def is_gpu_available() -> bool:
    return detect_gpu().device_count > 0
