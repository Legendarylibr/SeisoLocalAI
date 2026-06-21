"""GPU platform detection — NVIDIA CUDA, AMD ROCm, or CPU."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache

from seiso.compat import StrEnum
from seiso.platform import detect_wsl2


class GpuVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    CPU = "cpu"


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
        torch = None  # type: ignore[assignment]

    triton_ok = False
    try:
        import triton  # noqa: F401

        triton_ok = True
    except ImportError:
        pass

    if torch is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if device_count else "unknown"
        is_amd = bool(getattr(torch.version, "hip", None))
        cc = _cuda_compute_capability()

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

    if platform.system().lower() in {"linux", "windows"}:
        try:
            from seiso.security.nvidia_boundary import query_nvidia_gpus

            smi_gpus = query_nvidia_gpus()
        except ImportError:
            smi_gpus = []
        if smi_gpus:
            first = smi_gpus[0]
            name = str(first.get("name") or "nvidia gpu")
            return GpuPlatform(
                vendor=GpuVendor.NVIDIA,
                device_name=name,
                device_count=len(smi_gpus),
                supports_native_cuda=False,
                supports_triton=triton_ok,
                is_wsl2=wsl2,
                cuda_compute_capability=None,
            )

    return GpuPlatform(GpuVendor.CPU, "cpu", 0, False, False, wsl2, None)
