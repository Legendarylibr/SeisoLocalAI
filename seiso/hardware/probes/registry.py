"""GPU probe registry — re-exports enumeration and backend probes."""

from seiso.hardware.gpus import (
    _mlx_apple_gpu,
    _nvidia_smi_gpus,
    _torch_gpus,
    clear_gpu_enumeration_cache,
    enumerate_compute_gpus,
    enumerate_gpus,
    gpu_count,
    primary_gpu_name,
)
from seiso.hardware.probes.apple import probe_apple_mlx_gpu
from seiso.hardware.probes.nvidia import probe_nvidia_gpus
from seiso.hardware.probes.torch_cuda import probe_torch_gpus

__all__ = [
    "_mlx_apple_gpu",
    "_nvidia_smi_gpus",
    "_torch_gpus",
    "clear_gpu_enumeration_cache",
    "enumerate_compute_gpus",
    "enumerate_gpus",
    "gpu_count",
    "primary_gpu_name",
    "probe_apple_mlx_gpu",
    "probe_nvidia_gpus",
    "probe_torch_gpus",
]
