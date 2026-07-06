"""Platform/backend hardware probes — nvidia-smi, torch.cuda, MLX."""

from seiso.hardware.probes.apple import probe_apple_mlx_gpu
from seiso.hardware.probes.common import GpuMemoryProcess, sanitize_hardware_label
from seiso.hardware.probes.nvidia import (
    nvidia_gpu_metrics,
    parse_nvidia_smi_process_csv,
    probe_nvidia_gpus,
    query_nvidia_compute_processes,
)
from seiso.hardware.probes.registry import (
    clear_gpu_enumeration_cache,
    enumerate_compute_gpus,
    enumerate_gpus,
    gpu_count,
    primary_gpu_name,
)
from seiso.hardware.probes.torch_cuda import probe_torch_gpus

__all__ = [
    "GpuMemoryProcess",
    "clear_gpu_enumeration_cache",
    "enumerate_compute_gpus",
    "enumerate_gpus",
    "gpu_count",
    "nvidia_gpu_metrics",
    "parse_nvidia_smi_process_csv",
    "primary_gpu_name",
    "probe_apple_mlx_gpu",
    "probe_nvidia_gpus",
    "probe_torch_gpus",
    "query_nvidia_compute_processes",
    "sanitize_hardware_label",
]
