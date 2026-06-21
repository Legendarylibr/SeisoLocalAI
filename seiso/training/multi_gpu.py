"""Multi-GPU training coordination via Accelerate DDP."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuLayout:
    world_size: int
    local_rank: int
    device: str
    use_ddp: bool
    device_count: int = 1


def detect_gpus() -> GpuLayout:
    """Detect GPUs and distributed rank from torchrun/accelerate env vars."""
    try:
        import torch
    except ImportError:
        return GpuLayout(world_size=1, local_rank=0, device="cpu", use_ddp=False, device_count=0)

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count == 0:
        return GpuLayout(world_size=1, local_rank=0, device="cpu", use_ddp=False, device_count=0)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    device = f"cuda:{local_rank}" if use_ddp else "cuda"
    return GpuLayout(
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        use_ddp=use_ddp,
        device_count=device_count,
    )


def configure_training_args(base_args: dict, layout: GpuLayout, multi_gpu: bool) -> dict:
    """Merge DDP settings into HuggingFace TrainingArguments dict."""
    from seiso.memory.protection import training_pin_memory

    args = dict(base_args)
    if multi_gpu and layout.use_ddp:
        args.update(
            {
                "local_rank": layout.local_rank,
                "ddp_find_unused_parameters": False,
                "dataloader_pin_memory": training_pin_memory(),
            }
        )
        logger.info(
            "Multi-GPU DDP enabled: world_size=%d rank=%d", layout.world_size, layout.local_rank
        )
    else:
        args["dataloader_pin_memory"] = training_pin_memory()
    return args


def launch_worker_command(config_path: str, nproc: int) -> list[str]:
    """Build torchrun command for Forge worker subprocess."""
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "-m",
        "seiso.training.worker",
        "--config",
        config_path,
    ]


def gpu_stats() -> list[dict]:
    """Return per-GPU utilization snapshot for SSE metrics (no device names)."""
    from seiso.security.hardware_privacy import sanitize_gpu_stats

    stats: list[dict] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return stats
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            stats.append(
                {
                    "index": i,
                    "total_bytes": props.total_memory,
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "utilization_pct": round(100 * allocated / max(props.total_memory, 1), 1),
                }
            )
    except ImportError:
        pass
    return sanitize_gpu_stats(stats)
