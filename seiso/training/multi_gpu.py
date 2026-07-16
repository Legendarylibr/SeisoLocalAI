"""Distributed training coordination via Hugging Face Accelerate."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GpuLayout:
    world_size: int
    local_rank: int
    device: str
    use_ddp: bool
    device_count: int = 1


@dataclass(frozen=True)
class DistributedPlan:
    enabled: bool
    strategy: str
    nproc_per_node: int
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    reason: str = ""

    @property
    def world_size(self) -> int:
        return self.nproc_per_node * self.nnodes if self.enabled else 1


def detect_training_layout() -> GpuLayout:
    """Detect GPUs and distributed rank from Accelerate/PyTorch env vars."""
    try:
        import torch
    except ImportError:
        return GpuLayout(
            world_size=1, local_rank=0, device="cpu", use_ddp=False, device_count=0
        )

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_count == 0:
        return GpuLayout(
            world_size=1, local_rank=0, device="cpu", use_ddp=False, device_count=0
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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


def distributed_requested(config: Any) -> bool:
    """Return whether config asks Seiso to launch or honor distributed training."""
    strategy = str(getattr(config, "distributed_strategy", "auto")).lower()
    if strategy == "none":
        return False
    if strategy == "ddp":
        return True
    return bool(
        getattr(config, "multi_gpu", False)
        or getattr(config, "extra", {}).get("multi_gpu", False)
    )


def resolve_distributed_plan(
    config: Any,
    layout: GpuLayout | None = None,
) -> DistributedPlan:
    """Resolve high-level distributed settings into a concrete Accelerate plan."""
    layout = layout or detect_training_layout()
    strategy = str(getattr(config, "distributed_strategy", "auto")).lower()
    requested = distributed_requested(config)
    nnodes = int(getattr(config, "distributed_num_nodes", 1) or 1)
    node_rank = int(getattr(config, "distributed_node_rank", 0) or 0)
    nproc = getattr(config, "distributed_nproc_per_node", None)
    nproc_per_node = int(nproc or max(layout.device_count, 1))

    if strategy == "none" or not requested:
        return DistributedPlan(
            enabled=False,
            strategy="none",
            nproc_per_node=1,
            reason="distributed training not requested",
        )
    if strategy not in {"auto", "ddp"}:
        raise ValueError(f"Unsupported distributed strategy: {strategy}")
    if node_rank >= nnodes:
        raise ValueError(
            "distributed_node_rank must be less than distributed_num_nodes"
        )
    if layout.device_count <= 0:
        return DistributedPlan(
            enabled=False,
            strategy="none",
            nproc_per_node=1,
            reason="no CUDA/ROCm training GPUs detected",
        )
    if nproc_per_node > layout.device_count:
        raise ValueError(
            f"distributed_nproc_per_node={nproc_per_node} exceeds visible GPU count {layout.device_count}"
        )
    if nnodes == 1 and nproc_per_node <= 1:
        return DistributedPlan(
            enabled=False,
            strategy="none",
            nproc_per_node=1,
            reason="only one local training process requested",
        )

    return DistributedPlan(
        enabled=True,
        strategy="ddp",
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        master_addr=str(getattr(config, "distributed_master_addr", "127.0.0.1")),
        master_port=int(getattr(config, "distributed_master_port", 29500)),
    )


def configure_training_args(
    base_args: dict, layout: GpuLayout, multi_gpu: bool
) -> dict:
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
            "Multi-GPU DDP enabled: world_size=%d rank=%d",
            layout.world_size,
            layout.local_rank,
        )
    else:
        args["dataloader_pin_memory"] = training_pin_memory()
    return args


def configure_distributed_training_args(
    base_args: dict,
    layout: GpuLayout,
    config: Any,
    enabled: bool,
) -> dict:
    """Merge configured DDP settings into HuggingFace TrainingArguments."""
    args = configure_training_args(base_args, layout, enabled)
    if enabled and layout.use_ddp:
        ddp_backend = getattr(config, "ddp_backend", None)
        if ddp_backend:
            args["ddp_backend"] = str(ddp_backend)
        args["ddp_find_unused_parameters"] = bool(
            getattr(config, "ddp_find_unused_parameters", False)
        )
        # Expert LoRA leaves unused params each step — force find_unused for MoE.
        if getattr(config, "moe_finetune", False) or getattr(config, "is_moe", False):
            args["ddp_find_unused_parameters"] = True
        model_id = str(getattr(config, "model_id", "") or "")
        if "moe" in model_id.lower() or "mixtral" in model_id.lower():
            args["ddp_find_unused_parameters"] = True
    return args


def launch_worker_command(
    config_path: str,
    nproc: int | DistributedPlan,
) -> list[str]:
    """Build an Accelerate launch command for distributed worker subprocesses."""
    if isinstance(nproc, DistributedPlan):
        plan = nproc
        nproc_value = plan.nproc_per_node
    else:
        plan = None
        nproc_value = int(nproc)

    cmd = [
        "accelerate",
        "launch",
        "--multi_gpu",
        f"--num_processes={nproc_value * (plan.nnodes if plan else 1)}",
    ]
    if plan and plan.nnodes > 1:
        cmd.extend(
            [
                f"--num_machines={plan.nnodes}",
                f"--machine_rank={plan.node_rank}",
                f"--main_process_ip={plan.master_addr}",
                f"--main_process_port={plan.master_port}",
            ]
        )
    cmd.extend(
        [
            "--module",
            "seiso.training.worker",
            "--config",
            config_path,
        ]
    )
    return [
        *cmd,
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
                    "utilization_pct": round(
                        100 * allocated / max(props.total_memory, 1), 1
                    ),
                }
            )
    except ImportError:
        pass
    return sanitize_gpu_stats(stats)
