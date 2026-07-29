"""Distributed training coordination via Hugging Face Accelerate."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Env key set by seiso.training.worker before run_training (Accelerate child).
SEISO_DISTRIBUTED_WORKER_ENV = "SEISO_DISTRIBUTED_WORKER"

# Env keys set by torchrun / Accelerate / elastic launch (not bare WORLD_SIZE).
_DISTRIBUTED_LAUNCH_MARKERS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "TORCHELASTIC_RUN_ID",
)

# Proof we are inside a real launched worker — not a parent with leftover MASTER_*.
_DISTRIBUTED_WORKER_PROOF = (
    SEISO_DISTRIBUTED_WORKER_ENV,
    "TORCHELASTIC_RUN_ID",
    "GROUP_RANK",
)


@dataclass
class GpuLayout:
    world_size: int
    local_rank: int
    device: str
    use_ddp: bool
    device_count: int = 1


@dataclass(frozen=True)
class DistributedEnv:
    """Parsed torchrun/Accelerate process identity (global world size)."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    stale: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1 and not self.stale


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_nonempty(name: str) -> bool:
    return bool(str(os.environ.get(name, "") or "").strip())


def _dist_initialized() -> bool:
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_initialized())
    except ImportError:
        return False


def mark_distributed_worker() -> None:
    """Mark this process as an Accelerate/torchrun training worker."""
    os.environ[SEISO_DISTRIBUTED_WORKER_ENV] = "1"


def resolve_distributed_env(device_count: int | None = None) -> DistributedEnv:
    """Parse distributed env without breaking multi-node or CVD-isolated ranks.

    Stale (ignore → single process) when:
    - ``WORLD_SIZE < 1``
    - ``LOCAL_RANK`` or ``RANK`` unset while ``WORLD_SIZE > 1``
    - no non-empty launch markers while ``WORLD_SIZE > 1``
    - no worker proof (``SEISO_DISTRIBUTED_WORKER`` / elastic / initialized PG)
      — rejects parent shells with leftover ``MASTER_ADDR``
    - ``LOCAL_RANK >=`` visible GPU count
    - ``LOCAL_WORLD_SIZE >`` visible GPU count (when set)

    Global ``WORLD_SIZE > device_count`` is **not** stale — that is normal multi-node
    and per-rank ``CUDA_VISIBLE_DEVICES`` launches.
    """
    if device_count is None:
        try:
            import torch

            device_count = (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            )
        except ImportError:
            device_count = 0

    world_size = _env_int("WORLD_SIZE", 1)
    local_rank_set = "LOCAL_RANK" in os.environ
    rank_set = "RANK" in os.environ
    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", local_rank)
    local_world = _env_int("LOCAL_WORLD_SIZE", 0)

    def _single(*, stale: bool) -> DistributedEnv:
        return DistributedEnv(
            world_size=1,
            rank=0,
            local_rank=0,
            local_world_size=1,
            stale=stale,
        )

    if world_size < 1:
        return _single(stale=True)
    if world_size <= 1:
        return _single(stale=False)

    if not local_rank_set or not rank_set:
        return _single(stale=True)
    if not any(_env_nonempty(k) for k in _DISTRIBUTED_LAUNCH_MARKERS):
        return _single(stale=True)
    # Require nonempty proof — empty GROUP_RANK="" leftovers must not enable DDP.
    worker_proof = any(_env_nonempty(k) for k in _DISTRIBUTED_WORKER_PROOF)
    if not worker_proof and not _dist_initialized():
        return _single(stale=True)
    if device_count > 0 and local_rank >= device_count:
        return _single(stale=True)
    if local_world > 0 and device_count > 0 and local_world > device_count:
        return _single(stale=True)
    if rank < 0 or rank >= world_size:
        return _single(stale=True)

    if local_world <= 0:
        if device_count > 0 and world_size <= device_count:
            local_world = world_size
        elif device_count > 0:
            # Multi-node / CVD: local group size unknown; assume one local peer slot
            # per visible device for metadata only (placement uses local_rank).
            local_world = device_count
        else:
            local_world = world_size

    return DistributedEnv(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        local_world_size=max(1, local_world),
        stale=False,
    )


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

    dist_env = resolve_distributed_env(device_count)
    use_ddp = dist_env.enabled
    device = f"cuda:{dist_env.local_rank}" if use_ddp else "cuda"
    return GpuLayout(
        world_size=dist_env.world_size,
        local_rank=dist_env.local_rank,
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

    if nnodes > 1:
        from seiso.training.access import require_multinode_mesh_agent

        require_multinode_mesh_agent(nnodes)

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


def configure_distributed_training_args(
    base_args: dict,
    layout: GpuLayout,
    config: Any,
    enabled: bool,
) -> dict:
    """Merge configured DDP settings into HuggingFace TrainingArguments."""
    from seiso.memory.protection import training_pin_memory

    args = dict(base_args)
    args["dataloader_pin_memory"] = training_pin_memory()
    if not (enabled and layout.use_ddp):
        return args

    args["local_rank"] = layout.local_rank
    args["ddp_find_unused_parameters"] = bool(
        getattr(config, "ddp_find_unused_parameters", False)
    )
    ddp_backend = getattr(config, "ddp_backend", None)
    if ddp_backend:
        args["ddp_backend"] = str(ddp_backend)
    # Expert LoRA / MoE leave unused params each step.
    # MoE mode is stored on TrainConfig.extra["moe_finetune"], not a top-level attr.
    model_id = str(getattr(config, "model_id", "") or "")
    extra = getattr(config, "extra", None) or {}
    moe_from_extra = bool(extra.get("moe_finetune", False)) if isinstance(extra, dict) else False
    if (
        getattr(config, "moe_finetune", False)
        or moe_from_extra
        or getattr(config, "is_moe", False)
        or "moe" in model_id.lower()
        or "mixtral" in model_id.lower()
    ):
        args["ddp_find_unused_parameters"] = True
    logger.info(
        "Multi-GPU DDP enabled: world_size=%d rank=%d",
        layout.world_size,
        layout.local_rank,
    )
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
    return cmd


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
