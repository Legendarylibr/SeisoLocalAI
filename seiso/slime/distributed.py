"""Distributed (DDP) helpers for multi-GPU slime training."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.types import _DistributedSlimeContext


def _iter_distributed_sample_batches(
    config: SingleGpuSlimeConfig,
    rng: random.Random,
    dist_ctx: _DistributedSlimeContext,
    torch,
) -> Iterable[list[dict[str, Any]]]:
    from seiso.slime import trainer as _trainer

    _sampling_batch_size = _trainer._sampling_batch_size
    _count_rank_samples = _trainer._count_rank_samples
    _iter_sample_batches = _trainer._iter_sample_batches
    _batched_records = _trainer._batched_records
    _iter_shuffled_samples = _trainer._iter_shuffled_samples
    sample_batch_size = _sampling_batch_size(config)
    if config.balance_data:
        rank_samples = _balanced_rank_samples(config, dist_ctx, rng)
        local_batches = len(rank_samples) // sample_batch_size
        min_batches = _distributed_min_int(local_batches, torch, dist_ctx)
        if min_batches < 1:
            raise ValueError(
                "not enough balanced samples for distributed slime; "
                "need at least one sampling batch per rank"
            )
        yield from _batched_records(
            rank_samples[: min_batches * sample_batch_size], sample_batch_size
        )
        return

    local_count = _count_rank_samples(config, dist_ctx)
    local_batches = local_count // sample_batch_size
    min_batches = _distributed_min_int(local_batches, torch, dist_ctx)
    if min_batches < 1:
        raise ValueError(
            "not enough sharded samples for distributed slime; "
            "need at least one sampling batch per rank"
        )
    target_samples = min_batches * sample_batch_size
    yield from _batched_records(
        _iter_shuffled_samples(config, rng, dist_ctx, target_samples),
        sample_batch_size,
    )


def _balanced_rank_samples(
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
    rng: random.Random,
) -> list[dict[str, Any]]:
    from seiso.slime import trainer as _trainer

    _limited_samples = _trainer._limited_samples
    _sample_work_estimate = _trainer._sample_work_estimate
    samples = list(_limited_samples(config))
    rng.shuffle(samples)
    rank_loads = [0] * dist_ctx.world_size
    rank_samples: list[list[dict[str, Any]]] = [[] for _ in range(dist_ctx.world_size)]
    for sample in sorted(
        samples,
        key=lambda item: _sample_work_estimate(item, config),
        reverse=True,
    ):
        rank = min(range(dist_ctx.world_size), key=lambda idx: rank_loads[idx])
        rank_samples[rank].append(sample)
        rank_loads[rank] += _sample_work_estimate(sample, config)
    return rank_samples[dist_ctx.rank]


def _count_rank_samples(
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
) -> int:
    from seiso.slime import trainer as _trainer

    _limited_samples = _trainer._limited_samples
    count = 0
    for sample_index, _sample in enumerate(_limited_samples(config)):
        if sample_index % dist_ctx.world_size == dist_ctx.rank:
            count += 1
    return count


def _save_distributed(
    model,
    tokenizer,
    output_dir: Path,
    dist_ctx: _DistributedSlimeContext,
) -> None:
    from seiso.slime import trainer as _trainer

    if dist_ctx.is_main:
        _trainer._save(model, tokenizer, output_dir)
    _distributed_barrier(dist_ctx)


def _distributed_context(torch, config: SingleGpuSlimeConfig) -> _DistributedSlimeContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if world_size <= 1:
        return _DistributedSlimeContext(
            enabled=False,
            world_size=1,
            rank=0,
            local_rank=0,
            device=config.device,
        )
    if not _is_cuda_device(config.device):
        device = config.device
        backend = "gloo"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed slime requires CUDA when device='cuda'")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    dist = torch.distributed
    if not dist.is_available():
        raise RuntimeError("distributed slime requires torch.distributed")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return _DistributedSlimeContext(
        enabled=True,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )


def _maybe_wrap_ddp(
    model,
    torch,
    dist_ctx: _DistributedSlimeContext,
    config: SingleGpuSlimeConfig,
):
    if not dist_ctx.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    kwargs: dict[str, Any] = {"find_unused_parameters": False}
    if _is_cuda_device(config.device):
        kwargs["device_ids"] = [dist_ctx.local_rank]
        kwargs["output_device"] = dist_ctx.local_rank
    return DistributedDataParallel(model, **kwargs)


def _distributed_min_int(
    value: int,
    torch,
    dist_ctx: _DistributedSlimeContext,
) -> int:
    if not dist_ctx.enabled:
        return value
    tensor = torch.tensor([int(value)], device=dist_ctx.device, dtype=torch.long)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return int(tensor.item())


def _reduce_stats(
    stats: dict[str, float],
    torch,
    dist_ctx: _DistributedSlimeContext,
) -> dict[str, float]:
    if not dist_ctx.enabled:
        return stats
    reduced = dict(stats)
    sum_keys = {
        "rollout_groups_total",
        "rollout_groups_kept",
        "rollout_groups_trained",
        "dynamic_filtered_groups",
        "rollout_status_stop",
        "rollout_status_length",
        "rollout_status_empty",
    }
    mean_keys = [key for key in reduced if key not in {"reward_max", *sum_keys}]
    if mean_keys:
        values = torch.tensor(
            [float(reduced[key]) for key in mean_keys],
            device=dist_ctx.device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values /= dist_ctx.world_size
        for key, value in zip(mean_keys, values.tolist(), strict=False):
            reduced[key] = float(value)
    if active_sum_keys := [key for key in sum_keys if key in reduced]:
        values = torch.tensor(
            [float(reduced[key]) for key in active_sum_keys],
            device=dist_ctx.device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        for key, value in zip(active_sum_keys, values.tolist(), strict=False):
            reduced[key] = float(value)
    max_tensor = torch.tensor(
        [float(reduced.get("reward_max", float("-inf")))],
        device=dist_ctx.device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)
    reduced["reward_max"] = float(max_tensor.item())
    return reduced


def _distributed_barrier(dist_ctx: _DistributedSlimeContext | None) -> None:
    if not dist_ctx or not dist_ctx.enabled:
        return
    import torch

    torch.distributed.barrier()


def _rank_verifier_path(
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
) -> Path:
    base = config.output_dir / config.verifier_data_file
    if not dist_ctx.enabled:
        return base
    return base.with_name(f"{base.stem}.rank{dist_ctx.rank}{base.suffix}")


def _unwrap_model(model):
    return getattr(model, "module", model)


def _generation_model(model):
    return _unwrap_model(model)


def _is_cuda_device(device: str) -> bool:
    return device == "cuda" or device.startswith("cuda:")


def _cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"unsupported CUDA device {device!r}") from exc
