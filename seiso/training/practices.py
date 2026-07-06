"""Current supervised fine-tuning defaults (hardware-aware, behavior-preserving)."""

from __future__ import annotations

import os
from typing import Any

from seiso.training.config import DatasetFormat, TrainConfig, TrainMethod


def default_dataset_num_proc(explicit: int | None = None) -> int | None:
    """Parallel dataset preprocessing workers (None = single process)."""
    if explicit is not None and explicit > 0:
        return explicit
    if explicit == 0:
        return None
    cpu = os.cpu_count() or 4
    if cpu <= 2:
        return None
    return min(4, max(1, cpu // 2))


def default_pad_to_multiple_of(
    explicit: int | None, *, cuda_available: bool
) -> int | None:
    """Pad sequence lengths for tensor-core efficiency (8 on CUDA, unpadded on CPU)."""
    if explicit is not None:
        return explicit if explicit > 0 else None
    return 8 if cuda_available else None


def resolve_map_workers(config: TrainConfig) -> int | None:
    return default_dataset_num_proc(config.dataset_num_proc)


def default_dataloader_num_workers(
    explicit: int,
    *,
    cuda_available: bool,
    cpu_count: int | None = None,
) -> int:
    """Training DataLoader workers: parallelize on CUDA, stay single-process on CPU."""
    if explicit > 0:
        return explicit
    if not cuda_available:
        return 0
    cpu = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    if cpu <= 2:
        return 0
    return min(4, max(1, cpu // 2))


def default_dataloader_prefetch_factor(
    explicit: int | None,
    *,
    num_workers: int,
    cuda_available: bool,
) -> int | None:
    """Prefetch batches only when worker processes can overlap CPU input work with GPU compute."""
    if num_workers <= 0:
        return None
    if explicit is not None:
        return explicit if explicit > 0 else None
    return 2 if cuda_available else None


def resolve_dataloader_settings(
    config: TrainConfig,
    *,
    cuda_available: bool,
) -> tuple[int, bool, int | None]:
    """Return DataLoader worker, persistent-worker, and prefetch settings."""
    workers = default_dataloader_num_workers(
        config.dataloader_num_workers,
        cuda_available=cuda_available,
    )
    persistent = workers > 0 and config.dataloader_persistent_workers
    prefetch = default_dataloader_prefetch_factor(
        config.dataloader_prefetch_factor,
        num_workers=workers,
        cuda_available=cuda_available,
    )
    return workers, persistent, prefetch


def learning_rate_for_method(
    method: TrainMethod, *, explicit: float | None = None
) -> float:
    """Method-appropriate LR without model-specific tuning."""
    if explicit is not None and explicit > 0:
        return explicit
    if method == TrainMethod.FULL:
        return 1e-5
    if method == TrainMethod.EMBEDDING:
        return 2e-5
    if method == TrainMethod.SLIME:
        return 5e-6
    return 2e-4


def warmup_ratio_for_corpus(
    sample_count: int, *, explicit: float | None = None
) -> float:
    """Warmup fraction scaled to corpus size (modern linear warmup practice)."""
    if explicit is not None:
        return explicit
    if sample_count < 500:
        return 0.1
    if sample_count < 5_000:
        return 0.05
    return 0.03


def resolve_optimizer(quant: str, *, use_cpu: bool) -> str:
    """AdamW variant selection aligned with QLoRA / fused CUDA training."""
    import logging

    if use_cpu:
        return "adamw_torch"
    if quant in ("4bit", "8bit"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            logging.getLogger(__name__).warning(
                "quant=%s requested but bitsandbytes is not installed; "
                "falling back to adamw_torch_fused optimizer",
                quant,
            )
            return "adamw_torch_fused"
        return "paged_adamw_8bit"
    return "adamw_torch_fused"


def resolve_compute_dtype(
    *, cuda_available: bool, bf16_supported: bool, quant: str
) -> tuple[bool, bool]:
    """Return (use_bf16, use_fp16) for TrainingArguments."""
    if not cuda_available:
        return False, False
    if quant == "16bit":
        return False, True
    if bf16_supported:
        return True, False
    return False, quant == "16bit"


def sft_modern_kwargs(
    config: TrainConfig,
    *,
    train_on_responses_only: bool,
    dataset_format: DatasetFormat,
    cuda_available: bool,
    use_text_field: bool,
) -> dict[str, Any]:
    """TRL SFTConfig knobs that match current HF/TRL best practices."""
    kwargs: dict[str, Any] = {}
    num_proc = resolve_map_workers(config)
    if num_proc:
        kwargs["dataset_num_proc"] = num_proc

    pad = default_pad_to_multiple_of(
        config.pad_to_multiple_of, cuda_available=cuda_available
    )
    if pad:
        kwargs["pad_to_multiple_of"] = pad

    # TRL-native masking when the trainer tokenizes conversational rows itself.
    if use_text_field and config.assistant_only_loss is not False:
        if dataset_format in (
            DatasetFormat.CHAT,
            DatasetFormat.SHAREGPT,
            DatasetFormat.PREFERENCE,
        ):
            kwargs["assistant_only_loss"] = bool(
                config.assistant_only_loss
                if config.assistant_only_loss is not None
                else train_on_responses_only
            )
        elif dataset_format == DatasetFormat.ALPACA and train_on_responses_only:
            kwargs["completion_only_loss"] = True

    return kwargs


def training_args_modern_extras(
    config: TrainConfig, *, eval_enabled: bool
) -> dict[str, Any]:
    """Extra TrainingArguments fields supported on recent transformers releases."""
    extras: dict[str, Any] = {}
    if eval_enabled:
        extras["eval_accumulation_steps"] = max(1, config.gradient_accumulation_steps)
    extras["include_num_input_tokens_seen"] = True
    return extras
