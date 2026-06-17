"""TRL SFTTrainer builder for supervised fine-tuning."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_sft_trainer(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    *,
    training_args_dict: dict[str, Any],
    max_seq_length: int,
    packing: bool = False,
    dataset_text_field: str | None = None,
    data_collator=None,
):
    """Create TRL SFTTrainer when available; falls back to HF Trainer."""
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        return _fallback_trainer(model, tokenizer, train_ds, eval_ds, training_args_dict, data_collator)

    cfg_kwargs = dict(training_args_dict)
    cfg_kwargs.setdefault("max_seq_length", max_seq_length)
    if packing:
        cfg_kwargs["packing"] = True
    if dataset_text_field:
        cfg_kwargs["dataset_text_field"] = dataset_text_field

    args = SFTConfig(**cfg_kwargs)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
    }
    if eval_ds is not None:
        kwargs["eval_dataset"] = eval_ds
    if data_collator is not None:
        kwargs["data_collator"] = data_collator

    logger.info("Using TRL SFTTrainer (max_seq_length=%d, packing=%s)", max_seq_length, packing)
    try:
        kwargs["processing_class"] = tokenizer
        return SFTTrainer(**kwargs)
    except TypeError:
        kwargs.pop("processing_class", None)
        kwargs["tokenizer"] = tokenizer
        return SFTTrainer(**kwargs)


def _fallback_trainer(model, tokenizer, train_ds, eval_ds, training_args_dict, data_collator):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(**training_args_dict)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
