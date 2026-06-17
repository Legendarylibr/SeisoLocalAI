"""TRL SFTTrainer builder for supervised fine-tuning."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from trl import SFTConfig
    from trl import SFTTrainer as _SFTTrainer
except ImportError:
    _SFTTrainer = None  # type: ignore[misc, assignment]
    SFTConfig = None  # type: ignore[misc, assignment]


def _fused_compute_loss(trainer, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.get("labels")
    if labels is None:
        return trainer._seiso_super_compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    outputs = model(**model_inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    if not logits.is_cuda:
        return trainer._seiso_super_compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    from seiso.kernels.loss import fused_cross_entropy_loss, shift_logits_and_labels

    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    loss = fused_cross_entropy_loss(shift_logits, shift_labels)

    if num_items_in_batch is not None:
        loss = loss * (shift_labels != -100).sum() / num_items_in_batch

    return (loss, outputs) if return_outputs else loss


if _SFTTrainer is not None:

    class FusedSFTTrainer(_SFTTrainer):
        def __init__(self, *args, use_fused_ce: bool = True, **kwargs):
            self._seiso_use_fused_ce = use_fused_ce
            self._seiso_super_compute_loss = super().compute_loss
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not self._seiso_use_fused_ce:
                return self._seiso_super_compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            return _fused_compute_loss(
                self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

else:
    FusedSFTTrainer = None  # type: ignore[misc, assignment]


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
    use_fused_ce: bool = True,
    callbacks=None,
):
    """Create TRL SFTTrainer when available; falls back to HF Trainer."""
    if _SFTTrainer is None or SFTConfig is None:
        return _fallback_trainer(
            model, tokenizer, train_ds, eval_ds, training_args_dict, data_collator, use_fused_ce, callbacks
        )

    cfg_kwargs = dict(training_args_dict)
    cfg_kwargs.setdefault("max_seq_length", max_seq_length)
    if packing:
        cfg_kwargs["packing"] = True
    if dataset_text_field:
        cfg_kwargs["dataset_text_field"] = dataset_text_field

    args = SFTConfig(**cfg_kwargs)
    trainer_cls = FusedSFTTrainer if use_fused_ce else _SFTTrainer
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
    }
    if use_fused_ce:
        kwargs["use_fused_ce"] = True
    if eval_ds is not None:
        kwargs["eval_dataset"] = eval_ds
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    if callbacks:
        kwargs["callbacks"] = callbacks

    logger.info(
        "Using %s (max_seq_length=%d, packing=%s, fused_ce=%s)",
        trainer_cls.__name__,
        max_seq_length,
        packing,
        use_fused_ce,
    )
    try:
        kwargs["processing_class"] = tokenizer
        return trainer_cls(**kwargs)
    except TypeError:
        kwargs.pop("processing_class", None)
        kwargs["tokenizer"] = tokenizer
        return trainer_cls(**kwargs)


def _fallback_trainer(model, tokenizer, train_ds, eval_ds, training_args_dict, data_collator, use_fused_ce, callbacks):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(**training_args_dict)

    if not use_fused_ce:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    class _FusedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if not logits.is_cuda:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            from seiso.kernels.loss import fused_cross_entropy_loss, shift_logits_and_labels

            shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
            loss = fused_cross_entropy_loss(shift_logits, shift_labels)
            return (loss, outputs) if return_outputs else loss

    return _FusedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )
