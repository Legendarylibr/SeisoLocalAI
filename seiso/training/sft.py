"""TRL SFTTrainer builder for supervised fine-tuning."""

from __future__ import annotations

import logging
from typing import Any

from seiso.env import configure_transformers_env

configure_transformers_env()

logger = logging.getLogger(__name__)

try:
    from trl import SFTConfig
    from trl import SFTTrainer as _SFTTrainer
except ImportError:
    _SFTTrainer = None  # type: ignore[misc, assignment]
    SFTConfig = None  # type: ignore[misc, assignment]


def _compute_fused_or_delegate_loss(
    delegate,
    model,
    inputs,
    *,
    return_outputs=False,
    num_items_in_batch=None,
):
    labels = inputs.get("labels")
    if labels is None:
        return delegate(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    outputs = model(**model_inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    if not logits.is_cuda:
        return delegate(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    from seiso.kernels.loss import fused_cross_entropy_loss, shift_logits_and_labels

    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    loss = fused_cross_entropy_loss(shift_logits, shift_labels)

    if num_items_in_batch is not None:
        loss = loss * (shift_labels != -100).sum() / num_items_in_batch

    return (loss, outputs) if return_outputs else loss


if _SFTTrainer is not None:
    from seiso.kernels.cuda_graphs import CudaGraphTrainerMixin

    class FusedSFTTrainer(CudaGraphTrainerMixin, _SFTTrainer):
        def __init__(
            self,
            *args,
            use_fused_ce: bool = True,
            use_cuda_graphs: bool = False,
            **kwargs,
        ):
            self._seiso_use_fused_ce = use_fused_ce
            self._seiso_super_compute_loss = super().compute_loss
            super().__init__(*args, use_cuda_graphs=use_cuda_graphs, **kwargs)

        def training_step(self, model, inputs, num_items_in_batch=None):
            with self.maybe_activation_offload_context:
                return CudaGraphTrainerMixin.training_step(
                    self, model, inputs, num_items_in_batch=num_items_in_batch
                )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not self._seiso_use_fused_ce:
                return self._seiso_super_compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            return _compute_fused_or_delegate_loss(
                self._seiso_super_compute_loss,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

else:
    FusedSFTTrainer = None  # type: ignore[misc, assignment]


def _sft_max_length_key() -> str:
    """TRL 1.x uses max_length; older releases used max_seq_length."""
    if SFTConfig is None:
        return "max_seq_length"
    import inspect

    params = inspect.signature(SFTConfig.__init__).parameters
    return "max_length" if "max_length" in params else "max_seq_length"


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
    use_cuda_graphs: bool = False,
    callbacks=None,
    sft_extras: dict[str, Any] | None = None,
):
    """Create TRL SFTTrainer when available; falls back to HF Trainer."""
    if _SFTTrainer is None or SFTConfig is None:
        return _fallback_trainer(
            model,
            tokenizer,
            train_ds,
            eval_ds,
            training_args_dict,
            data_collator,
            use_fused_ce,
            use_cuda_graphs,
            callbacks,
        )

    cfg_kwargs = dict(training_args_dict)
    max_key = _sft_max_length_key()
    cfg_kwargs.pop("max_seq_length", None)
    cfg_kwargs.setdefault(max_key, max_seq_length)
    if packing:
        cfg_kwargs["packing"] = True
    if dataset_text_field:
        cfg_kwargs["dataset_text_field"] = dataset_text_field
    # padding_free is passed through to SFTConfig when present in args dict
    if training_args_dict.get("padding_free"):
        cfg_kwargs["padding_free"] = True
    if sft_extras:
        for key, value in sft_extras.items():
            if value is not None:
                cfg_kwargs[key] = value

    args = SFTConfig(**cfg_kwargs)
    # FusedSFTTrainer owns seiso-only kwargs; plain TRL SFTTrainer rejects them.
    use_fused_cls = bool(FusedSFTTrainer is not None and (use_fused_ce or use_cuda_graphs))
    trainer_cls = FusedSFTTrainer if use_fused_cls else _SFTTrainer
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
    }
    if use_fused_cls:
        kwargs["use_fused_ce"] = bool(use_fused_ce)
        kwargs["use_cuda_graphs"] = bool(use_cuda_graphs)
    if eval_ds is not None:
        kwargs["eval_dataset"] = eval_ds
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    if callbacks:
        kwargs["callbacks"] = callbacks

    logger.info(
        "Using %s (max_seq_length=%d, packing=%s, fused_ce=%s, cuda_graphs=%s)",
        trainer_cls.__name__,
        max_seq_length,
        packing,
        use_fused_ce,
        use_cuda_graphs,
    )
    try:
        kwargs["processing_class"] = tokenizer
        return trainer_cls(**kwargs)
    except TypeError:
        # Older TRL uses tokenizer= instead of processing_class=.
        # Keep seiso-only kwargs on FusedSFTTrainer (defaults would flip fused_ce on).
        kwargs.pop("processing_class", None)
        kwargs["tokenizer"] = tokenizer
        if not use_fused_cls:
            kwargs.pop("use_fused_ce", None)
            kwargs.pop("use_cuda_graphs", None)
        return trainer_cls(**kwargs)


def _fallback_trainer(
    model,
    tokenizer,
    train_ds,
    eval_ds,
    training_args_dict,
    data_collator,
    use_fused_ce,
    use_cuda_graphs,
    callbacks,
):
    from transformers import Trainer, TrainingArguments

    # Filter SFTConfig-only params that TrainingArguments doesn't accept
    sft_only_keys = {
        "padding_free",
        "packing",
        "dataset_text_field",
        "max_length",
        "max_seq_length",
        "packing_strategy",
        "eval_packing",
        "assistant_only_loss",
        "completion_only_loss",
        "dataset_kwargs",
        "dataset_num_proc",
        "chat_template_path",
        "eos_token",
        "pad_token",
        "pad_to_multiple_of",
        "truncation_mode",
        "shuffle_dataset",
        "activation_offloading",
        "loss_type",
        "model_init_kwargs",
    }
    filtered_args = {k: v for k, v in training_args_dict.items() if k not in sft_only_keys}
    args = TrainingArguments(**filtered_args)

    if not use_fused_ce:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    from seiso.kernels.cuda_graphs import CudaGraphTrainerMixin

    class _FusedTrainer(CudaGraphTrainerMixin, Trainer):
        def __init__(self, *args, **kwargs):
            use_cuda_graphs = kwargs.pop("use_cuda_graphs", False)
            super().__init__(*args, use_cuda_graphs=use_cuda_graphs, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return _compute_fused_or_delegate_loss(
                super().compute_loss,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    return _FusedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        use_cuda_graphs=use_cuda_graphs,
        callbacks=callbacks,
    )
