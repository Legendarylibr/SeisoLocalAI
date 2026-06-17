"""Robust Hugging Face training — QLoRA, SFT, embeddings."""

from __future__ import annotations

import gc
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.kernels.hooks import apply_training_kernels
from seiso.kernels.lifecycle import release_training_memory
from seiso.models.fast_model import FastModel
from seiso.security import resolve_data_dir
from seiso.training.config import QuantMode, TrainConfig, TrainMethod
from seiso.training.datasets import (
    format_dataset_text,
    load_training_dataset,
    prepare_tokenized_dataset,
)
from seiso.training.multi_gpu import configure_training_args, detect_gpus, gpu_stats
from seiso.training.sft import build_sft_trainer

logger = logging.getLogger(__name__)


class SeisoTrainer:
    def __init__(self, config: TrainConfig, *, on_metric: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._kernel_meta: dict = {}
        self._fast: FastModel | None = None
        self._on_metric = on_metric
        self._metrics_callback = None

    def run(self) -> Path:
        cfg = self.config
        layout = detect_gpus()
        multi_gpu = bool(cfg.multi_gpu or cfg.extra.get("multi_gpu", False)) and layout.use_ddp
        use_triton = cfg.use_triton
        use_fused_ce = cfg.use_fused_ce

        logger.info(
            "Training %s | method=%s quant=%s | world_size=%d",
            cfg.model_id,
            cfg.method.value,
            cfg.quant.value,
            layout.world_size,
        )

        if cfg.method == TrainMethod.EMBEDDING:
            out = self._train_embedding()
            self._cleanup_gpu(None)
            return out

        if cfg.method not in (TrainMethod.LORA, TrainMethod.FULL):
            raise ValueError(f"Unsupported training method: {cfg.method.value}")

        model, tokenizer = self._load_model()
        if use_triton:
            self._kernel_meta = apply_training_kernels(model, use_cuda=True, use_triton=True)
            self._kernel_meta["fused_ce"] = use_fused_ce

        if cfg.method == TrainMethod.LORA:
            model = self._apply_lora(model)
        elif cfg.method == TrainMethod.FULL and cfg.quant in (QuantMode.INT4, QuantMode.INT8):
            logger.warning("Full fine-tune with quantization — consider LoRA for memory efficiency")

        FastModel.for_training(model)

        ds_fmt = cfg.dataset_format
        sandbox = resolve_data_dir()
        raw_ds = load_training_dataset(cfg.dataset, sandbox_root=sandbox)
        if cfg.eval_split_ratio > 0 and len(raw_ds) > 10:
            split = raw_ds.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds, eval_ds = raw_ds, None

        detected_fmt = ds_fmt
        if cfg.packing and cfg.train_on_responses_only:
            logger.warning("Sequence packing disables train-on-responses-only masking")
        use_sft_text = cfg.packing
        data_collator = None
        dataset_text_field = None

        if use_sft_text:
            train_ds, detected_fmt = format_dataset_text(train_ds, tokenizer, ds_fmt)
            dataset_text_field = "text"
            tokenized_eval = None
            if eval_ds is not None:
                tokenized_eval, _ = format_dataset_text(eval_ds, tokenizer, detected_fmt)
                eval_ds = tokenized_eval
        else:
            train_ds, detected_fmt = prepare_tokenized_dataset(
                train_ds,
                tokenizer,
                max_seq_length=cfg.max_seq_length,
                dataset_format=ds_fmt,
                train_on_inputs=not cfg.train_on_responses_only,
            )
            tokenized_eval = None
            if eval_ds is not None:
                tokenized_eval, _ = prepare_tokenized_dataset(
                    eval_ds,
                    tokenizer,
                    max_seq_length=cfg.max_seq_length,
                    dataset_format=detected_fmt,
                    train_on_inputs=not cfg.train_on_responses_only,
                )
                eval_ds = tokenized_eval
            data_collator = self._make_collator(tokenizer)

        from seiso.training.metrics import build_metrics_callback

        emit_stdout = multi_gpu or bool(os.environ.get("SEISO_EMIT_METRICS_STDOUT"))
        metrics_cb = build_metrics_callback(
            cfg.output_dir,
            on_metric=self._on_metric,
            emit_stdout=emit_stdout,
        )
        self._metrics_callback = metrics_cb

        trainer = self._build_trainer(
            model,
            tokenizer,
            train_ds,
            eval_ds,
            layout,
            multi_gpu,
            data_collator=data_collator,
            dataset_text_field=dataset_text_field,
            callbacks=[metrics_cb],
        )

        if cfg.resume_from:
            trainer.train(resume_from_checkpoint=str(cfg.resume_from))
        else:
            trainer.train()

        is_main = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
        if not is_main:
            release_training_memory(model)
            logger.info("Non-main rank finished training (no checkpoint write)")
            return cfg.output_dir

        out = cfg.output_dir / f"checkpoint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        self._write_manifest(out, layout, multi_gpu, detected_fmt.value)
        release_training_memory(model)
        logger.info("Training complete: %s", out)
        return out

    def _load_model(self):
        cfg = self.config
        load_4bit = cfg.quant == QuantMode.INT4
        load_8bit = cfg.quant == QuantMode.INT8

        self._fast = FastModel.from_pretrained(
            cfg.model_id,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            dtype="float16" if cfg.quant == QuantMode.INT16 else None,
        )
        model, tokenizer = self._fast.model, self._fast.tokenizer

        if cfg.quant == QuantMode.INT4:
            try:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=cfg.gradient_checkpointing
                )
                self._fast.model = model
            except ImportError:
                pass

        return model, tokenizer

    def _apply_lora(self, model):
        cfg = self.config
        model = FastModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            random_state=cfg.seed,
            use_rslora=cfg.use_rslora,
            model_id=cfg.model_id,
        )
        if self._fast:
            self._fast.model = model
        return model

    @staticmethod
    def _make_collator(tokenizer):
        from dataclasses import dataclass

        import torch

        @dataclass
        class SFTCollator:
            tokenizer: Any

            def __call__(self, features):
                label_rows = [f.pop("labels") for f in features] if features and "labels" in features[0] else None
                batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
                if label_rows:
                    max_len = batch["input_ids"].shape[1]
                    padded = []
                    for lab in label_rows:
                        row = list(lab) + [-100] * (max_len - len(lab))
                        padded.append(row[:max_len])
                    batch["labels"] = torch.tensor(padded, dtype=torch.long)
                else:
                    labels = batch["input_ids"].clone()
                    labels[batch["attention_mask"] == 0] = -100
                    batch["labels"] = labels
                return batch

        return SFTCollator(tokenizer=tokenizer)

    def _build_trainer(
        self,
        model,
        tokenizer,
        train_ds,
        eval_ds,
        layout,
        multi_gpu,
        *,
        data_collator=None,
        dataset_text_field: str | None = None,
        callbacks=None,
    ):
        cfg = self.config
        import torch

        use_bf16 = False
        use_fp16 = False
        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported() and cfg.quant != QuantMode.INT16
            use_fp16 = cfg.quant == QuantMode.INT16 and not use_bf16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            use_fp16 = cfg.quant == QuantMode.INT16
        optim = "paged_adamw_8bit" if cfg.quant == QuantMode.INT4 else "adamw_torch"

        base = {
            "output_dir": str(cfg.output_dir),
            "num_train_epochs": cfg.epochs,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "logging_steps": cfg.logging_steps,
            "save_steps": cfg.save_steps,
            "eval_strategy": "steps" if eval_ds is not None else "no",
            "eval_steps": cfg.eval_steps or cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "seed": cfg.seed,
            "report_to": "none",
            "optim": optim,
            "lr_scheduler_type": cfg.lr_scheduler,
            "dataloader_pin_memory": True,
            "remove_unused_columns": False if dataset_text_field else True,
            "load_best_model_at_end": eval_ds is not None,
        }
        args_dict = configure_training_args(base, layout, multi_gpu)
        return build_sft_trainer(
            model,
            tokenizer,
            train_ds,
            eval_ds,
            training_args_dict=args_dict,
            max_seq_length=cfg.max_seq_length,
            packing=cfg.packing,
            dataset_text_field=dataset_text_field,
            data_collator=data_collator,
            use_fused_ce=cfg.use_fused_ce,
            callbacks=callbacks,
        )

    def _train_embedding(self) -> Path:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        cfg = self.config
        from seiso.security import resolve_data_dir

        raw = load_training_dataset(cfg.dataset, sandbox_root=resolve_data_dir())
        examples = []
        for row in raw:
            anchor = row.get("anchor") or row.get("query") or row.get("text", "")
            positive = row.get("positive") or row.get("answer") or row.get("output", "")
            if anchor and positive:
                examples.append(InputExample(texts=[anchor, positive]))

        if not examples:
            raise ValueError("Embedding dataset needs anchor/query + positive/answer columns")

        model = SentenceTransformer(cfg.model_id)
        loader = DataLoader(examples, shuffle=True, batch_size=cfg.batch_size)
        loss = losses.MultipleNegativesRankingLoss(model)
        out = cfg.output_dir / f"embed-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=cfg.epochs,
            warmup_steps=int(len(loader) * cfg.warmup_ratio),
            output_path=str(out),
            show_progress_bar=True,
        )
        return out

    @staticmethod
    def _cleanup_gpu(model) -> None:
        try:
            import torch

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            gc.collect()

    def _write_manifest(self, out: Path, layout, multi_gpu: bool, dataset_format: str) -> None:
        manifest = {
            "model_id": self.config.model_id,
            "method": self.config.method.value,
            "quant": self.config.quant.value,
            "dataset_format": dataset_format,
            "train_on_responses_only": self.config.train_on_responses_only,
            "multi_gpu": multi_gpu,
            "world_size": layout.world_size,
            "kernels": self._kernel_meta,
            "gpu_stats": gpu_stats(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (out / "seiso_manifest.json").write_text(json.dumps(manifest, indent=2))
