"""Robust Hugging Face training — QLoRA, SFT, embeddings."""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.kernels.hooks import apply_fused_lora_kernels, apply_training_kernels
from seiso.kernels.lifecycle import release_training_memory
from seiso.memory.protection import (
    apply_training_memory_guards,
    apply_training_oom_fallback,
    ensure_load_fits,
    is_oom_error,
    release_cached_memory,
    training_pin_memory,
)
from seiso.models.seiso_model import SeisoModel
from seiso.research.provenance import (
    apply_determinism,
    write_json,
)
from seiso.security.deps import sha256_file
from seiso.training.config import QuantMode, TrainConfig, TrainMethod
from seiso.training.datasets import (
    format_dataset_text,
    load_training_dataset,
    prepare_tokenized_dataset,
)
from seiso.training.multi_gpu import configure_training_args, detect_training_layout
from seiso.training.preprocess import (
    compute_eval_split_size,
    preprocess_training_dataset,
    validate_training_dataset,
)
from seiso.training.sft import build_sft_trainer

logger = logging.getLogger(__name__)


class SeisoTrainer:
    def __init__(
        self,
        config: TrainConfig,
        *,
        on_metric: Callable[[dict[str, Any]], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._kernel_meta: dict = {}
        self._loaded: SeisoModel | None = None
        self._on_metric = on_metric
        self._on_log = on_log
        self._metrics_callback = None

    def _log(self, message: str) -> None:
        logger.info(message)
        if self._on_log:
            try:
                self._on_log(message)
            except Exception:
                logger.exception("on_log callback failed")

    def run(self) -> Path:
        self.config = apply_training_memory_guards(self.config)
        cfg = self.config
        apply_determinism(cfg.seed, deterministic=cfg.deterministic)
        try:
            from seiso.kernels.training_profile import apply_cuda_speedopts

            apply_cuda_speedopts(deterministic=cfg.deterministic)
        except ImportError:
            pass
        write_json(
            cfg.output_dir / "train_config_snapshot.json",
            cfg.model_dump(mode="json"),
        )

        # Validate dataset *first* (before loading potentially huge model weights).
        # This ensures errors about bad formatting are shown before expensive work.
        try:
            val_stats = validate_training_dataset(
                cfg.dataset,
                dataset_format=cfg.dataset_format,
                sandbox_root=cfg.sandbox_root,
                max_check_samples=None,  # full check at actual train time
            )
            self._log(
                f"Dataset validation passed: {val_stats['kept']} usable samples "
                f"(format={val_stats['resolved_format']})"
            )
        except Exception as exc:
            raise ValueError(f"Dataset cannot be normalized for training: {exc}") from exc

        layout = detect_training_layout()
        multi_gpu = bool(cfg.multi_gpu or cfg.extra.get("multi_gpu", False)) and layout.use_ddp
        use_triton = cfg.use_triton
        use_fused_ce = cfg.use_fused_ce
        use_fused_lora = cfg.use_fused_lora

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
            self._kernel_meta = apply_training_kernels(
                model,
                use_cuda=True,
                use_triton=True,
            )
            self._kernel_meta["fused_ce"] = use_fused_ce
            try:
                from seiso.kernels.training_profile import last_cuda_training_profile

                profile = last_cuda_training_profile()
                if profile:
                    self._kernel_meta["cuda_training_profile"] = profile
            except ImportError:
                pass

        if cfg.method == TrainMethod.LORA:
            model = self._apply_lora(model)
            if use_fused_lora:
                lora_meta = apply_fused_lora_kernels(model, max_rank=64)
                self._kernel_meta.update(lora_meta)
        elif cfg.method == TrainMethod.FULL and cfg.quant in (QuantMode.INT4, QuantMode.INT8):
            logger.warning("Full fine-tune with quantization — consider LoRA for memory efficiency")

        SeisoModel.for_training(model)

        ds_fmt = cfg.dataset_format
        raw_ds = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)
        preprocess_stats: dict[str, Any] | None = None
        if cfg.preprocess_dataset:
            raw_ds, preprocess_stats, ds_fmt = preprocess_training_dataset(
                raw_ds,
                dataset_format=ds_fmt,
                deduplicate=cfg.deduplicate_dataset,
                min_chars=cfg.min_sample_chars,
            )
            self._log(
                f"Preprocessed dataset: {preprocess_stats['kept']}/{preprocess_stats['initial_samples']} "
                f"samples kept (format={preprocess_stats['resolved_format']})"
            )
        max_samples = cfg.extra.get("max_samples")
        if isinstance(max_samples, int) and max_samples > 0 and len(raw_ds) > max_samples:
            raw_ds = raw_ds.select(range(max_samples))
            logger.info("Limited dataset to %d samples (max_samples)", max_samples)
        eval_n = 0
        if len(raw_ds) > 10 and (cfg.early_stopping or cfg.eval_split_ratio > 0):
            split_ratio = cfg.eval_split_ratio if cfg.eval_split_ratio > 0 else 0.02
            eval_n = compute_eval_split_size(
                len(raw_ds),
                split_ratio,
                cfg.max_eval_samples,
            )
        if eval_n > 0:
            split = raw_ds.train_test_split(test_size=eval_n, seed=cfg.seed)
            train_ds, eval_ds = split["train"], split["test"]
            self._log(f"Train/eval split: {len(train_ds)} train, {len(eval_ds)} eval (max_eval={cfg.max_eval_samples})")
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

        trainer_callbacks: list[Any] = [metrics_cb]
        if eval_ds is not None and cfg.early_stopping:
            from transformers import EarlyStoppingCallback

            trainer_callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience,
                    early_stopping_threshold=cfg.early_stopping_threshold,
                )
            )

        trainer = self._build_trainer(
            model,
            tokenizer,
            train_ds,
            eval_ds,
            layout,
            multi_gpu,
            data_collator=data_collator,
            dataset_text_field=dataset_text_field,
            callbacks=trainer_callbacks,
        )

        if cfg.resume_from:
            self._train_with_oom_recovery(trainer, resume_from_checkpoint=str(cfg.resume_from))
        else:
            self._train_with_oom_recovery(trainer)

        is_main = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
        if not is_main:
            release_training_memory(model)
            logger.info("Non-main rank finished training (no checkpoint write)")
            return cfg.output_dir

        out = cfg.output_dir / f"checkpoint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        if cfg.method == TrainMethod.LORA:
            self._patch_adapter_metadata(out)
        self._write_manifest(
            out,
            layout,
            multi_gpu,
            detected_fmt.value,
            preprocess_stats=preprocess_stats,
            train_samples=len(train_ds),
            eval_samples=len(eval_ds) if eval_ds is not None else 0,
        )
        release_training_memory(model)
        self._cleanup_gpu(None)
        logger.info("Training complete: %s", out)
        return out

    def _train_with_oom_recovery(
        self, trainer, *, resume_from_checkpoint: str | None = None
    ) -> None:
        attempts = 0
        while True:
            try:
                if resume_from_checkpoint:
                    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                else:
                    trainer.train()
                return
            except Exception as exc:
                if not is_oom_error(exc) or attempts >= 1:
                    raise
                attempts += 1
                release_cached_memory(sync=True)
                self.config = apply_training_oom_fallback(self.config)
                cfg = self.config
                trainer.args.per_device_train_batch_size = cfg.batch_size
                trainer.args.per_device_eval_batch_size = cfg.batch_size
                trainer.args.gradient_accumulation_steps = cfg.gradient_accumulation_steps
                resume_from_checkpoint = None

    def _resolve_load_model_id(self) -> str:
        """Prefer cached local snapshot path for offline merge/export after training."""
        cfg = self.config
        return str(cfg.extra.get("resolved_model_path") or cfg.model_id)

    def _load_model(self):
        cfg = self.config
        load_4bit = cfg.quant == QuantMode.INT4
        load_8bit = cfg.quant == QuantMode.INT8
        model_ref = self._resolve_load_model_id()
        from seiso.models.trainable_snapshot import (
            GGUF_ONLY_REPO_MESSAGE,
            snapshot_has_trainable_weights,
        )

        if Path(model_ref).exists() and not snapshot_has_trainable_weights(Path(model_ref)):
            raise ValueError(GGUF_ONLY_REPO_MESSAGE)
        ensure_load_fits(model_ref, mode="train")

        from seiso.models.hf_env import configure_hf_hub_auth
        from seiso.models.trainable_mirrors import resolve_trainable_hub_id

        hub_token = configure_hf_hub_auth()
        if not Path(model_ref).exists():
            resolved_hub, mirror_note = resolve_trainable_hub_id(
                str(cfg.model_id),
                token=hub_token,
            )
            if mirror_note and "Training with mirror" in mirror_note:
                model_ref = resolved_hub
                cfg.extra.setdefault("training_mirror_note", mirror_note)
                logger.warning(mirror_note)
            elif mirror_note:
                raise ValueError(mirror_note)

        if not Path(model_ref).exists():
            self._log(
                f"Downloading trainable weights for {model_ref} from Hugging Face "
                "(large models can take 10-30 minutes before the first training step)..."
            )
        else:
            self._log(f"Loading trainable weights from {model_ref}")

        try:
            from seiso.models.hub_quant import is_native_hub_quant_model

            trust_remote_code = bool(cfg.extra.get("trust_remote_code", False))
            native_hub_quant = is_native_hub_quant_model(
                str(cfg.model_id),
                trust_remote_code=trust_remote_code,
                peek=True,
            )
            load_dtype = None if native_hub_quant else (
                "float16" if cfg.quant == QuantMode.INT16 else None
            )
            use_flash = (
                not native_hub_quant
                and bool(cfg.extra.get("use_flash_attention", True))
            )
            self._loaded = SeisoModel.from_pretrained(
                model_ref,
                max_seq_length=cfg.max_seq_length,
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                dtype=load_dtype,
                trust_remote_code=bool(cfg.extra.get("trust_remote_code", False)),
                use_flash_attention=use_flash,
            )
        except OSError as exc:
            from seiso.models.hub_errors import format_hub_error

            msg = format_hub_error(exc, context="download", repo_id=str(cfg.model_id))
            raise ValueError(msg) from exc
        model, tokenizer = self._loaded.model, self._loaded.tokenizer

        if cfg.method == TrainMethod.LORA and cfg.quant in (QuantMode.INT4, QuantMode.INT8):
            try:
                from peft import prepare_model_for_kbit_training

                # Let TrainingArguments handle gradient checkpointing via
                # gradient_checkpointing_kwargs (use_reentrant=False) to avoid
                # double-enablement and reentrant overhead.
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=False
                )
                if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                        "use_reentrant": False
                    })
                self._loaded.model = model
            except ImportError:
                logger.warning(
                    "prepare_model_for_kbit_training unavailable — install peft>=0.11; "
                    "QLoRA may train without k-bit preparation"
                )

        return model, tokenizer

    def _apply_lora(self, model):
        cfg = self.config
        model = SeisoModel.attach_lora(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            use_rslora=cfg.use_rslora,
            model_id=self._resolve_load_model_id(),
        )
        if self._loaded:
            self._loaded.model = model
        return model

    @staticmethod
    def _make_collator(tokenizer):
        from dataclasses import dataclass

        import torch

        @dataclass
        class SFTCollator:
            tokenizer: Any

            def __call__(self, features):
                label_rows = (
                    [f.pop("labels") for f in features]
                    if features and "labels" in features[0]
                    else None
                )
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
    ) -> Any:
        cfg = self.config
        import torch

        use_bf16 = False
        use_fp16 = False
        use_cpu = False
        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported() and cfg.quant != QuantMode.INT16
            use_fp16 = cfg.quant == QuantMode.INT16 and not use_bf16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            use_fp16 = cfg.quant == QuantMode.INT16
        else:
            use_cpu = True

        # ── Optimizer selection ──
        # adamw_torch_fused is the fastest on CUDA — it fuses the multi-tensor
        # AdamW update into a single CUDA kernel.  For 4-bit QLoRA, paged_adamw_8bit
        # is used only when bitsandbytes is available, as it pages optimizer states
        # to CPU and prevents OOM on large models with limited VRAM.
        if use_cpu:
            optim = "adamw_torch"
        elif cfg.quant == QuantMode.INT4:
            try:
                import bitsandbytes  # noqa: F401

                optim = "paged_adamw_8bit"
            except ImportError:
                logger.warning(
                    "quant=4bit requested but bitsandbytes is not installed; "
                    "falling back to adamw_torch_fused optimizer"
                )
                optim = "adamw_torch_fused"
        else:
            optim = "adamw_torch_fused"

        # ── Dataloader workers: auto-detect when not explicitly set ──
        num_workers = cfg.dataloader_num_workers
        if num_workers == 0 and torch.cuda.is_available():
            cpu_count = os.cpu_count() or 4
            num_workers = min(4, max(1, cpu_count // 2))
        persistent_workers = num_workers > 0 and cfg.dataloader_persistent_workers

        # ── Gradient checkpointing: use non-reentrant for better speed ──
        grad_ckpt_kwargs = None
        if cfg.gradient_checkpointing:
            grad_ckpt_kwargs = {"use_reentrant": False}

        # ── Padding-free packing: eliminates padding waste entirely on CUDA ──
        # Requires flash-attention (sdpa also works on recent transformers).  When
        # enabled, sequences are concatenated with position_ids and cu_seqlens,
        # so every token is a real token — no padding compute waste at all.
        padding_free = cfg.padding_free and torch.cuda.is_available() and cfg.packing

        # ── Check which TrainingArguments params are actually available ──
        # (transformers 5.x removed group_by_length; guard with hasattr)
        from transformers import TrainingArguments as _TA
        _ta_fields = set(inspect.signature(_TA.__init__).parameters.keys())

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
            "use_cpu": use_cpu,
            "seed": cfg.seed,
            "report_to": "none",
            "optim": optim,
            "lr_scheduler_type": cfg.lr_scheduler,
            "dataloader_pin_memory": training_pin_memory(),
            "remove_unused_columns": not dataset_text_field,
            "load_best_model_at_end": eval_ds is not None and cfg.early_stopping,
            # ── Performance optimizations ──
            "dataloader_num_workers": num_workers,
            "dataloader_persistent_workers": persistent_workers,
            "gradient_checkpointing": cfg.gradient_checkpointing,
        }
        if (
            eval_ds is not None
            and cfg.early_stopping
            and "metric_for_best_model" in _ta_fields
        ):
            base["metric_for_best_model"] = cfg.metric_for_best_model
            base["greater_is_better"] = cfg.metric_for_best_model != "eval_loss"
        # save_safetensors was removed in transformers 5.x — only add when available
        if "save_safetensors" in _ta_fields:
            base["save_safetensors"] = cfg.save_safetensors
        # group_by_length was removed in transformers 5.x — only add when available
        if "group_by_length" in _ta_fields:
            base["group_by_length"] = cfg.group_by_length and not dataset_text_field
        if cfg.dataloader_prefetch_factor is not None and num_workers > 0:
            base["dataloader_prefetch_factor"] = cfg.dataloader_prefetch_factor
        if grad_ckpt_kwargs is not None:
            base["gradient_checkpointing_kwargs"] = grad_ckpt_kwargs
        if cfg.neftune_noise_alpha is not None and not use_cpu:
            base["neftune_noise_alpha"] = cfg.neftune_noise_alpha
        if padding_free:
            base["padding_free"] = True
        if cfg.torch_compile and torch.cuda.is_available():
            base["torch_compile"] = True
            base["torch_compile_backend"] = "inductor"

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
        import torch
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader

        cfg = self.config
        apply_determinism(cfg.seed, deterministic=cfg.deterministic)
        raw = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)
        examples = []
        for row in raw:
            anchor = row.get("anchor") or row.get("query") or row.get("text", "")
            positive = row.get("positive") or row.get("answer") or row.get("output", "")
            if anchor and positive:
                examples.append(InputExample(texts=[anchor, positive]))

        if not examples:
            raise ValueError("Embedding dataset needs anchor/query + positive/answer columns")

        model = SentenceTransformer(cfg.model_id)
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        loader = DataLoader(examples, shuffle=True, batch_size=cfg.batch_size, generator=g)
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

    def _patch_adapter_metadata(self, out: Path) -> None:
        """Ensure adapter_config.json points at the local base for offline merge/export."""
        adapter_cfg_path = out / "adapter_config.json"
        if not adapter_cfg_path.is_file():
            return
        try:
            adapter_cfg = json.loads(adapter_cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            return

        base_path = self._resolve_load_model_id()
        original_id = str(self.config.extra.get("original_model_id") or self.config.model_id)
        adapter_cfg["base_model_name_or_path"] = base_path
        adapter_cfg["seiso_original_base_model"] = original_id
        adapter_cfg["seiso_quant_mode"] = self.config.quant.value
        adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2))

    def _write_manifest(
        self,
        out: Path,
        layout,
        multi_gpu: bool,
        dataset_format: str,
        *,
        preprocess_stats: dict[str, Any] | None = None,
        train_samples: int = 0,
        eval_samples: int = 0,
    ) -> None:
        cfg = self.config
        base_path = self._resolve_load_model_id()
        original_id = str(cfg.extra.get("original_model_id") or cfg.model_id)
        dataset_path = str(cfg.dataset)
        dataset_hash: str | None = None
        ds_path = Path(cfg.dataset)
        if ds_path.is_file():
            with contextlib.suppress(OSError, ValueError):
                dataset_hash = sha256_file(ds_path)

        manifest = {
            "model_id": original_id,
            "original_model_id": original_id,
            "base_model_path": base_path,
            "resolved_model_path": cfg.extra.get("resolved_model_path") or base_path,
            "method": cfg.method.value,
            "quant": cfg.quant.value,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_seq_length": cfg.max_seq_length,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "eval_split_ratio": cfg.eval_split_ratio,
            "max_eval_samples": cfg.max_eval_samples,
            "eval_steps": cfg.eval_steps,
            "preprocess_dataset": cfg.preprocess_dataset,
            "deduplicate_dataset": cfg.deduplicate_dataset,
            "early_stopping": cfg.early_stopping,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_threshold": cfg.early_stopping_threshold,
            "metric_for_best_model": cfg.metric_for_best_model,
            "preprocess_stats": preprocess_stats,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "lr_scheduler": cfg.lr_scheduler,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "dataset": dataset_path,
            "dataset_hash_sha256": dataset_hash,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "use_rslora": cfg.use_rslora,
            "dataset_format": dataset_format,
            "train_on_responses_only": cfg.train_on_responses_only,
            "packing": cfg.packing,
            "dataloader_num_workers": cfg.dataloader_num_workers,
            "group_by_length": cfg.group_by_length,
            "padding_free": cfg.padding_free,
            "neftune_noise_alpha": cfg.neftune_noise_alpha,
            "torch_compile": cfg.torch_compile,
            "save_safetensors": cfg.save_safetensors,
            "multi_gpu": multi_gpu,
            "world_size": layout.world_size,
            "kernels": self._kernel_meta,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(out / "seiso_manifest.json", manifest)
