"""Robust Hugging Face training — QLoRA, SFT, embeddings."""

from __future__ import annotations

import contextlib
import gc
import inspect
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.io.jsonl import read_json_file
from seiso.kernels.hooks import (
    apply_fused_lora_kernels,
    apply_fused_lora_qkv_kernels,
    apply_fused_residual_norm_kernels,
    apply_training_kernels,
)
from seiso.kernels.lifecycle import KernelPatchSession, release_training_memory
from seiso.memory.protection import (
    apply_training_memory_guards,
    apply_training_oom_fallback,
    describe_training_memory_policy,
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
from seiso.training.config import DatasetFormat, QuantMode, TrainConfig, TrainMethod
from seiso.training.dataset_analysis import analyze_training_dataset
from seiso.training.datasets import (
    format_dataset_text,
    load_training_dataset,
    prepare_tokenized_dataset,
)
from seiso.training.multi_gpu import (
    configure_distributed_training_args,
    detect_training_layout,
    resolve_distributed_plan,
)
from seiso.training.practices import (
    default_pad_to_multiple_of,
    resolve_compute_dtype,
    resolve_dataloader_settings,
    resolve_map_workers,
    resolve_optimizer,
    sft_modern_kwargs,
    training_args_modern_extras,
)
from seiso.training.preprocess import (
    compute_eval_split_size,
    preprocess_training_dataset,
)
from seiso.training.sft import build_sft_trainer

logger = logging.getLogger(__name__)


@dataclass
class PreparedTrainingDatasets:
    train_ds: Any
    eval_ds: Any | None
    detected_format: DatasetFormat
    data_collator: Any | None
    dataset_text_field: str | None
    preprocess_stats: dict[str, Any] | None


class SeisoTrainer:
    def __init__(
        self,
        config: TrainConfig,
        *,
        on_metric: Callable[[dict[str, Any]], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        job_id: str | None = None,
    ) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._kernel_meta: dict = {}
        self._loaded: SeisoModel | None = None
        self._on_metric = on_metric
        self._on_log = on_log
        self._job_id = job_id
        self._metrics_callback = None

    def _log(self, message: str) -> None:
        logger.info(message)
        if self._on_log:
            try:
                self._on_log(message)
            except Exception:
                logger.exception("on_log callback failed")

    def run(self) -> Path:
        original_config = self.config
        self.config = apply_training_memory_guards(self.config)
        policy = describe_training_memory_policy(
            original_config, self.config, reason="initial_guard"
        )
        if policy["changed"]:
            self._log("MEMORY_POLICY " + json.dumps(policy, sort_keys=True))
        cfg = self.config
        self._apply_cuda_training_profile()
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

        if cfg.method == TrainMethod.EMBEDDING:
            # Embedding pairs use a light schema check inside _train_embedding.
            out = self._train_embedding()
            self._cleanup_gpu(None)
            return out

        # Validate dataset *first* (before loading potentially huge model weights).
        # Reuse Forge-provided analysis when the UI already scanned the corpus.
        try:
            cached = cfg.extra.get("cached_dataset_analysis")
            if isinstance(cached, dict) and cached.get("valid"):
                analysis = cached
                reused = True
            else:
                # Sample-only scan — full preprocess runs once in _prepare_datasets
                # (or reuses the cleaned cache when UI analysis already ran).
                analysis = analyze_training_dataset(
                    cfg.dataset,
                    dataset_format=cfg.dataset_format,
                    sandbox_root=cfg.sandbox_root,
                    full_scan=False,
                )
                reused = False
            write_json(cfg.output_dir / "dataset_analysis.json", analysis)
            prefix = "Reused" if reused else "Dataset analysis"
            sample_note = (
                ""
                if analysis.get("uses_full_dataset", True)
                else ", sample estimate"
            )
            self._log(
                f"{prefix}: {analysis['kept']:,}/{analysis['initial_samples']:,} usable samples "
                f"(format={analysis['resolved_format']}, domain={analysis['domain']}{sample_note})"
            )
            # Stash for _prepare_datasets when preprocess params match analysis defaults.
            self._cached_analysis = analysis
        except Exception as exc:
            raise ValueError(
                f"Dataset cannot be normalized for training: {exc}"
            ) from exc

        layout = detect_training_layout()
        distributed_plan = resolve_distributed_plan(cfg, layout)
        multi_gpu = distributed_plan.enabled and layout.use_ddp
        use_triton = cfg.use_triton
        use_fused_ce = cfg.use_fused_ce
        use_fused_lora = cfg.use_fused_lora
        use_fused_lora_qkv = bool(cfg.extra.get("use_fused_lora_qkv", use_fused_lora))

        logger.info(
            "Training %s | method=%s quant=%s | world_size=%d",
            cfg.model_id,
            cfg.method.value,
            cfg.quant.value,
            distributed_plan.world_size,
        )

        if cfg.method not in (TrainMethod.LORA, TrainMethod.FULL):
            raise ValueError(f"Unsupported training method: {cfg.method.value}")

        model = None
        patch_session: KernelPatchSession | None = None
        try:
            model, tokenizer = self._load_model()
            patch_session = KernelPatchSession(model)
            patch_session.__enter__()
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

            if use_triton:
                residual_meta = apply_fused_residual_norm_kernels(model)
                self._kernel_meta.update(residual_meta)

            if cfg.method == TrainMethod.LORA:
                model = self._apply_lora(model)
                fused_lora_rank = min(cfg.lora_r, 64)
                if use_fused_lora_qkv:
                    qkv_meta = apply_fused_lora_qkv_kernels(model, max_rank=fused_lora_rank)
                    self._kernel_meta.update(qkv_meta)
                if use_fused_lora or use_fused_lora_qkv:
                    lora_meta = apply_fused_lora_kernels(
                        model,
                        max_rank=fused_lora_rank,
                        skip_qkv=use_fused_lora_qkv,
                    )
                    self._kernel_meta.update(lora_meta)
            elif cfg.method == TrainMethod.FULL and cfg.quant in (
                QuantMode.INT4,
                QuantMode.INT8,
            ):
                logger.warning(
                    "Full fine-tune with quantization — consider LoRA for memory efficiency"
                )

            SeisoModel.for_training(model)

            from seiso.training.torch_dynamo import apply_compile_checkpoint_workarounds

            model = apply_compile_checkpoint_workarounds(
                model,
                torch_compile=cfg.torch_compile,
                gradient_checkpointing=cfg.gradient_checkpointing,
            )
            if self._loaded:
                self._loaded.model = model

            from seiso.training.metrics import build_metrics_callback

            prepared: PreparedTrainingDatasets

            def build_current_trainer():
                nonlocal prepared
                prepared = self._prepare_datasets(tokenizer)
                current_cfg = self.config
                emit_stdout = multi_gpu or bool(
                    os.environ.get("SEISO_EMIT_METRICS_STDOUT")
                )
                metrics_cb = build_metrics_callback(
                    current_cfg.output_dir,
                    on_metric=self._on_metric,
                    emit_stdout=emit_stdout,
                )
                self._metrics_callback = metrics_cb
                trainer_callbacks = self._build_callbacks(
                    metrics_cb, eval_enabled=prepared.eval_ds is not None
                )
                return self._build_trainer(
                    model,
                    tokenizer,
                    prepared.train_ds,
                    prepared.eval_ds,
                    layout,
                    multi_gpu,
                    data_collator=prepared.data_collator,
                    dataset_text_field=prepared.dataset_text_field,
                    dataset_format=prepared.detected_format,
                    callbacks=trainer_callbacks,
                )

            trainer = build_current_trainer()

            if cfg.resume_from:
                self._train_with_oom_recovery(
                    trainer,
                    resume_from_checkpoint=str(cfg.resume_from),
                    rebuild_trainer=build_current_trainer,
                )
            else:
                self._train_with_oom_recovery(
                    trainer, rebuild_trainer=build_current_trainer
                )

            is_main = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0
            if not is_main:
                logger.info("Non-main rank finished training (no checkpoint write)")
                return cfg.output_dir

            out = (
                cfg.output_dir
                / f"checkpoint-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            )
            trainer.save_model(str(out))
            tokenizer.save_pretrained(str(out))
            if cfg.method == TrainMethod.LORA:
                self._patch_adapter_metadata(out)
            self._write_manifest(
                out,
                layout,
                multi_gpu,
                distributed_plan.strategy,
                prepared.detected_format.value,
                preprocess_stats=prepared.preprocess_stats,
                train_samples=len(prepared.train_ds),
                eval_samples=len(prepared.eval_ds) if prepared.eval_ds is not None else 0,
            )
            logger.info("Training complete: %s", out)
            return out
        finally:
            if patch_session is not None:
                patch_session.__exit__(None, None, None)
            release_training_memory(model)
            self._cleanup_gpu(None)

    def _prepare_datasets(self, tokenizer) -> PreparedTrainingDatasets:
        from seiso.training.dataset_analysis import (
            cleaned_dataset_cache_key,
            take_cleaned_dataset,
        )

        cfg = self.config
        ds_fmt = cfg.dataset_format
        preprocess_stats: dict[str, Any] | None = None
        raw_ds = None

        if cfg.preprocess_dataset:
            analysis = getattr(self, "_cached_analysis", None)
            reuse_ok = (
                isinstance(analysis, dict)
                and cfg.deduplicate_dataset is True
                and int(cfg.min_sample_chars) <= 1
            )
            if reuse_ok:
                key = analysis.get("cleaned_cache_key") or cleaned_dataset_cache_key(
                    cfg.dataset,
                    dataset_format=cfg.dataset_format,
                    sandbox_root=cfg.sandbox_root,
                    deduplicate=True,
                    min_chars=1,
                )
                cached_clean = take_cleaned_dataset(str(key))
                if cached_clean is not None:
                    raw_ds, preprocess_stats, ds_fmt = cached_clean
                    self._log(
                        "Reusing cleaned dataset from prior analysis "
                        f"({preprocess_stats['kept']}/{preprocess_stats['initial_samples']} samples)"
                    )

            if raw_ds is None:
                raw_ds = load_training_dataset(
                    cfg.dataset, sandbox_root=cfg.sandbox_root
                )
                raw_ds, preprocess_stats, ds_fmt = preprocess_training_dataset(
                    raw_ds,
                    dataset_format=ds_fmt,
                    deduplicate=cfg.deduplicate_dataset,
                    min_chars=cfg.min_sample_chars,
                    num_proc=resolve_map_workers(cfg),
                )
                self._log(
                    f"Preprocessed dataset: {preprocess_stats['kept']}/{preprocess_stats['initial_samples']} "
                    f"samples kept (format={preprocess_stats['resolved_format']})"
                )
        else:
            raw_ds = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)

        raw_ds = self._limit_training_samples(raw_ds)
        train_ds, eval_ds = self._split_train_eval(raw_ds)
        detected_fmt, train_ds, eval_ds, dataset_text_field, data_collator = (
            self._format_or_tokenize_datasets(train_ds, eval_ds, tokenizer, ds_fmt)
        )

        return PreparedTrainingDatasets(
            train_ds=train_ds,
            eval_ds=eval_ds,
            detected_format=detected_fmt,
            data_collator=data_collator,
            dataset_text_field=dataset_text_field,
            preprocess_stats=preprocess_stats,
        )

    def _limit_training_samples(self, raw_ds):
        max_samples = self.config.extra.get("max_samples")
        if (
            isinstance(max_samples, int)
            and max_samples > 0
            and len(raw_ds) > max_samples
        ):
            raw_ds = raw_ds.select(range(max_samples))
            logger.info("Limited dataset to %d samples (max_samples)", max_samples)
        return raw_ds

    def _split_train_eval(self, raw_ds):
        cfg = self.config
        eval_n = 0
        if len(raw_ds) > 10 and (cfg.early_stopping or cfg.eval_split_ratio > 0):
            split_ratio = cfg.eval_split_ratio if cfg.eval_split_ratio > 0 else 0.02
            eval_n = compute_eval_split_size(
                len(raw_ds),
                split_ratio,
                cfg.max_eval_samples,
            )
        if eval_n <= 0:
            return raw_ds, None

        split = raw_ds.train_test_split(test_size=eval_n, seed=cfg.seed)
        train_ds, eval_ds = split["train"], split["test"]
        self._log(
            f"Train/eval split: {len(train_ds)} train, {len(eval_ds)} eval (max_eval={cfg.max_eval_samples})"
        )
        return train_ds, eval_ds

    def _format_or_tokenize_datasets(self, train_ds, eval_ds, tokenizer, ds_fmt):
        cfg = self.config
        map_workers = resolve_map_workers(cfg)
        data_collator = None
        dataset_text_field = None

        if cfg.packing and cfg.train_on_responses_only:
            logger.warning("Sequence packing disables train-on-responses-only masking")

        if cfg.packing:
            train_ds, detected_fmt = format_dataset_text(
                train_ds,
                tokenizer,
                ds_fmt,
                num_proc=map_workers,
            )
            dataset_text_field = "text"
            if eval_ds is not None:
                eval_ds, _ = format_dataset_text(
                    eval_ds,
                    tokenizer,
                    detected_fmt,
                    num_proc=map_workers,
                )
            return detected_fmt, train_ds, eval_ds, dataset_text_field, data_collator

        train_ds, detected_fmt = prepare_tokenized_dataset(
            train_ds,
            tokenizer,
            max_seq_length=cfg.max_seq_length,
            dataset_format=ds_fmt,
            train_on_inputs=not cfg.train_on_responses_only,
            num_proc=map_workers,
        )
        if eval_ds is not None:
            eval_ds, _ = prepare_tokenized_dataset(
                eval_ds,
                tokenizer,
                max_seq_length=cfg.max_seq_length,
                dataset_format=detected_fmt,
                train_on_inputs=not cfg.train_on_responses_only,
                num_proc=map_workers,
            )
        import torch

        pad_multiple = default_pad_to_multiple_of(
            cfg.pad_to_multiple_of,
            cuda_available=torch.cuda.is_available(),
        )
        data_collator = self._make_collator(tokenizer, pad_to_multiple_of=pad_multiple)
        return detected_fmt, train_ds, eval_ds, dataset_text_field, data_collator

    def _apply_cuda_training_profile(self) -> None:
        cfg = self.config
        try:
            from seiso.hardware import hardware_profile, vram_headroom_mb
            from seiso.kernels.training_profile import prepare_cuda_training_profile

            profile_hw = hardware_profile()
            headroom = vram_headroom_mb(profile_hw)
            profile = prepare_cuda_training_profile(
                headroom_mb=headroom,
                model_id=str(cfg.model_id),
                batch_size=cfg.batch_size,
                max_seq_length=cfg.max_seq_length,
            )
            updates = {
                key: profile[key]
                for key in (
                    "gradient_checkpointing",
                    "use_fused_ce",
                    "use_triton",
                    "use_fused_lora",
                    "use_fused_lora_qkv",
                    "use_cuda_graphs",
                    "max_seq_length",
                )
                if key in profile and getattr(cfg, key, None) != profile[key]
            }
            if updates:
                self.config = cfg.model_copy(update=updates)
        except Exception:
            logger.debug("CUDA training profile skipped", exc_info=True)

    def _build_callbacks(self, metrics_cb, *, eval_enabled: bool) -> list[Any]:
        cfg = self.config
        from seiso.training.cancel import should_stop

        callbacks: list[Any] = [metrics_cb]
        stop_fn = should_stop(self._job_id)
        if self._job_id:
            from transformers import TrainerCallback

            class _CancelTrainingCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if stop_fn():
                        control.should_training_stop = True
                        control.should_epoch_stop = True

            callbacks.append(_CancelTrainingCallback())
        if eval_enabled and cfg.early_stopping:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience,
                    early_stopping_threshold=cfg.early_stopping_threshold,
                )
            )
        return callbacks

    def _train_with_oom_recovery(
        self,
        trainer,
        *,
        resume_from_checkpoint: str | None = None,
        rebuild_trainer: Callable[[], Any] | None = None,
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
                original_config = self.config
                self.config = apply_training_oom_fallback(self.config)
                policy = describe_training_memory_policy(
                    original_config, self.config, reason="oom_fallback"
                )
                self._log("MEMORY_POLICY " + json.dumps(policy, sort_keys=True))
                cfg = self.config
                if rebuild_trainer is not None:
                    trainer = rebuild_trainer()
                else:
                    trainer.args.per_device_train_batch_size = cfg.batch_size
                    trainer.args.per_device_eval_batch_size = cfg.batch_size
                    trainer.args.gradient_accumulation_steps = (
                        cfg.gradient_accumulation_steps
                    )
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

        if Path(model_ref).exists() and not snapshot_has_trainable_weights(
            Path(model_ref)
        ):
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
            load_dtype = (
                None
                if native_hub_quant
                else ("float16" if cfg.quant == QuantMode.INT16 else None)
            )
            use_flash = not native_hub_quant and bool(
                cfg.extra.get("use_flash_attention", True)
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

        if cfg.method == TrainMethod.LORA and cfg.quant in (
            QuantMode.INT4,
            QuantMode.INT8,
        ):
            try:
                from peft import prepare_model_for_kbit_training

                # Let TrainingArguments handle gradient checkpointing via
                # gradient_checkpointing_kwargs (use_reentrant=False) to avoid
                # double-enablement and reentrant overhead.
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=False
                )
                if cfg.gradient_checkpointing and hasattr(
                    model, "gradient_checkpointing_enable"
                ):
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                self._loaded.model = model
            except ImportError:
                logger.warning(
                    "prepare_model_for_kbit_training unavailable — install peft>=0.11; "
                    "QLoRA may train without k-bit preparation"
                )

        return model, tokenizer

    def _apply_lora(self, model):
        cfg = self.config
        moe_finetune = bool(cfg.extra.get("moe_finetune", False))
        target_override = cfg.extra.get("lora_target_modules")
        configured_targets = target_override if isinstance(target_override, list) else None
        model = SeisoModel.attach_lora(
            model,
            r=cfg.lora_r,
            target_modules=configured_targets,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            use_rslora=cfg.use_rslora,
            model_id=self._resolve_load_model_id(),
            freeze_moe_router=moe_finetune
            and bool(cfg.extra.get("freeze_moe_router", True)),
        )
        if moe_finetune:
            logger.info(
                "MoE-aware LoRA enabled (router frozen: %s)",
                bool(cfg.extra.get("freeze_moe_router", True)),
            )
        if self._loaded:
            self._loaded.model = model
        return model

    @staticmethod
    def _make_collator(tokenizer, *, pad_to_multiple_of: int | None = None):
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of,
        )

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
        dataset_format: DatasetFormat = DatasetFormat.AUTO,
        callbacks=None,
    ) -> Any:
        cfg = self.config
        import torch

        cuda_available = torch.cuda.is_available()
        use_cpu = not cuda_available and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
        use_bf16, use_fp16 = resolve_compute_dtype(
            cuda_available=cuda_available,
            bf16_supported=bf16_supported,
            quant=cfg.quant.value,
        )
        if (
            not cuda_available
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            use_fp16 = cfg.quant == QuantMode.INT16

        optim = resolve_optimizer(cfg.quant.value, use_cpu=use_cpu)

        # Overlap CPU input preparation with GPU compute when workers are available.
        num_workers, persistent_workers, prefetch_factor = resolve_dataloader_settings(
            cfg,
            cuda_available=cuda_available,
        )

        # ── Gradient checkpointing: use non-reentrant for better speed ──
        grad_ckpt_kwargs = None
        if cfg.gradient_checkpointing:
            grad_ckpt_kwargs = {"use_reentrant": False}

        # ── Padding-free packing: eliminates padding waste entirely on CUDA ──
        # Requires flash-attention (sdpa also works on recent transformers).  When
        # enabled, sequences are concatenated with position_ids and cu_seqlens,
        # so every token is a real token — no padding compute waste at all.
        padding_free = cfg.padding_free and cuda_available and cfg.packing

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
        if prefetch_factor is not None:
            base["dataloader_prefetch_factor"] = prefetch_factor
        if grad_ckpt_kwargs is not None:
            base["gradient_checkpointing_kwargs"] = grad_ckpt_kwargs
        if cfg.neftune_noise_alpha is not None and not use_cpu:
            base["neftune_noise_alpha"] = cfg.neftune_noise_alpha
        if padding_free:
            base["padding_free"] = True
        if cfg.torch_compile and cuda_available:
            base["torch_compile"] = True
            base["torch_compile_backend"] = "inductor"

        base.update(training_args_modern_extras(cfg, eval_enabled=eval_ds is not None))

        merged_callbacks = list(callbacks or [])
        use_cuda_graphs = bool(cfg.extra.get("use_cuda_graphs", not cfg.deterministic))
        if use_cuda_graphs and cfg.gradient_checkpointing:
            logger.info(
                "CUDA graphs disabled — incompatible with gradient_checkpointing "
                "(disable GC or set extra.use_cuda_graphs=false)"
            )
            use_cuda_graphs = False
        if use_cuda_graphs and cuda_available:
            try:
                from seiso.kernels.cuda_graphs import make_training_graph_callback

                cb = make_training_graph_callback(
                    deterministic=cfg.deterministic,
                    enabled=use_cuda_graphs,
                )
                if cb is not None:
                    merged_callbacks.append(cb)
            except ImportError:
                pass
            self._kernel_meta["cuda_graphs_requested"] = True

        args_dict = configure_distributed_training_args(base, layout, cfg, multi_gpu)
        modern_sft = sft_modern_kwargs(
            cfg,
            train_on_responses_only=cfg.train_on_responses_only,
            dataset_format=dataset_format,
            cuda_available=cuda_available,
            use_text_field=bool(dataset_text_field),
        )
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
            use_cuda_graphs=use_cuda_graphs,
            callbacks=merged_callbacks or None,
            sft_extras=modern_sft,
        )

    def _train_embedding(self) -> Path:
        import torch
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset as TorchDataset

        cfg = self.config
        apply_determinism(cfg.seed, deterministic=cfg.deterministic)
        # Light schema check — skip full analyze_training_dataset for embedding pairs.
        raw = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)

        class _PairDataset(TorchDataset):
            """Index into the HF dataset — avoids materializing all InputExamples."""

            def __init__(self, rows):
                self._rows = rows
                self._indices: list[int] = []
                for i in range(len(rows)):
                    row = rows[i]
                    anchor = (
                        row.get("anchor") or row.get("query") or row.get("text", "")
                    )
                    positive = (
                        row.get("positive")
                        or row.get("answer")
                        or row.get("output", "")
                    )
                    if anchor and positive:
                        self._indices.append(i)

            def __len__(self) -> int:
                return len(self._indices)

            def __getitem__(self, idx: int) -> InputExample:
                row = self._rows[self._indices[idx]]
                anchor = row.get("anchor") or row.get("query") or row.get("text", "")
                positive = (
                    row.get("positive") or row.get("answer") or row.get("output", "")
                )
                return InputExample(texts=[str(anchor), str(positive)])

        pair_ds = _PairDataset(raw)
        if len(pair_ds) == 0:
            raise ValueError(
                "Embedding dataset needs anchor/query + positive/answer columns"
            )

        model = SentenceTransformer(cfg.model_id)
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        cuda_available = torch.cuda.is_available()
        num_workers, persistent_workers, prefetch_factor = resolve_dataloader_settings(
            cfg,
            cuda_available=cuda_available,
        )
        loader_kwargs: dict[str, Any] = {
            "shuffle": True,
            "batch_size": cfg.batch_size,
            "generator": g,
            "num_workers": num_workers,
            "pin_memory": training_pin_memory(),
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loader = DataLoader(pair_ds, **loader_kwargs)
        loss = losses.MultipleNegativesRankingLoss(model)
        out = (
            cfg.output_dir
            / f"embed-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
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
        adapter_cfg = read_json_file(adapter_cfg_path, default=None)
        if not isinstance(adapter_cfg, dict):
            return

        base_path = self._resolve_load_model_id()
        original_id = str(
            self.config.extra.get("original_model_id") or self.config.model_id
        )
        adapter_cfg["base_model_name_or_path"] = base_path
        adapter_cfg["seiso_original_base_model"] = original_id
        adapter_cfg["seiso_quant_mode"] = self.config.quant.value
        adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2))

    def _write_manifest(
        self,
        out: Path,
        layout,
        multi_gpu: bool,
        distributed_strategy: str,
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

        from seiso.research.provenance import manifest_common_fields

        manifest = {
            **manifest_common_fields(
                config_snapshot={
                    "model_id": original_id,
                    "method": cfg.method.value,
                    "quant": cfg.quant.value,
                    "dataset": dataset_path,
                    "seed": cfg.seed,
                }
            ),
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
            "dataset_num_proc": cfg.dataset_num_proc,
            "pad_to_multiple_of": cfg.pad_to_multiple_of,
            "assistant_only_loss": cfg.assistant_only_loss,
            "save_safetensors": cfg.save_safetensors,
            "multi_gpu": multi_gpu,
            "distributed_strategy": distributed_strategy,
            "distributed_nproc_per_node": cfg.distributed_nproc_per_node,
            "distributed_num_nodes": cfg.distributed_num_nodes,
            "distributed_node_rank": cfg.distributed_node_rank,
            "ddp_backend": cfg.ddp_backend,
            "ddp_find_unused_parameters": cfg.ddp_find_unused_parameters,
            "cloud_gpu_enabled": cfg.cloud_gpu_enabled,
            "cloud_gpu_provider": cfg.cloud_gpu_provider.value,
            "cloud_gpu_region": cfg.cloud_gpu_region,
            "cloud_gpu_instance_type": cfg.cloud_gpu_instance_type,
            "cloud_gpu_count": cfg.cloud_gpu_count,
            "cloud_gpu_project": cfg.cloud_gpu_project,
            "cloud_gpu_credential_id": cfg.cloud_gpu_credential_id,
            "world_size": layout.world_size,
            "kernels": self._kernel_meta,
        }
        write_json(out / "seiso_manifest.json", manifest)
