"""Slime-style GRPO trainer for Hugging Face causal LMs.

Supports single-GPU HF rollouts, multi-GPU DDP, and remote SGLang/vLLM backends.
Types, policy math, and distributed helpers live in sibling modules; this module
re-exports them so existing imports and test patches keep working.
"""

from __future__ import annotations

import gc
import importlib
import itertools
import json
import logging
import math
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from seiso.io.jsonl import iter_jsonl
from seiso.models.lora_targets import (
    has_multimodal_language_model_backbone,
    resolve_lora_target_modules,
)
from seiso.research.provenance import apply_determinism
from seiso.rl_verify import score_completion as verify_score_completion
from seiso.rl_verify.extract import split_thinking_trace
from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.distributed import (  # noqa: F401
    _balanced_rank_samples,
    _count_rank_samples,
    _cuda_device_index,
    _distributed_barrier,
    _distributed_context,
    _distributed_min_int,
    _generation_model,
    _is_cuda_device,
    _iter_distributed_sample_batches,
    _maybe_wrap_ddp,
    _rank_verifier_path,
    _reduce_stats,
    _save_distributed,
    _unwrap_model,
)
from seiso.slime.policy import (  # noqa: F401
    _assign_grouped_advantages,
    _clipped_policy_loss,
    _empty_stats,
    _filter_rollout_groups,
    _group_reward_spread_mean,
    _group_verifier_stats,
    _keep_rollout_group,
    _masked_sequence_logprobs,
    _mean,
    _merge_stats,
    _pad_rollout_token_logprobs,
    _pad_rollouts,
    _policy_loss,
    _response_mask_for_sequence,
    _rollout_status,
    _rollout_status_stats,
    _sequence_logprobs,
    _sequence_token_logprobs,
    _truncate_rollout_groups,
)
from seiso.slime.rewards import resolve_reward

# Split modules (re-exported below for API/test stability)
from seiso.slime.types import (  # noqa: F401
    Rollout,
    T,
    _AutoStopController,
    _AutoStopDecision,
    _CompletionScore,
    _DistributedSlimeContext,
    _PushbackIterator,
    _RolloutBatch,
)
from seiso.training.metrics import METRIC_STDOUT_PREFIX

logger = logging.getLogger(__name__)

_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}


def train_slime(config: SingleGpuSlimeConfig) -> Path:
    """Run a compact slime-style rollout/reward/update loop.

    When launched under Accelerate with WORLD_SIZE > 1, this keeps the original
    local SLIME algorithm but shards prompts by rank and synchronizes policy
    updates with PyTorch DDP so multi-node jobs behave as one training run.

    When ``data_gen`` / ``data_gen_count`` is set, a high-level verifiable
    prompt corpus is materialized first (numeric / choice / code). Rewards
    always score *online model completions*, never stored answers as outputs.

    ``train_single_gpu_slime`` is a compatibility alias for this function.
    """

    config.validate()
    if config.kl_coef == 0.0 and config.epochs > 1:
        logger.warning(
            "kl_coef=0 with epochs=%s skips the frozen reference KL term (VRAM-saving). "
            "For multi-epoch GRPO prefer kl_coef in [0.02, 0.05] to limit policy drift.",
            config.epochs,
        )
    if config.dynamic_sampling_filter == "none":
        logger.warning(
            "dynamic_sampling_filter=none keeps zero-spread groups; GRPO advantages "
            "are then vacuous. Prefer reward_nonzero_std / outcome_nonzero_std and "
            "monitor group_nonzero_outcome_spread_frac."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist_ctx = _distributed_context(torch, config)
    config = _maybe_materialize_data_gen(config, dist_ctx)
    config = replace(config, device=dist_ctx.device)
    _require_single_gpu(config)
    _set_seed(config.seed + dist_ctx.rank)

    _configure_vram_cap(torch, config.max_vram_gb, config.device)
    dtype = _resolve_dtype(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map={"": config.device},
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    _freeze_multimodal_backbones(model)
    if config.use_lora:
        model = _apply_lora(model, config)
    elif config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=_GRADIENT_CHECKPOINTING_KWARGS
        )

    ref_model = None
    if config.kl_coef > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            device_map={"": config.device},
            trust_remote_code=config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)

    model = _maybe_wrap_ddp(model, torch, dist_ctx, config)
    _assert_vram_fit(torch, config.max_vram_gb, config.device)
    optimizer = _build_optimizer(model, config)
    # Validate reward checker name early (scoring uses config.reward directly).
    resolve_reward(config.reward)

    if dist_ctx.is_main:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier(dist_ctx)
    metrics_path = config.output_dir / "slime_single_gpu_metrics.jsonl"
    verifier_path = _rank_verifier_path(config, dist_ctx) if config.write_verifier_data else None
    final_output_dir = _final_output_dir(config)
    best_checkpoint_dir = config.output_dir / config.best_checkpoint_dir
    auto_stop = _AutoStopController.from_config(config)
    global_step = 0
    pending_accumulation_steps = 0
    saw_sample = False
    total_groups_seen = 0.0
    total_groups_kept = 0.0
    empty_trainable_batches = 0
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(config.seed)
    from seiso.slime.rollout_backend import WeightSyncState

    weight_sync_state = WeightSyncState()
    # slime: push actor → SGLang before the first rollout (critical with LoRA).
    _sync_rollout_engine_weights(
        model=model,
        tokenizer=tokenizer,
        config=config,
        dist_ctx=dist_ctx,
        step=0,
        sync_state=weight_sync_state,
    )

    for epoch in range(config.epochs):
        sample_batches = _PushbackIterator(iter(_iter_sample_batches(config, rng, dist_ctx, torch)))
        for batch_samples in sample_batches:
            saw_sample = True
            rollout_batch = _collect_training_rollout_batch(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                sample_batches=sample_batches,
                samples=batch_samples,
                config=config,
                torch=torch,
                epoch=epoch,
                global_step=global_step,
                verifier_path=verifier_path,
                dist_ctx=dist_ctx,
            )
            rollouts = rollout_batch.rollouts
            total_groups_seen += float(rollout_batch.stats.get("rollout_groups_total", 0.0))
            total_groups_kept += float(rollout_batch.stats.get("rollout_groups_kept", 0.0))
            if config.dynamic_sampling_filter != "none" and dist_ctx.enabled:
                target_groups = _distributed_min_int(
                    len(rollouts) // config.rollouts_per_prompt,
                    torch,
                    dist_ctx,
                )
                rollouts = _truncate_rollout_groups(
                    rollouts, config.rollouts_per_prompt, target_groups
                )
                rollout_batch = _RolloutBatch(
                    rollouts=rollouts,
                    stats={
                        **rollout_batch.stats,
                        "distributed_kept_groups_min": float(target_groups),
                    },
                )
            # DDP: all ranks must agree on whether this step has trainable data.
            has_local = 1 if rollouts else 0
            if dist_ctx.enabled:
                has_local = _distributed_min_int(has_local, torch, dist_ctx)
            if not has_local:
                empty_trainable_batches += 1
                if dist_ctx.is_main and empty_trainable_batches == 1:
                    _append_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "epoch": epoch,
                            **rollout_batch.stats,
                            "stop_reason": "awaiting_trainable_groups",
                            "rollout_groups_seen_total": total_groups_seen,
                            "rollout_groups_kept_total": total_groups_kept,
                        },
                    )
                _distributed_barrier(dist_ctx)
                continue
            trained_groups = len(rollouts) // config.rollouts_per_prompt
            rollout_batch = _RolloutBatch(
                rollouts=rollouts,
                stats={
                    **rollout_batch.stats,
                    "rollout_groups_trained": float(trained_groups),
                },
            )
            stats = _backprop_policy_step(
                model,
                rollouts,
                tokenizer.pad_token_id,
                config,
                torch,
                loss_scale=1.0 / config.gradient_accumulation_steps,
            )
            stats.update(rollout_batch.stats)
            stats = _reduce_stats(stats, torch, dist_ctx)
            pending_accumulation_steps += 1

            health_reason = _check_training_health(stats, config)
            if health_reason:
                optimizer.zero_grad(set_to_none=True)
                if dist_ctx.is_main and global_step % config.log_every_steps == 0:
                    _append_metrics(
                        metrics_path,
                        {
                            "step": global_step,
                            "epoch": epoch,
                            **stats,
                            **_auto_stop_stats(auto_stop),
                            "stop_reason": health_reason,
                        },
                    )
                global_step += 1
                _write_training_state(config, global_step, health_reason, auto_stop, dist_ctx)
                _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
                return final_output_dir

            if pending_accumulation_steps >= config.gradient_accumulation_steps:
                _optimizer_step(model, optimizer, torch, config)
                pending_accumulation_steps = 0
                # slime-style: push actor weights to SGLang after the optimizer step
                # so the next multi-GPU rollout is (near) on-policy.
                _sync_rollout_engine_weights(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    dist_ctx=dist_ctx,
                    step=global_step + 1,
                    sync_state=weight_sync_state,
                )

            decision = auto_stop.update(global_step, stats)
            if decision.improved:
                _save_distributed(model, tokenizer, best_checkpoint_dir, dist_ctx)

            if dist_ctx.is_main and global_step % config.log_every_steps == 0:
                _append_metrics(
                    metrics_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        **stats,
                        **_auto_stop_stats(auto_stop),
                        "stop_reason": decision.reason,
                    },
                )
            if (
                config.save_every_steps
                and global_step
                and global_step % config.save_every_steps == 0
            ):
                _save_distributed(
                    model, tokenizer, config.output_dir / f"checkpoint-{global_step}", dist_ctx
                )

            global_step += 1
            if decision.should_stop:
                optimizer.zero_grad(set_to_none=True)
                _write_training_state(config, global_step, decision.reason, auto_stop, dist_ctx)
                _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
                return final_output_dir
            if config.max_steps is not None and global_step >= config.max_steps:
                if pending_accumulation_steps:
                    _optimizer_step(model, optimizer, torch, config)
                    _sync_rollout_engine_weights(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        dist_ctx=dist_ctx,
                        step=global_step + 1,
                        sync_state=weight_sync_state,
                    )
                _write_training_state(config, global_step, "max_steps", auto_stop, dist_ctx)
                _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
                return final_output_dir

    if not saw_sample:
        raise ValueError(f"no samples found in {config.dataset}")
    if global_step == 0:
        # All oversampled groups were filtered — do not report a silent "complete".
        reason = "no_trainable_groups"
        if dist_ctx.is_main:
            _append_metrics(
                metrics_path,
                {
                    "step": 0,
                    "epoch": max(0, config.epochs - 1),
                    "rollout_groups_seen_total": total_groups_seen,
                    "rollout_groups_kept_total": total_groups_kept,
                    "empty_trainable_batches": float(empty_trainable_batches),
                    "stop_reason": reason,
                },
            )
        _write_training_state(config, global_step, reason, auto_stop, dist_ctx)
        raise RuntimeError(
            "no trainable rollout groups after dynamic sampling "
            f"(seen={total_groups_seen:.0f}, kept={total_groups_kept:.0f}). "
            "Outcome rewards may be uniform (all fail or all pass). "
            "Inspect outcome_pass_rate / group_pass_rate, ease the dataset, "
            "or set dynamic_sampling_filter: none only for debugging."
        )
    if pending_accumulation_steps:
        _optimizer_step(model, optimizer, torch, config)
        _sync_rollout_engine_weights(
            model=model,
            tokenizer=tokenizer,
            config=config,
            dist_ctx=dist_ctx,
            step=global_step,
            sync_state=weight_sync_state,
        )
    _write_training_state(config, global_step, "complete", auto_stop, dist_ctx)
    _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
    return final_output_dir


def _collect_training_rollout_batch(
    *,
    model,
    ref_model,
    tokenizer,
    sample_batches: Iterator[list[dict[str, Any]]],
    samples: list[dict[str, Any]],
    config: SingleGpuSlimeConfig,
    torch,
    epoch: int,
    global_step: int,
    verifier_path: Path | None,
    dist_ctx: _DistributedSlimeContext,
) -> _RolloutBatch:
    rollout_batch = _collect_rollouts(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        samples=samples,
        config=config,
        torch=torch,
        epoch=epoch,
        global_step=global_step,
        verifier_path=verifier_path,
        dist_ctx=dist_ctx,
    )
    if config.dynamic_sampling_filter == "none":
        return rollout_batch

    from seiso.slime.config import effective_train_batch_size

    stats = dict(rollout_batch.stats)
    rollouts = list(rollout_batch.rollouts)
    target_groups = effective_train_batch_size(config)
    refill_rounds = 0
    while _refill_visible_group_count(rollouts, config, torch, dist_ctx) < target_groups:
        try:
            refill_samples = next(sample_batches)
            has_refill = 1
        except StopIteration:
            refill_samples = []
            has_refill = 0
        if dist_ctx.enabled:
            has_refill = _distributed_min_int(has_refill, torch, dist_ctx)
        if not has_refill:
            # Re-queue any samples taken on ranks that still had data so the
            # next step (or epoch) does not permanently drop them.
            if refill_samples and isinstance(sample_batches, _PushbackIterator):
                sample_batches.push(refill_samples)
            break
        refill_rounds += 1
        refill_batch = _collect_rollouts(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            samples=refill_samples,
            config=config,
            torch=torch,
            epoch=epoch,
            global_step=global_step,
            verifier_path=verifier_path,
            dist_ctx=dist_ctx,
        )
        rollouts.extend(refill_batch.rollouts)
        _merge_rollout_collection_stats(stats, refill_batch.stats)

    kept_groups = len(rollouts) // config.rollouts_per_prompt
    if kept_groups > target_groups:
        rollouts = _truncate_rollout_groups(rollouts, config.rollouts_per_prompt, target_groups)
    stats["dynamic_refill_rounds"] = float(refill_rounds)
    stats["rollout_groups_target"] = float(target_groups)
    return _RolloutBatch(rollouts=rollouts, stats=stats)


def _refill_visible_group_count(
    rollouts: list[Rollout],
    config: SingleGpuSlimeConfig,
    torch,
    dist_ctx: _DistributedSlimeContext,
) -> int:
    local_groups = len(rollouts) // config.rollouts_per_prompt
    if not dist_ctx.enabled:
        return local_groups
    return _distributed_min_int(local_groups, torch, dist_ctx)


def _merge_rollout_collection_stats(
    target: dict[str, float],
    update: dict[str, float],
) -> None:
    for key, value in update.items():
        target[key] = float(target.get(key, 0.0)) + float(value)


def _collect_rollouts(
    *,
    model,
    ref_model,
    tokenizer,
    samples: list[dict[str, Any]],
    config: SingleGpuSlimeConfig,
    torch,
    epoch: int,
    global_step: int,
    verifier_path: Path | None,
    dist_ctx: _DistributedSlimeContext | None = None,
) -> _RolloutBatch:
    from seiso.slime.rollout_backend import (
        build_sequence_tensors,
        format_generation_prompt,
        generate_data_gen_chunk,
        generate_sglang_chunk,
        generate_vllm_chunk,
        resolve_rollout_backend,
    )

    model.eval()
    world_size = dist_ctx.world_size if dist_ctx is not None else 1
    backend = resolve_rollout_backend(config, world_size=world_size)
    # slime: rollout_batch_size is number of prompts
    prompt_batch_size = max(1, int(config.rollout_batch_size))
    rollouts: list[Rollout] = []
    filter_stats = {
        "rollout_groups_total": 0.0,
        "rollout_groups_kept": 0.0,
        "dynamic_filtered_groups": 0.0,
        "rollout_backend_is_sglang": 1.0 if backend == "sglang" else 0.0,
        "rollout_backend_is_vllm": 1.0 if backend == "vllm" else 0.0,
    }
    for sample_chunk in _chunked(samples, prompt_batch_size):
        prompt_chunk = [
            format_generation_prompt(
                tokenizer,
                _sample_prompt_value(sample, config),
                config,
            )
            for sample in sample_chunk
        ]
        if backend in {"sglang", "vllm"}:
            if backend == "vllm":
                gen = generate_vllm_chunk(
                    tokenizer=tokenizer,
                    prompts=prompt_chunk,
                    config=config,
                )
            else:
                gen = generate_sglang_chunk(
                    tokenizer=tokenizer,
                    prompts=prompt_chunk,
                    config=config,
                )
            seq_rows = build_sequence_tensors(
                tokenizer=tokenizer,
                prompts=gen.prompts,
                completions=gen.completions,
                config=config,
                torch=torch,
                device=config.device,
                completion_token_ids=getattr(gen, "completion_token_ids", None),
            )
            completions = gen.completions
            use_hf_sequences = False
            generated = None
            prompt_width = 0
        else:
            # hf (colocated generate) — slime single-GPU colocate analogue
            gen = generate_data_gen_chunk(
                generation_model=_generation_model(model),
                tokenizer=tokenizer,
                prompts=prompt_chunk,
                config=config,
                torch=torch,
            )
            generated = gen.sequences
            prompt_width = int(gen.prompt_width or 0)
            completions = gen.completions
            seq_rows = None
            use_hf_sequences = True
        chunk_rollouts: list[Rollout] = []
        verifier_records: list[dict[str, Any]] = []
        total = len(completions)
        for idx in range(total):
            sample_idx = idx // config.rollouts_per_prompt
            sample = sample_chunk[sample_idx]
            completion = completions[idx]
            if use_hf_sequences:
                # HF generate returns a tensor; keep getitem for pylint E1136 on Optional[Any].
                assert generated is not None
                row = generated.__getitem__(idx)
                input_ids = row.detach()
                attention_mask = (row != tokenizer.pad_token_id).detach()
                response_mask = _response_mask_for_sequence(
                    input_ids,
                    prompt_width=prompt_width,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    torch=torch,
                )
                status = _rollout_status(
                    generated.__getitem__((idx, slice(prompt_width, None))),
                    tokenizer.eos_token_id,
                )
            else:
                assert seq_rows is not None
                row = seq_rows[idx]
                input_ids = row["input_ids"]
                attention_mask = row["attention_mask"]
                response_mask = row["response_mask"]
                status = _rollout_status(
                    input_ids[int(row["prompt_len"]) :],
                    tokenizer.eos_token_id,
                )
            # Score the raw model completion only — never rewrite tags for reward.
            reward_sample = _reward_sample(sample, config)
            metadata = _sample_metadata(sample, config)
            score = _score_completion(completion, reward_sample, config)
            chunk_rollouts.append(
                Rollout(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    old_logprobs=None,
                    ref_logprobs=None,
                    reward=score["reward"],
                    outcome_reward=score["outcome_reward"],
                    format_reward=score["format_reward"],
                    process_reward=score["process_reward"],
                    thinking_penalty=score["thinking_penalty"],
                    final_answer=score["final_answer"],
                    thinking_trace=score["thinking_trace"],
                    status=status,
                    outcome_passed=score["outcome_passed"],
                    format_ok=score["format_ok"],
                    checker=score["checker"],
                    proof_passed=score["proof_passed"],
                    proof_score=score["proof_score"],
                    proof_detail=score["proof_detail"],
                    )
                )
            if verifier_path is not None:
                prompt_text = (
                    prompt_chunk[sample_idx] if sample_idx < len(prompt_chunk) else gen.prompts[idx]
                )
                verifier_records.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "sample_index": sample_idx,
                        "rollout_index": idx % config.rollouts_per_prompt,
                        "reward": score["reward"],
                        "outcome_reward": score["outcome_reward"],
                        "format_reward": score["format_reward"],
                        "process_reward": score["process_reward"],
                        "thinking_penalty": score["thinking_penalty"],
                        "outcome_passed": score["outcome_passed"],
                        "format_ok": score["format_ok"],
                        "checker": score["checker"],
                        "proof_passed": score.get("proof_passed"),
                        "proof_score": score.get("proof_score"),
                        "proof_detail": score.get("proof_detail"),
                        "code_reward_mode": config.code_reward_mode,
                        "extracted_answer": _truncate_text(
                            score.get("extracted_answer", ""),
                            config.verifier_max_text_chars,
                        ),
                        "detail": score.get("detail"),
                        "reward_name": config.reward,
                        "rollout_backend": backend,
                        "status": status,
                        "metadata": _bounded_verifier_metadata(
                            metadata, config.verifier_max_text_chars
                        ),
                        "prompt": _truncate_text(prompt_text, config.verifier_max_text_chars),
                        "answer": _truncate_text(
                            reward_sample.get("answer", ""),
                            config.verifier_max_text_chars,
                        ),
                        "completion": _truncate_text(completion, config.verifier_max_text_chars),
                        "thinking_trace": _truncate_text(
                            score["thinking_trace"], config.verifier_max_text_chars
                        ),
                        "final_answer": _truncate_text(
                            score["final_answer"], config.verifier_max_text_chars
                        ),
                    }
                )

        _finalize_auto_code_rewards(chunk_rollouts, config)
        if verifier_records:
            _sync_verifier_records_with_rollouts(
                verifier_records, chunk_rollouts, config.rollouts_per_prompt
            )

        filter_stats["rollout_groups_total"] += len(sample_chunk)
        kept_rollouts, kept_group_indexes, rejected_groups = _filter_rollout_groups(
            chunk_rollouts,
            config,
        )
        filter_stats["rollout_groups_kept"] += len(kept_group_indexes)
        filter_stats["dynamic_filtered_groups"] += rejected_groups
        if not kept_rollouts:
            del gen
            if generated is not None:
                del generated
            continue
        verifier_records = [
            record
            for record in verifier_records
            if int(record["sample_index"]) in kept_group_indexes
        ]

        padded = _pad_rollouts(kept_rollouts, tokenizer.pad_token_id, config.device, torch)
        with torch.no_grad():
            old_token_logprobs = _sequence_token_logprobs(model, padded, torch).detach()
            old_logprobs = _masked_sequence_logprobs(
                old_token_logprobs,
                padded["response_mask"][:, 1:].float(),
            ).detach()
            ref_token_logprobs = (
                _sequence_token_logprobs(ref_model, padded, torch).detach()
                if ref_model is not None
                else None
            )
            ref_logprobs = (
                _masked_sequence_logprobs(
                    ref_token_logprobs,
                    padded["response_mask"][:, 1:].float(),
                ).detach()
                if ref_token_logprobs is not None
                else None
            )
        for idx, rollout in enumerate(kept_rollouts):
            token_length = max(0, int(rollout.input_ids.numel()) - 1)
            rollout.old_logprobs = old_logprobs[idx]
            rollout.ref_logprobs = ref_logprobs[idx] if ref_logprobs is not None else None
            rollout.old_token_logprobs = old_token_logprobs[idx, :token_length]
            rollout.ref_token_logprobs = (
                ref_token_logprobs[idx, :token_length] if ref_token_logprobs is not None else None
            )
        if verifier_records:
            _append_jsonl_records(verifier_path, verifier_records)
        rollouts.extend(kept_rollouts)
        del gen, padded
        if generated is not None:
            del generated

    _assign_grouped_advantages(
        rollouts,
        config.rollouts_per_prompt,
        grpo_std_normalization=config.grpo_std_normalization,
    )
    model.train()
    return _RolloutBatch(rollouts=rollouts, stats=filter_stats)


def _format_rollout_prompt(prompt: str, config: SingleGpuSlimeConfig) -> str:
    """Legacy prompt formatter (thinking open-tag only; no chat template)."""
    if not config.require_thinking_trace:
        return prompt
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{config.thinking_instruction}\n<think>"


def _score_completion(
    completion: str,
    sample: dict[str, Any],
    config: SingleGpuSlimeConfig,
) -> _CompletionScore:
    """Score raw generated text via the shared verifier (no synthetic tags)."""
    result = verify_score_completion(
        completion,
        sample,
        checker=config.reward,
        require_thinking_trace=config.require_thinking_trace,
        outcome_weight=config.outcome_reward_weight,
        format_weight=config.format_reward_weight,
        process_weight=config.process_reward_weight,
        missing_format_penalty=config.missing_thinking_penalty,
        min_thinking_tokens=config.min_thinking_tokens,
        code_reward_mode=config.code_reward_mode,
    )
    penalty = (
        config.missing_thinking_penalty
        if config.require_thinking_trace and not result.format_ok
        else 0.0
    )
    return {
        "reward": float(result.reward),
        "outcome_reward": float(result.outcome),
        "format_reward": float(result.format_score),
        "process_reward": float(result.process_score),
        "thinking_penalty": float(penalty),
        "thinking_trace": result.thinking_trace,
        "final_answer": result.final_answer,
        "extracted_answer": result.extracted_answer,
        "outcome_passed": bool(result.passed),
        "format_ok": bool(result.format_ok),
        "checker": str(result.checker),
        "detail": result.detail,
        "proof_passed": result.proof_passed,
        "proof_score": (float(result.proof_score) if result.proof_score is not None else None),
        "proof_detail": result.proof_detail,
    }


def _finalize_auto_code_rewards(
    rollouts: list[Rollout],
    config: SingleGpuSlimeConfig,
) -> None:
    """For ``code_reward_mode=auto``, promote to binary once a group has a full passer.

    Until then, keep dense pass-fraction outcomes so early GRPO still has signal.
    """
    if config.code_reward_mode != "auto":
        return
    group_size = config.rollouts_per_prompt
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        code = [rollout for rollout in group if rollout.proof_score is not None]
        if not code:
            continue
        use_binary = any(bool(rollout.proof_passed) for rollout in code)
        for rollout in code:
            outcome = (
                (1.0 if rollout.proof_passed else 0.0)
                if use_binary
                else float(rollout.proof_score or 0.0)
            )
            rollout.outcome_reward = outcome
            rollout.outcome_passed = bool(rollout.proof_passed)
            rollout.reward = (
                config.outcome_reward_weight * outcome
                + config.format_reward_weight * float(rollout.format_reward)
                + config.process_reward_weight * float(rollout.process_reward)
                - float(rollout.thinking_penalty)
            )


def _sync_verifier_records_with_rollouts(
    records: list[dict[str, Any]],
    rollouts: list[Rollout],
    group_size: int,
) -> None:
    """Refresh verifier JSONL fields after auto code-reward finalization."""
    del group_size
    if len(records) != len(rollouts):
        return
    for record, rollout in zip(records, rollouts, strict=True):
        record["reward"] = float(rollout.reward)
        record["outcome_reward"] = float(rollout.outcome_reward)
        record["outcome_passed"] = bool(rollout.outcome_passed)


def _split_thinking_trace(completion: str) -> tuple[str, str, bool]:
    return split_thinking_trace(completion)


def _process_reward(
    thinking_trace: str,
    final_answer: str,
    config: SingleGpuSlimeConfig,
) -> float:
    """Deprecated lexical process helper — prefer ``process_reward_weight=0``."""
    from seiso.rl_verify.verify import experimental_process_reward

    return experimental_process_reward(
        thinking_trace,
        final_answer,
        min_thinking_tokens=config.min_thinking_tokens,
    )


def _backprop_policy_step(
    model,
    rollouts: list[Rollout],
    pad_token_id: int,
    config: SingleGpuSlimeConfig,
    torch,
    *,
    loss_scale: float,
):
    total = len(rollouts)
    stats = _empty_stats()
    for chunk in _chunked(rollouts, config.policy_micro_batch_size):
        loss, chunk_stats = _policy_loss(model, chunk, pad_token_id, config, torch)
        weighted = len(chunk) / total
        (loss * weighted * loss_scale).backward()
        _merge_stats(stats, chunk_stats, weight=weighted)
        del loss
    return stats


def _build_optimizer(model, config: SingleGpuSlimeConfig):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters found for slime optimizer")
    if config.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("use_8bit_optimizer requires bitsandbytes") from exc
        return bnb.optim.AdamW8bit(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    import torch

    return torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=_is_cuda_device(config.device),
    )


def _apply_lora(model, config: SingleGpuSlimeConfig):
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError(
            "use_lora requires PEFT. Install training extras with `pip install -e '.[train]'`."
        ) from exc

    if _model_loaded_in_kbit(model):
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    target_modules = resolve_lora_target_modules(
        config.model_id,
        model,
        configured=config.lora_target_modules,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=_GRADIENT_CHECKPOINTING_KWARGS
        )
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def _model_loaded_in_kbit(model) -> bool:
    try:
        parameters = model.parameters()
    except AttributeError:
        return False
    return any(param.__class__.__name__ == "Params4bit" for param in parameters)


def _freeze_multimodal_backbones(model) -> None:
    """Keep non-text towers frozen for text-only slime on multimodal wrappers."""
    if not has_multimodal_language_model_backbone(model):
        return
    for container in (model, getattr(model, "model", None)):
        if container is None:
            continue
        for name in (
            "vision_tower",
            "audio_tower",
            "vision_model",
            "audio_model",
            "visual",
            "multi_modal_projector",
            "mm_projector",
            "embed_vision",
            "embed_audio",
        ):
            tower = getattr(container, name, None)
            if tower is not None and hasattr(tower, "requires_grad_"):
                tower.requires_grad_(False)


def _optimizer_step(model, optimizer, torch, config: SingleGpuSlimeConfig) -> None:
    torch.nn.utils.clip_grad_norm_(
        (param for param in model.parameters() if param.requires_grad),
        config.max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _assert_vram_fit(torch, config.max_vram_gb, config.device)


def _iter_sample_batches(
    config: SingleGpuSlimeConfig,
    rng: random.Random,
    dist_ctx: _DistributedSlimeContext | None = None,
    torch=None,
) -> Iterable[list[dict[str, Any]]]:
    if dist_ctx and dist_ctx.enabled:
        if torch is None:
            import torch as torch_mod

            torch = torch_mod
        yield from _iter_distributed_sample_batches(config, rng, dist_ctx, torch)
        return
    yield from _batched_records(_iter_shuffled_samples(config, rng), _sampling_batch_size(config))


def _sampling_batch_size(config: SingleGpuSlimeConfig) -> int:
    from seiso.slime.config import effective_train_batch_size

    # slime: sample over_sampling_batch_size prompts, keep until rollout_batch_size
    train = effective_train_batch_size(config)
    if config.dynamic_sampling_filter == "none":
        return train
    return max(train, config.over_sampling_batch_size or train)


def _sample_work_estimate(sample: dict[str, Any], config: SingleGpuSlimeConfig) -> int:
    prompt = sample.get(config.prompt_field, "")
    return max(1, len(str(prompt)))


def _iter_shuffled_samples(
    config: SingleGpuSlimeConfig,
    rng: random.Random,
    dist_ctx: _DistributedSlimeContext | None = None,
    target_samples: int | None = None,
) -> Iterable[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []
    yielded = 0
    for sample_index, sample in enumerate(_limited_samples(config)):
        if dist_ctx and dist_ctx.enabled and sample_index % dist_ctx.world_size != dist_ctx.rank:
            continue
        buffer.append(sample)
        if len(buffer) < config.shuffle_buffer_size:
            continue
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)
        yielded += 1
        if target_samples is not None and yielded >= target_samples:
            return

    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)
        yielded += 1
        if target_samples is not None and yielded >= target_samples:
            return


def _load_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    for sample in iter_jsonl(config.dataset):
        if config.prompt_field not in sample:
            raise ValueError(f"sample missing prompt field {config.prompt_field!r}")
        yield sample


def _sample_prompt_value(sample: dict[str, Any], config: SingleGpuSlimeConfig) -> Any:
    """Return raw prompt (string or slime chat-message list)."""
    value = sample.get(config.prompt_field)
    if value is None:
        raise ValueError(f"sample missing prompt field {config.prompt_field!r}")
    return value


def _limited_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    samples = _load_samples(config)
    if config.max_samples_per_epoch is None:
        return samples
    return itertools.islice(samples, config.max_samples_per_epoch)


def _reward_sample(sample: dict[str, Any], config: SingleGpuSlimeConfig) -> dict[str, Any]:
    """Normalize a JSONL row into the shared verifier sample dict.

    Supports slime fields (``label``, ``metadata.rm_type``, chat ``prompt``)
    and Seiso aliases (``answer``, top-level ``tests``).
    """
    merged = dict(sample)
    if config.reward == "field":
        merged["reward"] = sample.get(config.reward_field, 0.0)
    # slime label / Seiso answer
    label = sample.get(config.answer_field)
    if label is None:
        label = sample.get("label", sample.get("answer", ""))
    merged["answer"] = label
    metadata = _sample_metadata(sample, config)
    if isinstance(metadata, dict):
        merged["metadata"] = metadata
        # Flatten slime metadata into verifier fields
        if "tests" in metadata and "tests" not in merged:
            merged["tests"] = metadata["tests"]
        if "test" in metadata and "test" not in merged:
            merged["test"] = metadata["test"]
        if "timeout_s" in metadata and "timeout_s" not in merged:
            merged["timeout_s"] = metadata["timeout_s"]
        if "setup" in metadata and "setup" not in merged:
            merged["setup"] = metadata["setup"]
        rm_type = metadata.get("rm_type") or metadata.get("benchmark")
        if rm_type and not merged.get("benchmark"):
            merged["benchmark"] = rm_type
    elif metadata is not None:
        merged["metadata"] = metadata
    return merged


def _sample_metadata(sample: dict[str, Any], config: SingleGpuSlimeConfig) -> Any | None:
    if config.metadata_field is None or config.metadata_field not in sample:
        return None
    value = sample[config.metadata_field]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _bounded_verifier_metadata(metadata: Any | None, max_chars: int) -> Any | None:
    if metadata is None:
        return None
    try:
        encoded = json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError):
        encoded = str(metadata)
    if len(encoded) <= max_chars:
        return metadata
    return {"_truncated": _truncate_text(encoded, max_chars)}


def _resolve_dtype(torch, dtype: str):
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    choices = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return choices[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}") from exc


def _require_single_gpu(config: SingleGpuSlimeConfig) -> None:
    if not _is_cuda_device(config.device):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("single-GPU slime training requires CUDA when device='cuda'")
    torch.cuda.set_device(_cuda_device_index(config.device))


def _configure_vram_cap(torch, max_vram_gb: float | None, device: str) -> None:
    if max_vram_gb is None or not _is_cuda_device(device):
        return
    device_index = _cuda_device_index(device)
    total_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    if max_vram_gb > total_gb:
        raise RuntimeError(
            f"VRAM cap {max_vram_gb:.2f} GiB exceeds device capacity {total_gb:.2f} GiB"
        )
    fraction = max_vram_gb / total_gb
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def _assert_vram_fit(torch, max_vram_gb: float | None, device: str) -> None:
    if max_vram_gb is None or not _is_cuda_device(device):
        return
    torch.cuda.synchronize()
    used_gb = torch.cuda.memory_allocated(_cuda_device_index(device)) / 1024**3
    if used_gb > max_vram_gb:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"VRAM cap exceeded: allocated {used_gb:.2f} GiB > {max_vram_gb:.2f} GiB"
        )


def _set_seed(seed: int) -> None:
    apply_determinism(seed, deterministic=True)


def _batched_records(
    records: Iterable[dict[str, Any]],
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _chunked(items: list[T], chunk_size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    metric = _metric_record(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
    if os.environ.get("SEISO_EMIT_METRICS_STDOUT") == "1":
        print(f"{METRIC_STDOUT_PREFIX}{json.dumps(metric)}", flush=True)


def _metric_record(record: dict[str, Any]) -> dict[str, Any]:
    metric = {
        "type": "training",
        "reward": record.get("reward_mean"),
        **record,
    }
    return {k: v for k, v in metric.items() if v is not None}


def _append_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _truncate_text(text: Any, max_chars: int) -> str:
    value = str(text)
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def _metric_is_minimized(metric: str) -> bool:
    return metric == "kl" or metric.endswith("loss") or metric.endswith("_loss")


def _check_training_health(
    stats: dict[str, float],
    config: SingleGpuSlimeConfig,
) -> str | None:
    if not config.stop_on_nonfinite:
        return None
    for key, value in stats.items():
        if isinstance(value, int | float) and not math.isfinite(float(value)):
            return f"nonfinite:{key}"
    return None


def _auto_stop_stats(
    controller: _AutoStopController,
) -> dict[str, float | int | str | None]:
    return {
        "best_metric": controller.best_value,
        "best_step": controller.best_step,
        "auto_stop_metric": controller.metric,
        "stale_steps": controller.stale_steps,
    }


def _final_output_dir(config: SingleGpuSlimeConfig) -> Path:
    if config.final_checkpoint_dir:
        return config.output_dir / config.final_checkpoint_dir
    return config.output_dir


def _write_training_state(
    config: SingleGpuSlimeConfig,
    global_step: int,
    stop_reason: str | None,
    controller: _AutoStopController,
    dist_ctx: _DistributedSlimeContext | None = None,
) -> None:
    if dist_ctx and not dist_ctx.is_main:
        _distributed_barrier(dist_ctx)
        return
    state = {
        "global_step": global_step,
        "stop_reason": stop_reason,
        "best_checkpoint_dir": str(config.output_dir / config.best_checkpoint_dir),
        "final_checkpoint_dir": str(_final_output_dir(config)),
        **_auto_stop_stats(controller),
    }
    if dist_ctx and dist_ctx.enabled:
        state.update(
            {
                "distributed": True,
                "world_size": dist_ctx.world_size,
                "rank_zero_only_artifacts": True,
            }
        )
    (config.output_dir / "slime_training_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _distributed_barrier(dist_ctx)


def _save(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _unwrap_model(model)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _sync_rollout_engine_weights(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
    step: int,
    sync_state: Any | None = None,
) -> None:
    """Push actor weights to SGLang/vLLM (initial + post-step; single- + multi-GPU).

    * SGLang — slime disk transport: full HF checkpoint and/or delta + multi-engine
      ``update_weights_from_disk`` / ``pull_weights``
    * vLLM — PEFT adapter via ``/v1/load_lora_adapter`` (preferred) or best-effort
      full disk reload endpoints

    No-op for ``rollout_backend=hf``. On failure, all ranks raise after a barrier
    so DDP does not hang.
    """
    import torch

    from seiso.slime.rollout_backend import (
        resolve_rollout_backend,
        sync_sglang_weights_from_actor,
        sync_vllm_weights_from_actor,
    )

    backend = resolve_rollout_backend(config, world_size=dist_ctx.world_size)
    if backend not in {"sglang", "vllm"}:
        return
    sync_enabled = (
        bool(getattr(config, "sglang_sync_weights", True))
        if backend == "sglang"
        else bool(getattr(config, "vllm_sync_weights", True))
    )
    if not sync_enabled:
        _distributed_barrier(dist_ctx)
        return

    error_msg = ""
    path: str | None = None
    if backend == "vllm":
        label = "vllm_weight_sync"
        weight_dir = getattr(config, "vllm_weight_dir", "vllm_weight_sync")
        disable_hint = "Set vllm_sync_weights: false only for debugging."
        engine_hint = (
            "Ensure each vLLM engine has --enable-lora (for LoRA mode) or "
            "exposes a disk weight reload endpoint (full mode) and can read "
            f"{config.output_dir / weight_dir}."
        )
        sync_fn = sync_vllm_weights_from_actor
    else:
        label = "sglang_weight_sync"
        weight_dir = config.sglang_weight_dir
        disable_hint = "Set sglang_sync_weights: false only for debugging."
        engine_hint = (
            "Ensure each engine exposes /update_weights_from_disk and can read "
            f"{config.output_dir / weight_dir}."
        )
        sync_fn = sync_sglang_weights_from_actor
    try:
        path = sync_fn(
            model=model,
            tokenizer=tokenizer,
            config=config,
            step=step,
            is_main=dist_ctx.is_main,
            active_backend=backend,
            sync_state=sync_state,
        )
        if dist_ctx.is_main and path is not None and config.log_every_steps:
            mode = getattr(sync_state, "last_mode", None) if sync_state else None
            print(
                f"{label} step={step} mode={mode} path={path}",
                flush=True,
            )
    except Exception as exc:
        error_msg = (
            f"{backend} weight sync failed at step={step}: {exc}. {engine_hint} {disable_hint}"
        )
    if dist_ctx.enabled:
        # Broadcast failure so every rank raises after the barrier.
        flag = torch.tensor(
            [0 if error_msg else 1],
            device=dist_ctx.device,
            dtype=torch.long,
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        _distributed_barrier(dist_ctx)
        if int(flag.item()) == 0:
            # Rank0 has the detailed message; others get a generic abort.
            if error_msg:
                raise RuntimeError(error_msg)
            raise RuntimeError(f"{backend} weight sync failed at step={step} on another rank")
        return

    if error_msg:
        raise RuntimeError(error_msg)


def _maybe_materialize_data_gen(
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
) -> SingleGpuSlimeConfig:
    """Build a verifiable prompt corpus when high-level data_gen is requested.

    Multi-GPU vLLM runs use NVIDIA NeMo Data Designer (local vLLM provider) for
    numeric/choice synth; other backends keep Seiso's deterministic generator.
    """
    enabled = config.data_gen or config.data_gen_count > 0
    if not enabled:
        return config
    count = config.data_gen_count if config.data_gen_count > 0 else 500
    out_path = config.output_dir / config.data_gen_filename
    if dist_ctx.is_main:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        from seiso.rl_verify.data_designer_gen import (
            materialize_for_slime_config,
            should_use_data_designer,
        )
        from seiso.rl_verify.data_gen import DataGenConfig, materialize_rl_corpus

        use_dd = should_use_data_designer(config, world_size=dist_ctx.world_size)
        generator = "seiso.rl_verify.data_gen"
        if use_dd:
            try:
                result = materialize_for_slime_config(
                    config,
                    out_path=out_path,
                    count=count,
                    world_size=dist_ctx.world_size,
                )
                generator = "nvidia.nemo.data_designer"
            except ImportError as exc:
                print(
                    f"Data Designer unavailable ({exc}); falling back to Seiso data_gen",
                    flush=True,
                )
                result = materialize_rl_corpus(
                    out_path,
                    DataGenConfig(
                        count=count,
                        seed=(
                            config.data_gen_seed
                            if config.data_gen_seed is not None
                            else config.seed
                        ),
                        mix=config.data_gen_mix,
                        difficulty=config.data_gen_difficulty,
                        require_thinking_trace=config.require_thinking_trace,
                        thinking_instruction=config.thinking_instruction,
                    ),
                )
        else:
            result = materialize_rl_corpus(
                out_path,
                DataGenConfig(
                    count=count,
                    seed=(
                        config.data_gen_seed if config.data_gen_seed is not None else config.seed
                    ),
                    mix=config.data_gen_mix,
                    difficulty=config.data_gen_difficulty,
                    require_thinking_trace=config.require_thinking_trace,
                    thinking_instruction=config.thinking_instruction,
                ),
            )
        summary = result.summary()
        (config.output_dir / "slime_data_gen_summary.json").write_text(
            json.dumps(
                {
                    **summary,
                    "path": str(out_path),
                    "generator": generator,
                    "data_designer": use_dd,
                    "note": (
                        "Prompts+labels only; completions come from online "
                        "rollouts (HF / SGLang / vLLM), not this file."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _distributed_barrier(dist_ctx)
    if not out_path.is_file():
        raise RuntimeError(f"data_gen did not produce {out_path}; rank0 materialization failed")
    return replace(config, dataset=out_path)


# Compatibility alias (historical name; supports multi-GPU / remote backends).
train_single_gpu_slime = train_slime

# Re-export private helpers so patches on seiso.slime.trainer.* still resolve.
for _mod_name in (
    "seiso.slime.types",
    "seiso.slime.policy",
    "seiso.slime.distributed",
):
    _mod = importlib.import_module(_mod_name)
    globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
