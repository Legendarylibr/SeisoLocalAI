"""Single-GPU slime-style GRPO trainer for Hugging Face causal LMs."""

from __future__ import annotations

import gc
import itertools
import json
import math
import os
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from seiso.io.jsonl import iter_jsonl
from seiso.models.lora_targets import (
    has_multimodal_language_model_backbone,
    resolve_lora_target_modules,
)
from seiso.research.provenance import apply_determinism
from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.rewards import resolve_reward
from seiso.training.metrics import METRIC_STDOUT_PREFIX

_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}

T = TypeVar("T")


@dataclass
class Rollout:
    input_ids: Any
    attention_mask: Any
    response_mask: Any
    old_logprobs: Any
    ref_logprobs: Any | None
    reward: float
    outcome_reward: float = 0.0
    process_reward: float = 0.0
    thinking_penalty: float = 0.0
    final_answer: str = ""
    thinking_trace: str = ""
    advantage: float = 0.0


@dataclass(frozen=True)
class _DistributedSlimeContext:
    enabled: bool
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    device: str = "cuda"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class _AutoStopDecision:
    improved: bool = False
    should_stop: bool = False
    reason: str | None = None


@dataclass
class _AutoStopController:
    enabled: bool
    metric: str
    patience: int
    min_delta: float
    warmup_steps: int
    best_value: float | None = None
    best_step: int | None = None
    stale_steps: int = 0

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> _AutoStopController:
        return cls(
            enabled=config.auto_stop,
            metric=config.auto_stop_metric,
            patience=config.auto_stop_patience,
            min_delta=config.auto_stop_min_delta,
            warmup_steps=config.auto_stop_warmup_steps,
        )

    def update(self, step: int, stats: dict[str, float]) -> _AutoStopDecision:
        value = stats.get(self.metric)
        if value is None or not math.isfinite(value):
            return _AutoStopDecision()

        improved = self._is_better(value)
        if improved:
            self.best_value = value
            self.best_step = step
            self.stale_steps = 0
            return _AutoStopDecision(improved=True)

        if not self.enabled or step < self.warmup_steps:
            return _AutoStopDecision()

        self.stale_steps += 1
        if self.stale_steps >= self.patience:
            return _AutoStopDecision(
                should_stop=True,
                reason=f"auto_stop:{self.metric}_plateau",
            )
        return _AutoStopDecision()

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if _metric_is_minimized(self.metric):
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta


def train_single_gpu_slime(config: SingleGpuSlimeConfig) -> Path:
    """Run a compact slime-style rollout/reward/update loop.

    When launched under Accelerate with WORLD_SIZE > 1, this keeps the original
    local SLIME algorithm but shards prompts by rank and synchronizes policy
    updates with PyTorch DDP so multi-node jobs behave as one training run.
    """

    config.validate()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist_ctx = _distributed_context(torch, config)
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
    reward_fn = resolve_reward(config.reward)

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
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(config.seed)

    for epoch in range(config.epochs):
        for batch_samples in _iter_sample_batches(config, rng, dist_ctx, torch):
            saw_sample = True
            rollouts = _collect_rollouts(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                samples=batch_samples,
                config=config,
                reward_fn=reward_fn,
                torch=torch,
                epoch=epoch,
                global_step=global_step,
                verifier_path=verifier_path,
            )
            if not rollouts:
                continue
            stats = _backprop_policy_step(
                model,
                rollouts,
                tokenizer.pad_token_id,
                config,
                torch,
                loss_scale=1.0 / config.gradient_accumulation_steps,
            )
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
                _write_training_state(config, global_step, "max_steps", auto_stop, dist_ctx)
                _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
                return final_output_dir

    if not saw_sample:
        raise ValueError(f"no samples found in {config.dataset}")
    if pending_accumulation_steps:
        _optimizer_step(model, optimizer, torch, config)
    _write_training_state(config, global_step, "complete", auto_stop, dist_ctx)
    _save_distributed(model, tokenizer, final_output_dir, dist_ctx)
    return final_output_dir


def _collect_rollouts(
    *,
    model,
    ref_model,
    tokenizer,
    samples: list[dict[str, Any]],
    config: SingleGpuSlimeConfig,
    reward_fn,
    torch,
    epoch: int,
    global_step: int,
    verifier_path: Path | None,
) -> list[Rollout]:
    model.eval()
    prompt_batch_size = max(1, config.rollout_batch_size // config.rollouts_per_prompt)
    rollouts: list[Rollout] = []
    for sample_chunk in _chunked(samples, prompt_batch_size):
        prompt_chunk = [
            _format_rollout_prompt(str(sample[config.prompt_field]), config)
            for sample in sample_chunk
        ]
        encoded = tokenizer(
            prompt_chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_tokens,
        ).to(config.device)
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = _generation_model(model).generate(
                **encoded,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=config.rollouts_per_prompt,
            )

        completions = tokenizer.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
        )
        chunk_rollouts: list[Rollout] = []
        verifier_records: list[dict[str, Any]] = []
        for idx in range(int(generated.shape[0])):
            sample_idx = idx // config.rollouts_per_prompt
            sample = sample_chunk[sample_idx]
            response_mask = torch.zeros_like(generated[idx], dtype=torch.bool)
            response_mask[prompt_width:] = generated[idx, prompt_width:] != tokenizer.pad_token_id
            completion = _force_completion_thinking_prefix(completions[idx], config)
            reward_sample = _reward_sample(sample, config)
            score = _score_completion(completion, reward_sample, config, reward_fn)
            # Keep rollout tensors on device to avoid GPU↔CPU staging per sample.
            chunk_rollouts.append(
                Rollout(
                    input_ids=generated[idx].detach(),
                    attention_mask=(generated[idx] != tokenizer.pad_token_id).detach(),
                    response_mask=response_mask.detach(),
                    old_logprobs=None,
                    ref_logprobs=None,
                    reward=score["reward"],
                    outcome_reward=score["outcome_reward"],
                    process_reward=score["process_reward"],
                    thinking_penalty=score["thinking_penalty"],
                    final_answer=score["final_answer"],
                    thinking_trace=score["thinking_trace"],
                )
            )
            if verifier_path is not None:
                verifier_records.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "sample_index": sample_idx,
                        "rollout_index": idx % config.rollouts_per_prompt,
                        "reward": score["reward"],
                        "outcome_reward": score["outcome_reward"],
                        "process_reward": score["process_reward"],
                        "thinking_penalty": score["thinking_penalty"],
                        "reward_name": config.reward,
                        "prompt": _truncate_text(
                            prompt_chunk[sample_idx], config.verifier_max_text_chars
                        ),
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

        padded = _pad_rollouts(chunk_rollouts, tokenizer.pad_token_id, config.device, torch)
        with torch.no_grad():
            old_logprobs = _sequence_logprobs(model, padded, torch).detach()
            ref_logprobs = (
                _sequence_logprobs(ref_model, padded, torch).detach()
                if ref_model is not None
                else None
            )
        for idx, rollout in enumerate(chunk_rollouts):
            rollout.old_logprobs = old_logprobs[idx]
            rollout.ref_logprobs = ref_logprobs[idx] if ref_logprobs is not None else None
        if verifier_records:
            _append_jsonl_records(verifier_path, verifier_records)
        rollouts.extend(chunk_rollouts)
        del encoded, generated, padded

    _assign_grouped_advantages(rollouts, config.rollouts_per_prompt)
    model.train()
    return rollouts


def _format_rollout_prompt(prompt: str, config: SingleGpuSlimeConfig) -> str:
    if not config.require_thinking_trace:
        return prompt
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{config.thinking_instruction}\n<think>"


def _force_completion_thinking_prefix(
    completion: str,
    config: SingleGpuSlimeConfig,
) -> str:
    if not config.require_thinking_trace:
        return completion
    if completion.lstrip().lower().startswith("<think>"):
        return completion
    return f"<think>{completion}"


def _score_completion(
    completion: str,
    sample: dict[str, Any],
    config: SingleGpuSlimeConfig,
    reward_fn,
) -> dict[str, float | str]:
    thinking_trace, final_answer, has_trace = _split_thinking_trace(completion)
    answer_for_outcome = final_answer if config.require_thinking_trace else completion
    outcome = float(reward_fn(answer_for_outcome, sample))
    process = (
        _process_reward(thinking_trace, final_answer, config)
        if config.require_thinking_trace
        else 0.0
    )
    penalty = (
        config.missing_thinking_penalty if config.require_thinking_trace and not has_trace else 0.0
    )
    reward = (
        config.outcome_reward_weight * outcome + config.process_reward_weight * process - penalty
    )
    return {
        "reward": float(reward),
        "outcome_reward": outcome,
        "process_reward": float(process),
        "thinking_penalty": float(penalty),
        "thinking_trace": thinking_trace,
        "final_answer": final_answer,
    }


def _split_thinking_trace(completion: str) -> tuple[str, str, bool]:
    match = re.search(
        r"<think>(?P<trace>.*?)</think>(?P<final>.*)",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        open_match = re.search(
            r"<think>(?P<trace>.*)",
            completion,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if open_match is not None:
            return open_match.group("trace").strip(), "", False
        return "", completion.strip(), False
    trace = match.group("trace").strip()
    final = match.group("final").strip()
    return trace, final, True


def _process_reward(
    thinking_trace: str,
    final_answer: str,
    config: SingleGpuSlimeConfig,
) -> float:
    tokens = re.findall(r"\w+", thinking_trace)
    if not tokens:
        return 0.0

    score = 0.0
    if len(tokens) >= config.min_thinking_tokens:
        score += 0.35
    else:
        score += 0.35 * (len(tokens) / max(config.min_thinking_tokens, 1))

    lower = thinking_trace.lower()
    transition_hits = sum(
        marker in lower
        for marker in (
            "because",
            "therefore",
            "so",
            "first",
            "next",
            "then",
            "check",
            "verify",
        )
    )
    score += min(0.35, 0.07 * transition_hits)

    if re.search(r"\b(wait|actually|however|but|correct|revise)\b", lower):
        score += 0.15
    if final_answer.strip():
        score += 0.15

    return min(1.0, score)


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


def _policy_loss(
    model,
    rollouts: list[Rollout],
    pad_token_id: int,
    config: SingleGpuSlimeConfig,
    torch,
):
    padded = _pad_rollouts(rollouts, pad_token_id, config.device, torch)
    new_logprobs = _sequence_logprobs(model, padded, torch)
    old_logprobs = torch.stack([r.old_logprobs for r in rollouts]).to(config.device)
    advantages = torch.tensor([r.advantage for r in rollouts], device=config.device)

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    kl_loss = torch.zeros((), device=config.device)
    if config.kl_coef > 0 and rollouts[0].ref_logprobs is not None:
        ref_logprobs = torch.stack([r.ref_logprobs for r in rollouts]).to(config.device)
        kl_loss = (new_logprobs - ref_logprobs).mean()

    rewards = [r.reward for r in rollouts]
    loss = policy_loss + config.kl_coef * kl_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "reward_mean": float(sum(rewards) / len(rewards)),
        "reward_max": float(max(rewards)),
        "outcome_reward_mean": _mean(r.outcome_reward for r in rollouts),
        "process_reward_mean": _mean(r.process_reward for r in rollouts),
        "thinking_penalty_mean": _mean(r.thinking_penalty for r in rollouts),
        "group_reward_spread_mean": _group_reward_spread_mean(rollouts, config.rollouts_per_prompt),
    }


def _empty_stats() -> dict[str, float]:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl": 0.0,
        "reward_mean": 0.0,
        "reward_max": float("-inf"),
        "outcome_reward_mean": 0.0,
        "process_reward_mean": 0.0,
        "thinking_penalty_mean": 0.0,
        "group_reward_spread_mean": 0.0,
    }


def _merge_stats(
    stats: dict[str, float],
    chunk_stats: dict[str, float],
    *,
    weight: float,
) -> None:
    for key in (
        "loss",
        "policy_loss",
        "kl",
        "reward_mean",
        "outcome_reward_mean",
        "process_reward_mean",
        "thinking_penalty_mean",
        "group_reward_spread_mean",
    ):
        stats[key] += chunk_stats.get(key, 0.0) * weight
    stats["reward_max"] = max(stats["reward_max"], chunk_stats.get("reward_max", 0.0))


def _sequence_logprobs(model, batch: dict[str, Any], torch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].float()
    token_logprobs = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(
            -1,
            labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (token_logprobs * mask).sum(dim=1) / denom


def _pad_rollouts(rollouts: list[Rollout], pad_token_id: int, device: str, torch) -> dict[str, Any]:
    max_len = max(int(r.input_ids.numel()) for r in rollouts)
    # Prefer stacking on-device tensors; fall back to per-row .to(device).
    same_device = all(
        getattr(r.input_ids, "device", None) is not None
        and str(r.input_ids.device) == str(torch.device(device))
        for r in rollouts
    )
    if same_device and all(int(r.input_ids.numel()) == max_len for r in rollouts):
        return {
            "input_ids": torch.stack([r.input_ids for r in rollouts], dim=0),
            "attention_mask": torch.stack([r.attention_mask for r in rollouts], dim=0),
            "response_mask": torch.stack([r.response_mask for r in rollouts], dim=0),
        }

    input_ids = torch.full((len(rollouts), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rollouts), max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((len(rollouts), max_len), dtype=torch.bool, device=device)
    for idx, rollout in enumerate(rollouts):
        length = int(rollout.input_ids.numel())
        input_ids[idx, :length] = rollout.input_ids.to(device)
        attention_mask[idx, :length] = rollout.attention_mask.to(device)
        response_mask[idx, :length] = rollout.response_mask.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def _assign_grouped_advantages(rollouts: list[Rollout], group_size: int) -> None:
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        rewards = [r.reward for r in group]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance) or 1.0
        for rollout in group:
            rollout.advantage = (rollout.reward - mean) / std


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _group_reward_spread_mean(rollouts: list[Rollout], group_size: int) -> float:
    spreads: list[float] = []
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        rewards = [r.reward for r in group]
        if rewards:
            spreads.append(max(rewards) - min(rewards))
    return _mean(spreads)


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
    yield from _batched_records(_iter_shuffled_samples(config, rng), config.train_batch_size)


def _iter_distributed_sample_batches(
    config: SingleGpuSlimeConfig,
    rng: random.Random,
    dist_ctx: _DistributedSlimeContext,
    torch,
) -> Iterable[list[dict[str, Any]]]:
    local_count = _count_rank_samples(config, dist_ctx)
    local_batches = local_count // config.train_batch_size
    min_batches = _distributed_min_int(local_batches, torch, dist_ctx)
    if min_batches < 1:
        raise ValueError(
            "not enough sharded samples for distributed slime; "
            "need at least train_batch_size samples per rank"
        )
    target_samples = min_batches * config.train_batch_size
    yield from _batched_records(
        _iter_shuffled_samples(config, rng, dist_ctx, target_samples),
        config.train_batch_size,
    )


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


def _count_rank_samples(
    config: SingleGpuSlimeConfig,
    dist_ctx: _DistributedSlimeContext,
) -> int:
    count = 0
    for sample_index, _sample in enumerate(_limited_samples(config)):
        if sample_index % dist_ctx.world_size == dist_ctx.rank:
            count += 1
    return count


def _load_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    for sample in iter_jsonl(config.dataset):
        if config.prompt_field not in sample:
            raise ValueError(f"sample missing prompt field {config.prompt_field!r}")
        yield sample


def _limited_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    samples = _load_samples(config)
    if config.max_samples_per_epoch is None:
        return samples
    return itertools.islice(samples, config.max_samples_per_epoch)


def _reward_sample(sample: dict[str, Any], config: SingleGpuSlimeConfig) -> dict[str, Any]:
    if config.reward == "field":
        merged = dict(sample)
        merged["reward"] = sample.get(config.reward_field, 0.0)
        return merged
    if config.answer_field == "answer":
        return sample
    merged = dict(sample)
    merged["answer"] = sample.get(config.answer_field, "")
    return merged


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


def _save_distributed(
    model,
    tokenizer,
    output_dir: Path,
    dist_ctx: _DistributedSlimeContext,
) -> None:
    if dist_ctx.is_main:
        _save(model, tokenizer, output_dir)
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
    mean_keys = [key for key in reduced if key != "reward_max"]
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
