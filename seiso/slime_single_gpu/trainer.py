"""Single-GPU slime-style GRPO trainer for Hugging Face causal LMs."""

from __future__ import annotations

import gc
import json
import math
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
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
    ema_alpha: float = 0.0
    best_value: float | None = None
    best_step: int | None = None
    stale_steps: int = 0
    ema_value: float | None = None

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> _AutoStopController:
        return cls(
            enabled=config.auto_stop,
            metric=config.auto_stop_metric,
            patience=config.auto_stop_patience,
            min_delta=config.auto_stop_min_delta,
            warmup_steps=config.auto_stop_warmup_steps,
            ema_alpha=float(getattr(config, "auto_stop_ema_alpha", 0.0) or 0.0),
        )

    def update(self, step: int, stats: dict[str, float]) -> _AutoStopDecision:
        value = stats.get(self.metric)
        if value is None or not math.isfinite(value):
            return _AutoStopDecision()

        score = self._smoothed(float(value))
        stats["metric_smooth"] = score
        improved = self._is_better(score)
        if improved:
            self.best_value = score
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

    def _smoothed(self, value: float) -> float:
        if self.ema_alpha <= 0.0:
            return value
        if self.ema_value is None:
            self.ema_value = value
        else:
            a = self.ema_alpha
            self.ema_value = a * value + (1.0 - a) * self.ema_value
        return float(self.ema_value)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if _metric_is_minimized(self.metric):
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta


def train_single_gpu_slime(config: SingleGpuSlimeConfig) -> Path:
    """Run a compact slime-style rollout/reward/update loop on one GPU."""

    config.validate()
    _require_single_gpu(config)
    _set_seed(config.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    if config.adapter_path:
        model = _load_existing_adapter(model, config)
    elif config.use_lora:
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

    _assert_vram_fit(torch, config.max_vram_gb, config.device)
    optimizer = _build_optimizer(model, config)
    reward_fn = resolve_reward(config.reward)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "slime_single_gpu_metrics.jsonl"
    verifier_path = (
        config.output_dir / config.verifier_data_file
        if config.write_verifier_data
        else None
    )
    final_output_dir = _final_output_dir(config)
    best_checkpoint_dir = config.output_dir / config.best_checkpoint_dir
    auto_stop = _AutoStopController.from_config(config)
    global_step = 0
    pending_accumulation_steps = 0
    saw_sample = False
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(config.seed)

    for epoch in range(config.epochs):
        for batch_samples in _iter_sample_batches(config, rng):
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
            pending_accumulation_steps += 1

            health_reason = _check_training_health(stats, config)
            if health_reason:
                optimizer.zero_grad(set_to_none=True)
                if global_step % config.log_every_steps == 0:
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
                _write_training_state(config, global_step, health_reason, auto_stop)
                _save(model, tokenizer, final_output_dir)
                return final_output_dir

            if pending_accumulation_steps >= config.gradient_accumulation_steps:
                _optimizer_step(model, optimizer, torch, config)
                pending_accumulation_steps = 0

            decision = auto_stop.update(global_step, stats)
            if decision.improved:
                _save(model, tokenizer, best_checkpoint_dir)

            if global_step % config.log_every_steps == 0:
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
                _save(model, tokenizer, config.output_dir / f"checkpoint-{global_step}")

            global_step += 1
            if decision.should_stop:
                optimizer.zero_grad(set_to_none=True)
                _write_training_state(config, global_step, decision.reason, auto_stop)
                _save(model, tokenizer, final_output_dir)
                return final_output_dir
            if config.max_steps is not None and global_step >= config.max_steps:
                if pending_accumulation_steps:
                    _optimizer_step(model, optimizer, torch, config)
                _write_training_state(config, global_step, "max_steps", auto_stop)
                _save(model, tokenizer, final_output_dir)
                return final_output_dir

    if not saw_sample:
        raise ValueError(f"no samples found in {config.dataset}")
    if pending_accumulation_steps:
        _optimizer_step(model, optimizer, torch, config)
    _write_training_state(config, global_step, "complete", auto_stop)
    _save(model, tokenizer, final_output_dir)
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
            _format_rollout_prompt(
                str(sample[config.prompt_field]), config, tokenizer=tokenizer
            )
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
            generated = _generate_rollout_sequences(
                model=model,
                encoded=encoded,
                tokenizer=tokenizer,
                config=config,
                torch=torch,
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
            response_mask[prompt_width:] = (
                generated[idx, prompt_width:] != tokenizer.pad_token_id
            )
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
                        "reward_name": score.get("reward_name", config.reward),
                        "prompt": _truncate_text(
                            prompt_chunk[sample_idx], config.verifier_max_text_chars
                        ),
                        "answer": _truncate_text(
                            reward_sample.get("answer", ""),
                            config.verifier_max_text_chars,
                        ),
                        "completion": _truncate_text(
                            completion, config.verifier_max_text_chars
                        ),
                        "thinking_trace": _truncate_text(
                            score["thinking_trace"], config.verifier_max_text_chars
                        ),
                        "final_answer": _truncate_text(
                            score["final_answer"], config.verifier_max_text_chars
                        ),
                    }
                )

        # Score log-probs one rollout at a time. Batching 4 long sequences and
        # materializing vocab log_softmax is a common 24GB OOM on 8B models.
        with torch.no_grad():
            for rollout in chunk_rollouts:
                single = _pad_rollouts(
                    [rollout], tokenizer.pad_token_id, config.device, torch
                )
                rollout.old_logprobs = _sequence_logprobs(model, single, torch)[
                    0
                ].detach()
                if ref_model is not None:
                    rollout.ref_logprobs = _sequence_logprobs(
                        ref_model, single, torch
                    )[0].detach()
                else:
                    rollout.ref_logprobs = None
                del single
                if config.device == "cuda":
                    torch.cuda.empty_cache()
        if verifier_records:
            _append_jsonl_records(verifier_path, verifier_records)
        rollouts.extend(chunk_rollouts)
        del encoded, generated
        if config.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    _assign_grouped_advantages(rollouts, config.rollouts_per_prompt)
    model.train()
    return rollouts


def _generate_rollout_sequences(
    *,
    model,
    encoded,
    tokenizer,
    config: SingleGpuSlimeConfig,
    torch,
):
    """Sample group rollouts, optionally one at a time to reduce peak VRAM."""
    gen_kwargs = {
        "do_sample": True,
        "temperature": max(float(config.temperature), 1e-5),
        "top_p": config.top_p,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    n = int(config.rollouts_per_prompt)
    if n <= 1 or not getattr(config, "sequential_rollouts", True):
        return model.generate(
            **encoded,
            num_return_sequences=n,
            **gen_kwargs,
        )

    # Sequential path: generate one full group member per call, then interleave
    # so layout matches HF num_return_sequences (p0r0, p0r1, ..., p1r0, ...).
    batch_size = int(encoded["input_ids"].shape[0])
    per_return: list[Any] = []
    for _ in range(n):
        gen = model.generate(
            **encoded,
            num_return_sequences=1,
            **gen_kwargs,
        )
        per_return.append(gen)
        if config.device == "cuda":
            torch.cuda.empty_cache()

    max_len = max(int(g.shape[1]) for g in per_return)
    pad_id = int(tokenizer.pad_token_id)
    # Left-pad prompts already; pad generation on the right to max_len.
    padded_returns = []
    for gen in per_return:
        if int(gen.shape[1]) == max_len:
            padded_returns.append(gen)
            continue
        pad = torch.full(
            (gen.shape[0], max_len - int(gen.shape[1])),
            pad_id,
            dtype=gen.dtype,
            device=gen.device,
        )
        padded_returns.append(torch.cat([gen, pad], dim=1))

    # Stack returns then reorder: [r0_b0, r0_b1, ...] -> [b0_r0, b0_r1, ...]
    stacked = torch.stack(padded_returns, dim=1)  # [B, R, L]
    interleaved = stacked.reshape(batch_size * n, max_len)
    return interleaved


def _format_rollout_prompt(
    prompt: str,
    config: SingleGpuSlimeConfig,
    tokenizer=None,
) -> str:
    """Build a generation prompt; use chat template when the tokenizer provides one.

    Qwen3-style models need ``apply_chat_template`` (and optional
    ``enable_thinking``) or they free-run without structured answers/code.
    """
    text = str(prompt)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # When we require a thinking trace, leave thinking enabled; otherwise
        # disable it so coding models emit the final solution sooner.
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=bool(config.require_thinking_trace),
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    if not config.require_thinking_trace:
        return text
    if "<think>" in text.lower():
        return text
    return f"{text.rstrip()}\n\n{config.thinking_instruction}\n<think>"


def _force_completion_thinking_prefix(
    completion: str,
    config: SingleGpuSlimeConfig,
) -> str:
    if not config.require_thinking_trace:
        return completion
    lower = completion.lower()
    if "<think>" in lower or "</think>" in lower:
        return completion
    return f"<think>{completion}"


def _score_completion(
    completion: str,
    sample: dict[str, Any],
    config: SingleGpuSlimeConfig,
    reward_fn,
) -> dict[str, float | str]:
    from seiso.slime_single_gpu.rewards import (
        infer_reward_name,
        uses_unit_tests_scoring,
    )

    thinking_trace, final_answer, has_trace = _split_thinking_trace(completion)
    answer_for_outcome = final_answer if config.require_thinking_trace else completion
    # For code unit tests, accept a fenced solution anywhere in the completion.
    # multi/auto also uses this path when the sample itself is coding.
    if uses_unit_tests_scoring(sample, config.reward):
        outcome = max(
            float(reward_fn(answer_for_outcome, sample)),
            float(reward_fn(completion, sample)),
            float(reward_fn(final_answer, sample)) if final_answer else 0.0,
        )
    else:
        outcome = float(reward_fn(answer_for_outcome, sample))
    process = (
        _process_reward(thinking_trace, final_answer, config)
        if config.require_thinking_trace
        else 0.0
    )
    penalty = (
        config.missing_thinking_penalty
        if config.require_thinking_trace and not has_trace
        else 0.0
    )
    reward = (
        config.outcome_reward_weight * outcome
        + config.process_reward_weight * process
        - penalty
    )
    reward_name = (
        infer_reward_name(sample)
        if config.reward in {"multi", "auto", "mixed"}
        else config.reward
    )
    return {
        "reward": float(reward),
        "outcome_reward": outcome,
        "process_reward": float(process),
        "thinking_penalty": float(penalty),
        "thinking_trace": thinking_trace,
        "final_answer": final_answer,
        "reward_name": reward_name,
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
    # Spread must be measured on full groups, not policy microbatches of size 1.
    group_spread = _group_reward_spread_mean(rollouts, config.rollouts_per_prompt)
    nonzero_adv = sum(1 for r in rollouts if abs(r.advantage) > 1e-8)
    for chunk in _chunked(rollouts, config.policy_micro_batch_size):
        loss, chunk_stats = _policy_loss(model, chunk, pad_token_id, config, torch)
        weighted = len(chunk) / total
        (loss * weighted * loss_scale).backward()
        _merge_stats(stats, chunk_stats, weight=weighted)
        del loss
    stats["group_reward_spread_mean"] = float(group_spread)
    stats["advantage_nonzero_frac"] = float(nonzero_adv / max(total, 1))
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
    clipped = (
        torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
        * advantages
    )
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
        # Placeholder; overwritten in _backprop_policy_step with full-group value.
        "group_reward_spread_mean": 0.0,
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
        "advantage_nonzero_frac": 0.0,
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
    """Mean response log-prob per sequence (memory-conscious).

    Avoids casting the full [B, T, V] logits tensor to float32 (which OOMs on
    8B + long coding completions). Processes oversized batches row-by-row.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    batch_size = int(input_ids.shape[0])

    # Keep micro-batches tiny: full-vocab log_softmax dominates VRAM.
    micro = 1 if batch_size > 1 else batch_size
    scores = []
    for start in range(0, batch_size, max(micro, 1)):
        end = min(start + max(micro, 1), batch_size)
        outputs = model(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
        )
        # Stay in model dtype (bf16/fp16); float32 log_softmax blows up memory.
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[start:end, 1:]
        mask = response_mask[start:end, 1:].to(dtype=logits.dtype)
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        scores.append((token_logprobs * mask).sum(dim=1) / denom)
        del outputs, logits, token_logprobs
    return torch.cat(scores, dim=0)


def _pad_rollouts(
    rollouts: list[Rollout], pad_token_id: int, device: str, torch
) -> dict[str, Any]:
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
            "attention_mask": torch.stack(
                [r.attention_mask for r in rollouts], dim=0
            ),
            "response_mask": torch.stack([r.response_mask for r in rollouts], dim=0),
        }

    input_ids = torch.full(
        (len(rollouts), max_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(rollouts), max_len), dtype=torch.long, device=device
    )
    response_mask = torch.zeros(
        (len(rollouts), max_len), dtype=torch.bool, device=device
    )
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
    """Group-relative advantages (GRPO-style).

    Uses leave-one-out mean when group_size >= 3 for lower-bias estimates; for
    pairs falls back to standard mean centering. Zero-variance groups get
    advantage 0 (no fictitious signal).
    """
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        rewards = [r.reward for r in group]
        if len(rewards) < 2:
            for rollout in group:
                rollout.advantage = 0.0
            continue
        if max(rewards) - min(rewards) < 1e-12:
            for rollout in group:
                rollout.advantage = 0.0
            continue
        if len(rewards) >= 3:
            total = sum(rewards)
            for rollout, reward in zip(group, rewards, strict=False):
                loo_mean = (total - reward) / (len(rewards) - 1)
                rollout.advantage = reward - loo_mean
        else:
            mean = sum(rewards) / len(rewards)
            for rollout, reward in zip(group, rewards, strict=False):
                rollout.advantage = reward - mean
        # Normalize within group for stable policy gradients.
        advs = [r.advantage for r in group]
        mean_a = sum(advs) / len(advs)
        var_a = sum((a - mean_a) ** 2 for a in advs) / len(advs)
        std_a = math.sqrt(var_a) or 1.0
        for rollout in group:
            rollout.advantage = (rollout.advantage - mean_a) / std_a


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
        fused=config.device == "cuda",
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
            "use_lora requires PEFT. Install training extras with "
            "`pip install -e '.[train]'`."
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
    return _enable_lora_training_hooks(model, config)


def _load_existing_adapter(model, config: SingleGpuSlimeConfig):
    """Resume multi-round RL from a saved PEFT adapter directory."""
    try:
        from peft import PeftModel, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "adapter_path requires PEFT. Install training extras with "
            "`pip install -e '.[train]'`."
        ) from exc

    adapter = Path(config.adapter_path or "")
    if not adapter.exists():
        raise FileNotFoundError(f"adapter_path not found: {adapter}")

    if _model_loaded_in_kbit(model):
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=False,
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    return _enable_lora_training_hooks(model, config)


def _enable_lora_training_hooks(model, config: SingleGpuSlimeConfig):
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=_GRADIENT_CHECKPOINTING_KWARGS
        )
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # Ensure adapter weights are trainable after load.
    if hasattr(model, "train"):
        model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
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
) -> Iterable[list[dict[str, Any]]]:
    yield from _batched_records(
        _iter_shuffled_samples(config, rng), config.train_batch_size
    )


def _iter_shuffled_samples(
    config: SingleGpuSlimeConfig,
    rng: random.Random,
) -> Iterable[dict[str, Any]]:
    buffer: list[dict[str, Any]] = []
    read = 0
    for sample in _load_samples(config):
        buffer.append(sample)
        read += 1
        if (
            config.max_samples_per_epoch is not None
            and read >= config.max_samples_per_epoch
        ):
            break
        if len(buffer) < config.shuffle_buffer_size:
            continue
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)

    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)


def _load_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    for sample in iter_jsonl(config.dataset):
        if config.prompt_field not in sample:
            raise ValueError(f"sample missing prompt field {config.prompt_field!r}")
        yield sample


def _reward_sample(
    sample: dict[str, Any], config: SingleGpuSlimeConfig
) -> dict[str, Any]:
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
    if config.device != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("single-GPU slime training requires CUDA when device='cuda'")
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)


def _configure_vram_cap(torch, max_vram_gb: float | None, device: str) -> None:
    if max_vram_gb is None or device != "cuda":
        return
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if max_vram_gb > total_gb:
        raise RuntimeError(
            f"VRAM cap {max_vram_gb:.2f} GiB exceeds device capacity {total_gb:.2f} GiB"
        )
    fraction = max_vram_gb / total_gb
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def _assert_vram_fit(torch, max_vram_gb: float | None, device: str) -> None:
    if max_vram_gb is None or device != "cuda":
        return
    torch.cuda.synchronize()
    used_gb = torch.cuda.memory_allocated() / 1024**3
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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
        "metric_ema": controller.ema_value,
        "auto_stop_ema_alpha": controller.ema_alpha,
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
) -> None:
    state = {
        "global_step": global_step,
        "stop_reason": stop_reason,
        "best_checkpoint_dir": str(config.output_dir / config.best_checkpoint_dir),
        "final_checkpoint_dir": str(_final_output_dir(config)),
        **_auto_stop_stats(controller),
    }
    (config.output_dir / "slime_training_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
