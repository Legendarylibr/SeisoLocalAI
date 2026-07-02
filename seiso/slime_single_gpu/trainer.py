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
from seiso.models.lora_targets import resolve_lora_target_modules
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
            generated = model.generate(
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
            response_mask[prompt_width:] = (
                generated[idx, prompt_width:] != tokenizer.pad_token_id
            )
            completion = _force_completion_thinking_prefix(completions[idx], config)
            reward_sample = _reward_sample(sample, config)
            score = _score_completion(completion, reward_sample, config, reward_fn)
            chunk_rollouts.append(
                Rollout(
                    input_ids=generated[idx].detach().cpu(),
                    attention_mask=(generated[idx] != tokenizer.pad_token_id)
                    .detach()
                    .cpu(),
                    response_mask=response_mask.detach().cpu(),
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

        padded = _pad_rollouts(
            chunk_rollouts, tokenizer.pad_token_id, config.device, torch
        )
        with torch.no_grad():
            old_logprobs = _sequence_logprobs(model, padded, torch).detach().cpu()
            ref_logprobs = (
                _sequence_logprobs(ref_model, padded, torch).detach().cpu()
                if ref_model is not None
                else None
            )
        for idx, rollout in enumerate(chunk_rollouts):
            rollout.old_logprobs = old_logprobs[idx]
            rollout.ref_logprobs = (
                ref_logprobs[idx] if ref_logprobs is not None else None
            )
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
        config.missing_thinking_penalty
        if config.require_thinking_trace and not has_trace
        else 0.0
    )
    reward = (
        config.outcome_reward_weight * outcome
        + config.process_reward_weight * process
        - penalty
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
        "group_reward_spread_mean": _group_reward_spread_mean(
            rollouts, config.rollouts_per_prompt
        ),
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
    outputs = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
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


def _pad_rollouts(
    rollouts: list[Rollout], pad_token_id: int, device: str, torch
) -> dict[str, Any]:
    max_len = max(int(r.input_ids.numel()) for r in rollouts)
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
    for param in parameters:
        if param.__class__.__name__ == "Params4bit":
            return True
    return False


def _freeze_multimodal_backbones(model) -> None:
    """Keep vision/audio towers frozen for text-only slime on multimodal Gemma4."""
    model_config = getattr(model, "config", None)
    if getattr(model_config, "model_type", None) != "gemma4":
        return
    backbone = getattr(model, "model", None)
    if backbone is None:
        return
    for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        tower = getattr(backbone, name, None)
        if tower is None:
            continue
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
