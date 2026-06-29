"""Single-GPU slime-style GRPO trainer for Hugging Face causal LMs."""

from __future__ import annotations

import gc
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.io.jsonl import iter_jsonl
from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
from seiso.slime_single_gpu.rewards import resolve_reward


@dataclass
class Rollout:
    sample: dict[str, Any]
    prompt: str
    completion: str
    input_ids: Any
    attention_mask: Any
    response_mask: Any
    old_logprobs: Any
    ref_logprobs: Any | None
    reward: float
    advantage: float = 0.0


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
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

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
    samples = list(_load_samples(config))
    if not samples:
        raise ValueError(f"no samples found in {config.dataset}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "slime_single_gpu_metrics.jsonl"
    global_step = 0
    pending_accumulation_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.epochs):
        random.shuffle(samples)
        for batch_samples in _batched(samples, config.train_batch_size):
            rollouts = _collect_rollouts(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                samples=batch_samples,
                config=config,
                reward_fn=reward_fn,
                torch=torch,
            )
            if not rollouts:
                continue
            loss, stats = _policy_step(model, rollouts, tokenizer.pad_token_id, config, torch)
            (loss / config.gradient_accumulation_steps).backward()
            pending_accumulation_steps += 1

            if pending_accumulation_steps >= config.gradient_accumulation_steps:
                _optimizer_step(model, optimizer, torch, config)
                pending_accumulation_steps = 0

            if global_step % config.log_every_steps == 0:
                _append_metrics(
                    metrics_path,
                    {"step": global_step, "epoch": epoch, **stats},
                )
            if config.save_every_steps and global_step and global_step % config.save_every_steps == 0:
                _save(model, tokenizer, config.output_dir / f"checkpoint-{global_step}")

            global_step += 1
            if config.max_steps is not None and global_step >= config.max_steps:
                if pending_accumulation_steps:
                    _optimizer_step(model, optimizer, torch, config)
                _save(model, tokenizer, config.output_dir)
                return config.output_dir

    if pending_accumulation_steps:
        _optimizer_step(model, optimizer, torch, config)
    _save(model, tokenizer, config.output_dir)
    return config.output_dir


def _collect_rollouts(
    *,
    model,
    ref_model,
    tokenizer,
    samples: list[dict[str, Any]],
    config: SingleGpuSlimeConfig,
    reward_fn,
    torch,
) -> list[Rollout]:
    model.eval()
    prompts: list[str] = []
    source_samples: list[dict[str, Any]] = []
    for sample in samples:
        prompt = str(sample[config.prompt_field])
        for _ in range(config.rollouts_per_prompt):
            prompts.append(prompt)
            source_samples.append(sample)

    encoded = tokenizer(
        prompts,
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
        )

    rollouts: list[Rollout] = []
    for idx, sample in enumerate(source_samples):
        response_mask = torch.zeros_like(generated[idx], dtype=torch.bool)
        response_mask[prompt_width:] = generated[idx, prompt_width:] != tokenizer.pad_token_id
        completion_ids = generated[idx, prompt_width:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        rollouts.append(
            Rollout(
                sample=sample,
                prompt=prompts[idx],
                completion=completion,
                input_ids=generated[idx].detach(),
                attention_mask=(generated[idx] != tokenizer.pad_token_id).detach(),
                response_mask=response_mask.detach(),
                old_logprobs=None,
                ref_logprobs=None,
                reward=reward_fn(completion, _reward_sample(sample, config)),
            )
        )

    padded = _pad_rollouts(rollouts, tokenizer.pad_token_id, config.device, torch)
    with torch.no_grad():
        old_logprobs = _sequence_logprobs(model, padded, torch).detach()
        ref_logprobs = (
            _sequence_logprobs(ref_model, padded, torch).detach() if ref_model is not None else None
        )
    for idx, rollout in enumerate(rollouts):
        rollout.old_logprobs = old_logprobs[idx]
        rollout.ref_logprobs = ref_logprobs[idx] if ref_logprobs is not None else None

    _assign_grouped_advantages(rollouts, config.rollouts_per_prompt)
    model.train()
    return rollouts


def _policy_step(
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
    }


def _sequence_logprobs(model, batch: dict[str, Any], torch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].float()
    token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
        -1,
        labels.unsqueeze(-1),
    ).squeeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (token_logprobs * mask).sum(dim=1) / denom


def _pad_rollouts(rollouts: list[Rollout], pad_token_id: int, device: str, torch) -> dict[str, Any]:
    max_len = max(int(r.input_ids.numel()) for r in rollouts)
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


def _build_optimizer(model, config: SingleGpuSlimeConfig):
    if config.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("use_8bit_optimizer requires bitsandbytes") from exc
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=config.device == "cuda",
    )


def _optimizer_step(model, optimizer, torch, config: SingleGpuSlimeConfig) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _assert_vram_fit(torch, config.max_vram_gb, config.device)


def _load_samples(config: SingleGpuSlimeConfig) -> Iterable[dict[str, Any]]:
    for sample in iter_jsonl(config.dataset):
        if config.prompt_field not in sample:
            raise ValueError(f"sample missing prompt field {config.prompt_field!r}")
        yield sample


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
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _save(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
