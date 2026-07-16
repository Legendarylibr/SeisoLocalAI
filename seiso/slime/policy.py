"""Policy loss, advantages, and rollout batch math for slime GRPO."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.types import Rollout


def _policy_loss(
    model,
    rollouts: list[Rollout],
    pad_token_id: int,
    config: SingleGpuSlimeConfig,
    torch,
):
    padded = _pad_rollouts(rollouts, pad_token_id, config.device, torch)
    old_logprobs = torch.stack([r.old_logprobs for r in rollouts]).to(config.device)
    advantages = torch.tensor([r.advantage for r in rollouts], device=config.device)
    response_mask = padded["response_mask"][:, 1:].float()
    response_token_counts = response_mask.sum(dim=1)

    new_token_logprobs = _sequence_token_logprobs(model, padded, torch)
    new_logprobs = _masked_sequence_logprobs(new_token_logprobs, response_mask)
    if config.calculate_per_token_loss:
        old_token_logprobs = _pad_rollout_token_logprobs(
            rollouts,
            "old_token_logprobs",
            int(new_token_logprobs.shape[1]),
            config.device,
            torch,
        )
        policy_loss = _clipped_policy_loss(
            new_token_logprobs,
            old_token_logprobs,
            advantages[:, None],
            response_mask,
            config.clip_ratio,
            torch,
            clip_ratio_high=config.clip_ratio_high,
        )
    else:
        policy_loss = _clipped_policy_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            torch.ones_like(new_logprobs),
            config.clip_ratio,
            torch,
            clip_ratio_high=config.clip_ratio_high,
        )

    kl_loss = torch.zeros((), device=config.device)
    if config.kl_coef > 0 and rollouts[0].ref_logprobs is not None:
        if config.calculate_per_token_loss:
            ref_token_logprobs = _pad_rollout_token_logprobs(
                rollouts,
                "ref_token_logprobs",
                int(new_token_logprobs.shape[1]),
                config.device,
                torch,
            )
            kl_loss = ((new_token_logprobs - ref_token_logprobs) * response_mask).sum()
            kl_loss = kl_loss / response_mask.sum().clamp_min(1.0)
        else:
            # Sequence log-probs are sums (not means). Normalize by response
            # length so KL does not grow with generation length and fight long
            # correct traces when calculate_per_token_loss is False.
            ref_logprobs = torch.stack([r.ref_logprobs for r in rollouts]).to(config.device)
            lengths = response_token_counts.clamp_min(1.0)
            kl_loss = ((new_logprobs - ref_logprobs) / lengths).mean()

    rewards = [r.reward for r in rollouts]
    loss = policy_loss + config.kl_coef * kl_loss
    group_stats = _group_verifier_stats(rollouts, config.rollouts_per_prompt)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "reward_mean": float(sum(rewards) / len(rewards)),
        "reward_max": float(max(rewards)),
        "outcome_reward_mean": _mean(r.outcome_reward for r in rollouts),
        "format_reward_mean": _mean(r.format_reward for r in rollouts),
        "process_reward_mean": _mean(r.process_reward for r in rollouts),
        "thinking_penalty_mean": _mean(r.thinking_penalty for r in rollouts),
        "outcome_pass_rate": _mean(1.0 if r.outcome_passed else 0.0 for r in rollouts),
        "format_ok_rate": _mean(1.0 if r.format_ok else 0.0 for r in rollouts),
        "proof_pass_rate": _mean(
            1.0 if r.proof_passed else 0.0 for r in rollouts if r.proof_passed is not None
        ),
        "proof_score_mean": _mean(
            float(r.proof_score) for r in rollouts if r.proof_score is not None
        ),
        "group_reward_spread_mean": group_stats["group_reward_spread_mean"],
        "group_outcome_spread_mean": group_stats["group_outcome_spread_mean"],
        "group_pass_rate": group_stats["group_pass_rate"],
        "group_nonzero_spread_frac": group_stats["group_nonzero_spread_frac"],
        "group_nonzero_outcome_spread_frac": group_stats["group_nonzero_outcome_spread_frac"],
        "response_tokens_mean": float(response_token_counts.mean().detach().cpu()),
        **_rollout_status_stats(rollouts),
    }


def _empty_stats() -> dict[str, float]:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl": 0.0,
        "reward_mean": 0.0,
        "reward_max": float("-inf"),
        "outcome_reward_mean": 0.0,
        "format_reward_mean": 0.0,
        "process_reward_mean": 0.0,
        "thinking_penalty_mean": 0.0,
        # Metric rates (not secrets); names trip bandit B105 on "*pass*".
        "outcome_pass_rate": 0.0,  # nosec B105
        "format_ok_rate": 0.0,
        "proof_pass_rate": 0.0,  # nosec B105
        "proof_score_mean": 0.0,
        "group_reward_spread_mean": 0.0,
        "group_outcome_spread_mean": 0.0,
        "group_pass_rate": 0.0,  # nosec B105
        "group_nonzero_spread_frac": 0.0,
        "group_nonzero_outcome_spread_frac": 0.0,
        "response_tokens_mean": 0.0,
        "rollout_status_stop": 0.0,
        "rollout_status_length": 0.0,
        "rollout_status_empty": 0.0,
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
        "format_reward_mean",
        "process_reward_mean",
        "thinking_penalty_mean",
        "outcome_pass_rate",
        "format_ok_rate",
        "proof_pass_rate",
        "proof_score_mean",
        "group_reward_spread_mean",
        "group_outcome_spread_mean",
        "group_pass_rate",
        "group_nonzero_spread_frac",
        "group_nonzero_outcome_spread_frac",
        "response_tokens_mean",
    ):
        stats[key] += chunk_stats.get(key, 0.0) * weight
    for key in ("rollout_status_stop", "rollout_status_length", "rollout_status_empty"):
        stats[key] += chunk_stats.get(key, 0.0)
    stats["reward_max"] = max(stats["reward_max"], chunk_stats.get("reward_max", 0.0))


def _sequence_logprobs(model, batch: dict[str, Any], torch):
    token_logprobs = _sequence_token_logprobs(model, batch, torch)
    mask = batch["response_mask"][:, 1:].float()
    return _masked_sequence_logprobs(token_logprobs, mask)


def _sequence_token_logprobs(model, batch: dict[str, Any], torch):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    return (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(
            -1,
            labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def _masked_sequence_logprobs(token_logprobs, mask):
    """Sum response-token log-probs (GRPO/PPO sequence likelihood, not mean)."""
    return (token_logprobs * mask).sum(dim=1)


def _clipped_policy_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    mask,
    clip_ratio: float,
    torch,
    *,
    clip_ratio_high: float | None = None,
):
    """PPO/GRPO clipped surrogate (slime ``compute_policy_loss`` without dual-clip-c).

    ``clip_ratio`` / ``clip_ratio_high`` map to slime ``eps_clip`` / ``eps_clip_high``.
    When ``clip_ratio_high`` is None, the high bound equals the low bound.
    """
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + high) * advantages
    objective = torch.minimum(unclipped, clipped)
    return -((objective * mask).sum() / mask.sum().clamp_min(1.0))


def _pad_rollout_token_logprobs(
    rollouts: list[Rollout],
    field_name: str,
    width: int,
    device: str,
    torch,
):
    values = torch.zeros((len(rollouts), width), dtype=torch.float32, device=device)
    for idx, rollout in enumerate(rollouts):
        token_values = getattr(rollout, field_name)
        if token_values is None:
            raise ValueError(f"rollout missing {field_name}")
        length = min(int(token_values.numel()), width)
        values[idx, :length] = token_values[:length].to(device)
    return values


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


def _assign_grouped_advantages(
    rollouts: list[Rollout],
    group_size: int,
    *,
    grpo_std_normalization: bool = True,
) -> None:
    """GRPO group-relative advantages matching THUDM/slime ``_post_process_rewards``.

    For each prompt group: subtract the group mean, then optionally divide by
    unbiased sample std + 1e-6 (slime ``grpo_std_normalization`` / Dr.GRPO toggle).
    """
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        rewards = [float(r.reward) for r in group]
        n = len(rewards)
        mean = sum(rewards) / n
        centered = [reward - mean for reward in rewards]
        if grpo_std_normalization and n > 1:
            # torch.std default is unbiased (divide by n-1); slime uses std + 1e-6.
            variance = sum(value * value for value in centered) / (n - 1)
            std = math.sqrt(variance)
            scale = std + 1e-6
            for rollout, value in zip(group, centered, strict=True):
                rollout.advantage = value / scale
        else:
            for rollout, value in zip(group, centered, strict=True):
                rollout.advantage = value


def _filter_rollout_groups(
    rollouts: list[Rollout],
    config: SingleGpuSlimeConfig,
) -> tuple[list[Rollout], set[int], int]:
    if config.dynamic_sampling_filter == "none":
        group_count = math.ceil(len(rollouts) / config.rollouts_per_prompt)
        return rollouts, set(range(group_count)), 0

    kept: list[Rollout] = []
    kept_group_indexes: set[int] = set()
    rejected = 0
    for group_index, start in enumerate(range(0, len(rollouts), config.rollouts_per_prompt)):
        group = rollouts[start : start + config.rollouts_per_prompt]
        if _keep_rollout_group(group, config):
            kept.extend(group)
            kept_group_indexes.add(group_index)
        else:
            rejected += 1
    return kept, kept_group_indexes, rejected


def _keep_rollout_group(
    group: list[Rollout],
    config: SingleGpuSlimeConfig,
) -> bool:
    """Keep groups that have nonzero *outcome* spread for GRPO.

    ``reward_nonzero_std`` / ``outcome_nonzero_std`` intentionally ignore pure
    format-shaping spread on the composite ``reward``. Format remains a small
    shaping term *after* a group already has outcome diversity.
    """
    if config.dynamic_sampling_filter in {
        "reward_nonzero_std",
        "outcome_nonzero_std",
    }:
        outcomes = [float(rollout.outcome_reward) for rollout in group]
        if len(outcomes) < 2:
            return False
        mean = sum(outcomes) / len(outcomes)
        variance = sum((value - mean) ** 2 for value in outcomes) / len(outcomes)
        return math.sqrt(variance) > config.dynamic_sampling_min_reward_std
    return True


def _truncate_rollout_groups(
    rollouts: list[Rollout],
    group_size: int,
    max_groups: int,
) -> list[Rollout]:
    if max_groups <= 0:
        return []
    return rollouts[: max_groups * group_size]


def _rollout_status(response_tokens, eos_token_id: int | None) -> str:
    if int(response_tokens.numel()) == 0:
        return "empty"
    if eos_token_id is not None and bool((response_tokens == eos_token_id).any().item()):
        return "stop"
    return "length"


def _response_mask_for_sequence(
    input_ids,
    *,
    prompt_width: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
    torch,
):
    """Build response mask; keep EOS when pad_token_id == eos_token_id.

    Hugging Face often sets pad=eos. Naively masking ``!= pad`` drops the EOS
    token from the likelihood, slightly biasing sequence log-probs.
    """
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if prompt_width >= int(input_ids.numel()):
        return mask
    resp = input_ids[prompt_width:]
    resp_mask = torch.ones_like(resp, dtype=torch.bool)
    if pad_token_id is not None and eos_token_id is not None and pad_token_id == eos_token_id:
        eos_hits = (resp == eos_token_id).nonzero(as_tuple=False)
        if eos_hits.numel() > 0:
            first = int(eos_hits[0].item())
            # Keep through first EOS; drop trailing pad/eos after that.
            resp_mask[first + 1 :] = False
    elif pad_token_id is not None:
        resp_mask = resp != pad_token_id
    mask[prompt_width:] = resp_mask
    return mask


def _rollout_status_stats(rollouts: list[Rollout]) -> dict[str, float]:
    counts = {
        "rollout_status_stop": 0.0,
        "rollout_status_length": 0.0,
        "rollout_status_empty": 0.0,
    }
    for rollout in rollouts:
        key = f"rollout_status_{rollout.status}"
        if key in counts:
            counts[key] += 1.0
    return counts


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _group_reward_spread_mean(rollouts: list[Rollout], group_size: int) -> float:
    return _group_verifier_stats(rollouts, group_size)["group_reward_spread_mean"]


def _group_verifier_stats(rollouts: list[Rollout], group_size: int) -> dict[str, float]:
    """Per-prompt group diagnostics for collapse / verifier pass monitoring."""
    spreads: list[float] = []
    outcome_spreads: list[float] = []
    group_passes: list[float] = []
    nonzero_spread: list[float] = []
    nonzero_outcome_spread: list[float] = []
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        if not group:
            continue
        rewards = [r.reward for r in group]
        outcomes = [float(r.outcome_reward) for r in group]
        spread = max(rewards) - min(rewards)
        outcome_spread = max(outcomes) - min(outcomes)
        spreads.append(spread)
        outcome_spreads.append(outcome_spread)
        # Primary health metric: outcome diversity (matches dynamic sampling).
        nonzero_spread.append(1.0 if outcome_spread > 1e-8 else 0.0)
        nonzero_outcome_spread.append(1.0 if outcome_spread > 1e-8 else 0.0)
        group_passes.append(1.0 if any(r.outcome_passed for r in group) else 0.0)
    return {
        "group_reward_spread_mean": _mean(spreads),
        "group_outcome_spread_mean": _mean(outcome_spreads),
        "group_pass_rate": _mean(group_passes),
        "group_nonzero_spread_frac": _mean(nonzero_spread),
        "group_nonzero_outcome_spread_frac": _mean(nonzero_outcome_spread),
    }
