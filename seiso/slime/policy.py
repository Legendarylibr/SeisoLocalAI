"""Policy loss, advantages, and rollout batch math for slime GRPO."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.types import Rollout


def _kl_k3_from_log_ratio(log_ratio, torch, *, mask=None):
    """Schulman k3 estimator: ``exp(δ) - δ - 1`` (non-negative).

    ``δ = log π − log π_ref``. Unlike the signed k1 mean of ``δ``, k3 cannot
    reward divergence when used as a penalty term.
    """
    clamped = log_ratio.clamp(-20.0, 20.0)
    k3 = torch.exp(clamped) - clamped - 1.0
    if mask is None:
        return k3.mean()
    weight = mask.sum().clamp_min(1.0)
    return (k3 * mask).sum() / weight


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
            clip_ratio_c=config.clip_ratio_c,
            aggregation=config.loss_aggregation,
        )
    else:
        # Length-normalize sequence log-probs before the importance ratio so
        # variable response lengths do not dominate exp(ΣΔlogπ).
        lengths = response_token_counts.clamp_min(1.0)
        policy_loss = _clipped_policy_loss(
            new_logprobs / lengths,
            old_logprobs / lengths,
            advantages,
            torch.ones_like(new_logprobs),
            config.clip_ratio,
            torch,
            clip_ratio_high=config.clip_ratio_high,
            clip_ratio_c=config.clip_ratio_c,
            aggregation=config.loss_aggregation,
        )

    kl_loss = torch.zeros((), device=config.device)
    kl_k1 = torch.zeros((), device=config.device)
    if config.kl_coef > 0 and rollouts[0].ref_logprobs is not None:
        if config.calculate_per_token_loss:
            ref_token_logprobs = _pad_rollout_token_logprobs(
                rollouts,
                "ref_token_logprobs",
                int(new_token_logprobs.shape[1]),
                config.device,
                torch,
            )
            log_ratio = new_token_logprobs - ref_token_logprobs
            kl_k1 = (log_ratio * response_mask).sum() / response_mask.sum().clamp_min(1.0)
            kl_loss = _kl_k3_from_log_ratio(log_ratio, torch, mask=response_mask)
        else:
            # Sequence log-probs are sums. Normalize by response length so KL
            # scale matches per-token practice when calculate_per_token_loss is False.
            ref_logprobs = torch.stack([r.ref_logprobs for r in rollouts]).to(config.device)
            lengths = response_token_counts.clamp_min(1.0)
            log_ratio = (new_logprobs - ref_logprobs) / lengths
            kl_k1 = log_ratio.mean()
            kl_loss = _kl_k3_from_log_ratio(log_ratio, torch)

    rewards = [r.reward for r in rollouts]
    loss = policy_loss + config.kl_coef * kl_loss
    group_stats = _group_verifier_stats(rollouts, config.rollouts_per_prompt)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "kl_k1": float(kl_k1.detach().cpu()),
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
        "kl_k1": 0.0,
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
        "kl_k1",
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
    clip_ratio_c: float | None = 3.0,
    aggregation: str = "seq_mean",
):
    """PPO/GRPO clipped surrogate (DeepSeek / slime / OpenRLHF dual-clip).

    ``clip_ratio`` / ``clip_ratio_high`` map to slime ``eps_clip`` / ``eps_clip_high``.
    ``clip_ratio_c`` (>1) is OpenRLHF/verl dual-clip for negative advantages.
    ``aggregation``:
      * ``seq_mean`` — DeepSeekMath: mean over sequences of (masked token mean)
      * ``token_mean`` — global masked token mean (length-biased)
    """
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    # Clamp log-ratio before exp (verl/OpenRLHF) to avoid IS overflow.
    log_ratio = (new_logprobs - old_logprobs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + high) * advantages
    objective = torch.minimum(unclipped, clipped)
    if clip_ratio_c is not None and float(clip_ratio_c) > 1.0:
        dual = float(clip_ratio_c) * advantages
        objective = torch.where(advantages < 0, torch.maximum(objective, dual), objective)

    mode = str(aggregation or "seq_mean").strip().lower()
    if mode == "token_mean":
        return -((objective * mask).sum() / mask.sum().clamp_min(1.0))
    if mode != "seq_mean":
        raise ValueError("aggregation must be 'seq_mean' or 'token_mean'")
    # DeepSeek GRPO: (1/G) Σ_i (1/|o_i|) Σ_t L_{i,t}
    if mask.ndim == 1:
        return -(objective * mask).sum() / mask.sum().clamp_min(1.0)
    token_counts = mask.sum(dim=-1).clamp_min(1.0)
    per_seq = (objective * mask).sum(dim=-1) / token_counts
    valid = mask.sum(dim=-1) > 0
    if bool(valid.any()):
        return -per_seq[valid].mean()
    return -per_seq.mean()


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


_INVALID_ADVANTAGE_STATUS = frozenset({"length", "empty"})


def _assign_grouped_advantages(
    rollouts: list[Rollout],
    group_size: int,
    *,
    grpo_std_normalization: bool = True,
) -> None:
    """GRPO group-relative advantages matching THUDM/slime ``_post_process_rewards``.

    For each prompt group: subtract the group mean, then optionally divide by
    unbiased sample std + 1e-6 (slime ``grpo_std_normalization`` / Dr.GRPO toggle).

    Truncated/empty rollouts (DAPO / OpenRLHF overlong practice) are excluded from
    the baseline and receive advantage 0 so wiped zeros cannot pull the group mean.

    Incomplete trailing groups are invalid (advantages would not be mean-zero over
    a full sample set) and raise ``ValueError``.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for GRPO advantages")
    if len(rollouts) % group_size != 0:
        raise ValueError(
            f"rollout count {len(rollouts)} is not divisible by group_size "
            f"{group_size}; incomplete GRPO groups are invalid"
        )
    for start in range(0, len(rollouts), group_size):
        group = rollouts[start : start + group_size]
        valid = [r for r in group if r.status not in _INVALID_ADVANTAGE_STATUS]
        if len(valid) < 2:
            for rollout in group:
                rollout.advantage = 0.0
            continue
        rewards = [float(r.reward) for r in valid]
        n = len(rewards)
        mean = sum(rewards) / n
        centered = [reward - mean for reward in rewards]
        if grpo_std_normalization and n > 1:
            # torch.std default is unbiased (divide by n-1); slime uses std + 1e-6.
            variance = sum(value * value for value in centered) / (n - 1)
            std = math.sqrt(variance)
            scale = std + 1e-6
            adv_by_id = {
                id(rollout): value / scale
                for rollout, value in zip(valid, centered, strict=True)
            }
        else:
            adv_by_id = {
                id(rollout): value
                for rollout, value in zip(valid, centered, strict=True)
            }
        for rollout in group:
            rollout.advantage = float(adv_by_id.get(id(rollout), 0.0))


def _filter_rollout_groups(
    rollouts: list[Rollout],
    config: SingleGpuSlimeConfig,
) -> tuple[list[Rollout], set[int], int]:
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
    """Keep groups that can form a GRPO baseline with outcome diversity.

    Always requires ≥2 non-truncated/non-empty rollouts — otherwise advantages
    are forced to 0 and the step is vacuous. ``reward_nonzero_std`` /
    ``outcome_nonzero_std`` additionally require nonzero *outcome* spread
    among those valid rollouts (wiped truncated zeros must not fake diversity;
    format-only composite spread is ignored).
    """
    valid = [rollout for rollout in group if rollout.status not in _INVALID_ADVANTAGE_STATUS]
    if len(valid) < 2:
        return False
    if config.dynamic_sampling_filter in {
        "reward_nonzero_std",
        "outcome_nonzero_std",
    }:
        outcomes = [float(rollout.outcome_reward) for rollout in valid]
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


def _http_rollout_status(
    *,
    finish_reason: str | None,
    response_tokens,
    eos_token_id: int | None,
    completion_text: str = "",
) -> str:
    """Status for HTTP rollouts: prefer engine finish_reason over EOS-in-ids.

    Stock OpenAI ``/v1/completions`` rarely returns token ids; retokenized text
    usually omits EOS, so EOS-only detection falsely marks stopped samples as
    truncated and wipes GRPO rewards.
    """
    n_tokens = (
        int(response_tokens.numel())
        if hasattr(response_tokens, "numel")
        else len(response_tokens)
    )
    if n_tokens == 0 and not str(completion_text or "").strip():
        return "empty"
    from seiso.slime.rollout_clients import _normalize_rollout_finish_status

    mapped = _normalize_rollout_finish_status(finish_reason)
    if mapped is not None:
        return mapped
    # Fall back to token/EOS heuristic when the engine omitted finish_reason.
    return _rollout_status(response_tokens, eos_token_id)


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
