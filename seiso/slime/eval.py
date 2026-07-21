"""Held-out eval for slime GRPO (unit-test pass rate, not train-stream metrics)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from seiso.io.jsonl import iter_jsonl
from seiso.rl_verify import score_completion as verify_score_completion
from seiso.slime.config import SingleGpuSlimeConfig
from seiso.slime.distributed import _generation_model
from seiso.slime.rollout_generate import format_generation_prompt, generate_data_gen_chunk

logger = logging.getLogger(__name__)


def load_eval_samples(
    path: Path,
    *,
    prompt_field: str = "prompt",
    max_prompts: int | None = None,
) -> list[dict[str, Any]]:
    """Load a frozen held-out JSONL; never shuffle (determinism for reports)."""
    samples: list[dict[str, Any]] = []
    for sample in iter_jsonl(path):
        if prompt_field not in sample:
            raise ValueError(f"eval sample missing prompt field {prompt_field!r}")
        samples.append(sample)
        if max_prompts is not None and len(samples) >= max_prompts:
            break
    if not samples:
        raise ValueError(f"no eval samples found in {path}")
    return samples


def score_held_out_completions(
    *,
    completions: list[str],
    samples: list[dict[str, Any]],
    config: SingleGpuSlimeConfig,
) -> dict[str, float]:
    """Score generated completions against unit tests / verifiers."""
    if len(completions) != len(samples):
        raise ValueError(
            f"completions ({len(completions)}) != samples ({len(samples)})"
        )
    outcomes: list[float] = []
    proof_passes = 0
    proof_total = 0
    outcome_passes = 0
    for completion, sample in zip(completions, samples, strict=True):
        reward_sample = _eval_reward_sample(sample, config)
        result = verify_score_completion(
            completion,
            reward_sample,
            checker=config.reward,
            require_thinking_trace=config.require_thinking_trace,
            outcome_weight=config.outcome_reward_weight,
            format_weight=0.0,
            process_weight=0.0,
            missing_format_penalty=0.0,
            code_reward_mode="binary",
        )
        outcomes.append(float(result.outcome))
        if result.passed:
            outcome_passes += 1
        if result.proof_passed is not None:
            proof_total += 1
            if result.proof_passed:
                proof_passes += 1
    n = max(1, len(samples))
    metrics = {
        "eval_prompt_count": float(len(samples)),
        "eval_outcome_pass_rate": float(outcome_passes) / float(n),
        "eval_outcome_mean": float(sum(outcomes) / n),
    }
    if proof_total:
        metrics["eval_proof_pass_rate"] = float(proof_passes) / float(proof_total)
    return metrics


def run_held_out_eval(
    *,
    model,
    tokenizer,
    config: SingleGpuSlimeConfig,
    torch,
    step: int,
) -> dict[str, float] | None:
    """Generate one completion per held-out prompt and report pass rates."""
    if config.eval_dataset is None:
        return None
    samples = load_eval_samples(
        Path(config.eval_dataset),
        prompt_field=config.prompt_field,
        max_prompts=config.eval_max_prompts,
    )
    # One greedy-ish sample per prompt — held-out pass rate, not GRPO groups.
    from dataclasses import replace

    eval_config = replace(
        config,
        rollouts_per_prompt=1,
        temperature=0.0,
        top_p=1.0,
    )
    prompts: list[str] = []
    for sample in samples:
        raw_prompt = sample[config.prompt_field]
        if not isinstance(raw_prompt, (str, list)):
            raise ValueError(
                f"eval sample prompt field {config.prompt_field!r} must be "
                f"str or list, got {type(raw_prompt).__name__}"
            )
        prompts.append(
            format_generation_prompt(tokenizer, raw_prompt, eval_config)
        )
    # Rollout collection restores train(); eval must disable dropout for stable
    # pass-rate reports, then restore the caller's mode.
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        gen = generate_data_gen_chunk(
            generation_model=_generation_model(model),
            tokenizer=tokenizer,
            prompts=prompts,
            config=eval_config,
            torch=torch,
        )
        metrics = score_held_out_completions(
            completions=list(gen.completions),
            samples=samples,
            config=config,
        )
        metrics["eval_step"] = float(step)
        _write_eval_report(config, metrics, step=step)
        logger.info(
            "held-out eval step=%s prompts=%s outcome_pass_rate=%.3f",
            step,
            int(metrics["eval_prompt_count"]),
            metrics["eval_outcome_pass_rate"],
        )
        return metrics
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _eval_reward_sample(
    sample: dict[str, Any], config: SingleGpuSlimeConfig
) -> dict[str, Any]:
    """Minimal verifier sample (tests/answer) without trainer-side imports cycle."""
    out = dict(sample)
    if "answer" not in out and config.answer_field in out:
        out["answer"] = out.get(config.answer_field)
    if out.get("tests") is None and out.get("test") is None:
        metadata = out.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("tests") is not None:
                out["tests"] = metadata.get("tests")
            elif metadata.get("test") is not None:
                out["test"] = metadata.get("test")
    return out


def _write_eval_report(
    config: SingleGpuSlimeConfig,
    metrics: dict[str, float],
    *,
    step: int,
) -> None:
    # Distinct from auto-split prompt corpus (slime_held_out_prompts.jsonl).
    path = Path(config.output_dir) / "slime_held_out_eval_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
