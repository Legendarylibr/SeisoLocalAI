"""Preference-pair construction from verifiable scores (code hard negatives).

Unit-test pass is the verifier. Failed rollouts can become DPO *rejected*
completions when a same-prompt candidate passes (or scores strictly higher).

This is a good idea for **offline preference / DPO** data:
- chosen = passes tests (or best score)
- rejected = fails tests, preferring *hard* fails (extractable code, partial pass)

It is a weaker idea for **on-policy GRPO** (group advantages already demote fails).
Do not inject fails as negatives into the GRPO loss separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoredCompletion:
    completion: str
    score: float
    passed: bool
    detail: str = ""
    has_code: bool = False
    tests_passed: int = 0
    tests_total: int = 0


@dataclass(frozen=True)
class PreferencePair:
    chosen: ScoredCompletion
    rejected: ScoredCompletion
    pair_kind: str
    # hard_negative | score_gap | skipped reasons live outside


def score_code_completion(completion: str, sample: dict[str, Any]) -> ScoredCompletion:
    """Score one completion with the sandboxed code verifier."""
    from seiso.rl_verify.code_proof import extract_python_code, verify_code_proof

    result = verify_code_proof(completion, sample)
    code = result.extracted_code or extract_python_code(completion)
    return ScoredCompletion(
        completion=completion,
        score=float(result.score),
        passed=bool(result.passed),
        detail=result.detail,
        has_code=bool(code.strip()),
        tests_passed=int(result.tests_passed),
        tests_total=int(result.tests_total),
    )


def select_preference_pair(
    candidates: list[ScoredCompletion],
    *,
    hard_negatives: bool = True,
    require_chosen_pass: bool = True,
    min_score_gap: float = 1e-6,
) -> PreferencePair | None:
    """Pick chosen/rejected from a same-prompt group.

    Policy (code-oriented):
    1. Prefer a **passing** completion as chosen when ``require_chosen_pass``.
    2. Prefer a **failing** completion with extractable code as rejected
       (hard negative). Among fails, take the *highest* score (almost-right).
    3. Require ``chosen.score > rejected.score + min_score_gap``.
    4. Skip if no usable pair (both pass, all fail with no pass when required, etc.).
    """
    usable = [c for c in candidates if str(c.completion).strip()]
    if len(usable) < 2:
        return None

    # Chosen: passers first, then score, then has_code.
    passers = [c for c in usable if c.passed]
    if require_chosen_pass:
        if not passers:
            return None
        chosen = max(passers, key=lambda c: (c.score, c.has_code, len(c.completion)))
    else:
        chosen = max(usable, key=lambda c: (c.score, c.passed, c.has_code, len(c.completion)))

    worse = [
        c
        for c in usable
        if c.completion != chosen.completion and c.score < chosen.score - min_score_gap
    ]
    if not worse:
        return None

    if hard_negatives:
        # Hard: failing + extractable structure + highest residual score (near miss).
        hard_fails = [c for c in worse if (not c.passed) and c.has_code]
        soft_fails = [c for c in worse if not c.passed]
        if hard_fails:
            rejected = max(
                hard_fails,
                key=lambda c: (c.score, c.tests_passed, len(c.completion)),
            )
            kind = "hard_negative"
        elif soft_fails:
            # Math/choice: prefer the strongest incorrect candidate as rejected.
            rejected = max(
                soft_fails,
                key=lambda c: (c.score, len(c.completion.strip())),
            )
            kind = "hard_negative"
        else:
            # Soft fallback: any lower-scoring completion (all may have passed).
            rejected = max(worse, key=lambda c: (c.score, c.has_code, len(c.completion)))
            kind = "score_gap"
    else:
        rejected = min(worse, key=lambda c: (c.score, c.has_code, len(c.completion)))
        kind = "score_gap"

    if chosen.score <= rejected.score + min_score_gap:
        return None
    if chosen.completion.strip() == rejected.completion.strip():
        return None
    return PreferencePair(chosen=chosen, rejected=rejected, pair_kind=kind)


def preference_row_from_pair(
    *,
    prompt_id: str,
    prompt: str,
    pair: PreferencePair,
    sample: dict[str, Any] | None = None,
    generation_seed: int | None = None,
    group_size: int | None = None,
    group_rewards: list[float] | None = None,
    reward_source: str | None = None,
) -> dict[str, Any]:
    """Serialize a DPO-style preference row with verifier provenance."""
    sample = sample or {}
    if reward_source is None:
        if sample.get("tests") is not None or sample.get("test") is not None:
            reward_source = "code_unit_tests"
        else:
            reward_source = "verifiable_outcome"
    row: dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "chosen": pair.chosen.completion,
        "rejected": pair.rejected.completion,
        "chosen_reward": pair.chosen.score,
        "rejected_reward": pair.rejected.score,
        "chosen_passed": pair.chosen.passed,
        "rejected_passed": pair.rejected.passed,
        "pair_kind": pair.pair_kind,
        "reward_source": reward_source,
        "hard_negative": pair.pair_kind == "hard_negative",
        "chosen_detail": pair.chosen.detail,
        "rejected_detail": pair.rejected.detail,
    }
    if group_size is not None:
        row["grpo_group_size"] = group_size
    if group_rewards is not None:
        row["group_rewards"] = list(group_rewards)
    if generation_seed is not None:
        row["generation_seed"] = generation_seed
    if sample.get("answer") is not None:
        row["answer"] = sample.get("answer")
    if sample.get("benchmark") is not None:
        row["benchmark"] = sample.get("benchmark")
    if sample.get("tests") is not None:
        row["tests"] = sample.get("tests")
    elif sample.get("test") is not None:
        row["test"] = sample.get("test")
    return row
