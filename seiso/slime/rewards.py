"""Reward helpers for local slime-style RL (thin wrappers over ``seiso.rl_verify``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from seiso.rl_verify import resolve_checker, verify_outcome

RewardFn = Callable[[str, dict[str, Any]], float]


def exact_match_reward(completion: str, sample: dict[str, Any]) -> float:
    score, _, _ = verify_outcome(
        completion,
        sample.get("answer"),
        checker="exact_match",
        prefer_final_answer=True,
    )
    return score


def contains_answer_reward(completion: str, sample: dict[str, Any]) -> float:
    score, _, _ = verify_outcome(
        completion,
        sample.get("answer"),
        checker="contains_answer",
        prefer_final_answer=True,
    )
    return score


def numeric_reward(completion: str, sample: dict[str, Any]) -> float:
    score, _, _ = verify_outcome(
        completion,
        sample.get("answer"),
        checker="numeric",
        prefer_final_answer=True,
    )
    return score


def field_reward(completion: str, sample: dict[str, Any]) -> float:
    del completion
    score, _, _ = verify_outcome(
        "",
        None,
        checker="field",
        field_reward=sample.get("reward", 0.0),
    )
    return score


def code_reward(completion: str, sample: dict[str, Any]) -> float:
    """Sandboxed unit-test outcome: ``1.0`` only if all tests pass."""
    score, _, _ = verify_outcome(
        completion,
        sample.get("answer"),
        checker="code",
        prefer_final_answer=True,
        sample=sample,
    )
    return score


def resolve_reward(name: str) -> RewardFn:
    """Resolve a named outcome checker used for outcome-only scoring."""
    rewards: dict[str, RewardFn] = {
        "exact_match": exact_match_reward,
        "contains_answer": contains_answer_reward,
        "numeric": numeric_reward,
        "field": field_reward,
        "code": code_reward,
        "choice": lambda completion, sample: verify_outcome(
            completion,
            sample.get("answer"),
            checker="choice",
            prefer_final_answer=True,
        )[0],
        "auto": lambda completion, sample: verify_outcome(
            completion,
            sample.get("answer"),
            checker="auto",
            benchmark=sample.get("benchmark") if isinstance(sample.get("benchmark"), str) else None,
            prefer_final_answer=True,
            sample=sample,
        )[0],
    }
    try:
        key = resolve_checker(name)
        return rewards[key]
    except (KeyError, ValueError) as exc:
        choices = ", ".join(sorted(rewards))
        raise ValueError(f"unknown reward {name!r}; expected one of: {choices}") from exc
