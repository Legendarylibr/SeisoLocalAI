"""Reward helpers for local slime-style RL."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

RewardFn = Callable[[str, dict[str, Any]], float]


def exact_match_reward(completion: str, sample: dict[str, Any]) -> float:
    expected = str(sample.get("answer", "")).strip()
    if not expected:
        return 0.0
    return 1.0 if completion.strip() == expected else 0.0


def contains_answer_reward(completion: str, sample: dict[str, Any]) -> float:
    expected = str(sample.get("answer", "")).strip().lower()
    if not expected:
        return 0.0
    return 1.0 if expected in completion.lower() else 0.0


def numeric_reward(completion: str, sample: dict[str, Any]) -> float:
    expected = _last_number(str(sample.get("answer", "")))
    actual = _last_number(completion)
    if expected is None or actual is None:
        return 0.0
    return 1.0 if math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-4) else 0.0


def field_reward(completion: str, sample: dict[str, Any]) -> float:
    del completion
    value = sample.get("reward", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve_reward(name: str) -> RewardFn:
    rewards: dict[str, RewardFn] = {
        "exact_match": exact_match_reward,
        "contains_answer": contains_answer_reward,
        "numeric": numeric_reward,
        "field": field_reward,
    }
    try:
        return rewards[name]
    except KeyError as exc:
        choices = ", ".join(sorted(rewards))
        raise ValueError(
            f"unknown reward {name!r}; expected one of: {choices}"
        ) from exc


def _last_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None
