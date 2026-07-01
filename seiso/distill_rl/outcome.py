"""Outcome-only scoring helpers for verifiable distill-RL tasks."""

from __future__ import annotations

import math
import re

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def format_thinking_prompt(prompt: str, instruction: str) -> str:
    """Append a thinking instruction and opening tag unless already present."""
    if "<think>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n\n{instruction}\n<think>"


def ensure_thinking_completion(completion: str, *, enabled: bool) -> str:
    if not enabled:
        return completion
    if completion.lstrip().lower().startswith("<think>"):
        return completion
    return f"<think>{completion}"


def final_answer_text(completion: str) -> str:
    """Return text after the final thinking block, falling back to tag-stripped text."""
    match = re.search(
        r"</think>(?P<final>.*)$",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("final").strip()
    return _THINK_RE.sub("", completion).strip()


def outcome_reward(completion: str, answer: str | None, *, benchmark: str | None) -> float:
    """Score verifiable tasks with a pure 0/1 outcome reward."""
    if answer is None or not str(answer).strip():
        return 0.0
    final = final_answer_text(completion)
    expected = str(answer).strip()
    bench = (benchmark or "").lower()
    if bench in {"gsm8k", "aime", "math", "numeric"}:
        return _numeric_match(final, expected)
    if bench in {"gpqa", "multiple_choice", "choice"}:
        return _choice_or_exact_match(final, expected)
    return _choice_or_exact_match(final, expected)


def _numeric_match(actual: str, expected: str) -> float:
    actual_value = _last_number(actual)
    expected_value = _last_number(expected)
    if actual_value is None or expected_value is None:
        return _choice_or_exact_match(actual, expected)
    return 1.0 if math.isclose(actual_value, expected_value, rel_tol=1e-4, abs_tol=1e-4) else 0.0


def _choice_or_exact_match(actual: str, expected: str) -> float:
    actual_norm = _normalize_answer(actual)
    expected_norm = _normalize_answer(expected)
    if not actual_norm or not expected_norm:
        return 0.0
    if actual_norm == expected_norm:
        return 1.0
    actual_choice = _extract_choice(actual_norm)
    expected_choice = _extract_choice(expected_norm)
    return 1.0 if actual_choice and actual_choice == expected_choice else 0.0


def _normalize_answer(text: str) -> str:
    lowered = final_answer_text(text).lower()
    lowered = re.sub(r"final answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"answer\s*(is|:)?", " ", lowered)
    lowered = re.sub(r"[^a-z0-9.+-]+", " ", lowered)
    return " ".join(lowered.split()).strip()


def _extract_choice(text: str) -> str | None:
    match = re.search(r"\b([a-d])\b", text.lower())
    return match.group(1) if match else None


def _last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None
