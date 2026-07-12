"""Reward helpers for local slime-style RL."""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

RewardFn = Callable[[str, dict[str, Any]], float]

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_DEFAULT_UNIT_TEST_TIMEOUT_SEC = 2.0
_DEFAULT_MAX_UNIT_TESTS = 8
_DEFAULT_MAX_OUTPUT_CHARS = 100_000

# Atomic reward names (not multi/auto aliases).
_ATOMIC_REWARDS = frozenset(
    {
        "exact_match",
        "contains_answer",
        "numeric",
        "math",
        "field",
        "unit_tests",
        "codebase_tests",
        "assert_tests",
    }
)


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
    expected = _extract_numeric_answer(str(sample.get("answer", "")))
    actual = _extract_numeric_answer(completion)
    if expected is None or actual is None:
        return 0.0
    return 1.0 if math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-4) else 0.0


def math_reward(completion: str, sample: dict[str, Any]) -> float:
    """Math outcome: numeric match if possible, else exact / contains fallback.

    Prefer final answers from \\boxed{...} or GSM8K-style #### markers.
    """
    expected_raw = str(sample.get("answer", "")).strip()
    if not expected_raw:
        return 0.0

    pred = _extract_final_answer_text(completion)
    exp = _extract_final_answer_text(expected_raw) or expected_raw

    exp_num = _extract_numeric_answer(exp)
    pred_num = _extract_numeric_answer(pred)
    if exp_num is not None and pred_num is not None:
        if math.isclose(pred_num, exp_num, rel_tol=1e-4, abs_tol=1e-4):
            return 1.0

    exp_n = _normalize_math_text(exp)
    pred_n = _normalize_math_text(pred)
    if exp_n and pred_n and (exp_n == pred_n or exp_n in pred_n or pred_n in exp_n):
        return 1.0
    # Soft credit when the gold answer appears anywhere in the completion.
    if exp_n and exp_n in _normalize_math_text(completion):
        return 0.5
    return 0.0


def field_reward(completion: str, sample: dict[str, Any]) -> float:
    del completion
    value = sample.get("reward", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def unit_tests_reward(completion: str, sample: dict[str, Any]) -> float:
    """Score extracted Python by fraction of stdin/stdout unit tests passed.

    Expected sample fields:
      - unit_tests: {inputs: [...], outputs: [...]} (Nemotron / NeMo Gym style)
      - optional unit_test_timeout_sec, max_unit_tests

    Reward shaping:
      - 0.0 if no extractable Python
      - small format credit when code extracts but fails all tests (keeps GRPO
        signal when one rollout formats correctly and another does not)
      - linear in pass rate otherwise
    """
    code = extract_python_code(completion)
    if not code:
        return 0.0

    # Modest credit for producing a fenced / program-like solution.
    format_credit = 0.05

    unit_tests = sample.get("unit_tests") or {}
    if not isinstance(unit_tests, dict):
        return format_credit
    inputs = unit_tests.get("inputs") or []
    outputs = unit_tests.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return format_credit

    try:
        timeout = float(sample.get("unit_test_timeout_sec", _DEFAULT_UNIT_TEST_TIMEOUT_SEC))
    except (TypeError, ValueError):
        timeout = _DEFAULT_UNIT_TEST_TIMEOUT_SEC
    try:
        max_tests = int(sample.get("max_unit_tests", _DEFAULT_MAX_UNIT_TESTS))
    except (TypeError, ValueError):
        max_tests = _DEFAULT_MAX_UNIT_TESTS
    max_tests = max(1, max_tests)

    pairs = list(zip(inputs, outputs, strict=False))[:max_tests]
    if not pairs:
        return format_credit

    passed = 0
    for stdin_text, expected in pairs:
        if _run_python_case(code, str(stdin_text), str(expected), timeout=timeout):
            passed += 1
    pass_rate = passed / len(pairs)
    # format_credit + (1 - format_credit) * pass_rate keeps full credit at 1.0
    return float(format_credit + (1.0 - format_credit) * pass_rate)


def codebase_tests_reward(completion: str, sample: dict[str, Any]) -> float:
    """Multi-file synthetic codebase judge (pytest). See codebase_judge.py."""
    from seiso.slime_single_gpu.codebase_judge import codebase_tests_reward as _cb

    return float(_cb(completion, sample))


def assert_tests_reward(completion: str, sample: dict[str, Any]) -> float:
    """Function-level assert / check() harness (MBPP, HumanEval, etc.)."""
    from seiso.slime_single_gpu.function_judge import assert_tests_reward as _fn

    return float(_fn(completion, sample))


def infer_reward_name(sample: dict[str, Any]) -> str:
    """Pick an atomic reward for a sample (multi-reward dispatch).

    Priority:
      1. sample['reward_name'] if it is an atomic reward
      2. codebase scaffold present -> codebase_tests
      3. unit_tests present -> unit_tests
      4. sample['task'] / sample['domain'] hints
      5. non-empty answer that looks numeric/math -> math
      6. non-empty answer -> contains_answer
      7. fallback exact_match
    """
    explicit = str(sample.get("reward_name") or sample.get("reward_type") or "").strip()
    if explicit in _ATOMIC_REWARDS:
        return explicit
    if explicit in {"multi", "auto", "mixed"}:
        explicit = ""

    codebase = sample.get("codebase")
    if isinstance(codebase, dict) and (codebase.get("tests") or codebase.get("files")):
        return "codebase_tests"

    if sample.get("assert_tests") or sample.get("test_list") or sample.get("check_code"):
        return "assert_tests"
    if sample.get("test") and sample.get("entry_point"):
        return "assert_tests"

    unit_tests = sample.get("unit_tests")
    if isinstance(unit_tests, dict):
        inputs = unit_tests.get("inputs") or []
        outputs = unit_tests.get("outputs") or []
        if inputs and outputs and len(inputs) == len(outputs):
            return "unit_tests"

    domain = str(sample.get("domain") or sample.get("task") or sample.get("task_type") or "").lower()
    if domain in {"codebase", "repo", "multi_file", "synthetic_codebase"}:
        return "codebase_tests"
    if domain in {"function", "mbpp", "humaneval", "assert_tests", "library"}:
        return "assert_tests"
    if domain in {"code", "coding", "programming", "competitive_coding"}:
        return "unit_tests"
    if domain in {"math", "mathematics", "gsm8k", "competition_math"}:
        return "math"

    answer = str(sample.get("answer", "")).strip()
    if answer:
        if _extract_numeric_answer(answer) is not None or "\\boxed" in answer or "####" in answer:
            return "math"
        return "contains_answer"
    return "exact_match"


def multi_reward(completion: str, sample: dict[str, Any]) -> float:
    """Per-sample multi-reward: dispatch to unit_tests / math / etc."""
    name = infer_reward_name(sample)
    return float(_ATOMIC_REWARD_FNS[name](completion, sample))


def uses_unit_tests_scoring(sample: dict[str, Any], config_reward: str) -> bool:
    """Whether outcome scoring should search the full completion for code."""
    code_rewards = {"unit_tests", "codebase_tests", "assert_tests"}
    if config_reward in code_rewards:
        return True
    if config_reward in {"multi", "auto", "mixed"}:
        return infer_reward_name(sample) in code_rewards
    return False


def extract_python_code(text: str) -> str:
    """Prefer fenced ```python blocks; fall back to whole completion."""
    if not text or not str(text).strip():
        return ""
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        # Prefer the last fenced block (final solution after reasoning).
        return matches[-1].strip()
    stripped = text.strip()
    # Heuristic: looks like a program (has def/import/input/print).
    if any(tok in stripped for tok in ("def ", "import ", "input(", "print(", "sys.")):
        return stripped
    return ""


_ATOMIC_REWARD_FNS: dict[str, RewardFn] = {
    "exact_match": exact_match_reward,
    "contains_answer": contains_answer_reward,
    "numeric": numeric_reward,
    "math": math_reward,
    "field": field_reward,
    "unit_tests": unit_tests_reward,
    "codebase_tests": codebase_tests_reward,
    "assert_tests": assert_tests_reward,
}


def resolve_reward(name: str) -> RewardFn:
    rewards: dict[str, RewardFn] = {
        **_ATOMIC_REWARD_FNS,
        "multi": multi_reward,
        "auto": multi_reward,
        "mixed": multi_reward,
    }
    try:
        return rewards[name]
    except KeyError as exc:
        choices = ", ".join(sorted(rewards))
        raise ValueError(
            f"unknown reward {name!r}; expected one of: {choices}"
        ) from exc


def _extract_final_answer_text(text: str) -> str:
    if not text:
        return ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    hashes = _HASH_ANSWER_RE.findall(text)
    if hashes:
        return hashes[-1].strip()
    return text.strip()


def _extract_numeric_answer(text: str) -> float | None:
    focused = _extract_final_answer_text(text)
    return _last_number(focused if focused else text)


def _normalize_math_text(text: str) -> str:
    s = str(text).strip().lower()
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\\left|\\right", "", s)
    return s


def _last_number(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _normalize_io(text: str) -> str:
    # Competitive programming judges usually ignore trailing whitespace per line.
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _run_python_case(
    code: str,
    stdin_text: str,
    expected: str,
    *,
    timeout: float,
) -> bool:
    if len(code) > 200_000:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="seiso-slime-ut-") as tmp:
            path = Path(tmp) / "solution.py"
            path.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
    except (subprocess.TimeoutExpired, OSError, UnicodeError):
        return False
    if proc.returncode != 0:
        return False
    actual = proc.stdout
    if len(actual) > _DEFAULT_MAX_OUTPUT_CHARS:
        return False
    return _normalize_io(actual) == _normalize_io(expected)
