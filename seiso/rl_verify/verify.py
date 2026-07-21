"""Hard outcome + format verification for local RL pipelines."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from seiso.rl_verify.extract import (
    extract_choice,
    final_answer_text,
    has_closed_thinking_trace,
    last_number,
    normalize_answer,
    split_thinking_trace,
)

# Lexical process markers are experimental only (weight 0 by default).
_TRANSITION_MARKERS = (
    "because",
    "therefore",
    "so",
    "first",
    "next",
    "then",
    "check",
    "verify",
)
_REVISION_RE = re.compile(r"\b(wait|actually|however|but|correct|revise)\b", re.I)


@dataclass(frozen=True)
class VerifierResult:
    """Structured decision from the shared verifier."""

    passed: bool
    outcome: float
    format_ok: bool
    format_score: float
    process_score: float
    reward: float
    extracted_answer: str
    thinking_trace: str
    final_answer: str
    checker: str
    detail: str | None = None
    # Optional checkable-proof channel (e.g. sandboxed code tests).
    proof_passed: bool | None = None
    proof_score: float | None = None
    proof_detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_checker(
    name: str | None = None,
    *,
    benchmark: str | None = None,
) -> str:
    """Map a reward name or benchmark label to a checker id."""
    if name:
        key = name.strip().lower()
        aliases = {
            "exact": "exact_match",
            "exact_match": "exact_match",
            "contains": "contains_answer",
            "contains_answer": "contains_answer",
            "numeric": "numeric",
            "math": "numeric",
            "choice": "choice",
            "multiple_choice": "choice",
            "field": "field",
            "code": "code",
            "python": "code",
            "code_exec": "code",
            "code_proof": "code",
            "humaneval": "code",
            "auto": "auto",
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(
            f"unknown checker {name!r}; expected one of: "
            f"{', '.join(sorted(set(aliases)))}"
        )
    bench = (benchmark or "").strip().lower()
    if bench in {"gsm8k", "aime", "math", "numeric"}:
        return "numeric"
    if bench in {"gpqa", "multiple_choice", "choice"}:
        return "choice"
    if bench in {"humaneval", "mbpp", "code", "python", "code_exec"}:
        return "code"
    return "auto"


def verify_outcome(
    completion: str,
    answer: str | None,
    *,
    checker: str = "auto",
    benchmark: str | None = None,
    field_reward: float | None = None,
    prefer_final_answer: bool = True,
    sample: dict[str, Any] | None = None,
) -> tuple[float, str, str]:
    """Return ``(outcome_0_1, checker_used, extracted_answer)``."""
    if checker == "field":
        if field_reward is None:
            return 0.0, "field", ""
        try:
            # Bound dataset-supplied floats so they cannot dominate GRPO advantages.
            return max(0.0, min(1.0, float(field_reward))), "field", ""
        except (TypeError, ValueError):
            return 0.0, "field", ""

    if checker == "auto":
        if sample is not None and (
            sample.get("tests") is not None or sample.get("test") is not None
        ):
            resolved = "code"
        else:
            resolved = resolve_checker(benchmark=benchmark)
            if resolved == "auto":
                if answer is None or not str(answer).strip():
                    return 0.0, "exact_match", ""
                text_probe = (
                    final_answer_text(completion) if prefer_final_answer else completion
                )
                if last_number(str(answer)) is not None and last_number(
                    text_probe
                ) is not None:
                    resolved = "numeric"
                else:
                    resolved = "exact_match"
    else:
        resolved = resolve_checker(checker, benchmark=benchmark)

    if resolved == "code":
        from seiso.rl_verify.code_proof import (
            code_outcome_score,
            is_checkable_test_body,
        )

        payload = dict(sample or {})
        # Only promote answer → tests when it is a real assert/check harness.
        # Bare answers like "42" would otherwise execute as always-true expressions.
        if (
            answer is not None
            and "tests" not in payload
            and "test" not in payload
            and is_checkable_test_body(answer)
        ):
            payload.setdefault("tests", answer)
        return code_outcome_score(completion, payload)

    if answer is None or not str(answer).strip():
        return 0.0, resolved if resolved != "auto" else "exact_match", ""

    text_for_outcome = (
        final_answer_text(completion) if prefer_final_answer else completion
    )
    expected = str(answer).strip()

    if resolved == "numeric":
        score, extracted = _numeric_match(text_for_outcome, expected)
        return score, "numeric", extracted
    if resolved == "choice":
        score, extracted = _choice_match(text_for_outcome, expected)
        return score, "choice", extracted
    if resolved == "contains_answer":
        score, extracted = _contains_answer(text_for_outcome, expected)
        return score, "contains_answer", extracted
    score, extracted = _exact_match(text_for_outcome, expected)
    return score, "exact_match", extracted


def outcome_reward(
    completion: str,
    answer: str | None,
    *,
    benchmark: str | None = None,
    checker: str = "auto",
) -> float:
    """Pure 0/1 (or dense field) outcome score — distill-RL compatible API."""
    score, _, _ = verify_outcome(
        completion,
        answer,
        checker=checker,
        benchmark=benchmark,
        prefer_final_answer=True,
    )
    return float(score)


def format_reward(
    completion: str,
    *,
    require_thinking_trace: bool,
) -> tuple[bool, float]:
    """Binary format score on *raw* generated tokens only."""
    if not require_thinking_trace:
        return True, 0.0
    ok = has_closed_thinking_trace(completion)
    return ok, (1.0 if ok else 0.0)


def experimental_process_reward(
    thinking_trace: str,
    final_answer: str,
    *,
    min_thinking_tokens: int = 8,
) -> float:
    """Lexical process heuristic — keep weight 0 unless explicitly experimenting."""
    tokens = re.findall(r"\w+", thinking_trace)
    if not tokens:
        return 0.0
    score = 0.0
    if len(tokens) >= min_thinking_tokens:
        score += 0.35
    else:
        score += 0.35 * (len(tokens) / max(min_thinking_tokens, 1))
    lower = thinking_trace.lower()
    transition_hits = sum(marker in lower for marker in _TRANSITION_MARKERS)
    score += min(0.35, 0.07 * transition_hits)
    if _REVISION_RE.search(lower):
        score += 0.15
    if final_answer.strip():
        score += 0.15
    return min(1.0, score)


def resolve_code_reward_mode(mode: str | None) -> str:
    """Normalize code reward mode: ``binary`` (default), ``dense``, or ``auto``."""
    # Use membership checks (not a value dict) so bandit B105 does not treat
    # mode tokens like ``binary`` / ``dense`` as hardcoded passwords.
    key = str(mode or "binary").strip().lower()
    if key in {"binary", "all_pass", "pass"}:
        return "binary"
    if key in {"dense", "fraction", "pass_fraction"}:
        return "dense"
    if key in {"auto", "curriculum"}:
        return "auto"
    raise ValueError(
        f"unknown code_reward_mode {mode!r}; expected one of: "
        "binary, dense, auto"
    )


def code_outcome_value(
    *,
    proof_passed: bool,
    proof_score: float,
    mode: str = "binary",
) -> float:
    """Map a code proof to a GRPO outcome under ``code_reward_mode``.

    - ``binary``: 1.0 iff all tests pass (correctness default)
    - ``dense``: pass fraction in [0, 1] (early-training signal)
    - ``auto``: provisional dense; callers should promote to binary once a
      same-prompt group contains any full pass
    """
    resolved = resolve_code_reward_mode(mode)
    if resolved == "dense" or resolved == "auto":
        return float(proof_score)
    return 1.0 if proof_passed else 0.0


def score_completion(
    completion: str,
    sample: dict[str, Any],
    *,
    checker: str = "auto",
    require_thinking_trace: bool = False,
    outcome_weight: float = 1.0,
    format_weight: float = 0.1,
    process_weight: float = 0.0,
    missing_format_penalty: float = 0.0,
    min_thinking_tokens: int = 8,
    code_reward_mode: str = "binary",
) -> VerifierResult:
    """Combine outcome + format (+ optional experimental process) into one decision.

    ``completion`` must be the model-generated string — do not prepend synthetic tags.
    For code, ``code_reward_mode`` selects binary / dense / auto outcome mapping.
    """
    thinking_trace, final_answer, has_closed = split_thinking_trace(completion)
    format_ok, format_score = format_reward(
        completion, require_thinking_trace=require_thinking_trace
    )

    answer = sample.get("answer")
    field_value = sample.get("reward")
    bench = sample.get("benchmark")
    resolved_name = (
        resolve_checker(checker, benchmark=bench if isinstance(bench, str) else None)
        if checker != "auto"
        else "auto"
    )
    # Prefer code proof when sample carries tests or checker is code.
    use_code = resolved_name == "code" or (
        checker in {"auto", "code", "python", "code_exec", "code_proof", "humaneval"}
        and (sample.get("tests") is not None or sample.get("test") is not None)
    )

    proof_passed: bool | None = None
    proof_score: float | None = None
    proof_detail: str | None = None

    if use_code:
        from seiso.rl_verify.code_proof import verify_code_proof

        proof = verify_code_proof(completion, sample)
        proof_passed = proof.passed
        proof_score = float(proof.score)
        proof_detail = proof.detail
        # Keep pass-fraction on proof_score for logs / hard negatives.
        # Outcome follows code_reward_mode (default binary = all tests pass).
        outcome = code_outcome_value(
            proof_passed=bool(proof_passed),
            proof_score=float(proof_score),
            mode=code_reward_mode,
        )
        used_checker = "code"
        extracted = proof.extracted_code
    else:
        outcome, used_checker, extracted = verify_outcome(
            completion,
            None if answer is None else str(answer),
            checker=checker,
            benchmark=bench if isinstance(bench, str) else None,
            field_reward=None if field_value is None else field_value,
            # Always score the final-answer span for text checkers. Thinking
            # format is orthogonal — models may emit <think> even when the
            # trainer does not require it.
            prefer_final_answer=True,
            sample=sample,
        )

    process = 0.0
    if process_weight > 0 and require_thinking_trace and has_closed:
        process = experimental_process_reward(
            thinking_trace,
            final_answer,
            min_thinking_tokens=min_thinking_tokens,
        )

    penalty = (
        missing_format_penalty
        if require_thinking_trace and not format_ok
        else 0.0
    )
    reward = (
        outcome_weight * outcome
        + format_weight * format_score
        + process_weight * process
        - penalty
    )
    # Code: passed iff all unit tests pass (outcome already binary).
    # Text/math: binary (or dense field) threshold.
    passed = (
        bool(proof_passed)
        if use_code and proof_passed is not None
        else outcome > 0.5
    )
    detail = None
    if use_code and proof_detail is not None:
        detail = proof_detail
    elif require_thinking_trace and not format_ok:
        detail = "missing_closed_think_trace"
    elif not passed and answer is not None:
        detail = "outcome_mismatch"

    return VerifierResult(
        passed=passed,
        outcome=float(outcome),
        format_ok=format_ok,
        format_score=float(format_score),
        process_score=float(process),
        reward=float(reward),
        extracted_answer=extracted if extracted else final_answer,
        thinking_trace=thinking_trace,
        final_answer=final_answer if require_thinking_trace else completion.strip(),
        checker=used_checker,
        detail=detail,
        proof_passed=proof_passed,
        proof_score=proof_score,
        proof_detail=proof_detail,
    )


def _exact_match(actual: str, expected: str) -> tuple[float, str]:
    actual_norm = normalize_answer(actual)
    expected_norm = normalize_answer(expected)
    if not actual_norm or not expected_norm:
        return 0.0, actual.strip()
    if actual_norm == expected_norm:
        return 1.0, actual.strip()
    if actual.strip() == expected.strip():
        return 1.0, actual.strip()
    return 0.0, actual.strip()


def _contains_answer(actual: str, expected: str) -> tuple[float, str]:
    """Token-boundary containment — rejects substring traps like ``42`` in ``420``.

    Preserves signed / ``+``-suffixed golds (``-3``, ``c++``). Only trailing
    sentence ``.`` is stripped so ``42.`` still matches gold ``42``.
    """
    extracted = actual.strip()
    expected_norm = normalize_answer(expected)
    if not expected_norm:
        return 0.0, extracted
    actual_norm = normalize_answer(actual)
    if not actual_norm:
        return 0.0, extracted

    def _tokens(text: str) -> list[str]:
        # Strip trailing sentence periods only — never +/- (would collapse
        # gold ``c++`` → ``c`` or ``-3`` → ``3``).
        return [tok.rstrip(".") for tok in text.split() if tok.rstrip(".")]

    actual_tokens = _tokens(actual_norm)
    expected_tokens = _tokens(expected_norm)
    if not expected_tokens:
        return 0.0, extracted
    width = len(expected_tokens)
    for start in range(0, len(actual_tokens) - width + 1):
        if actual_tokens[start : start + width] == expected_tokens:
            return 1.0, extracted
    return 0.0, extracted


def _numeric_match(actual: str, expected: str) -> tuple[float, str]:
    actual_value = last_number(actual)
    expected_value = last_number(expected)
    extracted = "" if actual_value is None else str(actual_value)
    if actual_value is None or expected_value is None:
        score, extracted_exact = _exact_match(actual, expected)
        return score, extracted_exact
    ok = math.isclose(actual_value, expected_value, rel_tol=1e-4, abs_tol=1e-4)
    return (1.0 if ok else 0.0), extracted


def _choice_match(actual: str, expected: str) -> tuple[float, str]:
    actual_norm = normalize_answer(actual)
    expected_norm = normalize_answer(expected)
    if actual_norm and expected_norm and actual_norm == expected_norm:
        return 1.0, actual_norm
    actual_choice = extract_choice(actual_norm or actual)
    expected_choice = extract_choice(expected_norm or expected)
    if actual_choice and expected_choice and actual_choice == expected_choice:
        return 1.0, actual_choice
    return 0.0, actual_choice or actual.strip()
