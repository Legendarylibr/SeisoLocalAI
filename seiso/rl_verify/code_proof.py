"""Checkable code proofs: extract Python and run sandboxed unit tests.

This is the hard process/outcome signal for code RL — not lexical reasoning.
Completions must define working code that passes the sample's tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Prefer the existing compress sandbox (subprocess + resource limits + import guard).
from seiso.codellama_compress.code_exec import run_python_sandboxed

_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_ASSERT_LINE_RE = re.compile(r"^\s*assert\b", flags=re.MULTILINE)
_DEFAULT_TIMEOUT_S = 3.0
_MAX_TESTS = 32


@dataclass(frozen=True)
class CodeProofResult:
    """Result of a sandboxed code proof check."""

    passed: bool
    score: float
    tests_total: int
    tests_passed: int
    extracted_code: str
    detail: str
    exit_code: int = 0
    stderr: str = ""
    reason: str = "ok"


def extract_python_code(completion: str) -> str:
    """Pull the primary Python snippet from a model completion."""
    text = (completion or "").strip()
    if not text:
        return ""

    # Prefer the last fenced block (models often restate then code).
    fences = list(_FENCE_RE.finditer(text))
    if fences:
        return fences[-1].group("body").strip()

    # Strip closed thinking, then try again on the final answer region.
    from seiso.rl_verify.extract import final_answer_text

    final = final_answer_text(text).strip()
    if final and final != text:
        fences = list(_FENCE_RE.finditer(final))
        if fences:
            return fences[-1].group("body").strip()
        if _looks_like_python(final):
            return final

    if _looks_like_python(text):
        return text
    return final if _looks_like_python(final) else ""


def _looks_like_python(text: str) -> bool:
    lowered = text.lstrip()
    if not lowered:
        return False
    markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "async def ",
        "@",
    )
    return any(lowered.startswith(m) or f"\n{m}" in f"\n{lowered}" for m in markers)


def _normalize_tests(sample: dict[str, Any]) -> list[str]:
    """Return individual test units (assert lines or full harness strings)."""
    raw = sample.get("tests", sample.get("test"))
    if raw is None:
        return []
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return items[:_MAX_TESTS]
    text = str(raw).strip()
    if not text:
        return []
    # Split multi-assert bodies into per-assert units when they are simple lines.
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    assert_lines = [ln for ln in lines if _ASSERT_LINE_RE.match(ln)]
    if assert_lines and len(assert_lines) == len(lines):
        return assert_lines[:_MAX_TESTS]
    return [text]


def _code_prefix(sample: dict[str, Any]) -> str:
    for key in ("prompt_code", "code_prefix", "prompt_prefix"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).rstrip() + "\n"
    return ""


def _setup(sample: dict[str, Any]) -> str:
    value = sample.get("setup")
    if value is None or not str(value).strip():
        return ""
    return str(value).rstrip() + "\n\n"


def _timeout_s(sample: dict[str, Any]) -> float:
    raw = sample.get("timeout_s", sample.get("timeout", _DEFAULT_TIMEOUT_S))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return max(0.1, min(value, 30.0))


def build_program(
    *,
    extracted_code: str,
    sample: dict[str, Any],
    test_unit: str | None = None,
) -> str:
    """Assemble sandboxed program: setup + prompt prefix + solution + tests."""
    parts: list[str] = []
    setup = _setup(sample)
    if setup:
        parts.append(setup.rstrip())
    prefix = _code_prefix(sample)
    if prefix:
        parts.append(prefix.rstrip())
    parts.append(extracted_code.rstrip())
    test_body = test_unit if test_unit is not None else "\n".join(_normalize_tests(sample))
    if test_body.strip():
        parts.append("# --- tests (verifier) ---\n" + test_body.strip())
    entry = sample.get("entry_point")
    # HumanEval-style: test often defines check(candidate); call it if present.
    if entry and test_body and "check(" in test_body and f"check({entry}" not in test_body:
        parts.append(f"check({entry})")
    return "\n\n".join(parts) + "\n"


def verify_code_proof(
    completion: str,
    sample: dict[str, Any],
    *,
    timeout_s: float | None = None,
) -> CodeProofResult:
    """Run sandboxed tests against extracted code; score is pass fraction in [0, 1]."""
    code = extract_python_code(completion)
    # HumanEval-style: prefix has the signature; completion may be only the body.
    if not code and _code_prefix(sample):
        # Preserve leading indentation; only trim trailing whitespace.
        body = (completion or "").rstrip()
        if body.strip():
            code = body
    if not code:
        return CodeProofResult(
            passed=False,
            score=0.0,
            tests_total=0,
            tests_passed=0,
            extracted_code="",
            detail="code:no_extractable_python",
            reason="internal_error",
        )

    tests = _normalize_tests(sample)
    if not tests:
        return CodeProofResult(
            passed=False,
            score=0.0,
            tests_total=0,
            tests_passed=0,
            extracted_code=code,
            detail="code:missing_tests",
            reason="internal_error",
        )

    limit = timeout_s if timeout_s is not None else _timeout_s(sample)
    # Budget total wall time across units.
    per_test = max(0.1, limit / max(1, len(tests)))

    passed = 0
    last_stderr = ""
    last_exit = 0
    last_reason = "ok"
    for unit in tests:
        program = build_program(
            extracted_code=code, sample=sample, test_unit=unit
        )
        result = run_python_sandboxed(code=program, timeout_s=per_test)
        last_stderr = result.stderr
        last_exit = result.exit_code
        last_reason = result.reason
        if result.ok:
            passed += 1

    total = len(tests)
    score = float(passed) / float(total) if total else 0.0
    all_ok = passed == total and total > 0
    detail = f"code:pass {passed}/{total}"
    if not all_ok and last_reason != "ok":
        detail = f"{detail}; last={last_reason}"
    elif not all_ok and last_stderr:
        detail = f"{detail}; stderr={last_stderr.splitlines()[0][:120]}"

    return CodeProofResult(
        passed=all_ok,
        score=score,
        tests_total=total,
        tests_passed=passed,
        extracted_code=code,
        detail=detail,
        exit_code=last_exit,
        stderr=last_stderr[:500],
        reason=last_reason if not all_ok else "ok",
    )


def code_outcome_score(completion: str, sample: dict[str, Any]) -> tuple[float, str, str]:
    """Adapter for verify_outcome: ``(score, checker, extracted_code)``."""
    result = verify_code_proof(completion, sample)
    return result.score, "code", result.extracted_code
