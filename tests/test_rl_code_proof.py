"""Tests for sandboxed code proof rewards."""

from __future__ import annotations

from seiso.rl_verify import (
    extract_python_code,
    score_completion,
    verify_code_proof,
    verify_outcome,
)
from seiso.slime.rewards import code_reward, resolve_reward


def test_extract_python_from_fence():
    text = "Sure.\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert "def add" in extract_python_code(text)


def test_code_proof_passes_all_asserts():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    sample = {
        "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }
    result = verify_code_proof(completion, sample)
    assert result.passed is True
    assert result.score == 1.0
    assert result.tests_passed == 2
    assert result.tests_total == 2
    assert result.detail.startswith("code:pass 2/2")


def test_code_proof_partial_score():
    completion = "def add(a, b):\n    return a + b\n"
    sample = {
        "tests": [
            "assert add(1, 2) == 3",
            "assert add(1, 2) == 999",  # fails
        ],
    }
    result = verify_code_proof(completion, sample)
    assert result.passed is False
    assert result.score == 0.5
    assert result.tests_passed == 1
    # Outcome adapters are binary — partial fraction is not GRPO credit.
    score, checker, _ = verify_outcome(
        completion, None, checker="code", sample=sample
    )
    assert checker == "code"
    assert score == 0.0
    scored = score_completion(completion, sample, checker="code")
    assert scored.outcome == 0.0
    assert scored.proof_score == 0.5


def test_code_proof_wrong_solution_zero():
    completion = "def add(a, b):\n    return a - b\n"
    sample = {"tests": ["assert add(1, 2) == 3"]}
    result = verify_code_proof(completion, sample)
    assert result.passed is False
    assert result.score == 0.0


def test_code_proof_with_prompt_prefix():
    # Model only completes the body; prefix provides the signature.
    completion = "    return s[::-1]\n"
    sample = {
        "prompt_code": "def reverse_string(s: str) -> str:\n",
        "tests": ["assert reverse_string('ab') == 'ba'"],
    }
    result = verify_code_proof(completion, sample)
    assert result.passed is True
    assert result.score == 1.0


def test_verify_outcome_code_checker():
    completion = "def is_even(n):\n    return n % 2 == 0\n"
    sample = {"tests": ["assert is_even(2) is True", "assert is_even(3) is False"]}
    score, checker, extracted = verify_outcome(
        completion, None, checker="code", sample=sample
    )
    assert checker == "code"
    assert score == 1.0
    assert "is_even" in extracted


def test_score_completion_code_sets_proof_fields():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    sample = {"tests": ["assert add(2, 2) == 4"]}
    result = score_completion(
        completion,
        sample,
        checker="code",
        require_thinking_trace=False,
        outcome_weight=1.0,
        format_weight=0.0,
    )
    assert result.passed is True
    assert result.checker == "code"
    assert result.proof_passed is True
    assert result.proof_score == 1.0
    assert result.proof_detail is not None
    assert result.reward == 1.0


def test_resolve_reward_code_alias():
    fn = resolve_reward("code")
    completion = "def add(a, b):\n    return a + b\n"
    sample = {"tests": ["assert add(1, 1) == 2"]}
    assert fn(completion, sample) == 1.0
    assert code_reward(completion, sample) == 1.0


def test_missing_tests_scores_zero():
    result = verify_code_proof("def f():\n    return 1\n", {})
    assert result.score == 0.0
    assert "missing_tests" in result.detail


def test_bare_answer_not_promoted_to_code_tests():
    """Misconfigured code rows must not treat '42' as a always-passing test."""
    completion = "def solve():\n    return 42\n"
    score, checker, _ = verify_outcome(
        completion,
        "42",
        checker="code",
        sample={},
    )
    assert checker == "code"
    assert score == 0.0

    # Explicit asserts still work via answer-as-tests compact form.
    score_ok, _, _ = verify_outcome(
        completion,
        "assert solve() == 42",
        checker="code",
        sample={},
    )
    assert score_ok == 1.0


def test_non_assert_tests_list_is_rejected():
    result = verify_code_proof(
        "def f():\n    return 1\n",
        {"tests": ["42", "hello"]},
    )
    assert result.score == 0.0
    assert "missing_tests" in result.detail


def test_example_code_dataset_rows_are_loadable():
    from pathlib import Path

    from seiso.io.jsonl import iter_jsonl

    rows = list(iter_jsonl(Path("data/slime_code_sample.jsonl")))
    assert len(rows) >= 12
    assert all("tests" in row for row in rows)


def test_example_math_and_choice_datasets_are_loadable():
    from pathlib import Path

    from seiso.io.jsonl import iter_jsonl

    math_rows = list(iter_jsonl(Path("data/slime_sample.jsonl")))
    choice_rows = list(iter_jsonl(Path("data/slime_choice_sample.jsonl")))
    distill_rows = list(iter_jsonl(Path("data/distill_verifiable_prompts.jsonl")))
    assert len(math_rows) >= 16
    assert all("answer" in row for row in math_rows)
    assert len(choice_rows) >= 8
    assert all(row.get("benchmark") == "gpqa" for row in choice_rows)
    assert len(distill_rows) >= 16
    assert any(row.get("tests") for row in distill_rows)
    assert any(row.get("benchmark") == "gsm8k" for row in distill_rows)
