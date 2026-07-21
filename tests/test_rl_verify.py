"""Tests for the shared RL verifier."""

from __future__ import annotations

import pytest

from seiso.distill_rl.outcome import ensure_thinking_completion, outcome_reward
from seiso.rl_verify import (
    final_answer_text,
    format_thinking_prompt,
    has_closed_thinking_trace,
    score_completion,
    verify_outcome,
)
from seiso.slime.rewards import contains_answer_reward, numeric_reward


def test_format_thinking_prompt_appends_instruction():
    text = format_thinking_prompt("Solve.", "Show work.")
    assert "Show work." in text
    assert text.endswith("<think>")


def test_ensure_thinking_completion_is_noop():
    assert ensure_thinking_completion("answer", enabled=True) == "answer"


def test_closed_thinking_trace_detection():
    assert has_closed_thinking_trace("<think>x</think>42") is True
    assert has_closed_thinking_trace("<think>x") is False
    assert has_closed_thinking_trace("42") is False
    # Prompt already opened <think>; model continues and closes.
    assert has_closed_thinking_trace("step by step\n</think>\n42") is True
    assert has_closed_thinking_trace("</think>42") is True


def test_final_answer_text_after_think():
    assert final_answer_text("<think>reason</think> 42") == "42"
    assert final_answer_text("reason\n</think>\n42") == "42"


def test_numeric_and_choice_outcome():
    assert verify_outcome("The answer is 42.", "42", checker="numeric")[0] == 1.0
    assert verify_outcome("I pick B.", "B", checker="choice")[0] == 1.0
    assert verify_outcome("nope", "42", checker="numeric")[0] == 0.0


def test_field_reward_is_clamped_to_unit_interval():
    assert verify_outcome("", None, checker="field", field_reward=2.5)[0] == 1.0
    assert verify_outcome("", None, checker="field", field_reward=-1.0)[0] == 0.0
    assert verify_outcome("", None, checker="field", field_reward=0.4)[0] == 0.4


def test_extract_choice_prefers_final_letter():
    from seiso.rl_verify.extract import extract_choice

    # First letter is a distractor; final pick is B.
    assert extract_choice("A is wrong; the answer is B") == "b"
    assert extract_choice("a better option is B") == "b"
    assert extract_choice("select a better option is B") == "b"
    assert extract_choice("Answer: C") == "c"
    assert extract_choice("I pick D.") == "d"
    # Unique free-form letter still works.
    assert extract_choice("Definitely B only.") == "b"
    assert verify_outcome(
        "A is wrong; the answer is B", "B", checker="choice"
    )[0] == 1.0


def test_last_number_ignores_confidence_noise():
    from seiso.rl_verify.extract import last_number

    assert last_number("The answer is 42, confidence 100") == 42.0
    assert last_number("answer 42 after checking 3 cases") == 42.0
    assert last_number("Final answer: 7") == 7.0
    assert last_number("\\boxed{99}") == 99.0
    assert verify_outcome(
        "The answer is 42, confidence 100", "42", checker="numeric"
    )[0] == 1.0


def test_outcome_reward_distill_api():
    assert outcome_reward("<think>x</think>7", "7", benchmark="gsm8k") == 1.0
    assert outcome_reward("<think>x</think>C", "C", benchmark="gpqa") == 1.0


def test_score_completion_outcome_first():
    good = score_completion(
        "<think>because</think>42",
        {"answer": "42"},
        checker="numeric",
        require_thinking_trace=True,
        outcome_weight=1.0,
        format_weight=0.1,
        process_weight=0.0,
        missing_format_penalty=0.5,
    )
    # Prompt ended with open <think>; model only continues + closes.
    continued = score_completion(
        "because arithmetic\n</think>\n42",
        {"answer": "42"},
        checker="numeric",
        require_thinking_trace=True,
        outcome_weight=1.0,
        format_weight=0.1,
        process_weight=0.0,
        missing_format_penalty=0.5,
    )
    bad_format = score_completion(
        "42",
        {"answer": "42"},
        checker="numeric",
        require_thinking_trace=True,
        outcome_weight=1.0,
        format_weight=0.1,
        process_weight=0.0,
        missing_format_penalty=0.5,
    )

    assert good.passed is True
    assert good.format_ok is True
    assert good.reward == 1.1
    assert good.process_score == 0.0
    assert continued.passed is True
    assert continued.format_ok is True
    assert continued.reward == 1.1
    assert continued.final_answer == "42"
    assert bad_format.passed is True
    assert bad_format.format_ok is False
    assert bad_format.reward == 0.5  # 1.0 outcome - 0.5 penalty
    assert bad_format.detail == "missing_closed_think_trace"


def test_score_completion_prefers_final_answer_even_without_thinking_requirement():
    """Gold only inside <think> must not count when the final answer is wrong."""
    wrong_final = score_completion(
        "<think>Final answer: 42</think>\n41",
        {"answer": "42"},
        checker="numeric",
        require_thinking_trace=False,
    )
    assert wrong_final.outcome == 0.0
    assert wrong_final.passed is False

    contains_wrong = score_completion(
        "<think>The answer might be 42 but I doubt it</think>\nActually 41",
        {"answer": "42"},
        checker="contains_answer",
        require_thinking_trace=False,
    )
    assert contains_wrong.outcome == 0.0


def test_contains_answer_rejects_substring_traps():
    assert verify_outcome("420", "42", checker="contains_answer")[0] == 0.0
    assert verify_outcome("The answer is 42.", "42", checker="contains_answer")[0] == 1.0
    assert verify_outcome("new york city", "new york", checker="contains_answer")[0] == 1.0


def test_contains_answer_preserves_signed_and_plus_golds():
    # Must not collapse ``c++`` → ``c`` or ``-3`` → ``3``.
    assert verify_outcome("select option c", "c++", checker="contains_answer")[0] == 0.0
    assert verify_outcome("use c++ please", "c++", checker="contains_answer")[0] == 1.0
    assert verify_outcome("version 3", "-3", checker="contains_answer")[0] == 0.0
    assert verify_outcome("answer is -3", "-3", checker="contains_answer")[0] == 1.0
    assert verify_outcome("I got +5 points", "+5", checker="contains_answer")[0] == 1.0


def test_code_partial_credit_does_not_mark_passed():
    sample = {"tests": ["assert add(1,2)==3", "assert add(2,2)==4", "assert add(0,0)==0"]}
    partial = "def add(a,b):\n    return 3 if a==1 else (4 if a==2 else 9)\n"
    result = score_completion(partial, sample, checker="code")
    # Default binary: GRPO outcome is all-tests-pass; fraction stays on proof_score.
    assert result.outcome == 0.0
    assert result.proof_score == pytest.approx(2.0 / 3.0)
    assert result.proof_passed is False
    assert result.passed is False
    assert result.reward == 0.0

    dense = score_completion(
        partial, sample, checker="code", code_reward_mode="dense"
    )
    assert dense.outcome == pytest.approx(2.0 / 3.0)
    assert dense.passed is False
    assert dense.reward == pytest.approx(2.0 / 3.0)

    auto = score_completion(
        partial, sample, checker="code", code_reward_mode="auto"
    )
    # auto is provisional dense until the trainer promotes a group to binary.
    assert auto.outcome == pytest.approx(2.0 / 3.0)


def test_slime_named_rewards_prefer_final_answer():
    sample = {"answer": "42"}
    think_only = "<think>Final answer: 42</think>\n41"
    assert numeric_reward(think_only, sample) == 0.0
    assert contains_answer_reward(think_only, sample) == 0.0
    assert numeric_reward("<think>x</think>\n42", sample) == 1.0
