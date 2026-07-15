"""Tests for the shared RL verifier."""

from __future__ import annotations

from seiso.distill_rl.outcome import ensure_thinking_completion, outcome_reward
from seiso.rl_verify import (
    final_answer_text,
    format_thinking_prompt,
    has_closed_thinking_trace,
    score_completion,
    verify_outcome,
)


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


def test_final_answer_text_after_think():
    assert final_answer_text("<think>reason</think> 42") == "42"


def test_numeric_and_choice_outcome():
    assert verify_outcome("The answer is 42.", "42", checker="numeric")[0] == 1.0
    assert verify_outcome("I pick B.", "B", checker="choice")[0] == 1.0
    assert verify_outcome("nope", "42", checker="numeric")[0] == 0.0


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
    assert bad_format.passed is True
    assert bad_format.format_ok is False
    assert bad_format.reward == 0.5  # 1.0 outcome - 0.5 penalty
    assert bad_format.detail == "missing_closed_think_trace"
