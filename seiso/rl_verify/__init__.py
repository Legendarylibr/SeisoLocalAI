"""Shared verifiable rewards for slime GRPO and distill-RL.

Outcome scores come from hard checks (exact, numeric, choice). Format rewards
inspect the *generated* completion only — never post-hoc rewritten strings.
"""

from __future__ import annotations

from seiso.rl_verify.extract import (
    extract_choice,
    final_answer_text,
    format_thinking_prompt,
    has_closed_thinking_trace,
    last_number,
    normalize_answer,
    split_thinking_trace,
)
from seiso.rl_verify.verify import (
    VerifierResult,
    format_reward,
    outcome_reward,
    resolve_checker,
    score_completion,
    verify_outcome,
)

__all__ = [
    "VerifierResult",
    "extract_choice",
    "final_answer_text",
    "format_reward",
    "format_thinking_prompt",
    "has_closed_thinking_trace",
    "last_number",
    "normalize_answer",
    "outcome_reward",
    "resolve_checker",
    "score_completion",
    "split_thinking_trace",
    "verify_outcome",
]
