"""Outcome-only scoring helpers for verifiable distill-RL tasks.

Implementation lives in ``seiso.rl_verify``; this module re-exports the stable
distill-RL API and keeps backward-compatible names.
"""

from __future__ import annotations

from seiso.rl_verify.extract import (
    final_answer_text,
    format_thinking_prompt,
)
from seiso.rl_verify.verify import outcome_reward

__all__ = [
    "final_answer_text",
    "format_thinking_prompt",
    "outcome_reward",
]
