"""Shared chat helpers (prompts + output sanitize + thinking budget) for CLI and Forge."""

from __future__ import annotations

from seiso.chat.prompts import chat_system_prompt, resolve_model_key
from seiso.chat.sanitize import sanitize_llm_output
from seiso.chat.thinking import (
    ThinkingPolicy,
    ThinkingStreamGuard,
    apply_thinking_policy,
    classify_task,
    resolve_thinking_policy,
    thinking_max_tokens,
)

__all__ = [
    "ThinkingPolicy",
    "ThinkingStreamGuard",
    "apply_thinking_policy",
    "chat_system_prompt",
    "classify_task",
    "resolve_model_key",
    "resolve_thinking_policy",
    "sanitize_llm_output",
    "thinking_max_tokens",
]
