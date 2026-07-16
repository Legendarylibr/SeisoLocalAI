"""Chat system prompts — re-export from ``seiso.chat.prompts``.

Prefer importing from ``seiso.chat.prompts`` in new code. This module remains for
existing Forge imports.
"""

from __future__ import annotations

from seiso.chat.prompts import (  # noqa: F401
    chat_system_prompt,
    is_reasoning_prone_model,
    model_display_label,
    model_switch_system_prompt,
    resolve_model_key,
)

__all__ = [
    "chat_system_prompt",
    "is_reasoning_prone_model",
    "model_display_label",
    "model_switch_system_prompt",
    "resolve_model_key",
]
