"""Helpers for streaming assistant text to clients.

Implementation lives in ``seiso.chat.sanitize`` so the CLI does not need Forge.
This module re-exports the public surface for existing Forge imports.
"""

from __future__ import annotations

from seiso.chat.sanitize import (  # noqa: F401
    StreamingOutputSanitizer,
    chunk_sanitized_output,
    sanitize_llm_output,
    strip_spurious_chat_artifacts,
    strip_spurious_tool_syntax,
)

__all__ = [
    "StreamingOutputSanitizer",
    "chunk_sanitized_output",
    "sanitize_llm_output",
    "strip_spurious_chat_artifacts",
    "strip_spurious_tool_syntax",
]
