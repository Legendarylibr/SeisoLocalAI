"""Tool result sanitization — data-only envelope for LLM context."""

from __future__ import annotations

import re
import unicodedata

from forge.security.code_policy import flag_instruction_like, scrub_secrets

_ENVELOPE_START = "[TOOL_DATA source={source}]"
_ENVELOPE_END = "[/TOOL_DATA]"
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def normalize_text(text: str) -> str:
    """NFKC normalize and strip zero-width characters."""
    cleaned = _ZERO_WIDTH.sub("", text)
    return unicodedata.normalize("NFKC", cleaned)


def wrap_tool_result(source: str, data: str, *, max_len: int = 12_000) -> str:
    """Wrap tool output as untrusted data for the model."""
    body = scrub_secrets(normalize_text(data))[:max_len]
    body = flag_instruction_like(body)
    return f"{_ENVELOPE_START.format(source=source)}\n{body}\n{_ENVELOPE_END}"
