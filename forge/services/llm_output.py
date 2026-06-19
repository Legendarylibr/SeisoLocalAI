"""Guards for assistant text before it leaves the server."""

from __future__ import annotations

import re
from collections.abc import Iterator

_FALLBACK_RESPONSE = "I can't share hidden system or developer instructions."
_CHUNK_SIZE = 1_200
_STREAM_HOLD_CHARS = 512

_LEAK_PATTERNS = [
    # Common chat templates.
    re.compile(r"(?is)<\|im_start\|>\s*(?:system|developer)\b.*?(?:<\|im_end\|>|$)"),
    re.compile(
        r"(?is)<\|start_header_id\|>\s*(?:system|developer)\s*<\|end_header_id\|>.*?(?:<\|eot_id\|>|$)"
    ),
    re.compile(r"(?is)<<SYS>>.*?(?:<</SYS>>|$)"),
    re.compile(r"(?is)\[INST\]\s*<<SYS>>.*?(?:<</SYS>>|$)"),
    # Internal tool prompt used by the local tool loop.
    re.compile(
        r"(?is)(?:You have access to tools\.\s*To call a tool,\s*emit:|"
        r"Tools are available when needed\.\s*To call one,\s*reply only with:).*?"
        r"(?=(?:\n\s*(?:assistant|answer|final)\s*:)|\Z)"
    ),
    re.compile(r"(?ims)^\s*(?:Available tools|Tools):\s*\n(?:\s*-\s+.*(?:\n|$)){1,50}"),
    # Role-labelled dumps, e.g. "System prompt: ...".
    re.compile(
        r"(?ims)^\s*(?:system|developer)\s*(?:prompt|message|instructions?)?\s*:\s*.*?"
        r"(?=^\s*(?:assistant|user|answer|final)\s*:|\Z)"
    ),
    re.compile(
        r"(?is)\b(?:system|developer)\s*(?:prompt|message|instructions?)?\s*:\s*.*?"
        r"(?=(?:\n\s*(?:assistant|user|answer|final)\s*:)|\Z)"
    ),
    re.compile(
        r"(?ims)^\s*(?:BEGIN|START)\s+(?:SYSTEM|DEVELOPER)\s+(?:PROMPT|MESSAGE|INSTRUCTIONS?).*?"
        r"(?:^\s*(?:END|STOP)\s+(?:SYSTEM|DEVELOPER)\s+(?:PROMPT|MESSAGE|INSTRUCTIONS?).*?$|\Z)"
    ),
    # Tool-call markup should never be user-visible.
    re.compile(r"(?is)<tool_call>\s*\{.*?\}\s*</tool_call>"),
    re.compile(r"(?is)<tool_call>.*"),
]
_TEMPLATE_NOISE = [
    re.compile(r"(?is)<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>"),
    re.compile(r"(?is)<\|im_start\|>\s*assistant\b"),
    re.compile(r"(?is)<\|im_end\|>|<\|eot_id\|>"),
]
_LEAK_START_RE = re.compile(
    r"(?ims)"
    r"(<\|im_start\|>\s*(?:system|developer|assistant)\b|"
    r"<\|start_header_id\|>\s*(?:system|developer|assistant)\s*<\|end_header_id\|>|"
    r"<<SYS>>|\[INST\]\s*<<SYS>>|"
    r"You have access to tools\.\s*To call a tool,\s*emit:|"
    r"Tools are available when needed\.\s*To call one,\s*reply only with:|"
    r"^\s*(?:Available tools|Tools):\s*$|"
    r"\b(?:system|developer)\s*(?:prompt|message|instructions?)?\s*:|"
    r"^\s*(?:BEGIN|START)\s+(?:SYSTEM|DEVELOPER)\s+(?:PROMPT|MESSAGE|INSTRUCTIONS?)|"
    r"<tool_call>)"
)


def sanitize_llm_output(content: str) -> str:
    """Remove obvious hidden-prompt/template leaks from assistant output."""
    if not content:
        return content

    cleaned = content
    for pattern in _LEAK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in _TEMPLATE_NOISE:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or _FALLBACK_RESPONSE


def chunk_sanitized_output(content: str, *, chunk_size: int = _CHUNK_SIZE) -> Iterator[str]:
    """Yield sanitized assistant text in bounded chunks for SSE clients."""
    safe = sanitize_llm_output(content)
    for start in range(0, len(safe), chunk_size):
        yield safe[start : start + chunk_size]


class StreamingOutputSanitizer:
    """Small rolling buffer that keeps obvious prompt leaks from live SSE output."""

    def __init__(self, *, hold_chars: int = _STREAM_HOLD_CHARS) -> None:
        self._hold_chars = hold_chars
        self._pending = ""

    def feed(self, text: str) -> list[str]:
        """Append generated text and return chunks safe to emit now."""
        if not text:
            return []
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[str]:
        """Return the sanitized tail once generation is complete."""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        if not self._pending:
            return []
        if final:
            safe = sanitize_llm_output(self._pending)
            self._pending = ""
            return [safe] if safe else []

        limit = len(self._pending) - self._hold_chars
        leak_start = _LEAK_START_RE.search(self._pending)
        if leak_start:
            limit = min(limit, leak_start.start())
        if limit <= 0:
            return []

        chunk = self._pending[:limit]
        self._pending = self._pending[limit:]
        return [chunk] if chunk else []
