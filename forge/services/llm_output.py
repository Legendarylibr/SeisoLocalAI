"""Helpers for streaming assistant text to clients."""

from __future__ import annotations

from collections.abc import Iterator

_CHUNK_SIZE = 1_200


def sanitize_llm_output(content: str) -> str:
    """Return assistant text unchanged (no server-side content filtering)."""
    return content


def chunk_sanitized_output(content: str, *, chunk_size: int = _CHUNK_SIZE) -> Iterator[str]:
    """Yield assistant text in bounded chunks for SSE clients."""
    for start in range(0, len(content), chunk_size):
        yield content[start : start + chunk_size]


class StreamingOutputSanitizer:
    """Passthrough stream helper — emits tokens as they arrive."""

    def feed(self, text: str) -> list[str]:
        return [text] if text else []

    def finish(self) -> list[str]:
        return []
