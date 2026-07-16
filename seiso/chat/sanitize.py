"""Sanitize assistant text for plain chat (CLI + Forge).

Shared core used by ``forge.services.llm_output`` (re-export) and CLI chat.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_CALL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN)}\s*(\{{.*?\}})\s*{re.escape(TOOL_CALL_CLOSE)}",
    re.DOTALL,
)
_XML_FUNCTION_TOOL_PATTERN = re.compile(
    rf"{re.escape(TOOL_CALL_OPEN)}\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*{re.escape(TOOL_CALL_CLOSE)}",
    re.DOTALL,
)

_CHUNK_SIZE = 1_200
_TOOL_OPEN = TOOL_CALL_OPEN
_TOOL_CLOSE = TOOL_CALL_CLOSE
_PARTIAL_TOOL_PREFIXES = tuple(_TOOL_OPEN[:i] for i in range(1, len(_TOOL_OPEN) + 1))

_FUNCTION_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)


def strip_spurious_tool_syntax(content: str) -> str:
    """Remove accidental tool-call markup from plain chat replies."""
    if not content:
        return content
    cleaned = TOOL_CALL_PATTERN.sub("", content)
    cleaned = _XML_FUNCTION_TOOL_PATTERN.sub("", cleaned)
    cleaned = _FUNCTION_JSON_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[TOOL_CALLS?\]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|tool_call\|>.*?(?:<\|/tool_call\|>|$)", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def strip_spurious_chat_artifacts(content: str) -> str:
    """Strip only tool-call markup from plain chat replies."""
    return strip_spurious_tool_syntax(content).strip()


def sanitize_llm_output(content: str, *, strip_tool_calls: bool = False) -> str:
    """Return assistant text, optionally stripping spurious chat artifacts."""
    if strip_tool_calls:
        return strip_spurious_chat_artifacts(content)
    return content


def chunk_sanitized_output(content: str, *, chunk_size: int = _CHUNK_SIZE) -> Iterator[str]:
    """Yield assistant text in bounded chunks for SSE clients."""
    for start in range(0, len(content), chunk_size):
        yield content[start : start + chunk_size]


class StreamingOutputSanitizer:
    """Stream helper that can suppress spurious tool-call and reasoning markup."""

    def __init__(self, *, strip_tool_calls: bool = False) -> None:
        self._strip = strip_tool_calls
        self._buffer = ""
        self._in_tool_call = False
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self._strip:
            return [text]

        emitted: list[str] = []
        self._buffer += text

        while self._buffer:
            if self._in_tool_call:
                close_idx = self._buffer.find(_TOOL_CLOSE)
                if close_idx == -1:
                    self._buffer = ""
                    break
                self._buffer = self._buffer[close_idx + len(_TOOL_CLOSE) :]
                self._in_tool_call = False
                continue

            open_idx = self._buffer.find(_TOOL_OPEN)
            if open_idx == -1:
                safe, pending = self._split_pending_prefixes(self._buffer)
                if safe:
                    emitted.append(safe)
                    self._emitted = True
                self._buffer = pending
                break

            if open_idx > 0:
                emitted.append(self._buffer[:open_idx])
                self._emitted = True
            self._buffer = self._buffer[open_idx + len(_TOOL_OPEN) :]
            self._in_tool_call = True

        return emitted

    def finish(self) -> list[str]:
        if not self._strip:
            return []
        if self._in_tool_call:
            self._buffer = ""
            self._in_tool_call = False
        if not self._buffer:
            return []
        out = [self._buffer]
        self._buffer = ""
        self._emitted = True
        return out

    def finalize(self, *, full_text: str) -> str:
        """Sanitize the complete assistant reply after streaming finishes."""
        if not self._strip:
            return full_text
        return sanitize_llm_output(full_text, strip_tool_calls=True)

    @staticmethod
    def _split_pending_prefixes(text: str) -> tuple[str, str]:
        for prefix in reversed(_PARTIAL_TOOL_PREFIXES):
            if text.lower().endswith(prefix.lower()):
                return text[: -len(prefix)], prefix
        return text, ""
