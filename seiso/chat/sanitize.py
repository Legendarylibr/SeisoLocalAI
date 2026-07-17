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
_THINKING_CLOSE_RE = re.compile(r"</(?:redacted_thinking|think)>", re.IGNORECASE)
_THINKING_OPEN_RE = re.compile(r"<(?:redacted_thinking|think)>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<(?:redacted_thinking|think)>.*?</(?:redacted_thinking|think)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_leaked_reasoning(content: str, *, preserve_trailing: bool = False) -> str:
    """Remove model reasoning blocks from plain chat replies.

    When ``preserve_trailing`` is set (streaming path), only leading whitespace
    after a reasoning close-tag is trimmed so mid-stream chunk boundaries keep
    their trailing spaces.
    """
    if not content:
        return content
    edge = str.lstrip if preserve_trailing else str.strip
    match = re.search(
        r"</(?:redacted_thinking|think)>(?P<final>.*)$",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return edge(match.group("final"))
    cleaned = _THINK_BLOCK_RE.sub("", content)
    open_match = _THINKING_OPEN_RE.search(cleaned)
    if open_match is not None:
        cleaned = cleaned[: open_match.start()]
    return edge(cleaned)


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
    """Strip tool-call markup and leaked reasoning from plain chat replies."""
    cleaned = strip_spurious_tool_syntax(content)
    return strip_leaked_reasoning(cleaned)


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
        self._pending = ""
        self._visible = ""
        self._in_tool_call = False
        self._emitted = False
        self._thinking_resolved = False
        self._sanitized_emitted_len = 0

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self._strip:
            return [text]

        self._pending += text
        self._drain_pending()
        if not self._thinking_resolved:
            if not self._should_release_holdback(self._visible):
                return []
            self._thinking_resolved = True
        return self._emit_visible_delta()

    def finish(self) -> list[str]:
        if not self._strip:
            return []
        self._thinking_resolved = True
        if self._in_tool_call:
            self._pending = ""
            self._in_tool_call = False
        else:
            self._visible += self._pending
            self._pending = ""
        return self._emit_visible_delta()

    def finalize(self, *, full_text: str) -> str:
        """Sanitize the complete assistant reply after streaming finishes."""
        if not self._strip:
            return full_text
        return sanitize_llm_output(full_text, strip_tool_calls=True)

    def _drain_pending(self) -> None:
        while self._pending:
            if self._in_tool_call:
                close_idx = self._pending.find(_TOOL_CLOSE)
                if close_idx == -1:
                    self._pending = ""
                    break
                self._pending = self._pending[close_idx + len(_TOOL_CLOSE) :]
                self._in_tool_call = False
                continue

            open_idx = self._pending.find(_TOOL_OPEN)
            if open_idx == -1:
                safe, leftover = self._split_pending_prefixes(self._pending)
                if safe:
                    self._visible += safe
                self._pending = leftover
                break

            if open_idx > 0:
                self._visible += self._pending[:open_idx]
            self._pending = self._pending[open_idx + len(_TOOL_OPEN) :]
            self._in_tool_call = True

    def _emit_visible_delta(self) -> list[str]:
        sanitized = strip_leaked_reasoning(self._visible, preserve_trailing=True)
        if len(sanitized) <= self._sanitized_emitted_len:
            return []
        delta = sanitized[self._sanitized_emitted_len :]
        self._sanitized_emitted_len = len(sanitized)
        if delta:
            self._emitted = True
            return [delta]
        return []

    @staticmethod
    def _should_release_holdback(buffer: str) -> bool:
        if _THINKING_CLOSE_RE.search(buffer) or _THINKING_OPEN_RE.search(buffer):
            return True
        if "<" not in buffer:
            return True
        return len(buffer) >= 128

    @staticmethod
    def _split_pending_prefixes(text: str) -> tuple[str, str]:
        for prefix in reversed(_PARTIAL_TOOL_PREFIXES):
            if text.lower().endswith(prefix.lower()):
                return text[: -len(prefix)], prefix
        return text, ""
