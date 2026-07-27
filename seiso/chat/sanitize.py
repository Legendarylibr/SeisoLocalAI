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
_PARTIAL_TOOL_CLOSE_PREFIXES = tuple(
    _TOOL_CLOSE[:i] for i in range(1, len(_TOOL_CLOSE) + 1)
)

_FUNCTION_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)
_THINKING_CLOSE_RE = re.compile(r"</(?:redacted_thinking|think)>", re.IGNORECASE)
# Match attributed opens (e.g. <think channel="analysis">) like thinking.py.
_THINKING_OPEN_RE = re.compile(
    r"<(?:redacted_thinking|think)\b[^>]*>", re.IGNORECASE
)
_THINK_BLOCK_RE = re.compile(
    r"<(?:redacted_thinking|think)\b[^>]*>.*?</(?:redacted_thinking|think)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_leaked_reasoning(content: str, *, preserve_trailing: bool = False) -> str:
    """Remove model reasoning blocks from plain chat replies.

    When ``preserve_trailing`` is set (streaming path), only leading whitespace
    after a reasoning close-tag is trimmed so mid-stream chunk boundaries keep
    their trailing spaces.

    Only strips balanced ``<think>…</think>`` (or redacted_thinking) blocks, or
    an unclosed open tag. A lone ``</think>`` in prose/docs is left intact so
    real answers are not destroyed.
    """
    if not content:
        return content
    edge = str.lstrip if preserve_trailing else str.strip
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
    # Bare {"name","arguments"} JSON appears in docs/code; only strip when
    # accompanied by explicit tool-call markers.
    if re.search(r"\[TOOL_CALLS?\]|<\|tool_call\|>", cleaned, flags=re.I):
        cleaned = _FUNCTION_JSON_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[TOOL_CALLS?\]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|tool_call\|>.*?(?:<\|/tool_call\|>|$)", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def strip_spurious_chat_artifacts(content: str) -> str:
    """Strip tool-call markup and leaked reasoning from plain chat replies."""
    cleaned = strip_spurious_tool_syntax(content)
    return strip_leaked_reasoning(cleaned)


def sanitize_llm_output(content: str, *, strip_tool_calls: bool = False) -> str:
    """Return assistant text with reasoning stripped; optionally strip tool markup.

    Reasoning leaks (``<think>`` / ``redacted_thinking``) are always removed so
    tools-enabled chat cannot bypass the strip (CHAT-01). ``strip_tool_calls``
    only controls tool-call syntax removal.
    """
    cleaned = strip_leaked_reasoning(content)
    if strip_tool_calls:
        cleaned = strip_spurious_tool_syntax(cleaned)
    return cleaned


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
        self._pending += text
        if self._strip:
            self._drain_pending()
        else:
            # Tools allowed: keep tool markup, still accumulate for reasoning strip.
            self._visible += self._pending
            self._pending = ""
        if not self._thinking_resolved:
            if not self._should_release_holdback(self._visible):
                return []
            self._thinking_resolved = True
        return self._emit_visible_delta()

    def finish(self) -> list[str]:
        self._thinking_resolved = True
        if self._strip and self._in_tool_call:
            self._pending = ""
            self._in_tool_call = False
        else:
            self._visible += self._pending
            self._pending = ""
        return self._emit_visible_delta()

    def finalize(self, *, full_text: str) -> str:
        """Sanitize the complete assistant reply after streaming finishes."""
        return sanitize_llm_output(full_text, strip_tool_calls=self._strip)

    def _drain_pending(self) -> None:
        while self._pending:
            if self._in_tool_call:
                close_idx = self._pending.find(_TOOL_CLOSE)
                if close_idx == -1:
                    # Keep a trailing partial close tag across chunk boundaries.
                    _, leftover = self._split_pending_close_prefixes(self._pending)
                    self._pending = leftover
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

    @staticmethod
    def _split_pending_close_prefixes(text: str) -> tuple[str, str]:
        for prefix in reversed(_PARTIAL_TOOL_CLOSE_PREFIXES):
            if text.lower().endswith(prefix.lower()):
                return text[: -len(prefix)], prefix
        return text, ""
