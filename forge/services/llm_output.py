"""Helpers for streaming assistant text to clients."""

from __future__ import annotations

import re
from collections.abc import Iterator

from forge.tools.registry import (
    _XML_FUNCTION_TOOL_PATTERN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_PATTERN,
)

# Reasoning / thinking blocks — stripped for user-facing agent output only.
# Tool-call markup is handled separately so model interaction stays intact.
_REDACTED_THINKING_RE = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"\x3cthink\x3e.*?\x3c/think\x3e", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_AFTER_RE = re.compile(r"\x3c/think\x3e(?P<final>.*)$", re.IGNORECASE | re.DOTALL)
_QWEN_THINK_RE = re.compile(
    r"<\|im_start\|>think\s*[\s\S]*?<\|im_end\|>",
    re.IGNORECASE,
)
_GEMMA_THINK_RE = re.compile(
    r"(?:^|\n)\s*(?:thinking|reasoning)\s*:\s*[\s\S]*?(?=\n\s*(?:answer|response|final|```|\Z))",
    re.IGNORECASE,
)
_ARTIFACT_DUMP_RE = re.compile(
    r"(?:^|\n)\s*(?:artifact|scratchpad|working notes?)\s*:\s*[\s\S]*?(?=\n\s*(?:answer|response|final|```|\Z))",
    re.IGNORECASE,
)

_THINKING_OPEN_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("\x3cthink\x3e", "\x3c/think\x3e"),
)

_CHUNK_SIZE = 1_200
_TOOL_OPEN = TOOL_CALL_OPEN
_TOOL_CLOSE = TOOL_CALL_CLOSE
_PARTIAL_TOOL_PREFIXES = tuple(_TOOL_OPEN[:i] for i in range(1, len(_TOOL_OPEN) + 1))
_PARTIAL_THINK_PREFIXES = tuple(
    open_tag[:i]
    for open_tag, _close_tag in _THINKING_OPEN_TAGS
    for i in range(1, len(open_tag) + 1)
)

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
    cleaned = re.sub(
        r"<\|tool_call\|>.*?(?:<\|/tool_call\|>|$)", "", cleaned, flags=re.DOTALL
    )
    return cleaned.strip()


def strip_spurious_chat_artifacts(content: str) -> str:
    """Strip only tool-call markup from plain chat replies."""
    return strip_spurious_tool_syntax(content).strip()


def strip_reasoning_artifacts(content: str) -> str:
    """Remove model reasoning/thinking blocks while preserving final answers."""
    if not content:
        return content
    cleaned = content
    for pattern in (
        _REDACTED_THINKING_RE,
        _THINK_TAG_RE,
        _QWEN_THINK_RE,
        _GEMMA_THINK_RE,
        _ARTIFACT_DUMP_RE,
    ):
        cleaned = pattern.sub("", cleaned)
    match = _THINK_CLOSE_AFTER_RE.search(cleaned)
    if match is not None:
        cleaned = match.group("final")
    match = re.search(
        r"</think>(?P<final>.*)$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        cleaned = match.group("final")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_llm_output(
    content: str,
    *,
    strip_tool_calls: bool = False,
    strip_reasoning: bool = False,
) -> str:
    """Return assistant text, optionally stripping tool markup and reasoning."""
    cleaned = content
    if strip_tool_calls:
        cleaned = strip_spurious_chat_artifacts(cleaned)
    if strip_reasoning:
        cleaned = strip_reasoning_artifacts(cleaned)
    return cleaned


def sanitize_agent_output(content: str) -> str:
    """Final user-facing agent reply: hide tools, reasoning, and scratch artifacts."""
    return sanitize_llm_output(
        content,
        strip_tool_calls=True,
        strip_reasoning=True,
    )


def chunk_sanitized_output(
    content: str, *, chunk_size: int = _CHUNK_SIZE
) -> Iterator[str]:
    """Yield assistant text in bounded chunks for SSE clients."""
    for start in range(0, len(content), chunk_size):
        yield content[start : start + chunk_size]


class StreamingOutputSanitizer:
    """Stream helper that can suppress spurious tool-call and reasoning markup."""

    def __init__(
        self,
        *,
        strip_tool_calls: bool = False,
        strip_reasoning: bool = False,
    ) -> None:
        self._strip_tools = strip_tool_calls
        self._strip_reasoning = strip_reasoning
        self._buffer = ""
        self._in_tool_call = False
        self._in_thinking = False
        self._thinking_close = ""
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self._strip_tools and not self._strip_reasoning:
            return [text]

        emitted: list[str] = []
        self._buffer += text

        while self._buffer:
            if self._in_thinking:
                close_idx = self._buffer.lower().find(self._thinking_close.lower())
                if close_idx == -1:
                    self._buffer = ""
                    break
                self._buffer = self._buffer[close_idx + len(self._thinking_close) :]
                self._in_thinking = False
                self._thinking_close = ""
                continue

            if self._in_tool_call:
                close_idx = self._buffer.find(_TOOL_CLOSE)
                if close_idx == -1:
                    self._buffer = ""
                    break
                self._buffer = self._buffer[close_idx + len(_TOOL_CLOSE) :]
                self._in_tool_call = False
                continue

            next_block = self._next_block_start()
            if next_block is None:
                safe, pending = self._split_pending_prefixes(self._buffer)
                if safe:
                    cleaned = (
                        strip_reasoning_artifacts(safe)
                        if self._strip_reasoning
                        else safe
                    )
                    if cleaned:
                        emitted.append(cleaned)
                        self._emitted = True
                self._buffer = pending
                break

            idx, kind, open_tag, close_tag = next_block
            if idx > 0:
                prefix = self._buffer[:idx]
                if self._strip_reasoning:
                    prefix = strip_reasoning_artifacts(prefix)
                if prefix:
                    emitted.append(prefix)
                    self._emitted = True
            self._buffer = self._buffer[idx + len(open_tag) :]
            if kind == "tool":
                self._in_tool_call = True
            else:
                self._in_thinking = True
                self._thinking_close = close_tag

        return emitted

    def finish(self) -> list[str]:
        if not self._strip_tools and not self._strip_reasoning:
            return []
        if self._in_tool_call or self._in_thinking:
            self._buffer = ""
            self._in_tool_call = False
            self._in_thinking = False
            self._thinking_close = ""
        if not self._buffer:
            return []
        out_text = (
            strip_reasoning_artifacts(self._buffer)
            if self._strip_reasoning
            else self._buffer
        )
        self._buffer = ""
        if not out_text:
            return []
        self._emitted = True
        return [out_text]

    def finalize(self, *, full_text: str) -> str:
        """Sanitize the complete assistant reply after streaming finishes."""
        if not self._strip_tools and not self._strip_reasoning:
            return full_text
        return sanitize_llm_output(
            full_text,
            strip_tool_calls=self._strip_tools,
            strip_reasoning=self._strip_reasoning,
        )

    def _next_block_start(self) -> tuple[int, str, str, str] | None:
        candidates: list[tuple[int, str, str, str]] = []
        if self._strip_tools:
            tool_idx = self._buffer.find(_TOOL_OPEN)
            if tool_idx >= 0:
                candidates.append((tool_idx, "tool", _TOOL_OPEN, _TOOL_CLOSE))
        if self._strip_reasoning:
            lower = self._buffer.lower()
            for open_tag, close_tag in _THINKING_OPEN_TAGS:
                idx = lower.find(open_tag)
                if idx >= 0:
                    candidates.append((idx, "thinking", open_tag, close_tag))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])

    @staticmethod
    def _split_pending_prefixes(text: str) -> tuple[str, str]:
        for prefix in reversed(_PARTIAL_TOOL_PREFIXES + _PARTIAL_THINK_PREFIXES):
            if text.lower().endswith(prefix.lower()):
                return text[: -len(prefix)], prefix
        return text, ""
