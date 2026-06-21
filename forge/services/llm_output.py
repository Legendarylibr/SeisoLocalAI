"""Helpers for streaming assistant text to clients."""

from __future__ import annotations

import re
from collections.abc import Iterator

from forge.tools.registry import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, TOOL_CALL_PATTERN

_CHUNK_SIZE = 1_200
_TOOL_OPEN = TOOL_CALL_OPEN
_TOOL_CLOSE = TOOL_CALL_CLOSE
_PARTIAL_TOOL_PREFIXES = tuple(_TOOL_OPEN[:i] for i in range(1, len(_TOOL_OPEN) + 1))

# Tags used across Qwen, DeepSeek-R1, and other reasoning-tuned models.
_THINK_TAG_NAMES = (
    "think",
    "redacted_thinking",
    "reasoning",
    "thought",
    "analysis",
    "scratchpad",
    "channel",
)


def _paired_tag(name: str) -> tuple[str, str]:
    return f"<{name}>", f"</{name}>"


_THINK_TAG_PAIRS = tuple(_paired_tag(name) for name in _THINK_TAG_NAMES)
_PARTIAL_THINK_OPEN_PREFIXES = tuple(
    prefix
    for open_tag, _ in _THINK_TAG_PAIRS
    for prefix in (open_tag[:i] for i in range(1, len(open_tag) + 1))
)

_REASONING_HEADERS = (
    "thinking process:",
    "reasoning:",
    "analysis:",
    "thought process:",
    "chain of thought:",
    "let me think",
    "**thought:**",
    "**reasoning:**",
    "**analysis:**",
)
_PARTIAL_REASONING_PREFIXES = tuple(
    header[:i] for header in _REASONING_HEADERS for i in range(1, len(header) + 1)
)

_FUNCTION_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)
_THINK_TAG_PATTERN = re.compile(
    "|".join(
        re.escape(open_tag) + r".*?" + re.escape(close_tag)
        for open_tag, close_tag in _THINK_TAG_PAIRS
    ),
    re.DOTALL | re.IGNORECASE,
)
_PIPE_TAG_PATTERN = re.compile(
    r"<\|(think|reasoning|analysis|thought)\|>.*?<\|/\1\|>",
    re.DOTALL | re.IGNORECASE,
)
_REASONING_HEADER_PATTERN = re.compile(
    r"(?is)^\s*(?:\*\*)?(?:"
    r"Thinking Process|Reasoning|Analysis|Thought process|Chain of thought|Let me think"
    r")(?:\*\*)?\s*:"
)
_ANALYSIS_STEP_PATTERN = re.compile(
    r"(?i)\b\d+\.\s*\*\*(Analyze the Input|Determine the Appropriate Response|Drafting Options|Thought)\*\*"
)
_FINAL_ANSWER_PATTERNS = (
    re.compile(
        r"(?is)(?:Final Decision|Final Polish|Refining the Output|Let's go with|Final Answer|Answer|Response|Reply|Output)"
        r"\s*:+\s*(?:\*\*)?\s*\"([^\"]+)\""
    ),
    re.compile(
        r"(?is)(?:Final Decision|Final Polish|Refining the Output|Let's go with|Final Answer|Answer|Response|Reply|Output)"
        r"\s*:+\s*\*\*([^*]+)\*\*"
    ),
    re.compile(r"(?is)\*\*(?:Final Answer|Answer|Response|Reply|Output):\*\*\s*(.+)\Z"),
    re.compile(r"(?is)\*\*(?:Final Answer|Answer|Response|Reply|Output)\*\*\s*:+\s*(.+)\Z"),
    re.compile(
        r"(?is)(?:Final Decision|Final Polish|Final Answer|Answer|Response|Reply|Output)"
        r"\s*:+\s*([^\n\"]+?)\s*(?:Wait,|\Z)"
    ),
)
_ORPHAN_CLOSE_TAG_PATTERN = re.compile(
    r"</(?:think|redacted_thinking|reasoning|thought|analysis|scratchpad)>",
    re.IGNORECASE,
)
_NUMBERED_ANALYSIS_START = re.compile(
    r"(?is)^\s*\d+\.\s*\*\*(?:Analyze|Determine|Draft|Select|Refine|Thought|Reasoning|Option)"
)


def _looks_like_reasoning_leak(content: str) -> bool:
    if _REASONING_HEADER_PATTERN.match(content):
        return True
    if re.search(r"(?i)\*\*(?:reasoning|thought|analysis|response|reply):\*\*", content):
        return True
    if _ANALYSIS_STEP_PATTERN.search(content):
        return True
    if _NUMBERED_ANALYSIS_START.match(content):
        return True
    if _THINK_TAG_PATTERN.search(content):
        return True
    return bool(_PIPE_TAG_PATTERN.search(content))


def _extract_final_answer_from_reasoning(content: str) -> str | None:
    for pattern in _FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(content)
        if not matches:
            continue
        candidate = str(matches[-1]).strip().strip('"').strip("'")
        if (
            candidate
            and len(candidate) < 500
            and not candidate.lower().startswith(("thinking process", "reasoning", "analysis"))
        ):
            return candidate
    quoted = re.findall(r'"([^"]{3,200})"', content)
    for candidate in reversed(quoted):
        text = candidate.strip()
        if text and not re.search(
            r"(?i)(analyze|drafting|option|refining|determine|reasoning)", text
        ):
            return text
    return None


def _starts_reasoning_leak(text: str) -> bool:
    lower = text.lstrip().lower()
    return any(lower.startswith(header) for header in _REASONING_HEADERS)


def strip_reasoning_leakage(content: str) -> str:
    """Remove visible chain-of-thought / thinking-process output."""
    if not content:
        return content

    cleaned = _THINK_TAG_PATTERN.sub("", content)
    cleaned = _PIPE_TAG_PATTERN.sub("", cleaned)
    cleaned = _ORPHAN_CLOSE_TAG_PATTERN.sub("", cleaned)

    if _looks_like_reasoning_leak(cleaned):
        extracted = _extract_final_answer_from_reasoning(cleaned)
        if extracted:
            return extracted.strip()
        cleaned = _REASONING_HEADER_PATTERN.sub("", cleaned)

    cleaned = re.sub(
        r"(?im)^\s*\d+\.\s+\*\*(Analyze|Determine|Drafting|Selecting|Refining|Final|Thought|Reasoning|Option)[^*]*\*\*.*?"
        r"(?=^\s*\d+\.\s+\*\*|\Z)",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?im)^\s*\*\*(?:Reasoning|Thought|Analysis):\*\*.*?(?=^\s*\*\*|\Z)", "", cleaned
    )
    return cleaned.strip()


def strip_spurious_tool_syntax(content: str) -> str:
    """Remove accidental tool-call markup from plain chat replies."""
    if not content:
        return content
    cleaned = TOOL_CALL_PATTERN.sub("", content)
    cleaned = _FUNCTION_JSON_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[TOOL_CALLS?\]", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|tool_call\|>.*?(?:<\|/tool_call\|>|$)", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def strip_spurious_chat_artifacts(content: str) -> str:
    """Strip tool-call markup and leaked reasoning from plain chat replies."""
    cleaned = strip_spurious_tool_syntax(content)
    cleaned = strip_reasoning_leakage(cleaned)
    return cleaned.strip()


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
        self._in_think = False
        self._reasoning_mode = False
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self._strip:
            return [text]

        if self._reasoning_mode:
            self._buffer += text
            return []

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

            if self._in_think:
                close_idx = -1
                close_len = 0
                for _open_tag, close_tag in _THINK_TAG_PAIRS:
                    idx = self._buffer.lower().find(close_tag.lower())
                    if idx != -1 and (close_idx == -1 or idx < close_idx):
                        close_idx = idx
                        close_len = len(close_tag)
                if close_idx == -1:
                    self._buffer = ""
                    break
                self._buffer = self._buffer[close_idx + close_len :]
                self._in_think = False
                continue

            if not self._emitted and _starts_reasoning_leak(self._buffer):
                self._reasoning_mode = True
                break

            think_idx = -1
            think_open = ""
            for open_tag, _close_tag in _THINK_TAG_PAIRS:
                idx = self._buffer.lower().find(open_tag.lower())
                if idx != -1 and (think_idx == -1 or idx < think_idx):
                    think_idx = idx
                    think_open = open_tag
            if think_idx != -1:
                if think_idx > 0:
                    emitted.append(self._buffer[:think_idx])
                    self._emitted = True
                self._buffer = self._buffer[think_idx + len(think_open) :]
                self._in_think = True
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
        if self._in_tool_call or self._in_think:
            self._buffer = ""
            self._in_tool_call = False
            self._in_think = False
        if self._reasoning_mode:
            buf = self._buffer
            self._buffer = ""
            self._reasoning_mode = False
            answer = _extract_final_answer_from_reasoning(buf)
            if answer:
                self._emitted = True
                return [answer]
            if re.search(r"(?i)thinking process", buf) or re.search(r"\d+\.\s*\*\*[^*]+$", buf):
                return []
            cleaned = strip_reasoning_leakage(buf)
            if (
                cleaned
                and not _looks_like_reasoning_leak(cleaned)
                and not _starts_reasoning_leak(cleaned)
            ):
                self._emitted = True
                return [cleaned]
            return []
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
        for prefixes in (
            _PARTIAL_TOOL_PREFIXES,
            _PARTIAL_REASONING_PREFIXES,
            _PARTIAL_THINK_OPEN_PREFIXES,
        ):
            for prefix in reversed(prefixes):
                if text.lower().endswith(prefix.lower()):
                    return text[: -len(prefix)], prefix
        return text, ""
