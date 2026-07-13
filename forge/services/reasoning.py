"""Separate model-provided reasoning from user-facing answer text."""

from __future__ import annotations

from dataclasses import dataclass

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _prefix_suffix_length(value: str, marker: str) -> int:
    """Length of the longest suffix that may be the start of ``marker``."""
    lowered = value.lower()
    limit = min(len(value), len(marker) - 1)
    for size in range(limit, 0, -1):
        if lowered.endswith(marker[:size]):
            return size
    return 0


@dataclass
class ReasoningStreamParser:
    """Incrementally split ``<think>`` blocks, including split tag chunks."""

    in_reasoning: bool = False
    _buffer: str = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buffer += text
        output: list[tuple[str, str]] = []

        while self._buffer:
            marker = _CLOSE_TAG if self.in_reasoning else _OPEN_TAG
            index = self._buffer.lower().find(marker)
            if index >= 0:
                if index:
                    output.append(
                        ("reasoning" if self.in_reasoning else "answer", self._buffer[:index])
                    )
                self._buffer = self._buffer[index + len(marker) :]
                self.in_reasoning = not self.in_reasoning
                continue

            held = _prefix_suffix_length(self._buffer, marker)
            ready = self._buffer[:-held] if held else self._buffer
            if ready:
                output.append(("reasoning" if self.in_reasoning else "answer", ready))
            self._buffer = self._buffer[-held:] if held else ""
            break

        return output

    def finish(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        output = [("reasoning" if self.in_reasoning else "answer", self._buffer)]
        self._buffer = ""
        return output


def split_reasoning_text(text: str) -> tuple[str, str]:
    """Return ``(answer, reasoning)`` while preserving non-reasoning text."""
    parser = ReasoningStreamParser()
    parts = [*parser.feed(text), *parser.finish()]
    answer = "".join(value for kind, value in parts if kind == "answer").strip()
    reasoning = "".join(value for kind, value in parts if kind == "reasoning").strip()
    return answer, reasoning
