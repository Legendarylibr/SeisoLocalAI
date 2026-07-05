"""Streaming inference helpers — token-accurate throughput metering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamToken:
    """One decoded chunk from the model and how many tokens it represents."""

    text: str
    new_tokens: int = 1

    def __post_init__(self) -> None:
        if self.new_tokens < 1:
            object.__setattr__(self, "new_tokens", 1)


@dataclass(frozen=True)
class StreamUpdate:
    """Batched text chunk with cumulative output token count."""

    text: str
    output_tokens: int


def merge_stream_updates(updates: list[StreamUpdate]) -> StreamUpdate:
    """Coalesce consecutive updates into one — concatenated text, latest token count.

    ``output_tokens`` is cumulative, so the last update already carries the running
    total; joining the text avoids emitting many tiny SSE frames at high token rates.
    """
    if not updates:
        raise ValueError("merge_stream_updates requires at least one update")
    if len(updates) == 1:
        return updates[0]
    return StreamUpdate(
        text="".join(u.text for u in updates),
        output_tokens=updates[-1].output_tokens,
    )
