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
