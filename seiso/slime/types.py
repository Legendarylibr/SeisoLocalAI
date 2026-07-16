"""Shared slime trainer types and auto-stop helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict, TypeVar

from seiso.slime.config import SingleGpuSlimeConfig

T = TypeVar("T")


def _metric_is_minimized(metric: str) -> bool:
    return metric == "kl" or metric.endswith("loss") or metric.endswith("_loss")


class _CompletionScore(TypedDict):
    """Per-rollout verifier breakdown (key types for mypy, not a free-form dict)."""

    reward: float
    outcome_reward: float
    format_reward: float
    process_reward: float
    thinking_penalty: float
    thinking_trace: str
    final_answer: str
    extracted_answer: str
    outcome_passed: bool
    format_ok: bool
    checker: str
    detail: str | None
    proof_passed: bool | None
    proof_score: float | None
    proof_detail: str | None


@dataclass
class Rollout:
    input_ids: Any
    attention_mask: Any
    response_mask: Any
    old_logprobs: Any
    ref_logprobs: Any | None
    reward: float
    old_token_logprobs: Any | None = None
    ref_token_logprobs: Any | None = None
    outcome_reward: float = 0.0
    format_reward: float = 0.0
    process_reward: float = 0.0
    thinking_penalty: float = 0.0
    final_answer: str = ""
    thinking_trace: str = ""
    status: str = "unknown"
    advantage: float = 0.0
    outcome_passed: bool = False
    format_ok: bool = True
    checker: str = ""
    proof_passed: bool | None = None
    proof_score: float | None = None
    proof_detail: str | None = None


@dataclass(frozen=True)
class _RolloutBatch:
    rollouts: list[Rollout]
    stats: dict[str, float]


@dataclass
class _DistributedSlimeContext:
    enabled: bool
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    device: str = "cuda"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class _AutoStopDecision:
    improved: bool = False
    should_stop: bool = False
    reason: str | None = None


class _PushbackIterator:
    """Iterator that can re-queue a value so DDP refill does not drop batches."""

    def __init__(self, source: Iterator[list[dict[str, Any]]]) -> None:
        self._source = source
        self._buf: list[list[dict[str, Any]]] = []

    def __iter__(self) -> _PushbackIterator:
        return self

    def __next__(self) -> list[dict[str, Any]]:
        if self._buf:
            return self._buf.pop()
        return next(self._source)

    def push(self, item: list[dict[str, Any]]) -> None:
        if item:
            self._buf.append(item)


@dataclass
class _AutoStopController:
    enabled: bool
    metric: str
    patience: int
    min_delta: float
    warmup_steps: int
    best_value: float | None = None
    best_step: int | None = None
    stale_steps: int = 0

    @classmethod
    def from_config(cls, config: SingleGpuSlimeConfig) -> _AutoStopController:
        return cls(
            enabled=config.auto_stop,
            metric=config.auto_stop_metric,
            patience=config.auto_stop_patience,
            min_delta=config.auto_stop_min_delta,
            warmup_steps=config.auto_stop_warmup_steps,
        )

    def update(self, step: int, stats: dict[str, float]) -> _AutoStopDecision:
        value = stats.get(self.metric)
        if value is None or not math.isfinite(value):
            return _AutoStopDecision()

        improved = self._is_better(value)
        if improved:
            self.best_value = value
            self.best_step = step
            self.stale_steps = 0
            return _AutoStopDecision(improved=True)

        if not self.enabled or step < self.warmup_steps:
            return _AutoStopDecision()

        self.stale_steps += 1
        if self.stale_steps >= self.patience:
            return _AutoStopDecision(
                should_stop=True,
                reason=f"auto_stop:{self.metric}_plateau",
            )
        return _AutoStopDecision()

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if _metric_is_minimized(self.metric):
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta
