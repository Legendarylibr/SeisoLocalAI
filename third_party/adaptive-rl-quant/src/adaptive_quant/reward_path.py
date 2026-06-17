"""Per-prompt reward tracking for curriculum and reward-path-based task selection."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.types import PromptSample


@dataclass
class _PromptStats:
    count: int = 0
    total_reward: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=16))

    @property
    def mean_reward(self) -> float:
        if self.count <= 0:
            return float("-inf")
        return self.total_reward / self.count

    @property
    def recent_mean(self) -> float:
        if not self.recent:
            return self.mean_reward
        return mean(list(self.recent))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward if self.count > 0 else None,
            "recent_mean": self.recent_mean if self.recent else None,
            "recent": list(self.recent),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _PromptStats:
        recent_raw = payload.get("recent") or []
        recent: deque[float] = deque(maxlen=16)
        for value in recent_raw:
            recent.append(float(value))
        return cls(
            count=int(payload.get("count") or 0),
            total_reward=float(payload.get("total_reward") or 0.0),
            recent=recent,
        )


class RewardPathTracker:
    """Track reward history per prompt and select tasks from prior reward paths."""

    def __init__(self, *, recent_window: int = 16) -> None:
        self.recent_window = max(4, recent_window)
        self._by_prompt: dict[str, _PromptStats] = {}
        self._reward_path: deque[tuple[str, float]] = deque(maxlen=256)

    def record(self, prompt_id: str, reward: float, *, domain: str = "") -> None:
        stats = self._by_prompt.setdefault(prompt_id, _PromptStats())
        stats.count += 1
        stats.total_reward += float(reward)
        stats.recent.append(float(reward))
        self._reward_path.append((prompt_id, float(reward)))
        if domain:
            stats_domain = self._by_prompt.setdefault(f"domain:{domain}", _PromptStats())
            stats_domain.count += 1
            stats_domain.total_reward += float(reward)
            stats_domain.recent.append(float(reward))

    def select_prompt(
        self,
        library: PromptLibrary,
        rng: random.Random,
        *,
        explore_fraction: float = 0.15,
    ) -> PromptSample:
        prompts = library.prompts
        if not prompts:
            raise ValueError("prompt library is empty")
        if not self._by_prompt:
            return prompts[rng.randrange(len(prompts))]

        explore_fraction = max(0.0, min(1.0, float(explore_fraction)))
        if rng.random() < explore_fraction:
            counts = [self._by_prompt.get(p.prompt_id, _PromptStats()).count for p in prompts]
            min_count = min(counts)
            candidates = [p for p, c in zip(prompts, counts, strict=True) if c == min_count]
            return rng.choice(candidates)

        def score(prompt: PromptSample) -> float:
            stats = self._by_prompt.get(prompt.prompt_id)
            if stats is None or stats.count <= 0:
                return float("-inf")
            return stats.recent_mean

        best_score = max(score(p) for p in prompts)
        tied = [p for p in prompts if score(p) == best_score]
        return rng.choice(tied)

    def prompt_summary(self) -> dict[str, dict[str, Any]]:
        return {
            prompt_id: stats.to_dict()
            for prompt_id, stats in sorted(self._by_prompt.items())
            if not prompt_id.startswith("domain:")
        }

    def reward_path_tail(self, limit: int = 32) -> list[dict[str, float | str]]:
        tail = list(self._reward_path)[-max(1, limit) :]
        return [{"prompt_id": pid, "reward": reward} for pid, reward in tail]

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_stats": {
                prompt_id: stats.to_dict() for prompt_id, stats in self._by_prompt.items()
            },
            "reward_path": self.reward_path_tail(limit=256),
        }
        write_json(str(target), payload)
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> RewardPathTracker:
        payload = read_json(str(path))
        tracker = cls()
        stats_raw = payload.get("prompt_stats") or {}
        if isinstance(stats_raw, dict):
            for prompt_id, row in stats_raw.items():
                if isinstance(row, dict):
                    tracker._by_prompt[str(prompt_id)] = _PromptStats.from_dict(row)
        path_raw = payload.get("reward_path") or []
        if isinstance(path_raw, list):
            for row in path_raw:
                if isinstance(row, dict) and "prompt_id" in row:
                    tracker._reward_path.append(
                        (str(row["prompt_id"]), float(row.get("reward", 0.0)))
                    )
        return tracker


__all__ = ["RewardPathTracker"]
