"""RL routing policy — contextual UCB bandit over specialist routes."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_quant.route_policy import RouteContext  # type: ignore[attr-defined]

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute


@dataclass
class _ArmStats:
    pulls: int = 0
    mean_reward: float = 0.0
    m2: float = 0.0
    last_reward: float = 0.0

    def update(self, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward!r}")
        self.pulls += 1
        delta = reward - self.mean_reward
        self.mean_reward += delta / self.pulls
        self.m2 += delta * (reward - self.mean_reward)
        self.last_reward = reward

    @property
    def variance(self) -> float:
        if self.pulls < 2:
            return 0.0
        return self.m2 / float(self.pulls - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pulls": self.pulls,
            "mean_reward": self.mean_reward,
            "m2": self.m2,
            "last_reward": self.last_reward,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> _ArmStats:
        return cls(
            pulls=int(payload.get("pulls", 0)),
            mean_reward=float(payload.get("mean_reward", 0.0)),
            m2=float(payload.get("m2", 0.0)),
            last_reward=float(payload.get("last_reward", 0.0)),
        )


@dataclass
class SpecialistSelection:
    route: SpecialistRoute
    score: float
    explore: bool
    feasible: bool
    bucket_key: str
    reasoning: str


@dataclass
class SpecialistRouteBandit:
    """UCB bandit over :class:`SpecialistRoute` arms (mirrors adaptive_quant RouteBandit)."""

    catalog: SpecialistCatalog
    ucb_c: float = 1.5
    prior_weight: float = 4.0
    warmup_pulls: int = 3
    seed: int = 13
    _global: dict[str, _ArmStats] = field(default_factory=dict, init=False, repr=False)
    _buckets: dict[str, dict[str, _ArmStats]] = field(default_factory=dict, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)
    _total_pulls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        for route in self.catalog:
            self._global.setdefault(route.route_id, _ArmStats())

    def select(self, context: RouteContext, *, deterministic: bool = False) -> SpecialistSelection:
        feasible = [r for r in self.catalog if r.matches_hardware(context.hardware)]
        if not feasible:
            feasible = list(self.catalog)
            is_feasible = False
        else:
            is_feasible = True

        bucket_key = context.key()
        bucket = self._buckets.get(bucket_key, {})
        bucket_pulls = sum(stats.pulls for stats in bucket.values())

        scored: list[tuple[float, SpecialistRoute, _ArmStats, _ArmStats]] = []
        for route in feasible:
            arm = bucket.get(route.route_id, _ArmStats())
            global_arm = self._global.get(route.route_id, _ArmStats())
            score = self._ucb_score(arm, global_arm, bucket_pulls, deterministic)
            scored.append((score, route, arm, global_arm))

        best = max(s[0] for s in scored)
        ties = [i for i, s in enumerate(scored) if math.isclose(s[0], best)]
        winner = ties[0] if deterministic else ties[self._rng.randrange(len(ties))]
        score, route, arm, global_arm = scored[winner]

        explore = (not deterministic) and (
            arm.pulls < self.warmup_pulls or global_arm.pulls < self.warmup_pulls
        )
        reasoning = (
            f"bucket={bucket_key} score={score:.4f} arm_pulls={arm.pulls} "
            f"global_pulls={global_arm.pulls} route={route.route_id}"
        )

        return SpecialistSelection(
            route=route,
            score=score,
            explore=explore,
            feasible=is_feasible,
            bucket_key=bucket_key,
            reasoning=reasoning,
        )

    def recommend(self, context: RouteContext) -> SpecialistSelection:
        return self.select(context, deterministic=True)

    def update(self, route_id: str, context: RouteContext, reward: float) -> None:
        if route_id not in self._global:
            self._global[route_id] = _ArmStats()
        self._global[route_id].update(float(reward))
        bucket = self._buckets.setdefault(context.key(), {})
        arm = bucket.setdefault(route_id, _ArmStats())
        arm.update(float(reward))
        self._total_pulls += 1

    def _ucb_score(
        self,
        arm: _ArmStats,
        global_arm: _ArmStats,
        bucket_pulls: int,
        deterministic: bool,
    ) -> float:
        if arm.pulls == 0 and global_arm.pulls == 0:
            mean = 0.0
        elif arm.pulls == 0:
            mean = global_arm.mean_reward
        elif bucket_pulls < self.warmup_pulls:
            weight = self.prior_weight / max(1.0, bucket_pulls + self.prior_weight)
            mean = (1.0 - weight) * arm.mean_reward + weight * global_arm.mean_reward
        else:
            mean = arm.mean_reward

        if deterministic or bucket_pulls == 0:
            return mean

        total = max(1, bucket_pulls)
        bonus = self.ucb_c * math.sqrt(math.log(total + 1) / max(1, arm.pulls))
        return mean + bonus

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "ucb_c": self.ucb_c,
            "prior_weight": self.prior_weight,
            "warmup_pulls": self.warmup_pulls,
            "seed": self.seed,
            "total_pulls": self._total_pulls,
            "global": {rid: s.to_dict() for rid, s in self._global.items()},
            "buckets": {
                bk: {rid: s.to_dict() for rid, s in arms.items()}
                for bk, arms in self._buckets.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("Unsupported policy state version")
        self.ucb_c = float(state.get("ucb_c", self.ucb_c))
        self.prior_weight = float(state.get("prior_weight", self.prior_weight))
        self.warmup_pulls = int(state.get("warmup_pulls", self.warmup_pulls))
        self.seed = int(state.get("seed", self.seed))
        self._rng = random.Random(self.seed)
        self._total_pulls = int(state.get("total_pulls", 0))
        self._global = {
            rid: _ArmStats.from_dict(stats) for rid, stats in (state.get("global") or {}).items()
        }
        self._buckets = {
            bk: {rid: _ArmStats.from_dict(stats) for rid, stats in arms.items()}
            for bk, arms in (state.get("buckets") or {}).items()
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")

    def load(self, path: Path) -> None:
        if path.is_file():
            self.load_state_dict(json.loads(path.read_text(encoding="utf-8")))


def domain_hint_boost(route: SpecialistRoute, domain: str) -> float:
    """Small score boost when route domain_hints match classified domain."""
    if not route.domain_hints:
        return 0.0
    hints = {d.strip().lower() for d in route.domain_hints}
    if domain.strip().lower() in hints:
        return 0.15
    return 0.0


def pick_route_with_hints(
    bandit: SpecialistRouteBandit,
    context: RouteContext,
    domain: str,
    *,
    deterministic: bool = False,
) -> SpecialistSelection:
    """Select route; apply domain hint boost by re-scoring feasible arms."""
    selection = bandit.select(context, deterministic=deterministic)
    feasible = [r for r in bandit.catalog if r.matches_hardware(context.hardware)]
    if not feasible:
        feasible = list(bandit.catalog)

    bucket_key = context.key()
    bucket = bandit._buckets.get(bucket_key, {})
    bucket_pulls = sum(stats.pulls for stats in bucket.values())

    boosted: list[tuple[float, SpecialistRoute]] = []
    for route in feasible:
        arm = bucket.get(route.route_id, _ArmStats())
        global_arm = bandit._global.get(route.route_id, _ArmStats())
        score = bandit._ucb_score(arm, global_arm, bucket_pulls, deterministic=True)
        score += domain_hint_boost(route, domain)
        boosted.append((score, route))

    best_score = max(s for s, _ in boosted)
    winners = [r for s, r in boosted if math.isclose(s, best_score)]
    route = winners[0]
    if not deterministic:
        explore_sel = bandit.select(context, deterministic=False)
        explore = explore_sel.explore
    else:
        explore = False

    return SpecialistSelection(
        route=route,
        score=best_score,
        explore=explore,
        feasible=selection.feasible,
        bucket_key=bucket_key,
        reasoning=selection.reasoning + f"; domain_boost={domain}",
    )
