"""Explicit reinforcement-learning episode loop: reset → act → measure → reward → update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from adaptive_quant.trainer_utils import training_row
from adaptive_quant.types import (
    EpisodeResult,
    EpisodeState,
    HardwareType,
    PromptSample,
    QuantizationDecision,
)


@dataclass(frozen=True)
class RLEpisodeOutcome:
    """One completed RL step with policy trace and environment result."""

    episode_index: int
    reward: float
    result: EpisodeResult
    trace: Any
    state: EpisodeState
    decision: QuantizationDecision


class _RLTrainer(Protocol):
    config: Any
    env: Any
    policy: Any
    previous_action: list[float]
    offline_router: Any

    def _feedback_vector(self, decision: QuantizationDecision) -> list[float]: ...


def run_rl_episode(
    trainer: _RLTrainer,
    *,
    episode_index: int,
    phase: str = "train",
    forced_prompt: PromptSample | None = None,
    forced_hardware: HardwareType | None = None,
    deterministic: bool | None = None,
    apply_update: bool = True,
    log_episode: bool = True,
) -> RLEpisodeOutcome:
    """Run one RL episode: reset env, sample action, evaluate, optionally update policy."""
    state = trainer.env.reset(
        previous_action=trainer.previous_action,
        forced_hardware=forced_hardware,
        forced_prompt=forced_prompt,
        phase=phase,
        episode_index=episode_index,
    )
    if deterministic is None:
        deterministic = bool(trainer.config.rl_train_deterministic())

    decision, trace = _policy_act(trainer, state, deterministic=deterministic)
    routed_decision = decision
    if trainer.offline_router is not None:
        routed_decision = trainer.offline_router.prepare_decision(decision, state)

    result = trainer.env.evaluate_current(
        routed_decision,
        episode_index=episode_index,
        log_episode=log_episode,
    )
    if trainer.offline_router is not None:
        trainer.offline_router.complete_episode(
            state=state,
            policy_decision=decision,
            routed_result=result,
            env=trainer.env,
            episode_index=episode_index,
        )

    reward = float(result.metrics.reward)
    executed = result.decision
    if apply_update and not executed.fallback_applied:
        _apply_policy_update(trainer, trace, reward)

    trainer.previous_action = trainer._feedback_vector(executed)
    return RLEpisodeOutcome(
        episode_index=episode_index,
        reward=reward,
        result=result,
        trace=trace,
        state=state,
        decision=decision,
    )


def record_training_row(outcome: RLEpisodeOutcome) -> dict[str, float]:
    return training_row(float(outcome.episode_index), outcome.result)


def _policy_act(
    trainer: _RLTrainer, state: EpisodeState, *, deterministic: bool
) -> tuple[QuantizationDecision, Any]:
    backend = str(trainer.config.training_backend).strip().lower()
    if backend == "pytorch":
        state_vector = state.to_vector(trainer.config.ordered_hardware())
        decision, record = trainer.policy.act(
            state_vector,
            deterministic=deterministic,
            moe_context=state.moe_context,
        )
        return decision, record
    return trainer.policy.act(state, deterministic=deterministic)


def _apply_policy_update(trainer: _RLTrainer, trace: Any, reward: float) -> None:
    backend = str(trainer.config.training_backend).strip().lower()
    if backend == "pytorch":
        record = dict(trace)
        record["reward"] = float(reward)
        trainer._update_policy([record])  # type: ignore[attr-defined]
        return
    trainer.policy.update(trace, reward)


__all__ = [
    "RLEpisodeOutcome",
    "record_training_row",
    "run_rl_episode",
]
