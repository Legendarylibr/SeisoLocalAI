"""Continuous reinforcement learning over a streaming task sequence."""

from __future__ import annotations

import random
import sys
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.configuration.validation import validate_online_prompt_text
from adaptive_quant.logging_utils import (
    JsonlLogger,
    jsonl_integrity_chain_enabled,
    load_jsonl,
    write_json,
)
from adaptive_quant.math_utils import mean
from adaptive_quant.prompts import PromptLibrary
from adaptive_quant.reward_path import RewardPathTracker
from adaptive_quant.rl_loop import RLEpisodeOutcome, record_training_row, run_rl_episode
from adaptive_quant.router_training import resolve_prompt_library
from adaptive_quant.trainer_utils import reward_summary
from adaptive_quant.types import HardwareType, PromptSample


@dataclass(frozen=True)
class ContinuousTask:
    """One task in the continuous learning stream."""

    task_index: int
    prompt: PromptSample
    hardware: HardwareType


@dataclass
class _ExperienceEntry:
    payload: Any
    reward: float
    task_index: int
    prompt_id: str
    hardware_mode: str


class ContinuousExperienceBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = max(1, capacity)
        self.rng = rng
        self._entries: deque[_ExperienceEntry] = deque(maxlen=self.capacity)

    def add(self, entry: _ExperienceEntry) -> None:
        self._entries.append(entry)

    def sample(
        self,
        count: int,
        *,
        reward_tracker: RewardPathTracker | None = None,
    ) -> list[_ExperienceEntry]:
        if not self._entries:
            return []
        sample_size = min(max(1, count), len(self._entries))
        if reward_tracker is None or sample_size >= len(self._entries):
            return self.rng.sample(list(self._entries), sample_size)

        pool = list(self._entries)
        prioritized: list[_ExperienceEntry] = []
        if reward_tracker.prompt_summary():
            means = {
                pid: float(row.get("recent_mean") or row.get("mean_reward") or 0.0)
                for pid, row in reward_tracker.prompt_summary().items()
            }
            global_mean = mean([entry.reward for entry in pool]) if pool else 0.0
            underperformers = [
                entry for entry in pool if means.get(entry.prompt_id, global_mean) <= global_mean
            ]
            if underperformers:
                take = min(sample_size // 2 or 1, len(underperformers))
                prioritized = self.rng.sample(underperformers, take)

        remaining = sample_size - len(prioritized)
        if remaining <= 0:
            return prioritized[:sample_size]
        leftover = [entry for entry in pool if entry not in prioritized]
        if not leftover:
            return prioritized
        return prioritized + self.rng.sample(leftover, min(remaining, len(leftover)))

    def __len__(self) -> int:
        return len(self._entries)


class ContinuousTaskStream:
    """Task stream for continuous RL (library, sequential, JSONL, or reward-adaptive)."""

    SUPPORTED_MODES = frozenset({"library_cycle", "sequential", "jsonl", "reward_adaptive"})

    def __init__(
        self,
        config: FrameworkConfig,
        *,
        prompt_library: PromptLibrary | None = None,
        reward_tracker: RewardPathTracker | None = None,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed + 1901)
        self.prompt_library = prompt_library or resolve_prompt_library(config) or PromptLibrary()
        self.reward_tracker = reward_tracker or RewardPathTracker()
        self.hardware_options = config.ordered_hardware()
        mode = config.continuous_task_stream_mode.strip().lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"continuous_task_stream_mode must be one of {sorted(self.SUPPORTED_MODES)}, got {mode!r}"
            )
        self.mode = mode
        self._jsonl_tasks = self._load_jsonl_tasks() if mode == "jsonl" else []

    def tasks(self, max_tasks: int | None = None) -> Iterator[ContinuousTask]:
        limit = self.config.continuous_max_tasks if max_tasks is None else max_tasks
        if limit <= 0:
            raise ValueError("continuous max_tasks must be > 0")
        for task_index in range(limit):
            prompt, hardware = self._next_task_pair(task_index)
            yield ContinuousTask(task_index=task_index, prompt=prompt, hardware=hardware)

    def _next_task_pair(self, task_index: int) -> tuple[PromptSample, HardwareType]:
        if self.mode == "jsonl":
            row = self._jsonl_tasks[task_index % len(self._jsonl_tasks)]
            return row["prompt"], row["hardware"]
        if self.mode == "sequential":
            prompt = self.prompt_library.prompts[task_index % len(self.prompt_library.prompts)]
            hardware = self.hardware_options[task_index % len(self.hardware_options)]
            return prompt, hardware
        if self.mode == "reward_adaptive":
            prompt = self.reward_tracker.select_prompt(
                self.prompt_library,
                self.rng,
                explore_fraction=float(self.config.continuous_exploration_rate),
            )
            hardware = self.hardware_options[self.rng.randrange(len(self.hardware_options))]
            return prompt, hardware
        prompt = self.prompt_library.prompts[self.rng.randrange(len(self.prompt_library.prompts))]
        hardware = self.hardware_options[self.rng.randrange(len(self.hardware_options))]
        return prompt, hardware

    def _load_jsonl_tasks(self) -> list[dict[str, Any]]:
        path = self.config.continuous_task_jsonl_path
        if not path:
            raise ValueError(
                "continuous_task_stream_mode='jsonl' requires continuous_task_jsonl_path"
            )
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"continuous task JSONL not found: {source}")
        max_rows = max(1, int(self.config.continuous_max_tasks))
        rows: list[dict[str, Any]] = []
        for line_number, payload in enumerate(load_jsonl(str(source)), start=1):
            if len(rows) >= max_rows:
                break
            if not isinstance(payload, dict):
                raise TypeError(
                    f"continuous task JSONL line {line_number} must be a JSON object: {source}"
                )
            prompt_id = str(payload.get("prompt_id") or f"jsonl_{line_number:06d}")
            if "prompt_text" not in payload:
                raise ValueError(
                    f"continuous task JSONL line {line_number} missing prompt_text: {source}"
                )
            text = validate_online_prompt_text(str(payload["prompt_text"]))
            domain = str(payload.get("prompt_domain") or "custom")
            hardware_raw = str(payload.get("hardware") or self.hardware_options[0].value)
            hardware = HardwareType(hardware_raw)
            rows.append(
                {
                    "prompt": PromptSample(prompt_id=prompt_id, text=text, domain=domain),
                    "hardware": hardware,
                }
            )
        if not rows:
            raise ValueError(f"continuous task JSONL is empty: {source}")
        return rows


class ContinuousLearningLoop:
    """Runs real RL updates on a continuous task stream with replay, eval, and drift rollback."""

    def __init__(self, config: FrameworkConfig, trainer=None) -> None:
        from adaptive_quant.trainer import build_trainer

        if not config.continuous_learning_enabled:
            raise ValueError("continuous_learning_enabled must be true for ContinuousLearningLoop")

        self.config = config
        self.trainer = trainer or build_trainer(config)
        self._owns_trainer = trainer is None
        self.rng = random.Random(config.seed + 1907)
        self.reward_tracker = self._load_reward_tracker()
        self.task_stream = ContinuousTaskStream(
            config,
            reward_tracker=self.reward_tracker,
        )
        self.experience_buffer = ContinuousExperienceBuffer(
            config.continuous_replay_capacity, self.rng
        )
        self.telemetry_logger = JsonlLogger(
            config.continuous_telemetry_path(),
            buffered=bool(config.jsonl_buffered),
            flush_every=int(config.jsonl_flush_every),
            integrity_chain=bool(config.jsonl_integrity_chain) or jsonl_integrity_chain_enabled(),
        )
        self.tasks_completed = 0
        self.tasks_since_update = 0
        self.total_updates = 0
        self.total_rollbacks = 0
        self.recent_rewards: deque[float] = deque(
            maxlen=max(4, int(config.continuous_drift_window))
        )
        self.best_recent_reward = float("-inf")
        self.best_policy_snapshot = self.trainer.snapshot_policy()

    def _reward_path_state_path(self) -> Path:
        return Path(self.config.outputs_dir) / self.config.run_name / "reward_path_state.json"

    def _load_reward_tracker(self) -> RewardPathTracker:
        path = self._reward_path_state_path()
        if path.is_file():
            return RewardPathTracker.load(path)
        return RewardPathTracker()

    def run(self, max_tasks: int | None = None) -> dict[str, Any]:
        rewards: list[float] = []
        eval_summaries: list[dict[str, float]] = []

        for task in self.task_stream.tasks(max_tasks):
            explore = self.rng.random() < float(self.config.continuous_exploration_rate)
            outcome = run_rl_episode(
                self.trainer,
                episode_index=task.task_index,
                phase="train",
                forced_prompt=task.prompt,
                forced_hardware=task.hardware,
                deterministic=not explore,
                apply_update=False,
            )
            self.reward_tracker.record(
                task.prompt.prompt_id,
                outcome.reward,
                domain=task.prompt.domain,
            )
            self._record_experience(outcome)
            self._sync_trainer_counters(outcome)
            rewards.append(outcome.reward)
            self.tasks_completed += 1
            self.tasks_since_update += 1

            update_summary = self._maybe_apply_rl_update()
            drift_event = self._maybe_handle_drift(outcome.reward)
            eval_summary = self._maybe_evaluate()
            if eval_summary is not None:
                eval_summaries.append(eval_summary)
            self._maybe_checkpoint()

            self.telemetry_logger.log(
                {
                    "task_index": task.task_index,
                    "prompt_id": task.prompt.prompt_id,
                    "prompt_domain": task.prompt.domain,
                    "hardware_mode": task.hardware.value,
                    "reward": outcome.reward,
                    "explore": explore,
                    "stream_mode": self.task_stream.mode,
                    "tasks_completed": self.tasks_completed,
                    "total_updates": self.total_updates,
                    "replay_size": len(self.experience_buffer),
                    "drift_event": drift_event,
                    "continuous_update_applied": update_summary is not None,
                    "continuous_update_summary": update_summary,
                    "decision_mode": outcome.decision.mode.value,
                    "decision_bits": outcome.decision.base_bit_width,
                    "reward_path_tail": self.reward_tracker.reward_path_tail(limit=8),
                }
            )

        reward_path_file = self.reward_tracker.save(self._reward_path_state_path())
        summary = reward_summary(rewards, updates=self.total_updates)
        summary.update(
            {
                "tasks": self.tasks_completed,
                "total_updates": self.total_updates,
                "total_rollbacks": self.total_rollbacks,
                "replay_size": len(self.experience_buffer),
                "eval_runs": len(eval_summaries),
                "mean_eval_reward": mean([row.get("mean_reward", 0.0) for row in eval_summaries])
                if eval_summaries
                else 0.0,
                "telemetry_path": self.config.continuous_telemetry_path(),
                "reward_path_state": reward_path_file,
                "prompt_reward_summary": self.reward_tracker.prompt_summary(),
            }
        )
        write_json(self.config.continuous_summary_path(), summary)
        return summary

    def close(self) -> None:
        self.telemetry_logger.close()
        if self._owns_trainer:
            self.trainer.close()

    def _record_experience(self, outcome: RLEpisodeOutcome) -> None:
        self.experience_buffer.add(
            _ExperienceEntry(
                payload=outcome.trace,
                reward=outcome.reward,
                task_index=outcome.episode_index,
                prompt_id=outcome.state.prompt.prompt_id,
                hardware_mode=outcome.state.hardware_profile.hardware_type.value,
            )
        )
        if hasattr(self.trainer, "training_history"):
            self.trainer.training_history.append(record_training_row(outcome))

    def _sync_trainer_counters(self, outcome: RLEpisodeOutcome) -> None:
        if hasattr(self.trainer, "completed_episodes"):
            self.trainer.completed_episodes = max(
                int(getattr(self.trainer, "completed_episodes", 0)),
                outcome.episode_index + 1,
            )
        if hasattr(self.trainer, "global_episode"):
            self.trainer.global_episode = max(
                int(getattr(self.trainer, "global_episode", 0)),
                outcome.episode_index + 1,
            )

    def _maybe_apply_rl_update(self) -> dict[str, float] | None:
        if self.tasks_since_update < int(self.config.continuous_update_every_n_tasks):
            return None
        if len(self.experience_buffer) < int(self.config.continuous_min_replay_before_update):
            return None

        sampled = self.experience_buffer.sample(
            int(self.config.continuous_batch_size),
            reward_tracker=self.reward_tracker,
        )
        updates = [(entry.payload, entry.reward) for entry in sampled]
        summary = self.trainer.update_online(updates)
        self.total_updates += 1
        self.tasks_since_update = 0
        return summary

    def _maybe_evaluate(self) -> dict[str, float] | None:
        interval = int(self.config.continuous_eval_every_n_tasks)
        if interval <= 0 or self.tasks_completed % interval != 0:
            return None
        eval_summary = self.trainer.evaluate()
        print(
            f"[continuous task {self.tasks_completed:,}] "
            f"eval_reward={eval_summary.get('mean_reward', 0):.3f}",
            file=sys.stderr,
        )
        return eval_summary

    def _maybe_checkpoint(self) -> None:
        interval = int(self.config.continuous_checkpoint_every_n_tasks)
        if interval <= 0 or self.tasks_completed % interval != 0:
            return
        ckpt_path = self.config.final_checkpoint_path().replace(
            "_final", f"_task{self.tasks_completed}"
        )
        self.trainer.save_checkpoint(ckpt_path)
        self.reward_tracker.save(self._reward_path_state_path())

    def _maybe_handle_drift(self, reward: float) -> str:
        self.recent_rewards.append(float(reward))
        window = int(self.config.continuous_drift_window)
        if len(self.recent_rewards) < min(window, self.recent_rewards.maxlen or window):
            return "warming_up"

        recent_mean = mean(list(self.recent_rewards))
        if recent_mean > self.best_recent_reward:
            self.best_recent_reward = recent_mean
            self.best_policy_snapshot = self.trainer.snapshot_policy()
            return "improved"

        if recent_mean < self.best_recent_reward - float(self.config.continuous_drift_reward_delta):
            self.trainer.restore_policy(self.best_policy_snapshot)
            self.total_rollbacks += 1
            self.recent_rewards.clear()
            return "rollback"
        return "steady"


def build_continuous_learning_loop(config: FrameworkConfig, trainer=None) -> ContinuousLearningLoop:
    return ContinuousLearningLoop(config, trainer=trainer)


__all__ = [
    "ContinuousExperienceBuffer",
    "ContinuousLearningLoop",
    "ContinuousTask",
    "ContinuousTaskStream",
    "build_continuous_learning_loop",
]
