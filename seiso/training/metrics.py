"""Structured training metrics — HF TrainerCallback with JSONL + stdout emission."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRIC_STDOUT_PREFIX = "SEISO_METRIC:"
MAX_METRIC_HISTORY = 5000
_METRICS_FLUSH_INTERVAL = 10


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def normalize_training_log(state, logs: dict[str, Any]) -> dict[str, Any]:
    """Map HuggingFace Trainer log dict to a stable metrics payload."""
    loss = logs.get("loss")
    if loss is None:
        loss = logs.get("train_loss")

    reward = logs.get("reward")
    if reward is None and loss is not None:
        # Useful scalar for RL-style dashboards when no explicit reward is logged.
        reward = -float(loss)

    metric: dict[str, Any] = {
        "type": "training",
        "step": int(state.global_step),
        "epoch": round(float(logs.get("epoch", state.epoch or 0)), 4),
        "loss": float(loss) if loss is not None else None,
        "eval_loss": float(logs["eval_loss"]) if logs.get("eval_loss") is not None else None,
        "reward": float(reward) if reward is not None else None,
        "learning_rate": float(logs["learning_rate"]) if logs.get("learning_rate") is not None else None,
        "grad_norm": float(logs["grad_norm"]) if logs.get("grad_norm") is not None else None,
        "train_runtime": float(logs["train_runtime"]) if logs.get("train_runtime") is not None else None,
        "train_samples_per_second": (
            float(logs["train_samples_per_second"]) if logs.get("train_samples_per_second") is not None else None
        ),
        "train_steps_per_second": (
            float(logs["train_steps_per_second"]) if logs.get("train_steps_per_second") is not None else None
        ),
        "ts": time.time(),
    }
    return {k: v for k, v in metric.items() if v is not None or k in ("type", "step", "epoch", "ts")}


class TrainingMetricsCallback:
    """HuggingFace TrainerCallback that emits structured training metrics."""

    def __init__(
        self,
        *,
        on_metric: Callable[[dict[str, Any]], None] | None = None,
        metrics_path: Path | None = None,
        emit_stdout: bool = False,
    ) -> None:
        self.on_metric = on_metric
        self.metrics_path = metrics_path
        self.emit_stdout = emit_stdout
        self._history: list[dict[str, Any]] = []
        self._file = None
        self._writes_since_flush = 0

    def _open_file(self) -> None:
        if self.metrics_path and self._file is None:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.metrics_path.open("a", encoding="utf-8")

    def _emit(self, metric: dict[str, Any]) -> None:
        self._history.append(metric)
        if len(self._history) > MAX_METRIC_HISTORY:
            del self._history[: len(self._history) - MAX_METRIC_HISTORY]

        if self.on_metric:
            try:
                self.on_metric(metric)
            except Exception:
                logger.exception("on_metric callback failed")

        if self._file:
            self._file.write(json.dumps(metric) + "\n")
            self._writes_since_flush += 1
            if self._writes_since_flush >= _METRICS_FLUSH_INTERVAL:
                self._file.flush()
                self._writes_since_flush = 0

        if self.emit_stdout:
            print(f"{METRIC_STDOUT_PREFIX}{json.dumps(metric)}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # noqa: ARG002
        if not logs or not is_main_process():
            return
        metric = normalize_training_log(state, logs)
        self._open_file()
        self._emit(metric)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:  # noqa: ARG002
        if not metrics or not is_main_process():
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        metric = {
            "type": "eval",
            "step": int(state.global_step),
            "epoch": round(float(state.epoch or 0), 4),
            "eval_loss": float(eval_loss),
            "reward": -float(eval_loss),
            "ts": time.time(),
        }
        for key in ("eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
            if metrics.get(key) is not None:
                metric[key] = float(metrics[key])
        self._open_file()
        self._emit(metric)

    def on_train_end(self, args, state, control, **kwargs) -> None:  # noqa: ARG002
        if self._file:
            if self._writes_since_flush:
                self._file.flush()
                self._writes_since_flush = 0
            self._file.close()
            self._file = None

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        training = [m for m in self._history if m.get("type") in ("training", "eval")]
        losses = [m["loss"] for m in training if m.get("loss") is not None]
        eval_losses = [m["eval_loss"] for m in training if m.get("eval_loss") is not None]
        return {
            "total_steps": max((m.get("step", 0) for m in training), default=0),
            "final_loss": losses[-1] if losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "points": len(self._history),
        }


def build_metrics_callback(
    output_dir: Path,
    *,
    on_metric: Callable[[dict[str, Any]], None] | None = None,
    emit_stdout: bool = False,
) -> Any:
    """Return a HF-compatible callback instance (lazy import transformers)."""
    from transformers import TrainerCallback

    impl = TrainingMetricsCallback(
        on_metric=on_metric,
        metrics_path=output_dir / "metrics.jsonl",
        emit_stdout=emit_stdout,
    )

    class _Adapter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            impl.on_log(args, state, control, logs=logs, **kwargs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            impl.on_evaluate(args, state, control, metrics=metrics, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            impl.on_train_end(args, state, control, **kwargs)

    adapter = _Adapter()
    adapter._seiso_metrics = impl  # type: ignore[attr-defined]
    return adapter


def parse_metric_line(line: str) -> dict[str, Any] | None:
    """Parse orchestrator stdout lines emitted by distributed workers."""
    text = line.strip()
    if not text.startswith(METRIC_STDOUT_PREFIX):
        return None
    try:
        return json.loads(text[len(METRIC_STDOUT_PREFIX) :])
    except json.JSONDecodeError:
        return None
