"""Tests for structured training metrics."""

from seiso.training.metrics import METRIC_STDOUT_PREFIX, normalize_training_log, parse_metric_line


class _State:
    global_step = 42
    epoch = 1.5


def test_normalize_training_log_maps_loss_and_reward():
    logs = {"loss": 1.25, "learning_rate": 0.0002, "epoch": 1.5, "grad_norm": 0.8}
    metric = normalize_training_log(_State(), logs)
    assert metric["step"] == 42
    assert metric["loss"] == 1.25
    assert metric["reward"] == -1.25
    assert metric["learning_rate"] == 0.0002


def test_parse_metric_line_from_stdout():
    payload = {"type": "training", "step": 10, "loss": 2.0}
    line = f"{METRIC_STDOUT_PREFIX}{__import__('json').dumps(payload)}"
    assert parse_metric_line(line) == payload


def test_parse_metric_line_ignores_regular_logs():
    assert parse_metric_line("Training step 10") is None
