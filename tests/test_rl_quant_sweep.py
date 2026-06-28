"""Auto hyperparameter sweep tests for RL quant."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.rl_quant.bootstrap import require_adaptive_quant
from seiso.rl_quant.sweep import (
    apply_best_sweep_overrides,
    auto_sweep_enabled,
    default_sweep_grid,
    sweep_episode_budget,
)


def test_auto_sweep_enabled_defaults_true():
    assert auto_sweep_enabled({}) is True
    assert auto_sweep_enabled({"preset": "minimal"}) is True


def test_auto_sweep_can_be_disabled():
    assert auto_sweep_enabled({"auto_sweep": False}) is False
    assert auto_sweep_enabled({"sweep": False}) is False


def test_default_sweep_grid_by_preset():
    minimal = default_sweep_grid({"preset": "minimal"})
    assert "learning_rate" in minimal
    assert len(minimal["learning_rate"]) == 2

    reproducible = default_sweep_grid({"preset": "reproducible"})
    assert "learning_rate" in reproducible
    assert "value_learning_rate" in reproducible


def test_sweep_episode_budget_scales_down():
    train, eval_eps = sweep_episode_budget(
        {"training_episodes": 256, "evaluation_episodes": 64},
        training_episodes=256,
        evaluation_episodes=64,
    )
    assert train == 64
    assert eval_eps == 16


def test_sweep_episode_budget_respects_benchmark_fields():
    train, eval_eps = sweep_episode_budget(
        {"benchmark_training_episodes": 12, "benchmark_evaluation_episodes": 4},
        training_episodes=256,
        evaluation_episodes=64,
    )
    assert train == 12
    assert eval_eps == 4


def test_apply_best_sweep_overrides():
    require_adaptive_quant()
    from seiso.rl_quant.config_builder import build_framework_config

    cfg = build_framework_config(
        job_id="job-sweep",
        user_id="user-1",
        data_dir=Path("/tmp/seiso-test"),
        payload={"preset": "minimal", "training_episodes": 32, "evaluation_episodes": 8},
    )
    updated = apply_best_sweep_overrides(cfg, {"learning_rate": 0.02})
    assert updated.learning_rate == 0.02


def test_run_auto_hyperparameter_sweep_ranks_trials(tmp_path: Path):
    require_adaptive_quant()
    from seiso.rl_quant.config_builder import build_framework_config
    from seiso.rl_quant.sweep import run_auto_hyperparameter_sweep

    cfg = build_framework_config(
        job_id="job-sweep",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "minimal",
            "training_episodes": 32,
            "evaluation_episodes": 8,
            "sweep_grid": {"learning_rate": (0.02, 0.03)},
            "sweep_training_episodes": 4,
            "sweep_evaluation_episodes": 2,
        },
    )

    summaries = [
        {"evaluation": {"mean_reward": 0.5}},
        {"evaluation": {"mean_reward": 1.25}},
    ]

    with patch(
        "seiso.adaptive_quant.research_pipeline.run_pipeline_entrypoint",
        side_effect=summaries,
    ):
        result = run_auto_hyperparameter_sweep(
            cfg,
            payload={
                "preset": "minimal",
                "sweep_grid": {"learning_rate": (0.02, 0.03)},
                "sweep_training_episodes": 4,
                "sweep_evaluation_episodes": 2,
            },
        )

    assert result["trial_count"] == 2
    assert result["best_objective_value"] == pytest.approx(1.25)
    assert result["best_overrides"]["learning_rate"] == 0.03
    assert Path(str(result["aggregate_path"])).is_file()


def test_run_rl_quant_job_runs_sweep_before_pipeline(tmp_path: Path):
    from seiso.rl_quant.runner import run_rl_quant_job

    sweep_result = {
        "best_overrides": {"learning_rate": 0.03},
        "best_objective_value": 1.0,
        "trial_count": 2,
    }
    pipeline_summary = {"recommendation": {"decision": {"deploy": "fixed"}}}

    with (
        patch(
            "seiso.rl_quant.runner.run_auto_hyperparameter_sweep",
            return_value=sweep_result,
        ) as sweep,
        patch("seiso.adaptive_quant.research_pipeline.ResearchPipeline") as pipeline_cls,
    ):
        pipeline_cls.return_value.run.return_value = pipeline_summary
        result = run_rl_quant_job(
            job_id="sweep-wrap",
            user_id="tester",
            data_dir=tmp_path,
            payload={"preset": "minimal", "auto_sweep": True},
        )

    sweep.assert_called_once()
    pipeline_cls.assert_called_once()
    assert result["auto_sweep"] is True
    assert result["sweep"] == sweep_result


def test_run_rl_quant_job_skips_sweep_when_disabled(tmp_path: Path):
    from seiso.rl_quant.runner import run_rl_quant_job

    with (
        patch("seiso.rl_quant.runner.run_auto_hyperparameter_sweep") as sweep,
        patch("seiso.adaptive_quant.research_pipeline.ResearchPipeline") as pipeline_cls,
    ):
        pipeline_cls.return_value.run.return_value = {}
        result = run_rl_quant_job(
            job_id="no-sweep",
            user_id="tester",
            data_dir=tmp_path,
            payload={"preset": "minimal", "auto_sweep": False},
        )

    sweep.assert_not_called()
    assert result["auto_sweep"] is False
    assert "sweep" not in result
