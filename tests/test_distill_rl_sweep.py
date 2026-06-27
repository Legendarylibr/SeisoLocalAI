"""Auto hyperparameter sweep tests for distill-RL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.distill_rl.config import build_distill_rl_config
from seiso.distill_rl.sweep import (
    SharedStageContext,
    apply_best_sweep_overrides,
    auto_sweep_enabled,
    default_sweep_grid,
    extract_metric,
    run_auto_hyperparameter_sweep,
    sweep_dpo_max_steps,
)


def test_auto_sweep_enabled_defaults_true():
    assert auto_sweep_enabled({}) is True


def test_auto_sweep_can_be_disabled():
    assert auto_sweep_enabled({"auto_sweep": False}) is False


def test_default_sweep_grid_smoke():
    grid = default_sweep_grid({"preset": "smoke"})
    assert "dpo_beta" in grid
    assert min(grid["dpo_learning_rate"]) >= 2e-6


def test_sweep_dpo_max_steps_halves_config():
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=Path("/tmp"),
        payload={"preset": "smoke", "dpo_max_steps": 8},
    )
    assert sweep_dpo_max_steps(cfg, {}) == 4


def test_sweep_dpo_max_steps_uses_train_size_when_uncapped():
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=Path("/tmp"),
        payload={"preset": "full", "dpo_batch_size": 1, "dpo_gradient_accumulation_steps": 8},
    )
    assert sweep_dpo_max_steps(cfg, {}, train_example_count=96) == 12


def test_apply_best_sweep_overrides():
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=Path("/tmp"),
        payload={"preset": "smoke"},
    )
    updated = apply_best_sweep_overrides(cfg, {"dpo_beta": 0.2})
    assert updated.dpo_beta == 0.2


def test_extract_metric_nested_path():
    payload = {"checkpoints": {"dpo": {"val_preference_accuracy": 0.82}}}
    assert extract_metric(payload, "checkpoints.dpo.val_preference_accuracy") == pytest.approx(0.82)


def test_distill_rl_defaults_use_stable_dpo_values(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "reproducible"},
    )
    assert cfg.dpo_learning_rate == pytest.approx(5e-6)
    assert cfg.dpo_gradient_accumulation_steps == 8
    assert cfg.dpo_average_log_prob is True
    assert cfg.dpo_warmup_ratio == pytest.approx(0.1)


def test_run_auto_hyperparameter_sweep_ranks_trials(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-sweep",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke", "stages": ["distill", "rollout", "dpo", "evaluate"]},
    )
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    prefs_dir = cfg.preferences_dir
    prefs_dir.mkdir(parents=True, exist_ok=True)
    train = prefs_dir / "preferences_train.jsonl"
    val = prefs_dir / "preferences_val.jsonl"
    train.write_text('{"prompt":"p","chosen":"yes","rejected":"no"}\n', encoding="utf-8")
    val.write_text('{"prompt":"p","chosen":"yes","rejected":"no"}\n', encoding="utf-8")

    distilled = cfg.distilled_dir
    distilled.mkdir(parents=True, exist_ok=True)
    shared = SharedStageContext(distilled_dir=distilled, stage_results={})

    dpo_dirs = [
        cfg.output_root / "sweep" / "trial_001" / "checkpoint-1",
        cfg.output_root / "sweep" / "trial_002" / "checkpoint-1",
    ]
    for path in dpo_dirs:
        path.mkdir(parents=True, exist_ok=True)

    evaluations = [
        {
            "checkpoints": {
                "dpo": {
                    "alignment_score": 0.4,
                    "val_preference_accuracy": 0.4,
                    "val_preference_margin_mean": -0.1,
                }
            },
            "summary_path": "a.json",
        },
        {
            "checkpoints": {
                "dpo": {
                    "alignment_score": 0.9,
                    "val_preference_accuracy": 0.9,
                    "val_preference_margin_mean": 0.2,
                }
            },
            "summary_path": "b.json",
        },
    ]
    seen_output_dirs: list[Path] = []
    dpo_iter = iter(dpo_dirs)

    def fake_dpo(_cfg, *, model_dir, preferences_path, on_log=None):
        seen_output_dirs.append(_cfg.dpo_output_dir)
        return next(dpo_iter)

    with patch("seiso.distill_rl.evaluate.evaluate_pipeline", side_effect=evaluations):
        result = run_auto_hyperparameter_sweep(
            cfg,
            payload={"preset": "smoke", "sweep_grid": {"dpo_beta": (0.05, 0.1)}},
            shared=shared,
            run_dpo_fn=fake_dpo,
        )

    assert result["trial_count"] == 2
    assert result["best_objective_value"] == pytest.approx(0.9)
    assert result["best_overrides"]["dpo_beta"] == 0.1
    assert all(path.parent == cfg.output_root / "sweep" for path in seen_output_dirs)
    assert Path(str(result["aggregate_path"])).is_file()


def test_run_distill_rl_job_runs_sweep_before_final_dpo(tmp_path: Path):
    from seiso.distill_rl.runner import run_distill_rl_job

    sweep_result = {"best_overrides": {"dpo_beta": 0.1}, "best_objective_value": 0.9}
    job_root = tmp_path / "distill_rl" / "cli" / "job-sweep"
    distilled = job_root / "distilled"
    distilled.mkdir(parents=True)
    prefs = job_root / "preferences"
    prefs.mkdir(parents=True)
    train = prefs / "preferences_train.jsonl"
    val = prefs / "preferences_val.jsonl"
    train.write_text('{"prompt":"p","chosen":"yes","rejected":"no"}\n', encoding="utf-8")
    val.write_text('{"prompt":"p","chosen":"yes","rejected":"no"}\n', encoding="utf-8")
    dpo_dir = job_root / "dpo" / "checkpoint-1"
    dpo_dir.mkdir(parents=True)
    eval_summary = job_root / "evaluation" / "evaluation_summary.json"
    eval_summary.parent.mkdir(parents=True)
    eval_summary.write_text("{}", encoding="utf-8")

    with (
        patch(
            "seiso.distill_rl.runner.run_auto_hyperparameter_sweep", return_value=sweep_result
        ) as sweep,
        patch("seiso.models.hf_env.configure_hf_hub_cache"),
        patch("seiso.security.nvidia_boundary.enforce_nvidia_secure_boundary"),
        patch("seiso.distill_rl.runner.init_run_manifest", return_value={}),
        patch("seiso.distill_rl.runner.verify_run_manifest", return_value={"ok": True}),
        patch("seiso.distill_rl.runner.append_artifact"),
        patch("seiso.distill_rl.runner._run_shared_stages") as shared,
        patch("seiso.distill_rl.runner._run_distill", return_value=distilled),
        patch("seiso.distill_rl.runner._run_dpo", return_value=dpo_dir),
        patch(
            "seiso.distill_rl.evaluate.evaluate_pipeline",
            return_value={"summary_path": str(eval_summary), "checkpoints": {}},
        ),
        patch(
            "seiso.distill_rl.runner.create_paper_bundle",
            return_value={"paper_bundle_dir": str(job_root)},
        ),
    ):
        shared.return_value = SharedStageContext(
            distilled_dir=distilled,
            stage_results={"distilled": str(distilled), "preferences_train": str(train)},
        )
        result = run_distill_rl_job(
            job_id="job-sweep",
            user_id="cli",
            data_dir=tmp_path,
            payload={
                "preset": "smoke",
                "stages": ["distill", "rollout", "dpo", "evaluate"],
                "auto_sweep": True,
            },
        )

    sweep.assert_called_once()
    assert result["auto_sweep"] is True
    assert result["sweep"] == sweep_result
