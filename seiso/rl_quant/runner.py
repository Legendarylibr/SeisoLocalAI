"""Run bundled Adaptive RL Quantization research pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.rl_quant.bootstrap import require_adaptive_quant
from seiso.rl_quant.config_builder import build_framework_config
from seiso.rl_quant.sweep import (
    apply_best_sweep_overrides,
    auto_sweep_enabled,
    run_auto_hyperparameter_sweep,
)


def run_rl_quant_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute train → evaluate → recommend → benchmark → analysis."""
    require_adaptive_quant()
    from seiso.adaptive_quant.research_pipeline import ResearchPipeline

    config = build_framework_config(
        job_id=job_id,
        user_id=user_id,
        data_dir=data_dir,
        payload=payload,
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    requested_stages = payload.get("stages")
    if requested_stages:
        _log(f"Requested stages: {', '.join(str(s) for s in requested_stages)}")
        if "auto_sweep" not in requested_stages:
            payload = {**payload, "auto_sweep": False}

    sweep_result: dict[str, Any] | None = None
    if auto_sweep_enabled(payload):
        _log("Phase: auto hyperparameter sweep")
        sweep_result = run_auto_hyperparameter_sweep(
            config,
            payload=payload,
            on_log=on_log,
        )
        config = apply_best_sweep_overrides(
            config, sweep_result.get("best_overrides") or {}
        )

    _log(
        f"RL quant run: {config.run_name} backend={config.backend} trainer={config.training_backend}"
    )
    _log(
        f"Episodes: train={config.training_episodes} eval={config.evaluation_episodes}"
    )
    _log(f"Artifacts: {config.artifacts.outputs_dir}")

    enabled_stages = (
        frozenset(str(s) for s in requested_stages) if requested_stages else None
    )
    pipeline = ResearchPipeline(config, enabled_stages=enabled_stages)
    _log("Phase: research pipeline (train → eval → recommend → benchmark → analysis)")
    summary = pipeline.run()
    _log("RL quantization pipeline complete")

    artifacts = (
        summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    )
    raw_rec = summary.get("recommendation")
    recommendation: dict[str, Any] | None = (
        raw_rec if isinstance(raw_rec, dict) else None
    )
    recommendation_path: str | Path | None = None
    if isinstance(artifacts.get("recommendation"), str):
        recommendation_path = artifacts["recommendation"]
    elif isinstance(raw_rec, str):
        recommendation_path = raw_rec
    else:
        recommendation_path = config.recommendation_path()

    if recommendation is None and recommendation_path:
        path = Path(str(recommendation_path))
        if path.is_file():
            import json

            recommendation = json.loads(path.read_text(encoding="utf-8"))
            _log(f"Recommendation written: {recommendation_path}")

    result: dict[str, Any] = {
        "summary": summary,
        "output_dir": config.artifacts.outputs_dir,
        "recommendation_path": (
            str(recommendation_path) if recommendation_path else None
        ),
        "recommendation": recommendation,
        "run_name": config.run_name,
        "auto_sweep": auto_sweep_enabled(payload),
    }
    if sweep_result is not None:
        result["sweep"] = sweep_result
    return result
