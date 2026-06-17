"""Run vendored Adaptive RL Quantization research pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.rl_quant.bootstrap import require_adaptive_quant
from seiso.rl_quant.config_builder import build_framework_config


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
    from adaptive_quant.research_pipeline import ResearchPipeline

    config = build_framework_config(
        job_id=job_id,
        user_id=user_id,
        data_dir=data_dir,
        payload=payload,
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    _log(f"RL quant run: {config.run_name} backend={config.backend} trainer={config.training_backend}")
    _log(f"Episodes: train={config.training_episodes} eval={config.evaluation_episodes}")
    _log(f"Artifacts: {config.artifacts.outputs_dir}")

    pipeline = ResearchPipeline(config)
    _log("Phase: research pipeline (train → eval → recommend → benchmark → analysis)")
    summary = pipeline.run()
    _log("RL quantization pipeline complete")

    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    raw_rec = summary.get("recommendation")
    recommendation: dict[str, Any] | None = raw_rec if isinstance(raw_rec, dict) else None
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

    return {
        "summary": summary,
        "output_dir": config.artifacts.outputs_dir,
        "recommendation_path": str(recommendation_path) if recommendation_path else None,
        "recommendation": recommendation,
        "run_name": config.run_name,
    }
