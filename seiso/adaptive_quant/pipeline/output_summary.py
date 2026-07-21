"""Headline metrics helpers — keep ``*_summary.json`` readable."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seiso.adaptive_quant.configuration import FrameworkConfig
from seiso.adaptive_quant.math_utils import format_display


def resolve_analysis_log_path(config: FrameworkConfig, suffix: str) -> str:
    """Prefer benchmark variant logs ``{run_name}_{suffix}.jsonl``, else primary log."""
    specialized = Path(config.log_dir) / f"{config.run_name}_{suffix}.jsonl"
    if specialized.is_file():
        return str(specialized)
    primary = Path(config.primary_log_path())
    if primary.is_file():
        return str(primary)
    return str(specialized)


def _fmt(value: object, *, digits: int = 3) -> str:
    return format_display(value, style="report", digits=digits)


def slim_analysis_section(section: object) -> dict[str, object]:
    if not isinstance(section, dict):
        return {}
    slim: dict[str, object] = {}
    if "log_path" in section:
        slim["log_path"] = section["log_path"]
    for key in (
        "generalization_gap",
        "reward_by_hardware",
        "latency_by_hardware",
        "throughput_by_hardware",
        "perplexity_by_hardware",
        "by_complexity",
        "mean_swap_cost_ms",
        "mean_cache_miss_count",
        "mean_reward",
        "final_reward",
        "episodes",
    ):
        if key in section:
            slim[key] = section[key]
    return slim


def slim_analysis_for_summary(
    analysis: dict[str, object],
    config: FrameworkConfig,
) -> dict[str, object]:
    """Drop chart-ready blobs from ``*_summary.json``; full JSON lives under ``analysis_dir``."""
    root = Path(config.analysis_dir) / config.run_name
    slim: dict[str, object] = {"root": str(root)}
    for name, section in analysis.items():
        if not isinstance(section, dict):
            continue
        entry = slim_analysis_section(section)
        subdir = {
            "hardware": "hardware",
            "input": "inputs",
            "quant_function": "quant",
            "training_dynamics": "training",
            "moe_experts": "moe_experts",
            "moe_cache": "moe_cache",
        }.get(name, name)
        entry["artifacts_dir"] = str(root / subdir)
        slim[name] = entry
    return slim


def experiment_config_summary(config: FrameworkConfig) -> dict[str, object]:
    """Small config fingerprint for multiseed/sweep aggregate JSON (not full ``asdict``)."""
    return {
        "run_name": config.run_name,
        "backend": config.backend,
        "training_backend": config.training_backend,
        "training_episodes": config.training_episodes,
        "evaluation_episodes": config.evaluation_episodes,
        "quant_mode": config.quant_mode,
        "hardware_modes": list(config.hardware_modes),
        "moe_enabled": config.moe_enabled,
        "seed": config.seed,
    }


def headline_summary_for_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Curated summary subset for paper-bundle headline metrics (skip config blobs)."""
    curated: dict[str, Any] = {}
    for section in ("train", "evaluation", "bootstrap_train", "online"):
        block = summary.get(section)
        if isinstance(block, dict):
            curated[section] = block
    benchmarks = summary.get("benchmarks")
    if isinstance(benchmarks, dict):
        curated["benchmarks"] = {
            key: benchmarks[key]
            for key in ("single_vs_multi", "static_vs_dynamic", "discrete_vs_learned")
            if key in benchmarks
        }
    recommendation = summary.get("recommendation")
    if isinstance(recommendation, dict):
        slim_rec: dict[str, Any] = {}
        if "target_hardware" in recommendation:
            slim_rec["target_hardware"] = recommendation["target_hardware"]
        adaptive = recommendation.get("adaptive_policy")
        if isinstance(adaptive, dict):
            slim_rec["adaptive_policy"] = {
                key: adaptive[key] for key in ("mean_reward",) if key in adaptive
            }
        fixed = recommendation.get("recommended_quant")
        if isinstance(fixed, dict):
            slim_rec["recommended_quant"] = {
                "signature": fixed.get("signature"),
                "evaluation": (
                    fixed.get("evaluation")
                    if isinstance(fixed.get("evaluation"), dict)
                    else None
                ),
            }
        decision = recommendation.get("decision")
        if isinstance(decision, dict):
            slim_rec["decision"] = decision
        curated["recommendation"] = slim_rec
    analysis = summary.get("analysis")
    if isinstance(analysis, dict):
        curated["seiso.analysis"] = {
            name: slim_analysis_section(section)
            for name, section in analysis.items()
            if isinstance(section, dict)
        }
    return curated


def slim_online_analysis_for_summary(
    online_analysis: dict[str, object],
) -> dict[str, object]:
    """Drop chart paths from online summary JSON; figures live under ``analysis_dir``."""
    slim: dict[str, object] = {}
    for key in (
        "log_path",
        "records",
        "reward_by_hardware",
        "candidate_accept_rate",
        "online_update_rate",
        "rollback_count",
        "mean_served_reward",
    ):
        if key in online_analysis:
            slim[key] = online_analysis[key]
    return slim


def recommendation_decision_block(payload: dict[str, object]) -> dict[str, object]:
    adaptive = payload.get("adaptive_policy")
    recommended = payload.get("recommended_quant")
    adaptive_reward = (
        float((adaptive or {}).get("mean_reward", float("nan")))
        if isinstance(adaptive, dict)
        else float("nan")
    )
    fixed_reward = float("nan")
    signature: str | None = None
    if isinstance(recommended, dict):
        signature = str(recommended.get("signature", "")) or None
        evaluation = recommended.get("evaluation")
        if isinstance(evaluation, dict):
            fixed_reward = float(evaluation.get("mean_reward", float("nan")))

    use_adaptive = True
    delta: float | None = None
    if signature and math.isfinite(adaptive_reward) and math.isfinite(fixed_reward):
        delta = fixed_reward - adaptive_reward
        use_adaptive = fixed_reward <= adaptive_reward

    claimable = bool(payload.get("deploy_quality_claimable"))
    prefer_verb = "Deploy" if claimable else "Prefer"
    if use_adaptive:
        rationale = (
            f"{prefer_verb} the trained adaptive policy on the target hardware "
            "(fixed candidate did not beat adaptive on mean reward)."
        )
        deploy = "adaptive_policy"
    else:
        rationale = (
            f"{prefer_verb} fixed quant `{signature}` — mean reward {_fmt(fixed_reward)} "
            f"vs adaptive {_fmt(adaptive_reward)} (Δ {_fmt(delta)})."
        )
        deploy = signature or "fixed_quant"

    block: dict[str, object] = {
        "deploy": deploy,
        "use_adaptive_policy": use_adaptive,
        "rationale": rationale,
    }
    if delta is not None and delta == delta:
        block["reward_delta_vs_adaptive"] = delta
    return block


def build_research_artifact_index(
    config: FrameworkConfig,
    artifacts: Mapping[str, Any],
) -> dict[str, str | None]:
    """Stable artifact map for summary navigation."""
    paper_bundle = artifacts.get("paper_bundle")
    bundle_dir: str | None = None
    if isinstance(paper_bundle, Mapping):
        raw = paper_bundle.get("paper_bundle_dir")
        bundle_dir = str(raw) if raw else None
    return {
        "summary_json": config.summary_path(),
        "checkpoint": _artifact_path(artifacts, "final_checkpoint"),
        "recommendation_json": _artifact_path(artifacts, "recommendation"),
        "training_history": _artifact_path(artifacts, "training_history"),
        "exported_gguf": _artifact_path(artifacts, "exported_gguf"),
        "replay_manifest": _artifact_path(artifacts, "replay_manifest"),
        "online_telemetry": _artifact_path(artifacts, "online_telemetry"),
        "online_replay": _artifact_path(artifacts, "online_replay"),
        "continuous_telemetry": _artifact_path(artifacts, "continuous_telemetry"),
        "continuous_detail": _artifact_path(artifacts, "continuous_detail"),
        "frontier_reference": _artifact_path(artifacts, "frontier_reference"),
        "frontier_comparison": _artifact_path(artifacts, "frontier_comparison"),
        "frontier_eval": _artifact_path(artifacts, "frontier_eval"),
        "paper_bundle_dir": bundle_dir,
        "analysis_dir": f"{config.analysis_dir}/{config.run_name}/",
    }


def _artifact_path(artifacts: Mapping[str, Any], key: str) -> str | None:
    value = artifacts.get(key)
    if value is None:
        return None
    return str(value)


__all__ = [
    "build_research_artifact_index",
    "experiment_config_summary",
    "headline_summary_for_metrics",
    "recommendation_decision_block",
    "resolve_analysis_log_path",
    "slim_analysis_for_summary",
    "slim_online_analysis_for_summary",
]
