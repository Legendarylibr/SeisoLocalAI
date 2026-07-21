"""Research architecture contract — evidence tiers, learning scope, and claim boundaries.

Every pipeline summary embeds a ``research`` block built here so runs state *what was
trained* (quantization policy), *how metrics were measured*, and *what claims are valid*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from seiso.adaptive_quant.configuration import FrameworkConfig

SCHEMA_VERSION = 2

EVIDENCE_SIMULATOR = "simulator"
EVIDENCE_LOCAL_LLAMA_CPP = "local_llama_cpp"
EVIDENCE_MULTISEED = "multiseed_aggregate"
EVIDENCE_SWEEP = "sweep_aggregate"

LEARNING_TARGET_POLICY = "quantization_policy"
LEARNING_TARGET_GGUF_EXPORT = "exported_gguf"


def _gguf_export_enabled(config: FrameworkConfig) -> bool:
    return bool(config.llama_cpp_gguf_export_enabled)


def infer_evidence_level(config: FrameworkConfig) -> str:
    if config.backend == "llama_cpp":
        return EVIDENCE_LOCAL_LLAMA_CPP
    return EVIDENCE_SIMULATOR


def metric_sources_for_config(config: FrameworkConfig) -> dict[str, str]:
    """Per-metric provenance labels (shared by summaries, paper bundles, reports)."""
    has_external_quality = bool(config.external_quality_path)
    quality_source = (
        f"external:{config.external_quality_metric or 'perplexity'}"
        if has_external_quality
        else "simulator"
    )
    if config.backend == "llama_cpp":
        return {
            "latency_ms": "llama_cpp",
            "throughput_tps": "llama_cpp",
            "memory_mb": "llama_cpp_when_parseable_else_simulator",
            "perplexity": quality_source,
            "reward": (
                "mixed_llama_cpp_perf_plus_external_quality"
                if has_external_quality
                else "mixed_llama_cpp_perf_plus_simulator_quality"
            ),
        }
    from seiso.adaptive_quant.pipeline.topology import infer_simulator_engine

    sim_engine = infer_simulator_engine(config)
    perf_source = "simulator_rust_cli" if sim_engine == "rust_cli" else "simulator"
    return {
        "latency_ms": perf_source,
        "throughput_tps": perf_source,
        "memory_mb": perf_source,
        "perplexity": quality_source,
        "reward": (
            "simulator_plus_external_quality" if has_external_quality else perf_source
        ),
        "simulator_engine": sim_engine or "python",
    }


def _base_measurement_level(config: FrameworkConfig, evidence_level: str) -> str:
    """Map aggregate tiers back to the underlying measurement backend."""
    if evidence_level in {EVIDENCE_MULTISEED, EVIDENCE_SWEEP}:
        return infer_evidence_level(config)
    return evidence_level


def _claim_boundary(config: FrameworkConfig, evidence_level: str) -> dict[str, object]:
    valid: list[str] = []
    invalid: list[str] = [
        "llm_weight_updates",
        "multi_device_deployment_validation",
        "production_serving_sla",
    ]
    if not _gguf_export_enabled(config):
        invalid.append("automatic_gguf_requantization")
    base_level = _base_measurement_level(config, evidence_level)
    if evidence_level in {EVIDENCE_MULTISEED, EVIDENCE_SWEEP}:
        valid.append("seed_or_trial_aggregate_statistics")
    if base_level == EVIDENCE_SIMULATOR:
        valid.extend(
            [
                "policy_learning_dynamics",
                "benchmark_comparisons_within_simulator",
                "reward_engineering_ablations",
                "reproducible_rl_iteration",
            ]
        )
        invalid.append("real_hardware_latency_claims")
        invalid.append("real_inference_quality_claims_without_external_sidecar")
    elif base_level == EVIDENCE_LOCAL_LLAMA_CPP:
        valid.extend(
            [
                "single_machine_llama_cpp_latency_throughput",
                "route_selection_among_prebuilt_gguf",
                "policy_learning_with_local_measurements",
            ]
        )
        if config.external_quality_path:
            valid.append("external_quality_on_configured_sidecar")
        else:
            invalid.append("real_perplexity_claims_without_external_sidecar")
    if _gguf_export_enabled(config):
        valid.append("exported_gguf_from_recommendation_quant_type")
        valid.append("single_machine_gguf_export_provenance")
    return {"valid_claims": valid, "invalid_claims": invalid}


def _escalation_path(config: FrameworkConfig, evidence_level: str) -> list[str]:
    hints: list[str] = []
    base_level = _base_measurement_level(config, evidence_level)
    if base_level == EVIDENCE_SIMULATOR:
        hints.append(
            "Escalate to local llama.cpp: set backend='llama_cpp', llama_cpp_binary, "
            "llama_cpp_model, and pre-built GGUF routes (see docs/LOCAL_RESEARCH.md)."
        )
    if not config.external_quality_path:
        hints.append(
            "Add external_quality_path for dataset-grounded quality instead of simulator perplexity."
        )
    hints.append(
        "Run adaptive-rl-quant-multiseed before comparative claims to report mean ± std across seeds."
    )
    if config.backend == "llama_cpp" and not config.router_enabled:
        hints.append(
            "Enable router_enabled with router_routes to compare multiple GGUF quant variants in one run."
        )
    if not _gguf_export_enabled(config) and config.backend == "llama_cpp":
        hints.append(
            "Enable llama_cpp_gguf_export_enabled to write a recommendation-driven GGUF under outputs/gguf/."
        )
    if (
        config.rust_simulator_enabled
        and config.backend == "simulator"
        and not config.moe_enabled
    ):
        from seiso.adaptive_quant.rust_cli import resolve_rust_cli_binary

        if resolve_rust_cli_binary(config) is None:
            hints.append(
                "rust_simulator_enabled is set but no binary was found; run ./scripts/build_rust.sh "
                "from the repo root or set rust_cli_binary / ADAPTIVE_RL_RUST_CLI."
            )
    hints.append(
        "Use outputs/paper_bundles/<run>/manifest.json and claims_validation.md when citing results."
    )
    return hints


def build_research_contract(
    config: FrameworkConfig,
    *,
    git_commit: str | None = None,
    pipeline: str = "offline_research",
    phases: Sequence[str] | None = None,
    evidence_level: str | None = None,
) -> dict[str, object]:
    """Machine-readable research scope block for ``*_summary.json`` and reports."""
    level = evidence_level or infer_evidence_level(config)
    sources = metric_sources_for_config(config)
    boundary = _claim_boundary(config, level)
    export_enabled = _gguf_export_enabled(config)
    from seiso.adaptive_quant.pipeline.topology import build_pipeline_topology
    from seiso.adaptive_quant.rust_cli import rust_cli_status

    topology = build_pipeline_topology(config)

    does_not_train = ["llm_weights"]
    if not export_enabled:
        does_not_train.append("gguf_quantization_export")
    trained_artifacts = ["policy_checkpoint"]
    if export_enabled:
        trained_artifacts.append(LEARNING_TARGET_GGUF_EXPORT)
    summary_text = (
        "Trains an RL quantization/routing policy and exports a GGUF via llama.cpp quantize "
        "when llama_cpp_gguf_export_enabled is set."
        if export_enabled
        else (
            "Trains an RL quantization/routing policy; GGUF files are pre-built "
            "measurement inputs unless llama_cpp_gguf_export_enabled is set."
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "learning_target": {
            "object": LEARNING_TARGET_POLICY,
            "trained_artifacts": trained_artifacts,
            "trained_artifact": trained_artifacts[0],
            "secondary_artifacts": trained_artifacts[1:],
            "does_not_train": does_not_train,
            "gguf_export_enabled": export_enabled,
            "summary": summary_text,
        },
        "evidence": {
            "level": level,
            "deployment_grade": False,
            "metric_sources": sources,
            "claim_boundary": boundary,
        },
        "measurement": {
            "backend": config.backend,
            "training_backend": config.training_backend,
            "quant_mode": config.quant_mode,
            "moe_enabled": config.moe_enabled,
            "hardware_modes": list(config.hardware_modes),
            "router_enabled": config.router_enabled,
            "router_route_count": (
                len(config.router_routes) if config.router_enabled else 0
            ),
            "llama_cpp_configured": bool(
                config.backend == "llama_cpp"
                and config.llama_cpp_binary
                and config.llama_cpp_model
            ),
            "external_quality": bool(config.external_quality_path),
            "external_quality_metric": (
                config.external_quality_metric if config.external_quality_path else None
            ),
            "gguf_export_enabled": export_enabled,
            "rust_simulator_enabled": config.rust_simulator_enabled,
            "rust_cli_binary": config.rust_cli_binary,
            "rust_cli": rust_cli_status(config),
        },
        "reproducibility": {
            "git_commit": git_commit,
            "run_name": config.run_name,
            "seed": config.seed,
            "phases_completed": list(phases or ()),
        },
        "escalation_path": _escalation_path(config, level),
        "topology": topology,
    }


def build_claims_validation(
    *,
    config: FrameworkConfig,
    summary: Mapping[str, Any],
    metrics: Mapping[str, float],
    evidence_level: str | None = None,
) -> dict[str, Any]:
    """Paper-bundle claims block (aligned with ``research`` contract)."""
    level = evidence_level or infer_evidence_level(config)
    base_level = _base_measurement_level(config, level)
    warnings: list[str] = []
    has_external_quality = bool(config.external_quality_path)
    if base_level == EVIDENCE_LOCAL_LLAMA_CPP:
        if has_external_quality:
            warnings.append(
                "Latency/throughput are locally measured and quality uses an external sidecar, "
                "but this is still single-machine evidence."
            )
            warnings.append(
                "Verify the external quality sidecar was generated from real datasets and "
                "fixed scoring code before citing quality claims."
            )
        else:
            warnings.append(
                "Latency/throughput are locally measured, but perplexity remains "
                "simulator-derived unless an external quality metric is supplied."
            )
        warnings.append(
            "Local results are single-machine evidence, not deployment-grade multi-device validation."
        )
    else:
        if has_external_quality:
            warnings.append(
                "Systems metrics are simulator-backed; only the configured quality metric "
                "uses an external sidecar."
            )
        else:
            warnings.append("All headline metrics are simulator-backed.")
    boundary = _claim_boundary(config, level)
    return {
        "evidence_level": level,
        "learning_target": LEARNING_TARGET_POLICY,
        "deployment_grade": False,
        "external_quality": has_external_quality,
        "external_quality_metric": (
            config.external_quality_metric if has_external_quality else None
        ),
        "metric_count": len(metrics),
        "has_benchmark_summary": isinstance(summary.get("benchmarks"), Mapping),
        "has_evaluation_summary": isinstance(summary.get("evaluation"), Mapping),
        "valid_claims": boundary["valid_claims"],
        "invalid_claims": boundary["invalid_claims"],
        "warnings": warnings,
    }


__all__ = [
    "EVIDENCE_LOCAL_LLAMA_CPP",
    "EVIDENCE_MULTISEED",
    "EVIDENCE_SIMULATOR",
    "EVIDENCE_SWEEP",
    "LEARNING_TARGET_GGUF_EXPORT",
    "LEARNING_TARGET_POLICY",
    "SCHEMA_VERSION",
    "build_claims_validation",
    "build_research_contract",
    "infer_evidence_level",
    "metric_sources_for_config",
]
