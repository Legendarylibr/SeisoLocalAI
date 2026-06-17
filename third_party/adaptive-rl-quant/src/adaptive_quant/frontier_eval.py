"""Structured evaluation metrics for frontier vs local output comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.frontier_comparison import compare_frontier_to_local
from adaptive_quant.logging_utils import write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.prompts import PromptSample


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_frontier_eval_rows(
    rows: list[dict[str, Any]], config: FrameworkConfig
) -> dict[str, Any]:
    overlaps = [
        float(row["reference_overlap"]) for row in rows if row.get("reference_overlap") is not None
    ]
    length_ratios = [
        float(row["length_ratio"]) for row in rows if row.get("length_ratio") is not None
    ]
    local_latency = [
        float(row["local_latency_ms"]) for row in rows if row.get("local_latency_ms") is not None
    ]
    frontier_latency = [
        float(row["frontier_latency_ms"])
        for row in rows
        if row.get("frontier_latency_ms") is not None
    ]
    local_available = sum(1 for row in rows if row.get("local_source") not in {None, "unavailable"})
    prompt_total = len(rows)
    local_coverage = float(local_available) / float(prompt_total) if prompt_total else 0.0

    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        overlap = row.get("reference_overlap")
        if overlap is None:
            continue
        domain = str(row.get("prompt_domain") or "unknown")
        by_domain[domain].append(float(overlap))

    overlap_by_domain = {domain: mean(values) for domain, values in sorted(by_domain.items())}
    min_overlap = float(config.frontier_eval_min_overlap)
    min_coverage = float(config.frontier_eval_min_local_coverage)
    mean_overlap = mean(overlaps) if overlaps else 0.0
    passed = bool(
        prompt_total > 0
        and mean_overlap >= min_overlap
        and local_coverage >= min_coverage
        and len(overlaps) == prompt_total
    )
    failed_prompts: list[str] = []
    for row in rows:
        prompt_id = str(row.get("prompt_id") or "")
        if not prompt_id:
            continue
        overlap = row.get("reference_overlap")
        if overlap is None or float(overlap) < min_overlap:
            failed_prompts.append(prompt_id)
            continue
        if row.get("local_source") in {None, "unavailable"} and local_coverage < min_coverage:
            failed_prompts.append(prompt_id)

    return {
        "prompts_evaluated": prompt_total,
        "prompts_with_overlap": len(overlaps),
        "local_coverage_rate": local_coverage,
        "mean_reference_overlap": mean_overlap,
        "median_reference_overlap": _median(overlaps),
        "min_reference_overlap": min(overlaps) if overlaps else 0.0,
        "max_reference_overlap": max(overlaps) if overlaps else 0.0,
        "mean_length_ratio": mean(length_ratios) if length_ratios else 0.0,
        "mean_local_latency_ms": mean(local_latency) if local_latency else 0.0,
        "mean_frontier_latency_ms": mean(frontier_latency) if frontier_latency else 0.0,
        "overlap_by_domain": overlap_by_domain,
        "thresholds": {
            "min_overlap": min_overlap,
            "min_local_coverage": min_coverage,
        },
        "passed": passed,
        "failed_prompt_ids": failed_prompts,
    }


def evaluate_frontier_vs_local(
    config: FrameworkConfig,
    *,
    prompts: list[PromptSample] | None = None,
    reference_path: str | None = None,
    refresh_reference: bool = False,
) -> dict[str, Any]:
    """Run frontier/local comparison and emit structured eval metrics + artifacts."""
    comparison = compare_frontier_to_local(
        config,
        prompts=prompts,
        reference_path=reference_path,
        refresh_reference=refresh_reference,
    )
    rows = list(comparison.get("rows", []))
    eval_summary = summarize_frontier_eval_rows(rows, config)
    payload = {
        "run_name": config.run_name,
        "frontier_model": config.frontier_model,
        "local_backend": config.backend,
        "comparison_metric": config.frontier_comparison_metric,
        "comparison_path": comparison.get("comparison_path"),
        "reference_path": comparison.get("reference_path"),
        "external_quality_sidecar": comparison.get("external_quality_sidecar"),
        "eval": eval_summary,
        "rows": rows,
    }
    eval_path = config.frontier_eval_path()
    write_json(eval_path, payload)
    if config.frontier_eval_write_analysis:
        from analysis.analyzers import analyze_frontier

        analysis_root = f"{config.analysis_dir}/{config.run_name}/frontier"
        payload["analysis"] = analyze_frontier(eval_path, analysis_root)
        write_json(eval_path, payload)
    return {
        **comparison,
        "eval": eval_summary,
        "eval_path": eval_path,
        "passed": eval_summary["passed"],
    }


def maybe_run_frontier_eval(config: FrameworkConfig) -> dict[str, Any] | None:
    if not config.frontier_auto_compare_in_pipeline:
        return None
    return evaluate_frontier_vs_local(config)


__all__ = [
    "evaluate_frontier_vs_local",
    "maybe_run_frontier_eval",
    "summarize_frontier_eval_rows",
]
