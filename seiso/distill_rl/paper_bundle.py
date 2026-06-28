"""Research paper bundle for distill-rl runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def create_paper_bundle(
    *,
    output_root: Path,
    run_name: str,
    config: dict[str, Any],
    stage_results: dict[str, Any],
    evaluation: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> dict[str, str]:
    bundle_dir = output_root / "paper_bundles" / run_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metrics = _flatten_metrics(evaluation)
    manifest_path = bundle_dir / "manifest.json"
    metrics_json_path = bundle_dir / "metrics_summary.json"
    metrics_csv_path = bundle_dir / "metrics_summary.csv"
    appendix_path = bundle_dir / "appendix.md"

    bundle_manifest = {
        "pipeline": "distill_rl",
        "run_name": run_name,
        "config": config,
        "stage_results": stage_results,
        "environment": (manifest or {}).get("environment", {}),
        "config_fingerprint": (manifest or {}).get("config_fingerprint"),
    }
    manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8"
    )
    metrics_json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_metrics_csv(metrics_csv_path, metrics)
    appendix_path.write_text(
        _appendix_markdown(bundle_dir, stage_results, metrics), encoding="utf-8"
    )

    return {
        "paper_bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "metrics_summary_json": str(metrics_json_path),
        "metrics_summary_csv": str(metrics_csv_path),
        "appendix": str(appendix_path),
    }


def _flatten_metrics(evaluation: dict[str, Any] | None) -> dict[str, float]:
    if not evaluation:
        return {}
    flat: dict[str, float] = {}
    for name, metrics in evaluation.get("checkpoints", {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key in {"model_dir", "samples"}:
                continue
            if isinstance(value, (int, float)):
                flat[f"{name}.{key}"] = float(value)
    return flat


def _write_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])


def _appendix_markdown(
    bundle_dir: Path,
    stage_results: dict[str, Any],
    metrics: dict[str, float],
) -> str:
    lines = [
        "# Distill-RL appendix",
        "",
        f"- Bundle directory: `{bundle_dir}`",
        "",
        "## Stage artifacts",
    ]
    for stage, path in stage_results.items():
        lines.append(f"- {stage}: `{path}`")
    lines.extend(["", "## Metrics"])
    for key, value in sorted(metrics.items()):
        lines.append(
            f"- {key}: {value:.6g}" if isinstance(value, float) else f"- {key}: {value}"
        )
    return "\n".join(lines) + "\n"
