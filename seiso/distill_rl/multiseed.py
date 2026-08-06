"""Multi-seed aggregation for distill-rl research runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def aggregate_multiseed_runs(run_dirs: list[Path], *, output_dir: Path) -> dict[str, Any]:
    """Aggregate evaluation metrics across seed runs (mean/std)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "evaluation" / "evaluation_summary.json"
        if summary_path.is_file():
            per_seed.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate: dict[str, dict[str, Any]] = {}
    for summary in per_seed:
        for name, metrics in summary.get("checkpoints", {}).items():
            bucket = aggregate.setdefault(name, {})
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                bucket.setdefault(f"{key}_values", []).append(float(value))

    stats: dict[str, dict[str, float]] = {}
    for name, values in aggregate.items():
        stats[name] = {}
        for key, samples in values.items():
            if not key.endswith("_values"):
                continue
            metric = key.removesuffix("_values")
            stats[name][f"{metric}_mean"] = _mean(samples)
            stats[name][f"{metric}_std"] = _std(samples)
            stats[name][f"{metric}_n"] = float(len(samples))

    payload = {"seed_count": len(per_seed), "checkpoints": stats}
    out_path = output_dir / "multiseed_aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["summary_path"] = str(out_path)
    return payload


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var)
