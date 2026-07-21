"""Auto hyperparameter sweep for distill-RL (DPO alignment) runs."""

from __future__ import annotations

import csv
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from seiso.distill_rl.config import DistillRLConfig

DEFAULT_OBJECTIVE = "checkpoints.dpo.alignment_score"

_DEFAULT_SWEEP_GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "smoke": {
        "dpo_beta": (0.1, 0.2),
        "dpo_learning_rate": (2e-6, 5e-6),
    },
    "reproducible": {
        "dpo_beta": (0.1, 0.2, 0.3),
        "dpo_learning_rate": (1e-6, 2e-6, 5e-6),
    },
    "full": {
        "dpo_beta": (0.1, 0.2, 0.3),
        "dpo_learning_rate": (1e-6, 2e-6, 5e-6),
    },
}

_FALLBACK_GRID: dict[str, tuple[Any, ...]] = {
    "dpo_beta": (0.1, 0.2),
    "dpo_learning_rate": (2e-6, 5e-6),
}


@dataclass(frozen=True)
class SharedStageContext:
    distilled_dir: Path
    stage_results: dict[str, Any]


@dataclass(frozen=True)
class SweepTrialPlan:
    trial_id: int
    overrides: dict[str, Any]
    run_name_suffix: str


@dataclass(frozen=True)
class SweepTrialResult:
    plan: SweepTrialPlan
    evaluation: dict[str, Any]
    objective_value: float | None
    dpo_dir: Path


def auto_sweep_enabled(payload: dict[str, Any]) -> bool:
    if payload.get("auto_sweep") is False or payload.get("sweep") is False:
        return False
    if payload.get("auto_sweep") is True or payload.get("sweep") is True:
        return True
    return bool(payload.get("auto_sweep", True))


def default_sweep_grid(payload: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    raw_grid = payload.get("sweep_grid")
    if isinstance(raw_grid, dict) and raw_grid:
        return {
            str(key): tuple(values)
            for key, values in raw_grid.items()
            if isinstance(values, (list, tuple)) and values
        }

    vary = payload.get("vary") or payload.get("sweep_vary")
    if isinstance(vary, list) and vary:
        grid: dict[str, tuple[Any, ...]] = {}
        for item in vary:
            key, values = _parse_vary_argument(str(item))
            grid[key] = values
        return grid

    preset = str(payload.get("preset", "smoke")).lower().replace("-", "_")
    return dict(_DEFAULT_SWEEP_GRIDS.get(preset, _FALLBACK_GRID))


def sweep_dpo_max_steps(
    config: DistillRLConfig,
    payload: dict[str, Any],
    *,
    train_example_count: int | None = None,
) -> int | None:
    if payload.get("sweep_dpo_max_steps") is not None:
        return int(payload["sweep_dpo_max_steps"])
    if config.dpo_max_steps is not None:
        return max(
            2, min(int(config.dpo_max_steps), max(4, int(config.dpo_max_steps) // 3))
        )
    if train_example_count is None:
        return max(4, int(config.dpo_epochs))
    micro_batches = max(
        1, (train_example_count + config.dpo_batch_size - 1) // config.dpo_batch_size
    )
    optimizer_steps_per_epoch = max(
        1,
        (micro_batches + config.dpo_gradient_accumulation_steps - 1)
        // config.dpo_gradient_accumulation_steps,
    )
    return min(32, max(4, optimizer_steps_per_epoch))


def apply_best_sweep_overrides(
    config: DistillRLConfig,
    best_overrides: dict[str, Any],
) -> DistillRLConfig:
    if not best_overrides:
        return config
    data = config.model_dump()
    data.update(best_overrides)
    return cast(DistillRLConfig, DistillRLConfig.model_validate(data))


def extract_metric(payload: dict[str, Any], objective: str) -> float | None:
    parts = objective.split(".")
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool):
        return float(current)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def run_auto_hyperparameter_sweep(
    config: DistillRLConfig,
    *,
    payload: dict[str, Any],
    shared: SharedStageContext,
    on_log: Callable[[str], None] | None = None,
    run_dpo_fn: Callable[..., Path],
) -> dict[str, Any]:
    """Sweep DPO hyperparameters using shared distill/rollout artifacts."""
    from seiso.distill_rl.evaluate import evaluate_pipeline

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    objective = str(payload.get("sweep_objective", DEFAULT_OBJECTIVE))
    direction = str(payload.get("sweep_direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("sweep_direction must be 'maximize' or 'minimize'")

    grid = default_sweep_grid(payload)
    plans = _build_trial_plans(grid)
    sweep_root = config.output_root / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    train_path = config.preferences_train_path
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Preference train dataset missing for sweep: {train_path}"
        )
    train_example_count = _count_jsonl_rows(train_path)
    sweep_steps = sweep_dpo_max_steps(
        config, payload, train_example_count=train_example_count
    )

    _log(
        f"Auto hyperparameter sweep: {len(plans)} DPO trial(s)"
        + (f" @ dpo_max_steps={sweep_steps}" if sweep_steps is not None else "")
    )
    _log(f"Sweep objective: {objective} ({direction})")

    results: list[SweepTrialResult] = []
    for plan in plans:
        _log(f"Sweep trial {plan.trial_id}/{len(plans)}: {plan.run_name_suffix}")
        trial_config = apply_best_sweep_overrides(config, plan.overrides).model_copy(
            update={
                "dpo_output_dir_override": sweep_root
                / f"trial_{plan.trial_id:03d}_{plan.run_name_suffix}",
                "dpo_max_steps": sweep_steps,
            }
        )
        trial_config.dpo_output_dir.mkdir(parents=True, exist_ok=True)

        dpo_dir = run_dpo_fn(
            trial_config,
            model_dir=shared.distilled_dir,
            preferences_path=train_path,
            on_log=on_log,
        )
        evaluation = evaluate_pipeline(
            output_dir=trial_config.dpo_output_dir / "evaluation",
            checkpoints={"dpo": dpo_dir},
            val_preferences_path=config.preferences_val_path,
            prompt_library_path=config.prompt_library_path,
            eval_max_prompts=max(1, min(config.eval_max_prompts, 8)),
            on_log=on_log,
        )
        objective_value = extract_metric(evaluation, objective)
        results.append(
            SweepTrialResult(
                plan=plan,
                evaluation=evaluation,
                objective_value=objective_value,
                dpo_dir=dpo_dir,
            )
        )

    ranked = _rank_trials(results, direction=direction)
    best = ranked[0] if ranked else None
    sweep_run_name = f"seiso_{config.job_id[:8]}_sweep"
    aggregate_path = sweep_root / f"{sweep_run_name}_summary.json"
    leaderboard_path = sweep_root / f"{sweep_run_name}_leaderboard.csv"

    aggregate_payload = {
        "run_name": sweep_run_name,
        "objective": objective,
        "direction": direction,
        "grid": grid,
        "sweep_dpo_max_steps": sweep_steps,
        "leaderboard": [
            {
                "rank": rank,
                "trial_id": result.plan.trial_id,
                "run_name_suffix": result.plan.run_name_suffix,
                "objective_value": result.objective_value,
                "metrics": _leaderboard_metrics(result.evaluation),
                "overrides": result.plan.overrides,
                "dpo_dir": str(result.dpo_dir),
                "evaluation_path": result.evaluation.get("summary_path"),
            }
            for rank, result in enumerate(ranked, start=1)
        ],
    }
    aggregate_path.write_text(
        json.dumps(aggregate_payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_leaderboard_csv(ranked, leaderboard_path, objective=objective)

    best_overrides = best.plan.overrides if best is not None else {}
    if best is not None:
        _log(
            f"Sweep best trial #{best.plan.trial_id} ({best.plan.run_name_suffix})"
            f" objective={best.objective_value}"
        )
    else:
        _log("Sweep produced no ranked trials; continuing with base config")

    return {
        "sweep_run_name": sweep_run_name,
        "objective": objective,
        "direction": direction,
        "trial_count": len(plans),
        "best_trial_id": best.plan.trial_id if best is not None else None,
        "best_overrides": best_overrides,
        "best_objective_value": best.objective_value if best is not None else None,
        "aggregate_path": str(aggregate_path),
        "leaderboard_path": str(leaderboard_path),
        "leaderboard": aggregate_payload["leaderboard"],
    }


def _parse_vary_argument(raw: str) -> tuple[str, tuple[Any, ...]]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=val1,val2,... got {raw!r}")
    key, values_text = raw.split("=", 1)
    values = [
        _coerce_value(part.strip()) for part in values_text.split(",") if part.strip()
    ]
    if not values:
        raise ValueError(f"No values provided for sweep parameter {key!r}")
    return key.strip(), tuple(values)


def _coerce_value(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _build_trial_plans(grid: dict[str, tuple[Any, ...]]) -> list[SweepTrialPlan]:
    if not grid:
        raise ValueError("Sweep requires a non-empty parameter grid")
    keys = sorted(grid.keys())
    plans: list[SweepTrialPlan] = []
    for trial_id, values in enumerate(
        itertools.product(*(grid[key] for key in keys)), start=1
    ):
        overrides = dict(zip(keys, values, strict=True))
        plans.append(
            SweepTrialPlan(
                trial_id=trial_id,
                overrides=overrides,
                run_name_suffix=_trial_run_suffix(overrides),
            )
        )
    return plans


def _trial_run_suffix(overrides: dict[str, Any], *, max_len: int = 48) -> str:
    parts: list[str] = []
    for key in sorted(overrides):
        value = overrides[key]
        short_key = key.split(".")[-1]
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        text = re.sub(r"[^a-zA-Z0-9]+", "p", text)[:16]
        parts.append(f"{short_key}_{text}")
    slug = "_".join(parts)
    return slug[:max_len] or "default"


def _rank_trials(
    results: list[SweepTrialResult], *, direction: str
) -> list[SweepTrialResult]:
    reverse = direction == "maximize"

    def sort_key(result: SweepTrialResult) -> tuple[int, float]:
        value = result.objective_value
        if value is None:
            return (1, 0.0)
        return (0, -value if reverse else value)

    return sorted(results, key=sort_key)


def _write_leaderboard_csv(
    ranked_results: list[SweepTrialResult],
    output_path: Path,
    *,
    objective: str,
) -> None:
    rows: list[dict[str, str]] = []
    for rank, result in enumerate(ranked_results, start=1):
        metrics = _leaderboard_metrics(result.evaluation)
        rows.append(
            {
                "rank": str(rank),
                "trial_id": str(result.plan.trial_id),
                "suffix": result.plan.run_name_suffix,
                "objective": objective,
                "objective_value": (
                    ""
                    if result.objective_value is None
                    else str(result.objective_value)
                ),
                "val_accuracy": _csv_metric(metrics.get("val_preference_accuracy")),
                "mean_margin": _csv_metric(metrics.get("val_preference_margin_mean")),
                "alignment_score": _csv_metric(metrics.get("alignment_score")),
                "dpo_dir": str(result.dpo_dir),
            }
        )
    fieldnames = [
        "rank",
        "trial_id",
        "suffix",
        "objective",
        "objective_value",
        "val_accuracy",
        "mean_margin",
        "alignment_score",
        "dpo_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _leaderboard_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    checkpoints = evaluation.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return {}
    dpo = checkpoints.get("dpo")
    return dpo if isinstance(dpo, dict) else {}


def _csv_metric(value: Any) -> str:
    return "" if value is None else str(value)
