"""Auto hyperparameter sweep for integrated RL quant runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from seiso.bundled.config_builder import resolve_config_file_path
from seiso.rl_quant.bootstrap import bundle_root

# Compact grids tuned for smoke vs research presets.
_DEFAULT_SWEEP_GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "minimal": {
        "learning_rate": (0.025, 0.035),
    },
    "smoke": {
        "learning_rate": (0.025, 0.035),
    },
    "reproducible": {
        "learning_rate": (0.02, 0.035),
        "value_learning_rate": (0.015, 0.025),
    },
    "post_train": {
        "learning_rate": (0.02, 0.035),
        "value_learning_rate": (0.015, 0.025),
        "reward_weights.beta_throughput": (0.04, 0.08),
    },
    "posttrain": {
        "learning_rate": (0.02, 0.035),
        "value_learning_rate": (0.015, 0.025),
        "reward_weights.beta_throughput": (0.04, 0.08),
    },
}

_FALLBACK_GRID: dict[str, tuple[Any, ...]] = {
    "learning_rate": (0.02, 0.035),
    "value_learning_rate": (0.015, 0.025),
}


def auto_sweep_enabled(payload: dict[str, Any]) -> bool:
    """Return True when this job should run an automatic hyperparameter sweep first."""
    if payload.get("auto_sweep") is False or payload.get("sweep") is False:
        return False
    if payload.get("auto_sweep") is True or payload.get("sweep") is True:
        return True
    return payload.get("auto_sweep", True)


def default_sweep_grid(payload: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Resolve the parameter grid for an auto sweep."""
    raw_grid = payload.get("sweep_grid")
    if isinstance(raw_grid, dict) and raw_grid:
        return {
            str(key): tuple(values)
            for key, values in raw_grid.items()
            if isinstance(values, (list, tuple)) and values
        }

    vary = payload.get("vary") or payload.get("sweep_vary")
    if isinstance(vary, list) and vary:
        from seiso.adaptive_quant.sweep import parse_vary_argument

        grid: dict[str, tuple[Any, ...]] = {}
        for item in vary:
            key, values = parse_vary_argument(str(item))
            grid[key] = values
        return grid

    preset = str(payload.get("preset", "reproducible")).lower().replace("-", "_")
    return dict(_DEFAULT_SWEEP_GRIDS.get(preset, _FALLBACK_GRID))


def sweep_episode_budget(
    payload: dict[str, Any],
    *,
    training_episodes: int,
    evaluation_episodes: int,
) -> tuple[int, int]:
    """Training/eval episode counts used during sweep trials (shorter than the final run)."""
    if payload.get("sweep_training_episodes") is not None:
        train = int(payload["sweep_training_episodes"])
    elif payload.get("benchmark_training_episodes") is not None:
        train = int(payload["benchmark_training_episodes"])
    else:
        train = max(8, training_episodes // 4)

    if payload.get("sweep_evaluation_episodes") is not None:
        eval_eps = int(payload["sweep_evaluation_episodes"])
    elif payload.get("benchmark_evaluation_episodes") is not None:
        eval_eps = int(payload["benchmark_evaluation_episodes"])
    else:
        eval_eps = max(2, evaluation_episodes // 4)

    return train, eval_eps


def resolve_sweep_spec(payload: dict[str, Any], *, preset: str) -> Any:
    """Build a SweepSpec from payload overrides or preset defaults."""
    from seiso.adaptive_quant.sweep import DEFAULT_OBJECTIVE, SweepSpec, load_sweep_file

    sweep_config = payload.get("sweep_config")
    if sweep_config:
        path = resolve_config_file_path(str(sweep_config), bundle_root=bundle_root())
        if path is None:
            raise ValueError(f"Sweep config file not found: {sweep_config}")
        spec, _ = load_sweep_file(path)
        return spec

    objective = str(payload.get("sweep_objective", DEFAULT_OBJECTIVE))
    direction_raw = str(payload.get("sweep_direction", "maximize")).strip().lower()
    if direction_raw not in {"maximize", "minimize"}:
        raise ValueError("sweep_direction must be 'maximize' or 'minimize'")

    seeds_raw = payload.get("sweep_seeds")
    seeds: tuple[int, ...] | None = None
    if isinstance(seeds_raw, list) and seeds_raw:
        seeds = tuple(int(value) for value in seeds_raw)
    elif isinstance(seeds_raw, str) and seeds_raw.strip():
        from seiso.adaptive_quant.sweep import parse_seed_list

        parsed = parse_seed_list(seeds_raw)
        seeds = tuple(parsed) if parsed else None

    return SweepSpec(
        objective=objective,
        direction=direction_raw,  # type: ignore[arg-type]
        seed=(
            int(payload["sweep_seed"])
            if payload.get("sweep_seed") is not None
            else None
        ),
        seeds=seeds,
        grid=default_sweep_grid({**payload, "preset": preset}),
        trials=None,
        base_config_path=None,
    )


def run_auto_hyperparameter_sweep(
    config: Any,
    *,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run sweep trials, rank them, and return the best override mapping."""
    from seiso.adaptive_quant.cli.aggregate_reports import build_sweep_report
    from seiso.adaptive_quant.cli.startup_overrides import (
        apply_startup_overrides,
        enforce_privileged_override_policy,
    )
    from seiso.adaptive_quant.experiment_aggregate import extract_metric
    from seiso.adaptive_quant.logging_utils import write_json
    from seiso.adaptive_quant.paper_bundle import create_multiseed_paper_bundle
    from seiso.adaptive_quant.pipeline.output_summary import experiment_config_summary
    from seiso.adaptive_quant.pipeline.research_contract import (
        EVIDENCE_SWEEP,
        build_research_contract,
    )
    from seiso.adaptive_quant.pipeline.vcs import git_commit_hash
    from seiso.adaptive_quant.research_pipeline import run_pipeline_entrypoint
    from seiso.adaptive_quant.sweep import (
        SweepTrialResult,
        build_trial_plans,
        rank_trials,
        trial_run_name,
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    preset = str(payload.get("preset", "reproducible"))
    spec = resolve_sweep_spec(payload, preset=preset)
    plans = build_trial_plans(grid=spec.grid, explicit_trials=list(spec.trials or ()))

    sweep_train, sweep_eval = sweep_episode_budget(
        payload,
        training_episodes=int(config.training_episodes),
        evaluation_episodes=int(config.evaluation_episodes),
    )
    base_run_name = str(config.run_name)
    sweep_run_name = f"{base_run_name}_sweep"
    seeds = list(spec.seeds) if spec.seeds else None

    _log(
        f"Auto hyperparameter sweep: {len(plans)} trial(s)"
        + (f" × {len(seeds)} seed(s)" if seeds else "")
        + f" @ train={sweep_train} eval={sweep_eval}"
    )
    _log(f"Sweep objective: {spec.objective} ({spec.direction})")

    sweep_base = config.clone(
        run_name=sweep_run_name,
        training_episodes=sweep_train,
        evaluation_episodes=sweep_eval,
    )

    results: list[SweepTrialResult] = []

    def _apply_trial_overrides(base: Any, overrides: dict[str, Any]) -> Any:
        enforce_privileged_override_policy(overrides)
        return cast(Any, apply_startup_overrides(base, overrides))

    def _execute_trial(plan: Any) -> SweepTrialResult:
        trial_config = _apply_trial_overrides(sweep_base, plan.overrides)
        runs_skipped = 0

        if seeds:
            from seiso.adaptive_quant.sweep import (
                SweepSeedResult,
                aggregate_objective_values,
            )

            seed_results: list[SweepSeedResult] = []
            for seed in seeds:
                run_name = trial_run_name(base_run_name, plan, seed=seed)
                trial = trial_config.clone(run_name=run_name, seed=seed)
                summary = run_pipeline_entrypoint(trial, footer_mode="none")
                seed_results.append(
                    SweepSeedResult(
                        seed=seed,
                        summary=summary,
                        summary_path=trial.summary_path(),
                        objective_value=extract_metric(summary, spec.objective),
                    )
                )
            objective_values = [result.objective_value for result in seed_results]
            objective_mean, objective_std, objective_n = aggregate_objective_values(
                objective_values
            )
            representative = max(
                seed_results,
                key=lambda result: (
                    result.objective_value is None,
                    -(result.objective_value or float("-inf")),
                ),
            )
            return SweepTrialResult(
                plan=plan,
                summary=representative.summary,
                summary_path=representative.summary_path,
                objective_value=objective_mean,
                objective_std=objective_std,
                objective_n=objective_n,
                seed_results=tuple(seed_results),
                runs_skipped=runs_skipped,
            )

        run_name = trial_run_name(base_run_name, plan)
        trial = trial_config.clone(run_name=run_name)
        summary = run_pipeline_entrypoint(trial, footer_mode="none")
        return SweepTrialResult(
            plan=plan,
            summary=summary,
            summary_path=trial.summary_path(),
            objective_value=extract_metric(summary, spec.objective),
        )

    for plan in plans:
        _log(f"Sweep trial {plan.trial_id}/{len(plans)}: {plan.run_name_suffix}")
        results.append(_execute_trial(plan))

    ranked = rank_trials(results, objective=spec.objective, direction=spec.direction)
    best = ranked[0] if ranked else None

    output_json_path = f"{config.benchmark_dir}/{sweep_run_name}_summary.json"
    output_md_path = f"{config.report_dir}/{sweep_run_name}_report.md"
    output_csv_path = f"{config.report_dir}/{sweep_run_name}_leaderboard.csv"

    aggregate_payload: dict[str, Any] = {
        "run_name": sweep_run_name,
        "base_run_name": base_run_name,
        "objective": spec.objective,
        "direction": spec.direction,
        "config": experiment_config_summary(sweep_base),
        "git_commit": git_commit_hash(),
        "research": build_research_contract(
            sweep_base,
            git_commit=git_commit_hash(),
            pipeline="sweep_aggregate",
            evidence_level=EVIDENCE_SWEEP,
            phases=["trial_runs", "ranking", "report", "paper_bundle"],
        ),
        "sweep": {
            "auto": True,
            "grid": spec.grid,
            "sweep_training_episodes": sweep_train,
            "sweep_evaluation_episodes": sweep_eval,
            **({"seeds": seeds} if seeds else {}),
        },
        "leaderboard": [
            {
                "rank": rank,
                "trial_id": result.plan.trial_id,
                "run_name_suffix": result.plan.run_name_suffix,
                "objective_value": result.objective_value,
                "objective_std": result.objective_std,
                "objective_n": result.objective_n,
                "summary_path": result.summary_path,
                "overrides": result.plan.overrides,
            }
            for rank, result in enumerate(ranked, start=1)
        ],
        "artifacts": {
            "report": output_md_path,
            "leaderboard_csv": output_csv_path,
            "aggregate_json": output_json_path,
        },
    }
    write_json(output_json_path, aggregate_payload)

    def _objective_display(result: SweepTrialResult) -> str:
        if result.objective_value is None:
            return "n/a"
        if result.objective_n > 1 and result.objective_std is not None:
            return f"{result.objective_value:.4g} ± {result.objective_std:.4g} (n={result.objective_n})"
        return f"{result.objective_value:.4g}"

    build_sweep_report(
        run_name=sweep_run_name,
        objective=spec.objective,
        direction=spec.direction,
        ranked_results=ranked,
        output_path=output_md_path,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path,
        seeds=seeds,
        runs_skipped=0,
        objective_display=_objective_display,
        extract_metric_fn=extract_metric,
    )
    _write_sweep_leaderboard_csv(
        ranked_results=ranked,
        output_path=output_csv_path,
        objective=spec.objective,
    )

    sweep_stats: dict[str, dict[str, float | int]] = {}
    for result in results:
        if result.objective_value is None:
            continue
        sweep_stats[f"trial_{result.plan.trial_id:03d}.{spec.objective}"] = {
            "mean": result.objective_value,
            "std": float(result.objective_std or 0.0),
            "n": int(result.objective_n),
            "stderr": 0.0,
            "ci95_low": result.objective_value,
            "ci95_high": result.objective_value,
            "effect_size_vs_zero": 0.0,
        }
    paper_bundle = create_multiseed_paper_bundle(
        config=sweep_base,
        run_name=sweep_run_name,
        aggregate_payload=aggregate_payload,
        aggregate_stats=sweep_stats,
        report_path=output_md_path,
    )
    aggregate_payload["artifacts"]["paper_bundle"] = paper_bundle
    write_json(output_json_path, aggregate_payload)

    best_overrides = best.plan.overrides if best is not None else {}
    if best is not None:
        _log(
            f"Sweep best trial #{best.plan.trial_id} ({best.plan.run_name_suffix})"
            f" objective={_objective_display(best)}"
        )
    else:
        _log("Sweep produced no ranked trials; continuing with base config")

    return {
        "sweep_run_name": sweep_run_name,
        "objective": spec.objective,
        "direction": spec.direction,
        "trial_count": len(plans),
        "best_trial_id": best.plan.trial_id if best is not None else None,
        "best_overrides": best_overrides,
        "best_objective_value": best.objective_value if best is not None else None,
        "aggregate_path": output_json_path,
        "report_path": output_md_path,
        "leaderboard_path": output_csv_path,
        "leaderboard": aggregate_payload["leaderboard"],
    }


def apply_best_sweep_overrides(config: Any, best_overrides: dict[str, Any]) -> Any:
    """Merge ranked sweep overrides onto the full-episode config."""
    if not best_overrides:
        return config
    from seiso.adaptive_quant.cli.startup_overrides import (
        apply_startup_overrides,
        enforce_privileged_override_policy,
    )

    enforce_privileged_override_policy(best_overrides)
    return cast(Any, apply_startup_overrides(config, best_overrides))


def _write_sweep_leaderboard_csv(
    *,
    ranked_results: list[Any],
    output_path: str,
    objective: str,
) -> None:
    import csv

    rows: list[dict[str, str]] = []
    for rank, result in enumerate(ranked_results, start=1):
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
                "objective_std": (
                    "" if result.objective_std is None else str(result.objective_std)
                ),
                "objective_n": str(result.objective_n),
                "summary_path": result.summary_path,
            }
        )
    fieldnames = [
        "rank",
        "trial_id",
        "suffix",
        "objective",
        "objective_value",
        "objective_std",
        "objective_n",
        "summary_path",
    ]
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
