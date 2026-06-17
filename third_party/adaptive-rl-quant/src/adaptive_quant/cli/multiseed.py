from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from adaptive_quant.cli.aggregate_reports import build_multiseed_report
from adaptive_quant.cli.presets import apply_short_run_episodes, select_dense_moe_preset
from adaptive_quant.experiment_aggregate import (
    aggregate_numeric_maps,
    default_key_filter,
    flatten_numeric,
)
from adaptive_quant.logging_utils import write_json
from adaptive_quant.math_utils import parse_seed_list
from adaptive_quant.paper_bundle import create_multiseed_paper_bundle
from adaptive_quant.pipeline.output_summary import experiment_config_summary
from adaptive_quant.pipeline.research_contract import EVIDENCE_MULTISEED, build_research_contract
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.research_pipeline import run_pipeline_entrypoint


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a preset across multiple seeds and aggregate results."
    )
    parser.add_argument(
        "--preset", choices=["dense", "moe"], default="dense", help="Which config preset to run."
    )
    parser.add_argument("--seeds", default="13,17,23,29,31", help='Seeds as "a,b,c" or "a-b".')
    parser.add_argument(
        "--run-name",
        default=None,
        help="Base run name for the multiseed aggregate (defaults to preset run_name).",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        type=int,
        help="Override training_episodes (useful for fast smoke tests).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress end-of-run CLI banners (e.g. unit tests)."
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Override output root for all per-seed and aggregate artifacts.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from adaptive_quant.cli.common import enforce_cli_startup, validate_cli_output_dir

    enforce_cli_startup(context="multiseed CLI")
    validate_cli_output_dir("outputs-dir", args.outputs_dir)

    seeds = parse_seed_list(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided.")

    base_config = select_dense_moe_preset(args.preset)
    if args.outputs_dir:
        base_config = base_config.with_output_root(args.outputs_dir)
    if args.episodes is not None:
        base_config = apply_short_run_episodes(base_config, args.episodes)
    base_run_name = str(args.run_name or base_config.run_name)
    multiseed_run_name = f"{base_run_name}_multiseed"

    per_seed_paths: list[str] = []
    per_seed_numeric: list[dict[str, float]] = []

    for seed in seeds:
        seed_run_name = f"{base_run_name}_seed{seed}"
        config = base_config.clone(seed=seed, run_name=seed_run_name)
        summary = run_pipeline_entrypoint(config, footer_mode="none" if args.quiet else "minimal")
        per_seed_paths.append(config.summary_path())

        numeric = flatten_numeric(summary)
        per_seed_numeric.append(numeric)

    aggregated = aggregate_numeric_maps(per_seed_numeric)

    output_json_path = f"{base_config.benchmark_dir}/{multiseed_run_name}_summary.json"
    output_md_path = f"{base_config.report_dir}/{multiseed_run_name}_report.md"

    aggregate_payload: dict[str, Any] = {
        "run_name": multiseed_run_name,
        "preset": args.preset,
        "base_run_name": base_run_name,
        "config": experiment_config_summary(base_config),
        "git_commit": git_commit_hash(),
        "research": build_research_contract(
            base_config,
            git_commit=git_commit_hash(),
            pipeline="multiseed_aggregate",
            evidence_level=EVIDENCE_MULTISEED,
            phases=["per_seed_runs", "aggregate_stats", "report", "paper_bundle"],
        ),
        "seeds": seeds,
        "per_seed": [
            {"seed": seed, "run_name": f"{base_run_name}_seed{seed}", "summary_path": path}
            for seed, path in zip(seeds, per_seed_paths, strict=True)
        ],
        "artifacts": {
            "per_seed_summaries": per_seed_paths,
            "report": output_md_path,
        },
        "aggregates": {k: v.to_dict() for k, v in aggregated.items() if default_key_filter(k)},
    }
    paper_bundle = create_multiseed_paper_bundle(
        config=base_config,
        run_name=multiseed_run_name,
        aggregate_payload=aggregate_payload,
        aggregate_stats=aggregate_payload["aggregates"],
        report_path=output_md_path,
    )
    aggregate_payload["artifacts"]["paper_bundle"] = paper_bundle
    write_json(output_json_path, aggregate_payload)
    build_multiseed_report(
        run_name=multiseed_run_name,
        seeds=seeds,
        per_seed_paths=per_seed_paths,
        aggregated=aggregated,
        output_path=output_md_path,
        output_json_path=output_json_path,
    )

    if not args.quiet:
        from adaptive_quant.run_footer import print_multiseed_footer

        print_multiseed_footer(
            multiseed_run_name=multiseed_run_name,
            seeds=seeds,
            aggregate_json=output_json_path,
            report_md=output_md_path,
            per_seed_summary_paths=per_seed_paths,
        )


if __name__ == "__main__":
    main()
