"""CLI: frontier model reference scoring and local comparison."""

from __future__ import annotations

import argparse
import sys

from adaptive_quant.cli.common import (
    add_config_file_argument,
    add_config_override_arguments,
    load_config_or_fallback,
    resolve_startup_config,
)
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.frontier_comparison import compare_frontier_to_local, score_frontier_reference
from adaptive_quant.presets.baseline import CONFIG


def _frontier_config(base: FrameworkConfig) -> FrameworkConfig:
    return base.clone(frontier_enabled=True)


def _score_command(args: argparse.Namespace) -> None:
    from adaptive_quant.configuration.validation import validate_cli_path_argument

    if args.output:
        validate_cli_path_argument("output", args.output)
    cfg, _ = resolve_startup_config(
        _frontier_config(load_config_or_fallback(args.config, CONFIG)),
        args,
    )
    summary = score_frontier_reference(cfg, output_path=args.output)
    print(
        f"[frontier] scored {summary['prompt_count']} prompts → {summary['path']}", file=sys.stderr
    )


def _compare_command(args: argparse.Namespace) -> None:
    from adaptive_quant.configuration.validation import validate_cli_path_argument

    if args.reference:
        validate_cli_path_argument("reference", args.reference)
    cfg, _ = resolve_startup_config(
        _frontier_config(load_config_or_fallback(args.config, CONFIG)),
        args,
    )
    summary = compare_frontier_to_local(
        cfg,
        reference_path=args.reference,
        refresh_reference=bool(args.refresh_reference),
    )
    print(
        f"[frontier] compared {summary['prompts_compared']} prompts "
        f"(mean overlap={summary.get('mean_reference_overlap', 0):.3f}) "
        f"→ {summary['comparison_path']}",
        file=sys.stderr,
    )


def _eval_command(args: argparse.Namespace) -> None:
    from adaptive_quant.configuration.validation import validate_cli_path_argument
    from adaptive_quant.frontier_eval import evaluate_frontier_vs_local

    if args.reference:
        validate_cli_path_argument("reference", args.reference)
    cfg, _ = resolve_startup_config(
        _frontier_config(load_config_or_fallback(args.config, CONFIG)),
        args,
    )
    summary = evaluate_frontier_vs_local(
        cfg,
        reference_path=args.reference,
        refresh_reference=bool(args.refresh_reference),
    )
    eval_block = summary.get("eval", {})
    status = "PASS" if eval_block.get("passed") else "FAIL"
    print(
        f"[frontier eval] {status}  prompts={eval_block.get('prompts_evaluated', 0)}  "
        f"mean_overlap={eval_block.get('mean_reference_overlap', 0):.3f}  "
        f"local_coverage={eval_block.get('local_coverage_rate', 0):.3f}  "
        f"→ {summary.get('eval_path')}",
        file=sys.stderr,
    )
    if args.require_pass and not eval_block.get("passed"):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Frontier model reference workflow: score hosted model outputs and compare them "
            "against local llama.cpp generations."
        )
    )
    add_config_file_argument(parser)
    add_config_override_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser(
        "score",
        help="Call the configured frontier API for each prompt and write a reference JSON file.",
    )
    score_parser.add_argument(
        "--output",
        default=None,
        help="Override output path (defaults to config.frontier_reference_path()).",
    )
    score_parser.set_defaults(handler=_score_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare frontier reference outputs to local llama.cpp text (or a local sidecar).",
    )
    compare_parser.add_argument(
        "--reference",
        default=None,
        help="Existing frontier reference JSON (defaults to config.frontier_reference_path()).",
    )
    compare_parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="Re-score frontier outputs before comparing.",
    )
    compare_parser.set_defaults(handler=_compare_command)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run frontier vs local comparison and write structured eval metrics with pass/fail.",
    )
    eval_parser.add_argument(
        "--reference",
        default=None,
        help="Existing frontier reference JSON (defaults to config.frontier_reference_path()).",
    )
    eval_parser.add_argument(
        "--refresh-reference",
        action="store_true",
        help="Re-score frontier outputs before evaluating.",
    )
    eval_parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with code 1 when eval thresholds are not met.",
    )
    eval_parser.set_defaults(handler=_eval_command)

    args = parser.parse_args()
    from adaptive_quant.cli.common import enforce_cli_startup

    enforce_cli_startup(context="frontier CLI")
    args.handler(args)


if __name__ == "__main__":
    main()
