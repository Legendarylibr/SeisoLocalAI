"""Shared markdown report builders for multiseed and sweep aggregate CLIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from adaptive_quant.experiment_aggregate import AggregateStat, default_key_filter
from adaptive_quant.logging_utils import md_table, write_text_file
from adaptive_quant.math_utils import fmt_float

MULTISEED_HEADLINE_KEYS: tuple[str, ...] = (
    "benchmarks.single_vs_multi.generalization_gap_improvement",
    "benchmarks.static_vs_dynamic.quality_variance_delta",
    "benchmarks.discrete_vs_learned.reward_delta",
    "evaluation.mean_reward",
    "evaluation.mean_latency_ms",
    "evaluation.mean_memory_mb",
)

SWEEP_HEADLINE_METRICS: tuple[str, ...] = (
    "evaluation.mean_reward",
    "evaluation.mean_latency_ms",
    "evaluation.mean_throughput_tps",
    "evaluation.mean_memory_mb",
    "benchmarks.single_vs_multi.generalization_gap_improvement",
)


def report_title(run_name: str, kind: str) -> str:
    return f"# {run_name} ({kind})"


def overview_section(bullets: Sequence[str]) -> list[str]:
    lines = ["## Overview"]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.append("")
    return lines


def notes_section(notes: Sequence[str]) -> list[str]:
    lines = ["## Notes"]
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    return lines


def headline_aggregate_section(
    aggregated: dict[str, AggregateStat],
    keys: Sequence[str],
    *,
    title: str = "## Headline aggregates (mean ± std)",
) -> list[str] | None:
    rows: list[list[object]] = []
    for key in keys:
        stat = aggregated.get(key)
        if stat is None:
            continue
        rows.append([key, fmt_float(stat.mean), fmt_float(stat.std), str(stat.n)])
    if not rows:
        return None
    return [
        title,
        "\n".join(md_table(["metric", "mean", "std", "n"], rows)),
        "",
    ]


def filtered_aggregate_section(
    aggregated: dict[str, AggregateStat],
    *,
    title: str = "## Aggregate metrics (filtered)",
) -> list[str]:
    filtered = {k: v for k, v in aggregated.items() if default_key_filter(k)}
    rows: list[list[object]] = [
        [
            key,
            fmt_float(stat.mean),
            fmt_float(stat.std),
            fmt_float(stat.ci95_low),
            fmt_float(stat.ci95_high),
            str(stat.n),
        ]
        for key, stat in filtered.items()
    ]
    return [
        title,
        "\n".join(md_table(["metric", "mean", "std", "ci95_low", "ci95_high", "n"], rows)),
        "",
    ]


def headline_value_section(
    metrics: Sequence[str],
    value_for: Callable[[str], float | None],
    *,
    title: str = "## Headline metrics",
    preamble: Sequence[str] | None = None,
) -> list[str] | None:
    rows: list[list[object]] = []
    for metric in metrics:
        value = value_for(metric)
        if value is not None:
            rows.append([metric, fmt_float(value)])
    if not rows:
        return None
    lines = [title]
    if preamble:
        lines.extend(preamble)
    lines.extend(
        [
            "\n".join(md_table(["metric", "value"], rows)),
            "",
        ]
    )
    return lines


def markdown_table_section(
    title: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> list[str]:
    return [
        title,
        "\n".join(md_table(list(headers), [list(row) for row in rows])),
        "",
    ]


def write_markdown_report(output_path: str, lines: Iterable[str]) -> None:
    write_text_file(output_path, "\n".join(lines) + "\n")


def build_multiseed_report(
    *,
    run_name: str,
    seeds: list[int],
    per_seed_paths: list[str],
    aggregated: dict[str, AggregateStat],
    output_path: str,
    output_json_path: str,
) -> None:
    lines: list[str] = [
        report_title(run_name, "multi-seed"),
        "",
        *overview_section(
            [
                f"seeds: `{seeds}`",
                f"per-seed summaries: `{len(per_seed_paths)}`",
                f"aggregate JSON: `{output_json_path}`",
            ]
        ),
    ]

    headline = headline_aggregate_section(aggregated, MULTISEED_HEADLINE_KEYS)
    if headline is not None:
        lines.extend(headline)

    per_seed_rows: list[list[object]] = [
        [str(seed), f"`{summary_path}`"]
        for seed, summary_path in zip(seeds, per_seed_paths, strict=True)
    ]
    lines.extend(filtered_aggregate_section(aggregated))
    lines.extend(
        markdown_table_section("## Per-seed artifacts", ["seed", "summary"], per_seed_rows)
    )
    lines.extend(
        notes_section(
            [
                "These statistics summarize the metrics produced by the pipeline (simulator by default unless you switch backends in the preset config).",
                "For deeper inspection, open a per-seed `outputs/reports/*_report.md` and the per-seed figures under `outputs/analysis/<run_name>/...`.",
            ]
        )
    )
    write_markdown_report(output_path, lines)


def build_sweep_report(
    *,
    run_name: str,
    objective: str,
    direction: str,
    ranked_results: list[Any],
    output_path: str,
    output_json_path: str,
    output_csv_path: str,
    seeds: list[int] | None,
    runs_skipped: int,
    objective_display: Callable[[Any], str],
    extract_metric_fn: Callable[[dict[str, Any], str], float | None],
) -> None:
    leaderboard_rows: list[list[object]] = []
    for rank, result in enumerate(ranked_results, start=1):
        override_bits = ", ".join(
            f"{key}={value!r}" for key, value in sorted(result.plan.overrides.items())
        )
        leaderboard_rows.append(
            [
                str(rank),
                str(result.plan.trial_id),
                result.plan.run_name_suffix,
                objective_display(result),
                override_bits or "(defaults)",
                f"`{result.summary_path}`",
            ]
        )

    lines: list[str] = [
        report_title(run_name, "hyperparameter sweep"),
        "",
        *overview_section(
            [
                f"objective: `{objective}` ({direction})",
                f"trial settings: `{len(ranked_results)}`",
                f"seeds: `{seeds if seeds else 'single run per setting'}`",
                f"resumed/skipped pipeline runs: `{runs_skipped}`",
                f"aggregate JSON: `{output_json_path}`",
                f"leaderboard CSV: `{output_csv_path}`",
            ]
        ),
        *markdown_table_section(
            "## Leaderboard",
            ["rank", "trial_id", "suffix", "objective", "overrides", "summary"],
            leaderboard_rows,
        ),
    ]

    best = ranked_results[0] if ranked_results else None
    if best is not None:
        flat = extract_metric_fn(best.summary, objective)
        headline = headline_value_section(
            SWEEP_HEADLINE_METRICS,
            lambda metric: extract_metric_fn(best.summary, metric),
            title="## Best trial headline metrics",
            preamble=[
                f"- trial: `#{best.plan.trial_id}` ({best.plan.run_name_suffix})",
                f"- ranking objective: `{objective}` = {objective_display(best)}",
            ],
        )
        if headline is not None:
            lines.extend(headline)
        elif flat is not None:
            lines.extend(
                [
                    "## Best trial",
                    f"- trial: `#{best.plan.trial_id}` ({best.plan.run_name_suffix})",
                    f"- `{objective}` = {objective_display(best)}",
                    "",
                ]
            )

    lines.extend(
        notes_section(
            [
                "Trials are ranked by the mean objective across seeds when `--seeds` / sweep `seeds` is set.",
                "Use `--resume` to skip pipeline runs whose summary JSON already exists.",
                "Open per-trial `outputs/reports/*_report.md` files for full benchmark and analysis artifacts.",
            ]
        )
    )
    write_markdown_report(output_path, lines)
