"""Pipeline for continuous RL over a streaming task sequence."""

from __future__ import annotations

from pathlib import Path

from adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from adaptive_quant.continuous_learning import ContinuousLearningLoop
from adaptive_quant.logging_utils import md_table, write_json, write_text_file
from adaptive_quant.pipeline.research_contract import build_research_contract
from adaptive_quant.pipeline.vcs import git_commit_hash
from adaptive_quant.research_pipeline import maybe_save_final_checkpoint, write_training_history
from adaptive_quant.security_audit import build_security_audit_record
from adaptive_quant.security_bypass import enforce_security_bypass_policy
from adaptive_quant.trainer import build_trainer


def run_continuous_pipeline(
    config: FrameworkConfig,
    *,
    max_tasks: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    if not config.continuous_learning_enabled:
        raise ValueError("continuous_learning_enabled must be true for the continuous pipeline")

    summary_path = config.summary_path()
    trainer = build_trainer(config)
    git_commit = git_commit_hash()
    loop: ContinuousLearningLoop | None = None
    pipeline_error: Exception | None = None
    continuous_summary: dict[str, object] = {}
    eval_summary: dict[str, object] = {}
    history_path: str | None = None
    checkpoint_path: str | None = None
    report_path: str | None = None

    enforce_security_bypass_policy(context="continuous learning pipeline")

    try:
        loop = ContinuousLearningLoop(config, trainer=trainer)
        continuous_summary = loop.run(max_tasks=max_tasks)
        eval_summary = trainer.evaluate()
        history_path = write_training_history(config, trainer)
        checkpoint_path = maybe_save_final_checkpoint(config, trainer)
        report_path = _write_continuous_report(
            config,
            git_commit=git_commit,
            summary_path=summary_path,
            continuous_summary=continuous_summary,
            eval_summary=eval_summary,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        pipeline_error = exc
        from adaptive_quant.research_pipeline import write_pipeline_failure_artifact

        write_pipeline_failure_artifact(config, exc)
    finally:
        if loop is not None:
            loop.close()
        trainer.close()

    if pipeline_error is not None:
        raise pipeline_error

    summary = {
        "config": config_to_flat_dict(config),
        "git_commit": git_commit,
        "research": build_research_contract(
            config,
            git_commit=git_commit,
            pipeline="continuous_learning",
            phases=["continuous_stream", "evaluate", "analysis", "report"],
        ),
        "security_audit": build_security_audit_record(
            config,
            cli_startup_overrides=cli_startup_overrides,
        ),
        "continuous": continuous_summary,
        "evaluation": eval_summary,
        "artifacts": {
            "training_history": history_path,
            "final_checkpoint": checkpoint_path,
            "continuous_detail": config.continuous_summary_path(),
            "continuous_telemetry": config.continuous_telemetry_path(),
            "report": report_path,
        },
    }
    from adaptive_quant.pipeline.output_summary import build_research_artifact_index

    summary["artifact_index"] = build_research_artifact_index(config, summary["artifacts"])
    write_json(summary_path, summary)
    return summary


def run_continuous_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    max_tasks: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
    footer_mode: str = "full",
) -> dict[str, object]:
    from adaptive_quant.run_footer import print_continuous_footer

    summary = run_continuous_pipeline(
        config,
        max_tasks=max_tasks,
        cli_startup_overrides=cli_startup_overrides,
    )
    print_continuous_footer(config, summary, mode=footer_mode)
    return summary


def _write_continuous_report(
    config: FrameworkConfig,
    *,
    git_commit: str | None,
    summary_path: str,
    continuous_summary: dict[str, object],
    eval_summary: dict[str, object],
    history_path: str | None,
    checkpoint_path: str | None,
) -> str | None:
    if not config.write_research_report:
        return None

    report_path = config.report_path()
    target = Path(report_path)
    continuous_rows = [
        [key, str(continuous_summary.get(key))]
        for key in (
            "tasks",
            "mean_reward",
            "total_updates",
            "total_rollbacks",
            "replay_size",
            "mean_eval_reward",
        )
        if key in continuous_summary
    ]
    eval_rows = [
        [key, str(eval_summary.get(key))]
        for key in (
            "mean_reward",
            "mean_latency_ms",
            "mean_throughput_tps",
            "mean_memory_mb",
            "mean_perplexity",
        )
        if key in eval_summary
    ]
    lines = [
        "# Continuous Learning Report",
        "",
        "## Overview",
        f"- run_name: `{config.run_name}`",
        f"- git_commit: `{git_commit or 'unknown'}`",
        f"- training_backend: `{config.training_backend}`",
        f"- task_stream_mode: `{config.continuous_task_stream_mode}`",
        f"- summary_json: `{summary_path}`",
        f"- continuous_detail_json: `{config.continuous_summary_path()}`",
        f"- telemetry_jsonl: `{config.continuous_telemetry_path()}`",
        f"- history: `{history_path or 'not written'}`",
        f"- checkpoint: `{checkpoint_path or 'not written'}`",
        "",
        "## Continuous stream",
        *(md_table(["metric", "value"], continuous_rows) if continuous_rows else ["_not written_"]),
        "",
        "## Evaluation",
        *(md_table(["metric", "value"], eval_rows) if eval_rows else ["_not written_"]),
    ]
    write_text_file(report_path, "\n".join(lines) + "\n")
    return str(target)


__all__ = ["run_continuous_pipeline", "run_continuous_pipeline_entrypoint"]
