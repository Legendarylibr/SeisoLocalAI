"""Pipeline for continuous RL over a streaming task sequence."""

from __future__ import annotations

from seiso.adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from seiso.adaptive_quant.continuous_learning import ContinuousLearningLoop
from seiso.adaptive_quant.logging_utils import write_json
from seiso.adaptive_quant.pipeline.research_contract import build_research_contract
from seiso.adaptive_quant.pipeline.vcs import git_commit_hash
from seiso.adaptive_quant.research_pipeline import (
    maybe_save_final_checkpoint,
    write_training_history,
)
from seiso.adaptive_quant.security_audit import build_security_audit_record
from seiso.adaptive_quant.security_bypass import enforce_security_bypass_policy
from seiso.adaptive_quant.trainer import build_trainer


def run_continuous_pipeline(
    config: FrameworkConfig,
    *,
    max_tasks: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    if not config.continuous_learning_enabled:
        raise ValueError(
            "continuous_learning_enabled must be true for the continuous pipeline"
        )

    summary_path = config.summary_path()
    trainer = build_trainer(config)
    git_commit = git_commit_hash()
    loop: ContinuousLearningLoop | None = None
    pipeline_error: Exception | None = None
    continuous_summary: dict[str, object] = {}
    eval_summary: dict[str, object] = {}
    history_path: str | None = None
    checkpoint_path: str | None = None

    enforce_security_bypass_policy(context="continuous learning pipeline")

    try:
        loop = ContinuousLearningLoop(config, trainer=trainer)
        continuous_summary = loop.run(max_tasks=max_tasks)
        eval_summary = trainer.evaluate()
        history_path = write_training_history(config, trainer)
        checkpoint_path = maybe_save_final_checkpoint(config, trainer)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        pipeline_error = exc
        from seiso.adaptive_quant.research_pipeline import (
            write_pipeline_failure_artifact,
        )

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
            phases=["continuous_stream", "evaluate", "analysis"],
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
        },
    }
    from seiso.adaptive_quant.pipeline.output_summary import (
        build_research_artifact_index,
    )

    summary["artifact_index"] = build_research_artifact_index(
        config, summary["artifacts"]
    )
    write_json(summary_path, summary)
    return summary


def run_continuous_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    max_tasks: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
    footer_mode: str = "full",
) -> dict[str, object]:
    from seiso.adaptive_quant.run_footer import print_continuous_footer

    summary = run_continuous_pipeline(
        config,
        max_tasks=max_tasks,
        cli_startup_overrides=cli_startup_overrides,
    )
    print_continuous_footer(config, summary, mode=footer_mode)
    return summary


__all__ = ["run_continuous_pipeline", "run_continuous_pipeline_entrypoint"]
