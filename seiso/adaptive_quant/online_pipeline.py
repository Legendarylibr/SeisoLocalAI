from __future__ import annotations

from seiso.adaptive_quant.configuration import FrameworkConfig, config_to_flat_dict
from seiso.adaptive_quant.logging_utils import write_json
from seiso.adaptive_quant.online_learning import (
    OnlineLearningLoop,
    build_request_stream,
)
from seiso.adaptive_quant.pipeline.output_summary import slim_online_analysis_for_summary
from seiso.adaptive_quant.pipeline.research_contract import build_research_contract
from seiso.adaptive_quant.pipeline.vcs import git_commit_hash
from seiso.adaptive_quant.research_pipeline import (
    maybe_save_final_checkpoint,
    write_training_history,
)
from seiso.adaptive_quant.security_audit import build_security_audit_record
from seiso.adaptive_quant.security_bypass import enforce_security_bypass_policy
from seiso.adaptive_quant.trainer import build_trainer


def run_online_pipeline(
    config: FrameworkConfig,
    *,
    request_count: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    summary_path = config.summary_path()
    trainer = build_trainer(config)
    git_commit = git_commit_hash()
    loop: OnlineLearningLoop | None = None
    pipeline_error: Exception | None = None
    bootstrap_summary: dict[str, object] = {}
    online_summary: dict[str, object] = {}
    eval_summary: dict[str, object] = {}
    online_analysis: dict[str, object] = {}
    history_path: str | None = None
    checkpoint_path: str | None = None

    enforce_security_bypass_policy(context="online pipeline")

    try:
        bootstrap_summary = trainer.train()
        loop = OnlineLearningLoop(config, trainer=trainer)
        online_summary = loop.run_stream(
            build_request_stream(config, request_count=request_count)
        )
        eval_summary = trainer.evaluate()

        analysis_root = f"{config.analysis_dir}/{config.run_name}"
        from seiso.analysis.analyzers import analyze_online

        online_analysis = analyze_online(
            config.online_telemetry_path(), f"{analysis_root}/online"
        )
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
            pipeline="online_adaptation",
            phases=[
                "bootstrap_train",
                "online_stream",
                "evaluate",
                "analysis",
            ],
        ),
        "security_audit": build_security_audit_record(
            config,
            cli_startup_overrides=cli_startup_overrides,
        ),
        "bootstrap_train": bootstrap_summary,
        "online": online_summary,
        "evaluation": eval_summary,
        "analysis": {
            "online_learning": slim_online_analysis_for_summary(online_analysis),
        },
        "artifacts": {
            "training_history": history_path,
            "final_checkpoint": checkpoint_path,
            "online_detail": config.online_summary_path(),
            "online_telemetry": config.online_telemetry_path(),
            "online_replay": config.online_replay_path(),
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


def run_online_pipeline_entrypoint(
    config: FrameworkConfig,
    *,
    request_count: int | None = None,
    cli_startup_overrides: dict[str, object] | None = None,
    footer_mode: str = "full",
) -> dict[str, object]:
    from seiso.adaptive_quant.run_footer import print_online_footer

    summary = run_online_pipeline(
        config,
        request_count=request_count,
        cli_startup_overrides=cli_startup_overrides,
    )
    print_online_footer(config, summary, mode=footer_mode)
    return summary


__all__ = ["run_online_pipeline", "run_online_pipeline_entrypoint"]
