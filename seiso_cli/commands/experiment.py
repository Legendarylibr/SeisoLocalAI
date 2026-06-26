"""Experiment sub-app and research benchmark commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from seiso_cli.console import console

experiment_app = typer.Typer(
    name="experiment",
    help="Research benchmarks and regression studies.",
    no_args_is_help=True,
)


@experiment_app.command("quant-regression")
def experiment_quant_regression(
    config: str = typer.Option(
        "configs/examples/quant_regression_study.yaml",
        "--config",
        "-c",
        help="Base training YAML (quant is overridden per run)",
    ),
    quants: str = typer.Option("4bit,8bit,16bit", help="Training quants to compare"),
    gguf_quants: str = typer.Option("q4_k_m,q8_0,f16", help="GGUF variants for route regression"),
    deploy_quants: str = typer.Option(
        "4bit,8bit,16bit", help="Deployment quants to compare (HF eval)"
    ),
    measurement: str = typer.Option(
        "both",
        "--measurement",
        help="hf = GPU eval on merged weights; llama_cpp = GGUF route eval; both = HF + llama.cpp",
    ),
    study_dir: str | None = typer.Option(None, help="Override study output directory"),
    rl_backend: str | None = typer.Option(
        None, help="Deprecated alias for --measurement hf|llama_cpp"
    ),
    llama_cpp_binary: str | None = typer.Option(None, help="Path to llama-cli"),
    llama_cpp_timeout_s: float = typer.Option(600.0, help="Per-prompt llama.cpp timeout"),
    route_prompt_limit: int = typer.Option(16, help="Prompts per llama.cpp route eval"),
    max_eval_samples: int = typer.Option(64, help="Eval samples for HF deploy-quant regression"),
    max_reward_regression: float | None = typer.Option(
        None, help="Max deploy reward regression vs best (default from config extra)"
    ),
    max_perplexity_regression: float | None = typer.Option(
        None, help="Max deploy perplexity regression vs best (default from config extra)"
    ),
    skip_training: bool = typer.Option(False, help="Reuse existing checkpoints under study_dir"),
    skip_rl: bool = typer.Option(False, help="Train/export only; skip route regression"),
    json_out: bool = typer.Option(False, "--json", help="Print JSON report"),
) -> None:
    """Train one model at several QLoRA quants, export GGUFs, measure deployment-quant regression."""
    from forge.config import get_settings
    from seiso.experiments.quant_regression import format_report_table, run_quant_regression_study
    from seiso.training.config import TrainConfig

    root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    if not os.environ.get("LLAMA_CPP_DIR"):
        for candidate in (Path.home() / "llama.cpp", Path("/opt/llama.cpp")):
            if (candidate / "convert_hf_to_gguf.py").is_file():
                os.environ.setdefault("LLAMA_CPP_DIR", str(candidate))
                break

    cfg = TrainConfig.from_yaml(cfg_path)
    settings = get_settings()

    report = run_quant_regression_study(
        cfg,
        data_dir=settings.data_dir,
        study_dir=Path(study_dir) if study_dir else None,
        train_quants=[q.strip() for q in quants.split(",") if q.strip()],
        gguf_quants=[q.strip() for q in gguf_quants.split(",") if q.strip()],
        deploy_quants=[q.strip() for q in deploy_quants.split(",") if q.strip()],
        measurement=rl_backend or measurement,
        llama_cpp_binary=llama_cpp_binary,
        llama_cpp_timeout_s=llama_cpp_timeout_s,
        route_prompt_limit=route_prompt_limit,
        max_eval_samples=max_eval_samples,
        max_reward_regression=max_reward_regression,
        max_perplexity_regression=max_perplexity_regression,
        skip_training=skip_training,
        skip_rl=skip_rl,
        on_log=lambda msg: console.print(msg),
    )

    if json_out:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    console.print("")
    console.print(format_report_table(report))
    console.print(f"\n[green]Report:[/] {report.study_dir}/quant_regression_report.json")
    if any(row.error for row in report.rows):
        raise typer.Exit(1)