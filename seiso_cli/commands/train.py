"""Training command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from seiso_cli.console import console


def train(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML config"),
) -> None:
    """Fine-tune a model from config."""
    from seiso.memory.gpu_task import gpu_task
    from seiso.training.config import TrainConfig, run_training
    from seiso.training.multi_gpu import (
        detect_training_layout,
        distributed_requested,
        launch_worker_command,
        resolve_distributed_plan,
    )

    cfg = TrainConfig.from_yaml(Path(config))
    console.print(f"Training [cyan]{cfg.model_id}[/] ({cfg.method.value})")
    layout = detect_training_layout()
    plan = resolve_distributed_plan(cfg, layout)
    with gpu_task("training"):
        if plan.enabled and not layout.use_ddp:
            console.print(
                "[cyan]Accelerate distributed launch:[/] "
                f"{plan.strategy} x{plan.world_size} "
                f"({plan.nproc_per_node} process(es) per node)"
            )
            subprocess.run(launch_worker_command(str(Path(config)), plan), check=True)
            console.print(f"[green]Done:[/] {cfg.output_dir}")
            return

        if distributed_requested(cfg) and not plan.enabled:
            console.print(f"[yellow]distributed launch skipped:[/] {plan.reason}")

        out = run_training(cfg)
    console.print(f"[green]Done:[/] {out}")
