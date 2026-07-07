"""Single-GPU slime-style training command."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console


def slime(
    config: str = typer.Option(
        ..., "--config", "-c", help="Single-GPU slime YAML config"
    ),
) -> None:
    """Run a compact slime-style RL loop on one local GPU."""
    from seiso.memory.gpu_task import gpu_task
    from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
    from seiso.slime_single_gpu.trainer import train_single_gpu_slime

    cfg = SingleGpuSlimeConfig.from_yaml(Path(config))
    console.print(f"Single-GPU slime training [cyan]{cfg.model_id}[/]")
    with gpu_task("slime"):
        out = train_single_gpu_slime(cfg)
    console.print(f"[green]Done:[/] {out}")
