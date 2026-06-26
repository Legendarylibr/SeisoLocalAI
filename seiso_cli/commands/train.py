"""Training command."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console


def train(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML config"),
) -> None:
    """Fine-tune a model from config."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory
    from seiso.training.config import TrainConfig, run_training

    pool = get_model_pool()
    if pool.active_key:
        pool.cancel_and_unload()
    release_cached_memory(sync=False)

    cfg = TrainConfig.from_yaml(Path(config))
    console.print(f"Training [cyan]{cfg.model_id}[/] ({cfg.method.value})")
    out = run_training(cfg)
    console.print(f"[green]Done:[/] {out}")