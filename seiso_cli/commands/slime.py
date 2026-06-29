"""Single-GPU slime-style training command."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console


def slime(
    config: str = typer.Option(..., "--config", "-c", help="Single-GPU slime YAML config"),
) -> None:
    """Run a compact slime-style RL loop on one local GPU."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory
    from seiso.slime_single_gpu.config import SingleGpuSlimeConfig
    from seiso.slime_single_gpu.trainer import train_single_gpu_slime

    pool = get_model_pool()
    if pool.active_key:
        pool.cancel_and_unload()
    release_cached_memory(sync=False)

    cfg = SingleGpuSlimeConfig.from_yaml(Path(config))
    console.print(f"Single-GPU slime training [cyan]{cfg.model_id}[/]")
    out = train_single_gpu_slime(cfg)
    console.print(f"[green]Done:[/] {out}")
