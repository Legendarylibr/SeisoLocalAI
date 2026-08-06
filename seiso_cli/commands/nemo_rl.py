"""NVIDIA NeMo RL training command (external checkout)."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console


def nemo_rl(
    config: str = typer.Option(..., "--config", "-c", help="NeMo RL / TrainConfig YAML"),
) -> None:
    """Run NVIDIA NeMo RL via an external checkout (SEISO_NEMO_RL_ROOT)."""
    from seiso.memory.gpu_task import gpu_task
    from seiso.nemo_rl.config import NeMoRLConfig
    from seiso.nemo_rl.runner import train_nemo_rl
    from seiso.training.config import TrainConfig, TrainMethod

    path = Path(config)
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    method = str(raw.get("method", "")).lower().strip()
    if method == TrainMethod.NEMO_RL.value or method == "nemo_rl":
        cfg = TrainConfig.from_yaml(path).to_nemo_rl_config()
    else:
        cfg = NeMoRLConfig.from_yaml(path)

    console.print(
        f"NeMo RL [cyan]{cfg.recipe}[/] model=[cyan]{cfg.model_id}[/] (external NVIDIA-NeMo/RL)"
    )
    with gpu_task("nemo_rl"):
        out = train_nemo_rl(cfg)
    console.print(f"[green]Done:[/] {out}")
