"""Single-GPU slime-style training command."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from seiso_cli.console import console


def slime(
    config: str = typer.Option(..., "--config", "-c", help="Single-GPU slime YAML config"),
) -> None:
    """Run a compact slime-style RL loop on one local GPU."""
    from seiso.memory.gpu_task import gpu_task
    from seiso.slime.config import SingleGpuSlimeConfig
    from seiso.slime.trainer import train_slime

    path = Path(config)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict):
        multi = bool(raw.get("multi_gpu"))
        strategy = str(raw.get("distributed_strategy", "auto") or "auto").lower()
        backend = str(raw.get("rollout_backend", "hf") or "hf").lower()
        if multi or strategy == "ddp":
            console.print(
                "[yellow]Note:[/] this YAML requests multi-GPU/DDP. "
                "`seiso slime` is single-process only — use "
                "`seiso train -c …` (or the DDP launch scripts) for policy DDP."
            )
        elif backend in {"sglang", "vllm"}:
            console.print(
                f"[yellow]Note:[/] rollout_backend={backend} with single-process "
                "policy. For Accelerate DDP policy updates use `seiso train`."
            )

    cfg = SingleGpuSlimeConfig.from_yaml(path)
    console.print(f"Single-GPU slime training [cyan]{cfg.model_id}[/]")
    with gpu_task("slime"):
        out = train_slime(cfg)
    console.print(f"[green]Done:[/] {out}")
