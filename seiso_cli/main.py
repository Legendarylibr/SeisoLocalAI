"""Seiso CLI — forge, train, chat, export."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="seiso",
    help="Seiso — local AI platform for training, inference, and export.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def forge(
    host: str | None = typer.Option(None, help="Bind host (default: 127.0.0.1)"),
    port: int | None = typer.Option(None, help="Port (default: 8765)"),
    reload: bool = typer.Option(False, help="Dev auto-reload"),
) -> None:
    """Launch Seiso Forge web server (UI + API)."""
    import uvicorn

    from forge.config import get_settings

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port
    console.print(f"[bold green]Seiso Forge[/] → http://{bind_host}:{bind_port}")
    uvicorn.run(
        "forge.main:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="info",
    )


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML config"),
) -> None:
    """Fine-tune a model from config."""
    from seiso.training.config import TrainConfig, run_training

    cfg = TrainConfig.from_yaml(Path(config))
    console.print(f"Training [cyan]{cfg.model_id}[/] ({cfg.method.value})")
    out = run_training(cfg)
    console.print(f"[green]Done:[/] {out}")


async def _run_chat(model: str, messages: list[dict]) -> str:
    from seiso.inference.runner import run_chat

    return await run_chat({"model_path": model, "messages": messages})


@app.command()
def chat(
    model: str = typer.Option(..., help="Model ID or GGUF path"),
    prompt: str = typer.Option("", help="Single-turn prompt"),
) -> None:
    """Terminal chat with a local model."""
    from seiso.models.loader import detect_backend

    backend = detect_backend()
    console.print(f"Backend: {backend.value}")

    if prompt:
        reply = asyncio.run(_run_chat(model, [{"role": "user", "content": prompt}]))
        console.print(f"[bold]Assistant:[/] {reply}")
        return

    async def _interactive() -> None:
        messages: list[dict] = []
        console.print("Interactive chat (Ctrl+C to exit)")
        while True:
            try:
                user_input = typer.prompt("You")
            except (EOFError, KeyboardInterrupt):
                break
            messages.append({"role": "user", "content": user_input})
            reply = await _run_chat(model, messages)
            console.print(f"[bold]Assistant:[/] {reply}")
            messages.append({"role": "assistant", "content": reply})

    asyncio.run(_interactive())


@app.command()
def export_cmd(
    checkpoint: str = typer.Option(..., help="Checkpoint directory"),
    formats: str = typer.Option("merged", help="Comma-separated: merged,lora,gguf"),
    hub_repo: str | None = typer.Option(None, help="Hugging Face repo to push"),
) -> None:
    """Export checkpoint to merged/GGUF/LoRA."""
    from forge.config import get_settings
    from seiso.export.formats import ExportFormat, ExportOptions, export_checkpoint

    fmt_list = [ExportFormat(f.strip()) for f in formats.split(",")]
    settings = get_settings()
    opts = ExportOptions(
        checkpoint=Path(checkpoint),
        output_dir=settings.exports_dir,
        formats=fmt_list,
        hub_repo=hub_repo,
        hub_token=settings.hf_token or None,
        sandbox_root=settings.data_dir,
    )
    results = export_checkpoint(opts, on_log=lambda m: console.print(m))
    for k, v in results.items():
        console.print(f"  [green]{k}[/] → {v}")


app.command(name="export")(export_cmd)


@app.command(name="inference")
def inference_cmd(
    model: str = typer.Option(..., help="Model path or ID"),
    prompt: str = typer.Option(..., help="Prompt text"),
) -> None:
    """Run one-shot inference (alias for single-turn chat)."""
    reply = asyncio.run(_run_chat(model, [{"role": "user", "content": prompt}]))
    console.print(reply)


if __name__ == "__main__":
    app()
