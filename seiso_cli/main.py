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
    """Launch Seiso web server (UI + API)."""
    import uvicorn

    from forge.config import get_settings

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port
    console.print(f"[bold green]Seiso[/] → http://{bind_host}:{bind_port}")
    uvicorn.run(
        "forge.main:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level="info",
        proxy_headers=settings.trust_proxy,
        forwarded_allow_ips="127.0.0.1,::1" if settings.trust_proxy else None,
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
    formats: str = typer.Option("merged", help="Comma-separated: merged,lora,full,gguf"),
    profile: str | None = typer.Option(None, help="Export profile: lora_bundle, full_bundle, inference, ..."),
    hub_repo: str | None = typer.Option(None, help="Hugging Face repo to push"),
    precheck_only: bool = typer.Option(False, help="Run Hub precheck without exporting"),
) -> None:
    """Export checkpoint to merged/GGUF/LoRA/full fine-tune."""
    from forge.config import get_settings
    from seiso.export.pipeline import prepare_export, profile_catalog, run_export_plan

    settings = get_settings()
    ckpt = Path(checkpoint)
    fmt_list = [f.strip() for f in formats.split(",")] if not profile else None

    if profile and profile == "list":
        for entry in profile_catalog():
            console.print(f"  [cyan]{entry['id']}[/] → {', '.join(entry['formats'])}")
        return

    plan = prepare_export(
        checkpoint=ckpt,
        output_dir=settings.exports_dir / ckpt.name,
        formats=fmt_list,
        profile=profile,
        hub_repo=hub_repo,
        hub_token=settings.hf_token or None,
        on_log=lambda m: console.print(m),
    )

    if precheck_only:
        if plan.precheck:
            console.print(plan.precheck.to_dict())
        else:
            console.print("No Hub precheck requested (set --hub-repo)")
        return

    results = run_export_plan(plan, hub_token=settings.hf_token or None, sandbox_root=settings.data_dir)
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


compress_app = typer.Typer(
    name="compress",
    help="Code Llama compression pipeline (distill, prune, finetune, export).",
    no_args_is_help=True,
)
app.add_typer(compress_app, name="compress")


@compress_app.command("run")
def compress_run(
    preset: str = typer.Option(
        "smoke", help="smoke | full | distill_only | prune_recover | quantize"
    ),
    model_dir: str | None = typer.Option(
        None, help="Starting model dir for prune/finetune presets"
    ),
    teacher_model: str = typer.Option("codellama/CodeLlama-13b-hf"),
    student_model: str = typer.Option("codellama/CodeLlama-7b-hf"),
    distill_steps: int | None = typer.Option(None),
    finetune_steps: int | None = typer.Option(None),
    prune_ratio: float | None = typer.Option(None),
    seed: int = typer.Option(42),
) -> None:
    """Run compression pipeline from CLI."""
    from forge.config import get_settings
    from seiso.compress.runner import run_compress_job

    settings = get_settings()
    job_id = "cli"
    user_id = "local"
    payload: dict = {
        "preset": preset,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "seed": seed,
    }
    if model_dir:
        payload["model_dir"] = model_dir
    if distill_steps is not None:
        payload["distill_steps"] = distill_steps
    if finetune_steps is not None:
        payload["finetune_steps"] = finetune_steps
    if prune_ratio is not None:
        payload["prune_ratio"] = prune_ratio

    console.print(f"[bold]Compression pipeline[/] preset={preset}")
    result = run_compress_job(
        job_id=job_id,
        user_id=user_id,
        data_dir=settings.data_dir,
        payload=payload,
        on_log=lambda m: console.print(m),
    )
    console.print(f"[green]Done:[/] {result.get('run_dir')}")


@compress_app.command("manifest-verify")
def compress_manifest_verify(
    run_dir: str = typer.Option(..., help="Run directory with manifest.json"),
) -> None:
    """Verify hash-chained manifest for a compression run."""
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from codellama_compress.replay import verify_manifest

    report = verify_manifest(Path(run_dir))
    console.print(report)
    if not report.get("ok"):
        raise typer.Exit(1)


@compress_app.command("speculative")
def compress_speculative(
    target_model: str = typer.Option(...),
    draft_model: str = typer.Option(...),
    prompt: str = typer.Option("def fibonacci(n):"),
    max_new_tokens: int = typer.Option(256),
    num_speculative_tokens: int = typer.Option(5),
) -> None:
    """Run speculative decoding with draft + target models."""
    from seiso.compress.bootstrap import require_codellama_compress

    require_codellama_compress()
    from codellama_compress.speculative import speculative_generate

    text, stats = speculative_generate(
        prompt=prompt,
        target_model=target_model,
        draft_model=draft_model,
        max_new_tokens=max_new_tokens,
        num_speculative_tokens=num_speculative_tokens,
    )
    console.print(text)
    console.print(stats.to_dict())


if __name__ == "__main__":
    app()
