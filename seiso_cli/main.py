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
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open Forge in the system browser when /health is ready",
    ),
) -> None:
    """Launch Seiso web server (UI + API)."""
    import os

    import uvicorn

    from forge.config import get_settings
    from forge.instance_lock import ForgeAlreadyRunningError, acquire_forge_instance_locks
    from forge.launch import schedule_browser_open

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port
    try:
        instance_locks = acquire_forge_instance_locks(
            host=bind_host,
            port=bind_port,
            data_dir=settings.data_dir,
        )
    except ForgeAlreadyRunningError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    forge_url = f"http://{bind_host}:{bind_port}"
    should_open = open_browser or os.environ.get("SEISO_OPEN_BROWSER", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if should_open:
        schedule_browser_open(forge_url)
    console.print(f"[bold green]Seiso[/] → {forge_url}")
    try:
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
    finally:
        instance_locks.release()


@app.command()
def doctor(
    network: bool = typer.Option(False, "--network", help="Also probe huggingface.co reachability"),
) -> None:
    """Diagnose install, runtime, and Hugging Face setup."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "doctor.sh"
    if not script.is_file():
        console.print(f"[red]Doctor script not found:[/] {script}")
        raise typer.Exit(1)
    args = [str(script)]
    if network:
        args.append("--network")
    raise typer.Exit(subprocess.call(args))


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Training YAML config"),
) -> None:
    """Fine-tune a model from config."""
    from seiso.memory.protection import apply_training_memory_guards
    from seiso.training.config import TrainConfig, run_training

    cfg = apply_training_memory_guards(TrainConfig.from_yaml(Path(config)))
    console.print(f"Training [cyan]{cfg.model_id}[/] ({cfg.method.value})")
    out = run_training(cfg)
    console.print(f"[green]Done:[/] {out}")


async def _run_chat(model: str, messages: list[dict], *, tools_enabled: bool = False) -> str:
    from forge.services.llm_output import sanitize_llm_output
    from forge.services.model_prompts import chat_system_prompt, resolve_model_key
    from seiso.inference.runner import run_chat

    model_key = resolve_model_key(model_path=model)
    system = chat_system_prompt(model_key, tools_enabled=tools_enabled)
    payload_messages = list(messages)
    if system and not any(m.get("role") == "system" for m in payload_messages):
        payload_messages = [{"role": "system", "content": system}, *payload_messages]

    raw = await run_chat({"model_path": model, "messages": payload_messages})
    if tools_enabled:
        return raw
    return sanitize_llm_output(raw, strip_tool_calls=True)


def _one_shot_reply(model: str, prompt: str) -> str:
    return asyncio.run(_run_chat(model, [{"role": "user", "content": prompt}]))


@app.command()
def chat(
    model: str = typer.Option(..., help="Model ID or GGUF path"),
    prompt: str = typer.Option("", help="Single-turn prompt"),
) -> None:
    """Terminal chat with a local model."""
    from seiso.memory.protection import MemoryLoadBlockedError, ensure_load_fits
    from seiso.models.loader import detect_backend

    backend = detect_backend()
    console.print(f"Backend: {backend.value}")
    try:
        ensure_load_fits(model, mode="chat")
    except MemoryLoadBlockedError as exc:
        console.print(f"[red]Memory guard:[/] {exc}")
        raise typer.Exit(1) from exc

    if prompt:
        console.print(f"[bold]Assistant:[/] {_one_shot_reply(model, prompt)}")
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
    profile: str | None = typer.Option(
        None, help="Export profile: lora_bundle, full_bundle, inference, ..."
    ),
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

    results = run_export_plan(
        plan, hub_token=settings.hf_token or None, sandbox_root=settings.data_dir
    )
    for k, v in results.items():
        console.print(f"  [green]{k}[/] → {v}")


app.command(name="export")(export_cmd)


@app.command(name="inference")
def inference_cmd(
    model: str = typer.Option(..., help="Model path or ID"),
    prompt: str = typer.Option(..., help="Prompt text"),
) -> None:
    """Run one-shot inference (alias for single-turn chat)."""
    chat(model=model, prompt=prompt)


@app.command(name="bench-inference")
def bench_inference_cmd(
    model: str = typer.Option(..., help="Model path or GGUF file"),
    prompt: str = typer.Option("", help="Benchmark prompt (default: built-in paragraph)"),
    max_tokens: int = typer.Option(128, help="Tokens to generate per run"),
    backend: str = typer.Option("auto", help="auto | llamacpp | mlx | torch"),
    compare: bool = typer.Option(
        False,
        "--compare",
        help="Run baseline (CPU/no flash) vs optimized and print speedup",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Measure load time, time-to-first-token, and generation tok/s."""
    from seiso.inference.benchmark import (
        DEFAULT_PROMPT,
        run_bench_inference,
        run_compare_inference_profiles,
    )

    text = prompt or DEFAULT_PROMPT
    console.print(f"[bold]Inference benchmark[/] backend={backend} max_tokens={max_tokens}")

    if compare:
        report = run_compare_inference_profiles(
            model_path=model,
            prompt=text,
            max_tokens=max_tokens,
            backend=backend,
        )
        if json_out:
            import json

            console.print(json.dumps(report, indent=2))
            return

        base = report["baseline"]
        opt = report["optimized"]
        console.print("\n[bold]Baseline[/] (CPU llama.cpp / no flash / no fused kernels)")
        _print_bench_row(base)
        console.print("\n[bold]Optimized[/] (current Seiso defaults)")
        _print_bench_row(opt)
        console.print(
            f"\n[green]Speedup:[/] {report['speedup_tokens_per_sec']:.2f}x tok/s  "
            f"TTFT improved by {report['ttft_improvement_ms']:.1f} ms"
        )
        return

    result = run_bench_inference(
        model_path=model,
        prompt=text,
        max_tokens=max_tokens,
        backend=backend,
        warmup=True,
    )
    if json_out:
        import json

        console.print(json.dumps(result.to_dict(), indent=2))
        return

    _print_bench_row(result.to_dict())


def _print_bench_row(row: dict) -> None:
    load = row.get("load_ms")
    load_txt = f"{load:.0f} ms" if load is not None else "n/a"
    console.print(f"  backend:       {row.get('backend')}")
    console.print(f"  load (cold):   {load_txt}")
    console.print(f"  TTFT:          {row.get('ttft_ms'):.1f} ms")
    console.print(f"  generate:      {row.get('generate_ms'):.1f} ms")
    console.print(f"  output tokens: {row.get('output_tokens')} (~estimate)")
    console.print(
        f"  throughput:    [cyan]{row.get('tokens_per_sec'):.1f} tok/s[/]  ({row.get('ms_per_token'):.1f} ms/tok)"
    )


rl_quant_app = typer.Typer(
    name="rl-quant",
    help="Adaptive RL quantization — train quant + CUDA kernel policies.",
    no_args_is_help=True,
)
app.add_typer(rl_quant_app, name="rl-quant")


@rl_quant_app.command("run")
def rl_quant_run(
    preset: str = typer.Option("minimal", help="minimal | reproducible | post_train"),
    training_episodes: int | None = typer.Option(None, help="Training episode count"),
    evaluation_episodes: int | None = typer.Option(None, help="Evaluation episode count"),
    backend: str = typer.Option("simulator", help="simulator | llama_cpp"),
    training_backend: str = typer.Option("stdlib", help="stdlib | pytorch"),
    seed: int = typer.Option(13, help="RNG seed"),
    checkpoint_path: str | None = typer.Option(
        None, help="Fine-tune checkpoint for quality sidecar"
    ),
    gguf_path: str | None = typer.Option(None, help="GGUF path for llama.cpp backend"),
    gguf_export: bool = typer.Option(False, help="Export GGUF after recommendation"),
    moe_enabled: bool = typer.Option(False, help="Enable MoE expert variants"),
    kernel_rl: bool = typer.Option(
        False, "--kernel-rl", help="Co-train CUDA kernel launch profiles"
    ),
    kernel_live_benchmark: bool = typer.Option(
        False, "--kernel-live-benchmark", help="Live CUDA micro-benchmarks (NVIDIA GPU)"
    ),
    kernel_hidden_dim: int = typer.Option(4096, help="Hidden dim for kernel bench shapes"),
    kernel_batch_rows: int = typer.Option(4096, help="Token rows for kernel bench shapes"),
    write_report: bool = typer.Option(False, help="Write research markdown report"),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable summary JSON"),
) -> None:
    """Run RL quantization pipeline locally (no Forge server required)."""
    import json
    import uuid

    from forge.config import get_settings
    from seiso.rl_quant.runner import run_rl_quant_job

    settings = get_settings()
    job_id = str(uuid.uuid4())[:12]
    user_id = "cli"

    payload: dict = {
        "preset": preset,
        "backend": backend,
        "training_backend": training_backend,
        "seed": seed,
        "gguf_export": gguf_export,
        "moe_enabled": moe_enabled,
        "write_research_report": write_report,
    }
    if training_episodes is not None:
        payload["training_episodes"] = training_episodes
    if evaluation_episodes is not None:
        payload["evaluation_episodes"] = evaluation_episodes
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    if gguf_path:
        payload["gguf_path"] = gguf_path
    if kernel_rl:
        payload["kernel_rl_enabled"] = True
        payload["kernel_live_benchmark"] = kernel_live_benchmark
        payload["kernel_hidden_dim"] = kernel_hidden_dim
        payload["kernel_batch_rows"] = kernel_batch_rows

    console.print(
        f"[bold]RL quantization[/] preset={preset} backend={backend} trainer={training_backend}"
        + (" kernel_rl=on" if kernel_rl else "")
    )

    result = run_rl_quant_job(
        job_id=job_id,
        user_id=user_id,
        data_dir=settings.data_dir,
        payload=payload,
        on_log=lambda m: console.print(m),
    )

    if json_out:
        console.print(json.dumps(result, indent=2, default=str))
        return

    console.print(f"[green]Output:[/] {result.get('output_dir')}")
    rec_path = result.get("recommendation_path")
    if rec_path:
        console.print(f"[green]Recommendation:[/] {rec_path}")
    rec = result.get("recommendation")
    if isinstance(rec, dict):
        decision = rec.get("decision") or rec.get("recommended_quant")
        if isinstance(decision, dict):
            kernel = decision.get("kernel_profile_name") or (
                (decision.get("metadata") or {}).get("kernel_profile_name")
            )
            if kernel:
                console.print(f"[cyan]CUDA kernel profile:[/] {kernel}")


@rl_quant_app.command("profiles")
def rl_quant_profiles() -> None:
    """List CUDA kernel profiles available to the RL policy."""
    from seiso.kernels.tuning import KERNEL_PROFILES

    for profile in KERNEL_PROFILES:
        console.print(
            f"  [cyan]{profile['id']}[/] {profile['name']:16} "
            f"rms={profile['rms_mode']} swiglu_vec={profile['swiglu_vec']} lora_tile={profile['lora_tile']}"
        )


compress_app = typer.Typer(
    name="compress",
    help="Code Llama compression pipeline (distill, prune, finetune, export).",
    no_args_is_help=True,
)
app.add_typer(compress_app, name="compress")

distill_rl_app = typer.Typer(
    name="distill-rl",
    help="Distill a teacher model into a student, then apply preference RL (DPO).",
    no_args_is_help=True,
)
app.add_typer(distill_rl_app, name="distill-rl")


@distill_rl_app.command("presets")
def distill_rl_presets() -> None:
    """List distill-RL presets and pipeline stages."""
    from seiso.distill_rl.config import PRESETS, STAGE_ORDER

    for name, preset in PRESETS.items():
        stages = preset.get("stages", [])
        console.print(f"  [cyan]{name}[/] stages={','.join(stages)}")
    console.print(f"Stage order: {', '.join(STAGE_ORDER)}")


@distill_rl_app.command("run")
def distill_rl_run(
    preset: str = typer.Option("smoke", help="smoke | reproducible | full"),
    config: str | None = typer.Option(None, "--config", "-c", help="JSON/YAML job config"),
    teacher_model: str | None = typer.Option(None, help="Teacher HF model ID or path"),
    student_model: str | None = typer.Option(None, help="Student HF model ID or path"),
    distilled_path: str | None = typer.Option(
        None, help="Existing distilled checkpoint (skip distill stage)"
    ),
    distill_steps: int | None = typer.Option(None, help="KL distillation steps"),
    rollout_prompts: int | None = typer.Option(None, help="Max prompts for preference rollouts"),
    dpo_epochs: int | None = typer.Option(None, help="DPO training epochs"),
    prompt_library: str | None = typer.Option(None, help="Prompt JSON/JSONL for rollouts"),
    stages: str | None = typer.Option(
        None, help="Comma-separated stages: distill,rollout,dpo,evaluate"
    ),
    seeds: str | None = typer.Option(
        None, help="Comma-separated seeds for multi-seed research runs"
    ),
    seed: int = typer.Option(42, help="RNG seed (ignored when --seeds is set)"),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable summary JSON"),
) -> None:
    """Distill teacher → student, build teacher/student preferences, run DPO."""
    import json as json_mod
    import uuid

    from forge.config import get_settings
    from seiso.distill_rl.runner import run_distill_rl_job

    settings = get_settings()
    job_id = f"cli-{uuid.uuid4().hex[:8]}"
    payload: dict = {"preset": preset, "seed": seed}
    if config:
        payload["config_file"] = config
    if teacher_model:
        payload["teacher_model"] = teacher_model
    if student_model:
        payload["student_model"] = student_model
    if distilled_path:
        payload["distilled_path"] = distilled_path
    if distill_steps is not None:
        payload["distill_steps"] = distill_steps
    if rollout_prompts is not None:
        payload["rollout_max_prompts"] = rollout_prompts
    if dpo_epochs is not None:
        payload["dpo_epochs"] = dpo_epochs
    if prompt_library:
        payload["prompt_library"] = prompt_library
    if stages:
        payload["stages"] = [s.strip() for s in stages.split(",") if s.strip()]
    if seeds:
        payload["seeds"] = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    console.print(
        f"[bold]Distill-RL[/] preset={preset} "
        f"teacher={teacher_model or '(preset default)'} "
        f"student={student_model or '(preset default)'}"
    )
    result = run_distill_rl_job(
        job_id=job_id,
        user_id="cli",
        data_dir=settings.data_dir,
        payload=payload,
        on_log=lambda m: console.print(m),
    )
    if json_out:
        console.print(json_mod.dumps(result, indent=2, default=str))
    else:
        console.print(f"[green]Done:[/] {result.get('final_model_dir')}")
        console.print(f"Artifacts: {result.get('output_dir')}")


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
