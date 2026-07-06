"""Pipeline sub-apps: rl-quant, compress, distill-rl."""

from __future__ import annotations

from pathlib import Path

import typer

from seiso_cli.console import console

rl_quant_app = typer.Typer(
    name="rl-quant",
    help="Adaptive RL quantization — train quant + CUDA kernel policies.",
    no_args_is_help=True,
)


@rl_quant_app.command("run")
def rl_quant_run(
    preset: str = typer.Option("minimal", help="minimal | reproducible | post_train"),
    training_episodes: int | None = typer.Option(None, help="Training episode count"),
    evaluation_episodes: int | None = typer.Option(
        None, help="Evaluation episode count"
    ),
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
    kernel_hidden_dim: int = typer.Option(
        4096, help="Hidden dim for kernel bench shapes"
    ),
    kernel_batch_rows: int = typer.Option(
        4096, help="Token rows for kernel bench shapes"
    ),
    write_report: bool = typer.Option(False, help="Write research markdown report"),
    auto_sweep: bool = typer.Option(
        True,
        "--auto-sweep/--no-auto-sweep",
        help="Grid-search key hyperparameters before the full run (default: on)",
    ),
    sweep_config: str | None = typer.Option(
        None, help="Optional sweep grid JSON/TOML (defaults to preset auto grid)"
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Print machine-readable summary JSON"
    ),
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
        "auto_sweep": auto_sweep,
    }
    if sweep_config:
        payload["sweep_config"] = sweep_config
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
    help="LLM compression pipeline (distill, prune, finetune, export).",
    no_args_is_help=True,
)

distill_rl_app = typer.Typer(
    name="distill-rl",
    help="Distill a teacher model into a student, then apply preference RL (DPO).",
    no_args_is_help=True,
)


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
    config: str | None = typer.Option(
        None, "--config", "-c", help="JSON/YAML job config"
    ),
    teacher_model: str | None = typer.Option(None, help="Teacher HF model ID or path"),
    student_model: str | None = typer.Option(None, help="Student HF model ID or path"),
    distilled_path: str | None = typer.Option(
        None, help="Existing distilled checkpoint (skip distill stage)"
    ),
    distill_steps: int | None = typer.Option(None, help="KL distillation steps"),
    rollout_prompts: int | None = typer.Option(
        None, help="Max prompts for preference rollouts"
    ),
    dpo_epochs: int | None = typer.Option(None, help="DPO training epochs"),
    prompt_library: str | None = typer.Option(
        None, help="Prompt JSON/JSONL for rollouts"
    ),
    require_thinking_trace: bool = typer.Option(
        True,
        "--thinking-trace/--no-thinking-trace",
        help="Force <think>...</think> before final answers during distill/rollout.",
    ),
    verifiable_outcome_rewards: bool = typer.Option(
        True,
        "--outcome-rewards/--teacher-preferences",
        help="Use pure final-answer rewards for prompts with answers.",
    ),
    grpo_group_size: int = typer.Option(
        4,
        help="Number of sampled reasoning traces per verifiable prompt.",
    ),
    benchmark_verifiable: bool = typer.Option(
        True,
        "--benchmark-verifiable/--no-benchmark-verifiable",
        help="Run GSM8K/GPQA/AIME-style strict outcome benchmarks.",
    ),
    benchmark_tasks: str = typer.Option(
        "gsm8k,gpqa,aime",
        help="Comma-separated verifiable benchmark tasks.",
    ),
    stages: str | None = typer.Option(
        None, help="Comma-separated stages: distill,rollout,dpo,evaluate"
    ),
    seeds: str | None = typer.Option(
        None, help="Comma-separated seeds for multi-seed research runs"
    ),
    seed: int = typer.Option(42, help="RNG seed (ignored when --seeds is set)"),
    auto_sweep: bool = typer.Option(
        True,
        "--auto-sweep/--no-auto-sweep",
        help="Grid-search DPO hyperparameters before the full run (default: on)",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Print machine-readable summary JSON"
    ),
) -> None:
    """Distill teacher → student, build teacher/student preferences, run DPO."""
    import json as json_mod
    import uuid

    from forge.config import get_settings
    from seiso.distill_rl.runner import run_distill_rl_job

    settings = get_settings()
    job_id = f"cli-{uuid.uuid4().hex[:8]}"
    payload: dict = {
        "preset": preset,
        "seed": seed,
        "auto_sweep": auto_sweep,
        "require_thinking_trace": require_thinking_trace,
        "verifiable_outcome_rewards": verifiable_outcome_rewards,
        "grpo_group_size": grpo_group_size,
        "benchmark_verifiable": benchmark_verifiable,
        "benchmark_tasks": [
            task.strip() for task in benchmark_tasks.split(",") if task.strip()
        ],
    }
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
        jumps = result.get("benchmark_jumps") or []
        if jumps:
            console.print("Benchmark jumps:")
            for line in jumps:
                console.print(f"  {line}")


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
    import uuid

    from forge.config import get_settings
    from seiso.compress.runner import run_compress_job

    settings = get_settings()
    job_id = f"cli-{uuid.uuid4().hex[:8]}"
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
    from seiso.codellama_compress.replay import verify_manifest

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
    from seiso.codellama_compress.speculative import speculative_generate

    text, stats = speculative_generate(
        prompt=prompt,
        target_model=target_model,
        draft_model=draft_model,
        max_new_tokens=max_new_tokens,
        num_speculative_tokens=num_speculative_tokens,
    )
    console.print(text)
    console.print(stats.to_dict())
