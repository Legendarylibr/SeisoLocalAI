"""Chat and inference benchmark commands."""

from __future__ import annotations

import asyncio

import typer

from seiso_cli.console import console


async def _run_chat(
    model: str, messages: list[dict], *, tools_enabled: bool = False
) -> str:
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


def inference_cmd(
    model: str = typer.Option(..., help="Model path or ID"),
    prompt: str = typer.Option(..., help="Prompt text"),
) -> None:
    """Run one-shot inference (alias for single-turn chat)."""
    chat(model=model, prompt=prompt)


def bench_inference_cmd(
    model: str = typer.Option(..., help="Model path or GGUF file"),
    prompt: str = typer.Option(
        "", help="Benchmark prompt (default: built-in paragraph)"
    ),
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
    from seiso.memory.gpu_task import gpu_task

    text = prompt or DEFAULT_PROMPT
    console.print(
        f"[bold]Inference benchmark[/] backend={backend} max_tokens={max_tokens}"
    )

    if compare:
        with gpu_task("inference"):
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
        console.print(
            "\n[bold]Baseline[/] (CPU llama.cpp / no flash / no fused kernels)"
        )
        _print_bench_row(base)
        console.print("\n[bold]Optimized[/] (current Seiso defaults)")
        _print_bench_row(opt)
        console.print(
            f"\n[green]Speedup:[/] {report['speedup_tokens_per_sec']:.2f}x tok/s  "
            f"TTFT improved by {report['ttft_improvement_ms']:.1f} ms"
        )
        return

    with gpu_task("inference"):
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
    console.print(f"  output tokens: {row.get('output_tokens')}")
    console.print(
        f"  throughput:    [cyan]{row.get('tokens_per_sec'):.1f} tok/s[/]  ({row.get('ms_per_token'):.1f} ms/tok)"
    )
