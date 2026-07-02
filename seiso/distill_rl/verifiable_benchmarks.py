"""Small verifiable benchmark harness for Distill-RL reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seiso.distill_rl.outcome import outcome_reward
from seiso.distill_rl.prompts import RolloutPrompt, load_rollout_prompts
from seiso.distill_rl.rollouts import generate_completions

_BUILTIN_BENCHMARKS: dict[str, list[RolloutPrompt]] = {
    "gsm8k": [
        RolloutPrompt(
            prompt_id="gsm8k_builtin_1",
            text="Jan has 3 apples and buys 4 more. How many apples does Jan have?",
            answer="7",
            benchmark="gsm8k",
        ),
        RolloutPrompt(
            prompt_id="gsm8k_builtin_2",
            text="A box has 12 pencils. Sam removes 5. How many pencils remain?",
            answer="7",
            benchmark="gsm8k",
        ),
    ],
    "gpqa": [
        RolloutPrompt(
            prompt_id="gpqa_builtin_1",
            text=(
                "Which option is the noble gas? "
                "A. Sodium B. Oxygen C. Neon D. Chlorine"
            ),
            answer="C",
            benchmark="gpqa",
        ),
        RolloutPrompt(
            prompt_id="gpqa_builtin_2",
            text=(
                "Which particle has negative electric charge? "
                "A. Proton B. Neutron C. Electron D. Photon"
            ),
            answer="C",
            benchmark="gpqa",
        ),
    ],
    "aime": [
        RolloutPrompt(
            prompt_id="aime_builtin_1",
            text="Compute the integer value of 15 squared.",
            answer="225",
            benchmark="aime",
        ),
        RolloutPrompt(
            prompt_id="aime_builtin_2",
            text="What is the remainder when 1000 is divided by 7?",
            answer="6",
            benchmark="aime",
        ),
    ],
}


def evaluate_verifiable_benchmarks(
    *,
    output_dir: Path,
    checkpoints: dict[str, str],
    prompt_library_path: Path | None,
    benchmark_tasks: list[str],
    max_prompts_per_task: int,
    trust_remote_code: bool,
    require_thinking_trace: bool,
    thinking_instruction: str,
    on_log=None,
) -> dict[str, Any]:
    """Run strict outcome benchmarks and report checkpoint deltas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_benchmark_tasks(
        prompt_library_path=prompt_library_path,
        benchmark_tasks=benchmark_tasks,
        max_prompts_per_task=max_prompts_per_task,
    )
    checkpoints_out: dict[str, Any] = {}
    for checkpoint_name, model_path in checkpoints.items():
        if on_log:
            on_log(f"Benchmark verifiable tasks: {checkpoint_name}")
        task_results: dict[str, Any] = {}
        for task_name, prompts in tasks.items():
            try:
                outputs = _generate_benchmark_outputs(
                    model_path,
                    prompts,
                    max_new_tokens=128,
                    temperature=0.0,
                    seed=0,
                    use_chat_template=False,
                    trust_remote_code=trust_remote_code,
                    require_thinking_trace=require_thinking_trace,
                    thinking_instruction=thinking_instruction,
                )
            except Exception as exc:
                task_results[task_name] = {
                    "accuracy": 0.0,
                    "correct": 0,
                    "total": len(prompts),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            scores = [
                outcome_reward(output, prompt.answer, benchmark=prompt.benchmark)
                for prompt, output in zip(prompts, outputs, strict=True)
            ]
            accuracy = sum(scores) / len(scores) if scores else 0.0
            task_results[task_name] = {
                "accuracy": accuracy,
                "correct": int(sum(scores)),
                "total": len(scores),
            }
        checkpoints_out[checkpoint_name] = task_results

    jumps = _accuracy_jumps(checkpoints_out)
    result = {
        "tasks": sorted(tasks),
        "checkpoints": checkpoints_out,
        "accuracy_jumps": jumps,
    }
    summary_path = output_dir / "verifiable_benchmarks.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def _generate_benchmark_outputs(
    model_path: str,
    prompts: list[RolloutPrompt],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    use_chat_template: bool,
    trust_remote_code: bool,
    require_thinking_trace: bool,
    thinking_instruction: str,
) -> list[str]:
    path = Path(model_path).expanduser()
    if path.is_file() and path.name.lower().endswith(".gguf"):
        return _generate_gguf_outputs(
            path,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            require_thinking_trace=require_thinking_trace,
            thinking_instruction=thinking_instruction,
        )
    return generate_completions(
        model_path,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
    )


def _generate_gguf_outputs(
    model_path: Path,
    prompts: list[RolloutPrompt],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    require_thinking_trace: bool,
    thinking_instruction: str,
) -> list[str]:
    from llama_cpp import Llama

    from seiso.distill_rl.outcome import (
        ensure_thinking_completion,
        format_thinking_prompt,
    )

    last_error: Exception | None = None
    llm = None
    for n_ctx in (2048, 1024, 512):
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=0,
                verbose=False,
                seed=seed,
            )
            break
        except Exception as exc:
            last_error = exc
    if llm is None:
        assert last_error is not None
        raise last_error
    outputs: list[str] = []
    for prompt in prompts:
        prompt_text = (
            format_thinking_prompt(prompt.text, thinking_instruction)
            if require_thinking_trace
            else prompt.text
        )
        response = llm.create_completion(
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stop=["</s>", "<|eot_id|>"],
        )
        text = str(response["choices"][0]["text"]).strip()
        outputs.append(ensure_thinking_completion(text, enabled=require_thinking_trace))
    return outputs


def summarize_accuracy_jumps(report: dict[str, Any] | None) -> list[str]:
    """Return human-readable benchmark jump lines from a benchmark report."""
    if not report:
        return []
    jumps = report.get("accuracy_jumps")
    if not isinstance(jumps, dict):
        return []
    by_checkpoint = jumps.get("by_checkpoint")
    if not isinstance(by_checkpoint, dict):
        return []
    lines: list[str] = []
    for checkpoint, task_jumps in sorted(by_checkpoint.items()):
        if not isinstance(task_jumps, dict):
            continue
        parts = [
            f"{task}={float(delta):+.3f}"
            for task, delta in sorted(task_jumps.items())
            if isinstance(delta, int | float)
        ]
        if parts:
            lines.append(f"{checkpoint}: " + ", ".join(parts))
    return lines


def _load_benchmark_tasks(
    *,
    prompt_library_path: Path | None,
    benchmark_tasks: list[str],
    max_prompts_per_task: int,
) -> dict[str, list[RolloutPrompt]]:
    requested = [task.lower() for task in benchmark_tasks]
    from_library: dict[str, list[RolloutPrompt]] = {task: [] for task in requested}
    if prompt_library_path is not None and prompt_library_path.is_file():
        prompts = load_rollout_prompts(prompt_library_path, limit=10_000)
        for prompt in prompts:
            task = (prompt.benchmark or "").lower()
            if task in from_library and prompt.answer is not None:
                from_library[task].append(prompt)

    tasks: dict[str, list[RolloutPrompt]] = {}
    for task in requested:
        prompts = from_library.get(task) or _BUILTIN_BENCHMARKS.get(task, [])
        tasks[task] = prompts[: max(1, max_prompts_per_task)]
    return tasks


def _accuracy_jumps(checkpoints: dict[str, Any]) -> dict[str, Any]:
    if not checkpoints:
        return {}
    baseline_name = (
        "student_base" if "student_base" in checkpoints else next(iter(checkpoints))
    )
    baseline = checkpoints[baseline_name]
    jumps: dict[str, Any] = {"baseline": baseline_name, "by_checkpoint": {}}
    for name, task_metrics in checkpoints.items():
        if name == baseline_name:
            continue
        jumps["by_checkpoint"][name] = {
            task: metrics["accuracy"] - baseline.get(task, {}).get("accuracy", 0.0)
            for task, metrics in task_metrics.items()
        }
    return jumps
