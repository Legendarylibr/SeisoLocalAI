"""Local inference throughput benchmark — load, TTFT, and tokens/sec."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

from seiso.inference.backends import resolve_local_backend
from seiso.inference.model_pool import get_model_pool
from seiso.inference.runner import LocalInferenceRunner

DEFAULT_PROMPT = (
    "Explain how a binary search tree works. Cover insertion, lookup, "
    "and why average-case lookup is O(log n). Be concise but complete."
)

# Conservative CPU / no-GPU-offload settings for before/after comparison.
BASELINE_ENV: dict[str, str] = {
    "SEISO_LLAMA_GPU_LAYERS": "0",
    "SEISO_LLAMA_FLASH_ATTN": "false",
    "SEISO_LLAMA_BATCH": "512",
    "SEISO_LLAMA_OFFLOAD_KQV": "false",
    "SEISO_INFERENCE_FUSED_KERNELS": "false",
}


@dataclass
class InferenceBenchResult:
    backend: str
    model_path: str
    prompt_chars: int
    output_chars: int
    output_tokens: int
    max_tokens: int
    load_ms: float | None
    ttft_ms: float
    generate_ms: float
    total_ms: float
    tokens_per_sec: float
    ms_per_token: float
    profile: str = "optimized"
    notes: list[str] = field(default_factory=list)
    kv_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def _bench_env(overrides: dict[str, str] | None) -> Iterator[None]:
    if not overrides:
        yield
        return
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    # Rough English token estimate — stable enough for relative benchmarks.
    return max(1, int(len(stripped.split()) * 1.35))


async def _timed_stream(
    payload: dict[str, Any],
    *,
    runner: LocalInferenceRunner | None = None,
) -> tuple[str, float | None, float, float, int]:
    """Return output text, load_ms, ttft_ms, generate_ms, and output token count."""
    runner = runner or LocalInferenceRunner()

    t0 = time.perf_counter()
    first_at: float | None = None
    chunks: list[str] = []
    output_tokens = 0

    async for update in runner.stream_updates(payload):
        now = time.perf_counter()
        if first_at is None:
            first_at = now
        chunks.append(update.text)
        output_tokens = update.output_tokens

    t1 = time.perf_counter()
    output = "".join(chunks)

    load_ms: float | None = None
    if first_at is not None:
        ttft_ms = (first_at - t0) * 1000.0
        generate_ms = (t1 - first_at) * 1000.0
    else:
        ttft_ms = (t1 - t0) * 1000.0
        generate_ms = 0.0

    return output, load_ms, ttft_ms, generate_ms, output_tokens


def _build_payload(
    model_path: str,
    *,
    prompt: str,
    max_tokens: int,
    backend: str,
) -> dict[str, Any]:
    resolved = resolve_local_backend(
        model_path=model_path,
        model_format=None,
        requested=backend,
    )
    return {
        "model_path": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "inference_backend": resolved,
    }


async def bench_inference(
    model_path: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 128,
    backend: str = "auto",
    warmup: bool = True,
    profile: str = "optimized",
    env_overrides: dict[str, str] | None = None,
) -> InferenceBenchResult:
    """Benchmark one inference profile; optionally warm the model pool first."""
    payload = _build_payload(model_path, prompt=prompt, max_tokens=max_tokens, backend=backend)
    resolved_backend = payload["inference_backend"]
    notes: list[str] = []
    load_ms: float | None = None

    with _bench_env(env_overrides):
        runner = LocalInferenceRunner()
        pool = get_model_pool()

        if warmup:
            pool.cancel_and_unload()
            cold_payload = {**payload, "max_tokens": min(16, max_tokens)}
            cold_t0 = time.perf_counter()
            await runner.chat(cold_payload)
            load_ms = (time.perf_counter() - cold_t0) * 1000.0
            notes.append("warmup=16tok cold load")

        output, _, ttft_ms, generate_ms, output_tokens = await _timed_stream(payload, runner=runner)
        kv_metadata = dict(getattr(runner, "last_inference_stats", {}))

    if output_tokens <= 0:
        output_tokens = _estimate_tokens(output)
    tokens_per_sec = (
        output_tokens / (generate_ms / 1000.0) if generate_ms > 0 and output_tokens > 0 else 0.0
    )
    ms_per_token = generate_ms / output_tokens if output_tokens > 0 else 0.0
    total_ms = (
        (load_ms or 0.0) + ttft_ms + generate_ms if load_ms is not None else ttft_ms + generate_ms
    )

    return InferenceBenchResult(
        backend=resolved_backend,
        model_path=model_path,
        prompt_chars=len(prompt),
        output_chars=len(output),
        output_tokens=output_tokens,
        max_tokens=max_tokens,
        load_ms=load_ms,
        ttft_ms=round(ttft_ms, 2),
        generate_ms=round(generate_ms, 2),
        total_ms=round(total_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        ms_per_token=round(ms_per_token, 2),
        profile=profile,
        notes=notes,
        kv_metadata=kv_metadata,
    )


async def compare_inference_profiles(
    model_path: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 128,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run baseline then optimized profiles, unloading between them."""
    pool = get_model_pool()
    pool.cancel_and_unload()

    baseline = await bench_inference(
        model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        backend=backend,
        warmup=True,
        profile="baseline",
        env_overrides=BASELINE_ENV,
    )

    pool.cancel_and_unload()

    optimized = await bench_inference(
        model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        backend=backend,
        warmup=True,
        profile="optimized",
        env_overrides=None,
    )

    speedup = (
        optimized.tokens_per_sec / baseline.tokens_per_sec if baseline.tokens_per_sec > 0 else 0.0
    )
    ttft_delta = baseline.ttft_ms - optimized.ttft_ms

    return {
        "baseline": baseline.to_dict(),
        "optimized": optimized.to_dict(),
        "speedup_tokens_per_sec": round(speedup, 2),
        "ttft_improvement_ms": round(ttft_delta, 2),
    }


async def benchmark_kv_scenarios(
    model_path: str,
    *,
    backend: str = "auto",
    max_tokens: int = 64,
    short_prompt: str = DEFAULT_PROMPT,
    long_prompt: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure short/long prefill and repeated-conversation prefix behavior."""
    runner = LocalInferenceRunner()
    pool = get_model_pool()
    pool.cancel_and_unload()
    long_prompt = long_prompt or "\n".join([DEFAULT_PROMPT] * 32)

    async def run_scenario(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        output, _, ttft_ms, generate_ms, output_tokens = await _timed_stream(
            payload, runner=runner
        )
        tokens_per_sec = (
            output_tokens / (generate_ms / 1000.0)
            if generate_ms > 0 and output_tokens > 0
            else 0.0
        )
        return output, {
            "ttft_ms": round(ttft_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "output_tokens": output_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "kv_metadata": runner.last_inference_stats,
        }

    short_payload = _build_payload(
        model_path,
        prompt=short_prompt,
        max_tokens=max_tokens,
        backend=backend,
    )
    long_payload = _build_payload(
        model_path,
        prompt=long_prompt,
        max_tokens=max_tokens,
        backend=backend,
    )
    _, short_result = await run_scenario(short_payload)
    _, long_result = await run_scenario(long_payload)

    first_output, repeated_first = await run_scenario(short_payload)
    repeated_payload = {
        **short_payload,
        "messages": [
            {"role": "user", "content": short_prompt},
            {"role": "assistant", "content": first_output},
            {"role": "user", "content": "Continue with one practical example."},
        ],
    }
    _, repeated_second = await run_scenario(repeated_payload)
    return {
        "short": short_result,
        "long": long_result,
        "repeated_first": repeated_first,
        "repeated_second": repeated_second,
    }


def run_bench_inference(**kwargs: Any) -> InferenceBenchResult:
    return asyncio.run(bench_inference(**kwargs))


def run_compare_inference_profiles(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(compare_inference_profiles(**kwargs))


def run_benchmark_kv_scenarios(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return asyncio.run(benchmark_kv_scenarios(**kwargs))
