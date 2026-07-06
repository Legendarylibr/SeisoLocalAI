#!/usr/bin/env python3
"""Live NVIDIA inference OOM guard + throughput check on real GGUF models."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_MODELS = [
    "/home/c/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf",
    "/home/c/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/b3a58239d8d40b953e34936c9afeb28baa518230/Qwen3.6-27B-UD-Q4_K_XL.gguf",
]


@dataclass
class GpuSnapshot:
    used_mb: int
    free_mb: int
    total_mb: int


@dataclass
class ModelLiveResult:
    model_path: str
    ok: bool
    error: str = ""
    load_kwargs: dict = field(default_factory=dict)
    tight_fit: bool = False
    prefill_reload: bool = False
    safe_batch: int = 0
    safe_ubatch: int = 0
    gpu_layers: int = 0
    vram_before_mb: int = 0
    vram_peak_mb: int = 0
    vram_after_mb: int = 0
    ttft_ms: float = 0.0
    generate_ms: float = 0.0
    tokens_per_sec: float = 0.0
    output_tokens: int = 0
    long_prefill_reload: bool = False


def gpu_snapshot() -> GpuSnapshot:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, free, total = (int(x.strip()) for x in out.split(","))
        return GpuSnapshot(used_mb=used, free_mb=free, total_mb=total)
    except Exception:
        return GpuSnapshot(used_mb=0, free_mb=0, total_mb=0)


def _long_messages() -> list[dict[str, str]]:
    chunk = (
        "Summarize the key ideas from this paragraph in one sentence. "
        "Binary search trees support O(log n) lookup on average. "
    )
    messages = [{"role": "user", "content": chunk * 40}]
    messages.append({"role": "assistant", "content": "Noted."})
    messages.append(
        {
            "role": "user",
            "content": "Now explain insertion and deletion briefly, with one example each.",
        }
    )
    return messages


def _analyze_model(model_path: str, *, n_ctx: int = 4096) -> dict:
    from seiso.inference.model_pool import fit_llama_gpu_layers, llama_load_kwargs
    from seiso.memory.platform_profile import apply_platform_memory_profile
    from seiso.memory.protection import (
        gpu_batch_tier_caps,
        headroom_mb,
        llama_model_is_tight_vram_fit,
        llama_prefill_needs_reload,
    )
    from seiso.platform import use_linux_nvidia_inference_guards

    apply_platform_memory_profile()
    free_mb = headroom_mb()
    kwargs = llama_load_kwargs(n_ctx, model_path=model_path)
    kwargs["_model_path"] = model_path
    from seiso.memory.protection import clamp_llama_load_kwargs

    kwargs = clamp_llama_load_kwargs(kwargs)
    gpu_layers = fit_llama_gpu_layers(model_path, -1, free_mb, n_ctx=n_ctx)
    tight = llama_model_is_tight_vram_fit(
        model_path=model_path,
        free_mb=free_mb,
        n_gpu_layers=gpu_layers,
        n_ctx=n_ctx,
    )
    messages = [{"role": "user", "content": "Reply with exactly: OK"}]
    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=model_path,
        messages=messages,
        n_ctx=n_ctx,
        loaded_n_batch=int(kwargs.get("n_batch") or 0),
        loaded_n_ubatch=int(kwargs.get("n_ubatch") or 0),
        loaded_n_gpu_layers=gpu_layers,
        loaded_headroom_mb=free_mb,
    )
    long_reload, _, _ = llama_prefill_needs_reload(
        model_path=model_path,
        messages=_long_messages(),
        n_ctx=n_ctx,
        loaded_n_batch=int(kwargs.get("n_batch") or 0),
        loaded_n_ubatch=int(kwargs.get("n_ubatch") or 0),
        loaded_n_gpu_layers=gpu_layers,
        loaded_headroom_mb=free_mb,
    )
    gpu_total = free_mb
    try:
        from seiso.memory.protection import discrete_gpu_total_mb

        gpu_total = discrete_gpu_total_mb() or free_mb
    except Exception:
        pass
    tier_batch, tier_ubatch = gpu_batch_tier_caps(gpu_total, "normal")
    return {
        "native_guards": use_linux_nvidia_inference_guards(),
        "free_mb": free_mb,
        "gpu_total_mb": gpu_total,
        "tier_batch": tier_batch,
        "tier_ubatch": tier_ubatch,
        "kwargs": {
            k: kwargs[k]
            for k in (
                "n_batch",
                "n_ubatch",
                "n_gpu_layers",
                "n_ctx",
                "flash_attn",
                "offload_kqv",
                "op_offload",
            )
            if k in kwargs
        },
        "gpu_layers": gpu_layers,
        "tight_fit": tight,
        "prefill_reload": needs_reload,
        "long_prefill_reload": long_reload,
        "safe_batch": safe_batch,
        "safe_ubatch": safe_ubatch,
    }


async def _run_inference(model_path: str, *, max_tokens: int = 64) -> dict:
    from seiso.inference.benchmark import _timed_stream
    from seiso.inference.model_pool import get_model_pool
    from seiso.inference.runner import LocalInferenceRunner
    from seiso.memory.platform_profile import apply_platform_memory_profile

    apply_platform_memory_profile()
    pool = get_model_pool()
    pool.cancel_and_unload()
    runner = LocalInferenceRunner()
    payload = {
        "model_path": model_path,
        "messages": [{"role": "user", "content": "Explain binary search in 3 sentences."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "inference_backend": "llamacpp",
    }
    before = gpu_snapshot()
    peak = before.used_mb
    t0 = time.perf_counter()
    try:
        output, _, ttft_ms, generate_ms, output_tokens = await _timed_stream(
            payload, runner=runner
        )
        elapsed = time.perf_counter() - t0
        snap = gpu_snapshot()
        peak = max(peak, snap.used_mb)
        tps = (
            output_tokens / (generate_ms / 1000.0)
            if generate_ms > 0 and output_tokens > 0
            else 0.0
        )
        return {
            "ok": bool(output.strip()),
            "error": "",
            "ttft_ms": round(ttft_ms, 2),
            "generate_ms": round(generate_ms, 2),
            "elapsed_s": round(elapsed, 2),
            "tokens_per_sec": round(tps, 2),
            "output_tokens": output_tokens,
            "output_chars": len(output),
            "vram_before_mb": before.used_mb,
            "vram_peak_mb": peak,
            "vram_after_mb": snap.used_mb,
        }
    except Exception as exc:
        snap = gpu_snapshot()
        peak = max(peak, snap.used_mb)
        return {
            "ok": False,
            "error": str(exc),
            "ttft_ms": 0.0,
            "generate_ms": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "tokens_per_sec": 0.0,
            "output_tokens": 0,
            "output_chars": 0,
            "vram_before_mb": before.used_mb,
            "vram_peak_mb": peak,
            "vram_after_mb": snap.used_mb,
        }
    finally:
        pool.cancel_and_unload()


async def run_live(models: list[str]) -> list[ModelLiveResult]:
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory

    results: list[ModelLiveResult] = []
    pool = get_model_pool()
    for model_path in models:
        path = Path(model_path)
        name = path.name
        print(f"\n=== {name} ===", flush=True)
        if not path.is_file():
            results.append(
                ModelLiveResult(model_path=model_path, ok=False, error="model file missing")
            )
            print("SKIP: missing file", flush=True)
            continue

        pool.cancel_and_unload()
        release_cached_memory(sync=True)
        import gc

        gc.collect()
        await asyncio.sleep(2.5)

        analysis = _analyze_model(model_path)
        print("analysis:", json.dumps(analysis, indent=2, default=str), flush=True)

        infer = await _run_inference(model_path)
        print("inference:", json.dumps(infer, indent=2), flush=True)

        ok = infer["ok"] and not analysis["prefill_reload"]
        if analysis["tight_fit"] and analysis["kwargs"].get("n_batch", 0) > 512:
            ok = False
            infer["error"] = infer["error"] or "tight model loaded with batch > 512"

        result = ModelLiveResult(
            model_path=model_path,
            ok=ok,
            error=infer["error"],
            load_kwargs=analysis["kwargs"],
            tight_fit=analysis["tight_fit"],
            prefill_reload=analysis["prefill_reload"],
            long_prefill_reload=analysis["long_prefill_reload"],
            safe_batch=analysis["safe_batch"],
            safe_ubatch=analysis["safe_ubatch"],
            gpu_layers=analysis["gpu_layers"],
            vram_before_mb=infer["vram_before_mb"],
            vram_peak_mb=infer["vram_peak_mb"],
            vram_after_mb=infer["vram_after_mb"],
            ttft_ms=infer["ttft_ms"],
            generate_ms=infer["generate_ms"],
            tokens_per_sec=infer["tokens_per_sec"],
            output_tokens=infer["output_tokens"],
        )
        results.append(result)
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}: {name}", flush=True)
    return results


def main() -> int:
    models = [p for p in (sys.argv[1:] or DEFAULT_MODELS)]
    print("GPU:", gpu_snapshot(), flush=True)
    results = asyncio.run(run_live(models))
    summary = {
        "gpu": asdict(gpu_snapshot()),
        "results": [asdict(r) for r in results],
        "passed": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
