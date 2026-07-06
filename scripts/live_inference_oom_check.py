#!/usr/bin/env python3
"""Live NVIDIA inference OOM guard check using Hugging Face Hub repo ids only."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class GpuSnapshot:
    used_mb: int
    free_mb: int
    total_mb: int


@dataclass
class ModelLiveResult:
    repo_id: str
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


def _init_seiso_hub(*, token: str | None = None) -> Path:
    from seiso.models.hf_env import configure_hf_hub_auth, configure_hf_hub_cache

    cache = configure_hf_hub_cache()
    configure_hf_hub_auth(token)
    return cache


def _seiso_hub_cache() -> Path:
    from seiso.models.hf_env import resolve_hf_cache_dir

    return resolve_hf_cache_dir()


def _validate_repo_id(repo_id: str) -> str:
    repo = repo_id.strip()
    if not _HF_REPO_RE.match(repo):
        raise ValueError(f"Invalid Hugging Face repo id: {repo_id!r} (expected org/name)")
    return repo


def _is_chat_gguf_filename(name: str) -> bool:
    lower = name.lower()
    return (
        lower.endswith(".gguf")
        and "mmproj" not in lower
        and not lower.startswith("mtp-")
    )


def _is_chat_gguf_file(path: Path) -> bool:
    if not _is_chat_gguf_filename(path.name):
        return False
    try:
        from seiso.inference.backends import gguf_is_supported_by_llamacpp, is_dflash_draft

        if is_dflash_draft(str(path)):
            return False
        return gguf_is_supported_by_llamacpp(str(path))
    except Exception:
        return True


def _pick_repo_gguf_file(repo_files: list[str]) -> str | None:
    from seiso.models.gguf_quant import rank_gguf_filenames

    ggufs = [name for name in repo_files if _is_chat_gguf_filename(name)]
    if not ggufs:
        return None

    def is_huge(name: str) -> bool:
        lower = name.lower()
        return any(x in lower for x in ("270b", "235b", "100b"))

    pool = [name for name in ggufs if not is_huge(name)] or ggufs
    ranked = rank_gguf_filenames(pool, preferred="Q4_K_M")
    return ranked[0] if ranked else None


def resolve_hf_repo(repo_id: str, *, token: str | None = None) -> str | None:
    """Download or refresh a compatible chat GGUF from a Hugging Face repo id."""
    repo_id = _validate_repo_id(repo_id)
    cache = _seiso_hub_cache()

    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        filename = _pick_repo_gguf_file(api.list_repo_files(repo_id))
        if not filename:
            print(f"SKIP: {repo_id} has no compatible chat GGUF files on Hugging Face", flush=True)
            return None
        downloaded = hf_hub_download(
            repo_id,
            filename,
            token=token,
            cache_dir=str(cache),
        )
        path = Path(downloaded)
        if not _is_chat_gguf_file(path):
            print(f"SKIP: {repo_id}/{filename} is not supported by llama.cpp", flush=True)
            return None
        return str(path.resolve())
    except Exception as exc:
        print(f"SKIP: failed to fetch {repo_id} from Hugging Face: {exc}", flush=True)
        return None


def discover_hf_repos(*, limit: int = 2, token: str | None = None) -> list[str]:
    """Pick compatible GGUF repos from live Hugging Face Hub search (not local disk)."""
    from seiso.models.catalog import search_catalog

    result = search_catalog(limit=max(limit * 4, 8), token=token)
    repos: list[str] = []
    for row in result.models:
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not row.get("gguf_repo"):
            continue
        if not _HF_REPO_RE.match(repo_id):
            continue
        repos.append(repo_id)
        if len(repos) >= limit:
            break
    return repos


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
        clamp_llama_load_kwargs,
        discrete_gpu_total_mb,
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
    gpu_total = discrete_gpu_total_mb() or free_mb
    tier_batch, tier_ubatch = gpu_batch_tier_caps(gpu_total, "normal")
    compact_batch, _ = gpu_batch_tier_caps(gpu_total, "compact")
    return {
        "native_guards": use_linux_nvidia_inference_guards(),
        "free_mb": free_mb,
        "gpu_total_mb": gpu_total,
        "tier_batch": tier_batch,
        "tier_ubatch": tier_ubatch,
        "compact_batch_cap": compact_batch,
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


async def run_live(
    repo_jobs: Iterable[tuple[str, str]], *, max_tokens: int
) -> list[ModelLiveResult]:
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory

    results: list[ModelLiveResult] = []
    pool = get_model_pool()
    for repo_id, model_path in repo_jobs:
        name = Path(model_path).name
        print(f"\n=== {repo_id} -> {name} ===", flush=True)

        pool.cancel_and_unload()
        release_cached_memory(sync=True)
        gc.collect()
        await asyncio.sleep(2.5)

        analysis = _analyze_model(model_path)
        print("analysis:", json.dumps(analysis, indent=2, default=str), flush=True)

        infer = await _run_inference(model_path, max_tokens=max_tokens)
        print("inference:", json.dumps(infer, indent=2), flush=True)

        compact_cap = int(analysis.get("compact_batch_cap") or 512)
        ok = infer["ok"] and not analysis["prefill_reload"]
        if analysis["tight_fit"] and analysis["kwargs"].get("n_batch", 0) > compact_cap:
            ok = False
            infer["error"] = (
                infer["error"]
                or f"tight model loaded with batch > compact cap ({compact_cap})"
            )

        results.append(
            ModelLiveResult(
                repo_id=repo_id,
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
        )
        print(f"{'PASS' if ok else 'FAIL'}: {repo_id}", flush=True)
    return results


def _resolve_repos(
    repo_ids: list[str], *, token: str | None, discover_limit: int
) -> list[tuple[str, str]]:
    if not repo_ids:
        repo_ids = discover_hf_repos(limit=discover_limit, token=token)
        if not repo_ids:
            print(
                "No compatible GGUF repos found on Hugging Face Hub. "
                "Pass one or more repo ids (org/model).",
                flush=True,
            )
            return []
        print("Selected from Hugging Face Hub:", flush=True)
        for repo_id in repo_ids:
            print(f"  - {repo_id}", flush=True)

    jobs: list[tuple[str, str]] = []
    for raw in repo_ids:
        try:
            repo_id = _validate_repo_id(raw)
        except ValueError as exc:
            print(f"SKIP: {exc}", flush=True)
            continue
        path = resolve_hf_repo(repo_id, token=token)
        if path:
            jobs.append((repo_id, path))
    return jobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live NVIDIA inference OOM guard check using Hugging Face repo ids only."
        )
    )
    parser.add_argument(
        "repos",
        nargs="*",
        metavar="ORG/MODEL",
        help="Hugging Face GGUF repo id(s), e.g. org/model-GGUF",
    )
    parser.add_argument(
        "--discover",
        type=int,
        default=2,
        metavar="N",
        help="When no repos are passed, pick N compatible GGUF repos from live Hub search.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Generation length for the live inference probe (default: 64).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Optional Hugging Face token for gated repo downloads.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = _init_seiso_hub(token=args.hf_token)
    print("GPU:", gpu_snapshot(), flush=True)
    print("Seiso HF cache:", cache, flush=True)

    jobs = _resolve_repos(args.repos, token=args.hf_token, discover_limit=args.discover)
    if not jobs:
        return 2

    results = asyncio.run(run_live(jobs, max_tokens=args.max_tokens))
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
