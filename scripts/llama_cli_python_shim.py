#!/usr/bin/env python3
"""Drop-in llama-cli shim backed by llama-cpp-python (GPU when available).

adaptive_quant invokes ``llama-cli`` as a subprocess; this shim accepts the same
core flags and prints timing lines that ``parse_llama_cpp_metrics`` understands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-m", "--model", dest="model")
    parser.add_argument("-p", "--prompt", dest="prompt", default="")
    parser.add_argument("-ngl", dest="ngl", type=int, default=-1)
    parser.add_argument("-t", dest="threads", type=int, default=8)
    parser.add_argument("-c", dest="context", type=int, default=2048)
    parser.add_argument("-n", dest="tokens", type=int, default=64)
    parser.add_argument("-st", action="store_true")
    parser.add_argument("--version", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))

    from seiso.platform import ensure_cuda_library_path

    ensure_cuda_library_path()
    from llama_cpp import Llama

    if args.version:
        gpu = False
        try:
            gpu = bool(Llama.__module__ and Llama) and bool(
                __import__("llama_cpp").llama_supports_gpu_offload()
            )
        except Exception:
            gpu = False
        print("llama_cli_python_shim (llama-cpp-python)")
        if gpu:
            print("CUDA : ON")
        else:
            print("CUDA : OFF")
        return 0

    if not args.model:
        print("error: -m model.gguf is required", file=sys.stderr)
        return 2

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(f"error: model not found: {model_path}", file=sys.stderr)
        return 2

    n_gpu_layers = int(args.ngl)
    llm = Llama(
        model_path=str(model_path),
        n_ctx=int(args.context),
        n_threads=int(args.threads),
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    prompt = args.prompt or ""
    started = time.perf_counter()
    out = llm(
        prompt,
        max_tokens=max(1, int(args.tokens)),
        echo=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = ""
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices and isinstance(choices[0], dict):
            text = str((choices[0].get("text") or "")).strip()
    token_count = max(1, int(args.tokens))
    per_token_ms = elapsed_ms / token_count
    tok_s = 1000.0 / per_token_ms if per_token_ms > 0 else 0.0
    if text:
        print(text)
    print(
        f"{elapsed_ms:.2f} ms / {token_count} tokens "
        f"( {per_token_ms:.2f} ms per token, {tok_s:.2f} tok/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
