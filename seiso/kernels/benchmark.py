#!/usr/bin/env python3
"""Benchmark Seiso fused kernels vs PyTorch baselines."""

from __future__ import annotations

import argparse
import time


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_rms_norm(rows: int, cols: int, dtype: str) -> None:
    import torch

    from seiso.kernels.dispatch import active_backend, fused_rms_norm

    device = "cuda"
    x = torch.randn(rows, cols, device=device, dtype=getattr(torch, dtype))
    w = torch.ones(cols, device=device, dtype=getattr(torch, dtype))
    r = torch.randn(rows, cols, device=device, dtype=getattr(torch, dtype))

    def pytorch():
        y = x + r
        v = y.pow(2).mean(dim=-1, keepdim=True)
        return y * torch.rsqrt(v + 1e-6) * w

    def fused():
        return fused_rms_norm(x, w, residual=r)

    pt_ms = _bench(pytorch)
    fused_ms = _bench(fused)
    print(
        f"RMSNorm [{active_backend()}] {rows}x{cols} {dtype}: pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
    )


def bench_swiglu(rows: int, cols: int, dtype: str) -> None:
    import torch

    from seiso.kernels.dispatch import active_backend, fused_swiglu

    device = "cuda"
    gate = torch.randn(rows, cols, device=device, dtype=getattr(torch, dtype))
    up = torch.randn(rows, cols, device=device, dtype=getattr(torch, dtype))

    def pytorch():
        return torch.nn.functional.silu(gate) * up

    def fused():
        return fused_swiglu(gate, up)

    pt_ms = _bench(pytorch)
    fused_ms = _bench(fused)
    print(
        f"SwiGLU  [{active_backend()}] {rows}x{cols} {dtype}: pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
    )


def bench_cross_entropy(rows: int, vocab: int, dtype: str) -> None:
    import torch

    from seiso.kernels.dispatch import active_backend, fused_cross_entropy_loss

    device = "cuda"
    logits = torch.randn(
        rows, vocab, device=device, dtype=getattr(torch, dtype), requires_grad=True
    )
    labels = torch.randint(0, vocab, (rows,), device=device)
    labels[::17] = -100

    def pytorch():
        local_logits = logits.detach().clone().requires_grad_(True)
        out = torch.nn.functional.cross_entropy(local_logits, labels, ignore_index=-100)
        out.backward()
        return out

    def fused():
        local_logits = logits.detach().clone().requires_grad_(True)
        out = fused_cross_entropy_loss(local_logits, labels)
        out.backward()
        return out

    pt_ms = _bench(pytorch)
    fused_ms = _bench(fused)
    print(
        f"CE      [{active_backend()}] {rows}x{vocab} {dtype}: pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Seiso fused kernels")
    parser.add_argument("--rows", type=int, default=4096, help="Token rows (batch*seq)")
    parser.add_argument(
        "--hidden", type=int, default=4096, help="Hidden / intermediate dim"
    )
    parser.add_argument("--vocab", type=int, default=32000, help="Vocab size for CE")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--op", choices=["all", "rms", "swiglu", "ce"], default="all")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm GPU required for kernel benchmarks")

    from seiso.kernels import kernel_metadata

    print("Kernel stack:", kernel_metadata())
    print()

    if args.op in ("all", "rms"):
        bench_rms_norm(args.rows, args.hidden, args.dtype)
    if args.op in ("all", "swiglu"):
        bench_swiglu(args.rows, args.hidden, args.dtype)
    if args.op in ("all", "ce"):
        bench_cross_entropy(args.rows, args.vocab, args.dtype)


if __name__ == "__main__":
    main()
