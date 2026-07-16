#!/usr/bin/env python3
"""Benchmark Seiso fused kernels vs PyTorch baselines.

Also supports ``--roofline``: shape → rough FLOP/byte estimates and
likely bandwidth vs compute labels for **Seiso fused ops only**.
Those estimates never block training.
"""

from __future__ import annotations

import argparse
import json
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
        f"RMSNorm [{active_backend()}] {rows}x{cols} {dtype}: "
        f"pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
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
        f"SwiGLU  [{active_backend()}] {rows}x{cols} {dtype}: "
        f"pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
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
        f"CE      [{active_backend()}] {rows}x{vocab} {dtype}: "
        f"pytorch {pt_ms:.3f}ms  fused {fused_ms:.3f}ms  speedup {pt_ms / fused_ms:.2f}x"
    )


def run_roofline_report(
    *,
    rows: int,
    hidden: int,
    vocab: int,
    dtype: str,
    lora_rank: int = 16,
    intermediate: int | None = None,
    as_json: bool = False,
) -> int:
    """Print intensity estimates for Seiso fused ops; return 0."""
    from seiso.kernels.roofline import (
        estimate_seiso_fused_ops,
        format_roofline_report,
    )

    estimates = estimate_seiso_fused_ops(
        rows=rows,
        hidden=hidden,
        vocab=vocab,
        intermediate=intermediate,
        lora_rank=lora_rank,
        dtype=dtype,
    )
    from seiso.kernels.roofline import REFERENCE_RIDGE_FLOP_PER_BYTE

    if as_json:
        print(
            json.dumps(
                {
                    "scope": "seiso_fused_ops_only",
                    "reference_ridge_flop_per_byte": REFERENCE_RIDGE_FLOP_PER_BYTE,
                    "performance_truth_rule": (
                        "source_of_truth only for FP16/BF16 GEMM-family ops with "
                        f"intensity >= {REFERENCE_RIDGE_FLOP_PER_BYTE:g} FLOP/byte "
                        "(H100-class dense TC/HBM reference bar); marks a strong "
                        "compute-bound *candidate* under efficient dense GEMM, "
                        "not a measured roofline; never blocks training"
                    ),
                    "disclaimer": (
                        "Elementwise/CE: lower-bound streams (heuristic). "
                        "GEMM: full classic traffic per matmul (conservative I). "
                        f"SoT bar: FP16/BF16 GEMM I>={REFERENCE_RIDGE_FLOP_PER_BYTE:g}; "
                        "never blocks training"
                    ),
                    "estimates": [e.to_dict() for e in estimates],
                    "source_of_truth_ops": [e.op for e in estimates if e.performance_truth],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_roofline_report(estimates))
        print("--- bound log (Seiso kernels) ---")
        for est in estimates:
            print(
                f"kernel={est.op} likely_bound={est.likely_bound} "
                f"confidence={est.confidence} performance_truth={est.performance_truth} "
                f"traffic_model={est.traffic_model} "
                f"intensity_flop_per_byte={est.intensity_flop_per_byte:.4g}"
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Seiso fused kernels, or report rough arithmetic-intensity "
            "estimates (--roofline) for those kernels only."
        )
    )
    parser.add_argument("--rows", type=int, default=4096, help="Token rows (batch*seq)")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden / intermediate base dim")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocab size for CE")
    parser.add_argument(
        "--intermediate",
        type=int,
        default=None,
        help=(
            "MLP intermediate size for roofline (default: 4*hidden textbook; "
            "many SwiGLU LLMs use ~8/3*hidden — pass real width for faithful I)"
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank for roofline LoRA estimates",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--op",
        choices=["all", "rms", "swiglu", "ce"],
        default="all",
        help="Timed benchmark op (ignored when --roofline-only or --json with roofline)",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help=(
            "Print shape→FLOP/byte estimates for Seiso fused ops. "
            "Uses a fixed H100-class ~300 FLOP/byte bar only as the shape-math "
            "SoT threshold for FP16/BF16 GEMM compute-bound *candidates* "
            "(not a per-GPU measured ridge; never gates training)"
        ),
    )
    parser.add_argument(
        "--roofline-only",
        action="store_true",
        help="Only run roofline estimates (no CUDA timing; CPU-only OK)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "With --roofline / --roofline-only: emit JSON only (no text, no timed "
            "benches). Without roofline flags, --json is ignored with a warning."
        ),
    )
    args = parser.parse_args()

    if args.json and not (args.roofline or args.roofline_only):
        print(
            "warning: --json applies only with --roofline / --roofline-only; ignoring",
            flush=True,
        )

    if args.roofline or args.roofline_only:
        run_roofline_report(
            rows=args.rows,
            hidden=args.hidden,
            vocab=args.vocab,
            dtype=args.dtype,
            lora_rank=args.lora_rank,
            intermediate=args.intermediate,
            as_json=args.json,
        )
        # JSON output must stay machine-readable: do not append timed benches.
        if args.roofline_only or args.json:
            return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA/ROCm GPU required for timed kernel benchmarks "
            "(use --roofline-only for intensity estimates without a GPU)"
        )

    from seiso.kernels import kernel_metadata

    print("Kernel stack:", kernel_metadata())
    print()

    if args.op in ("all", "rms"):
        bench_rms_norm(args.rows, args.hidden, args.dtype)
    if args.op in ("all", "swiglu"):
        # Elementwise SwiGLU bench uses intermediate width = hidden arg (historical).
        bench_swiglu(args.rows, args.hidden, args.dtype)
    if args.op in ("all", "ce"):
        bench_cross_entropy(args.rows, args.vocab, args.dtype)


if __name__ == "__main__":
    main()
