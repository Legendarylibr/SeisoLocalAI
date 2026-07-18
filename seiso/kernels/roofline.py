"""Arithmetic-intensity estimates for Seiso fused kernels.

Shape → FLOP/byte for **Seiso ops only**. Used by ``seiso-bench-kernels
--roofline``. Never gates training.

Performance source-of-truth *bar* (shape math only)
---------------------------------------------------
An estimate is marked ``performance_truth=true`` only when **all** of:

1. Op is GEMM-family (``fused_mlp_swiglu``, ``lora_delta``, ``lora_qkv_delta``)
2. Dtype is FP16 or BF16 (H100 TC ridge is defined for those paths)
3. Intensity **I ≥ 300 FLOP/byte**

**300** is the H100 SXM dense FP16/BF16 Tensor Core / HBM3 **reference ridge**
(~989 TFLOPS / 3.35 TB/s ≈ 295, bar rounded to 300). It is the bar at which
Seiso will call a shape a **strong compute-bound candidate** under efficient
dense GEMM — not a measured roofline, not every GPU's true ridge, and not a
training gate.

Below that bar (or for elementwise/CE/float32), labels are **heuristic only**.

Real backends vs shape math
---------------------------
These numbers assume *efficient* dense GEMM (cuBLAS / Tensor Cores). Production
routing matches that for GEMM-family ops:

* **fused_mlp_swiglu** — torch/cuBLAS ``x@W.T`` for gate/up + fused SwiGLU
  epilogue. Scalar CUDA matmul is opt-in only (``SEISO_KERNEL_ALLOW_NAIVE_MLP``).
* **lora_qkv_delta / lora_delta** — torch/cuBLAS by default. Naive custom CUDA
  only with ``SEISO_KERNEL_ALLOW_NAIVE_LORA`` on tiny no-grad shapes.
* **rms_norm / swiglu_elementwise / fused_cross_entropy** — custom elementwise
  or vocab kernels; intensity is heuristic bandwidth-class (never SoT).

SoT is a *shape* claim about FLOP/byte under efficient dense GEMM — which is
what production routing uses for MLP/LoRA.

Traffic accounting
------------------
* Elementwise / CE: lower-bound stream counts (heuristic only; never SoT).
* GEMM-family: full classic GEMM traffic **per matmul** (conservative for
  intensity — harder to hit SoT; no optimistic shared-activation discount).
  QKV is modeled as three independent LoRA pairs (pessimistic vs fused
  shared-``x``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

LikelyBound = Literal["bandwidth", "mixed", "compute"]
Confidence = Literal["heuristic", "source_of_truth"]

_BANDWIDTH_MAX = 10.0

# H100 SXM dense FP16/BF16 TC / HBM3 ≈ 295 FLOP/byte; SoT claim bar = 300.
REFERENCE_RIDGE_FLOP_PER_BYTE = 300.0

_SOT_ELIGIBLE_OPS = frozenset(
    {
        "fused_mlp_swiglu",
        "lora_delta",
        "lora_qkv_delta",
    }
)
_SOT_ELIGIBLE_DTYPES = frozenset(
    {
        "float16",
        "half",
        "bfloat16",
        "bf16",
    }
)


@dataclass(frozen=True)
class IntensityEstimate:
    """One kernel shape's intensity snapshot."""

    op: str
    shape: dict[str, Any]
    flops: float
    bytes_moved: float
    intensity_flop_per_byte: float
    likely_bound: LikelyBound
    confidence: Confidence
    performance_truth: bool
    reference_ridge_flop_per_byte: float
    note: str
    traffic_model: str = "unspecified"

    def summary_line(self) -> str:
        ridge = self.reference_ridge_flop_per_byte
        if self.performance_truth and self.likely_bound == "compute":
            bound = (
                f"COMPUTE-BOUND CANDIDATE [shape-math SoT bar: "
                f"I={self.intensity_flop_per_byte:.1f} ≥ {ridge:.0f} FLOP/byte "
                f"FP16/BF16 GEMM; not a measured roofline]"
            )
        else:
            soft = {
                "bandwidth": "likely bandwidth-limited",
                "mixed": "mixed / shape-sensitive",
                "compute": "compute-leaning",
            }[self.likely_bound]
            if self.intensity_flop_per_byte >= ridge and not self.performance_truth:
                reason = "not SoT-eligible (op/dtype gate)"
            else:
                reason = f"I < {ridge:.0f} or heuristic-only op"
            bound = f"{soft} [heuristic only; {reason}]"
        return (
            f"{self.op}: shape={self._shape_short()}  "
            f"I={self.intensity_flop_per_byte:.2f} FLOP/byte  → {bound}"
            f"  (flops={self.flops:.3g}, bytes={self.bytes_moved:.3g}, "
            f"traffic={self.traffic_model})"
        )

    def _shape_short(self) -> str:
        parts = [f"{k}={v}" for k, v in self.shape.items() if k != "dtype"]
        dtype = self.shape.get("dtype")
        if dtype:
            parts.append(str(dtype))
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dtype_nbytes(dtype: str) -> int:
    key = str(dtype).lower().replace("torch.", "")
    if key in {"float16", "half", "bfloat16", "bf16"}:
        return 2
    if key in {"float32", "float", "fp32"}:
        return 4
    if key in {"float64", "double"}:
        return 8
    raise ValueError(f"unsupported dtype for intensity estimate: {dtype!r}")


def _normalize_dtype_key(dtype: str) -> str:
    return str(dtype).lower().replace("torch.", "").strip()


def gemm_flops_bytes(
    *,
    m: int,
    n: int,
    k: int,
    elem_bytes: int,
) -> tuple[float, float]:
    """Classic dense GEMM: C[m,n] = A[m,k] @ B[k,n].

    FLOPs = 2·m·n·k (FMA as 2 FLOPs).
    Bytes = (m·k + k·n + m·n) · elem_bytes (full classic GEMM traffic).
    """
    m, n, k = int(m), int(n), int(k)
    eb = int(elem_bytes)
    flops = 2.0 * m * n * k
    bytes_moved = float(m * k + k * n + m * n) * eb
    return flops, bytes_moved


def is_performance_source_of_truth(
    *,
    op: str,
    intensity_flop_per_byte: float,
    dtype: str = "bfloat16",
    ridge: float = REFERENCE_RIDGE_FLOP_PER_BYTE,
) -> bool:
    """True only for FP16/BF16 GEMM-family ops with I ≥ reference ridge (300)."""
    if op not in _SOT_ELIGIBLE_OPS:
        return False
    if _normalize_dtype_key(dtype) not in _SOT_ELIGIBLE_DTYPES:
        return False
    return float(intensity_flop_per_byte) >= float(ridge)


def classify_bound(
    intensity: float,
    *,
    op: str = "",
    dtype: str = "bfloat16",
    ridge: float = REFERENCE_RIDGE_FLOP_PER_BYTE,
) -> tuple[LikelyBound, Confidence, bool, str]:
    """Classify bound + confidence.

    Shape-math SoT bar: GEMM-family + FP16/BF16 + I ≥ 300.
    That is a strong compute-bound *candidate*, not a measured roofline.
    """
    intensity = float(intensity)
    ridge = float(ridge)
    dtype_key = _normalize_dtype_key(dtype)
    sot = is_performance_source_of_truth(
        op=op,
        intensity_flop_per_byte=intensity,
        dtype=dtype,
        ridge=ridge,
    )

    if sot:
        margin = intensity - ridge
        return (
            "compute",
            "source_of_truth",
            True,
            (
                f"Shape-math SoT bar met: I={intensity:.1f} ≥ {ridge:.0f} FLOP/byte "
                f"on FP16/BF16 GEMM-family op (H100-class dense TC/HBM ridge ≈ 295; "
                f"bar=300). Margin +{margin:.1f}. Strong compute-bound *candidate* "
                f"if an efficient dense GEMM path runs (cuBLAS/Tensor Cores); "
                f"not a measured device roofline."
            ),
        )

    if dtype_key not in _SOT_ELIGIBLE_DTYPES and op in _SOT_ELIGIBLE_OPS:
        return (
            "mixed" if intensity >= _BANDWIDTH_MAX else "bandwidth",
            "heuristic",
            False,
            (
                f"Heuristic only: SoT bar is defined for FP16/BF16 TC paths; "
                f"dtype={dtype!r} uses a different peak/ridge. I={intensity:.2f}."
            ),
        )

    if intensity < _BANDWIDTH_MAX:
        return (
            "bandwidth",
            "heuristic",
            False,
            (
                f"Heuristic only (not SoT; I={intensity:.2f} < {ridge:.0f}). "
                f"Low reuse — typically HBM/launch limited "
                f"(band < {_BANDWIDTH_MAX:.0f} FLOP/byte)."
            ),
        )

    if intensity >= ridge and op and op not in _SOT_ELIGIBLE_OPS:
        return (
            "mixed",
            "heuristic",
            False,
            (
                f"Heuristic only: I={intensity:.1f} ≥ {ridge:.0f} but op {op!r} is "
                f"not GEMM-family SoT-eligible (elementwise/CE stay diagnostic)."
            ),
        )

    return (
        "mixed",
        "heuristic",
        False,
        (
            f"Heuristic only (not SoT; I={intensity:.2f} < {ridge:.0f}). "
            f"Moderate reuse — bound depends on GPU peak FLOPS vs bandwidth "
            f"(band {_BANDWIDTH_MAX:.0f}–{ridge:.0f} FLOP/byte)."
        ),
    )


def _pack(
    *,
    op: str,
    shape: dict[str, Any],
    flops: float,
    bytes_moved: float,
    traffic_model: str,
    extra_note: str = "",
) -> IntensityEstimate:
    flops = float(max(flops, 0.0))
    bytes_moved = float(max(bytes_moved, 1.0))
    intensity = flops / bytes_moved
    dtype = str(shape.get("dtype") or "bfloat16")
    bound, confidence, performance_truth, note = classify_bound(intensity, op=op, dtype=dtype)
    if extra_note:
        note = f"{extra_note} {note}"
    return IntensityEstimate(
        op=op,
        shape=shape,
        flops=flops,
        bytes_moved=bytes_moved,
        intensity_flop_per_byte=intensity,
        likely_bound=bound,
        confidence=confidence,
        performance_truth=performance_truth,
        reference_ridge_flop_per_byte=REFERENCE_RIDGE_FLOP_PER_BYTE,
        note=note.strip(),
        traffic_model=traffic_model,
    )


def estimate_rms_norm(
    *,
    rows: int,
    hidden: int,
    dtype: str = "bfloat16",
    residual: bool = True,
) -> IntensityEstimate:
    """Fused residual RMSNorm: elementwise — never SoT."""
    n = int(rows) * int(hidden)
    elem = dtype_nbytes(dtype)
    flops = 6.0 * n + float(rows)
    tensors = 3 if residual else 2
    bytes_moved = tensors * n * elem + int(hidden) * elem
    return _pack(
        op="rms_norm" + ("+residual" if residual else ""),
        shape={"rows": rows, "hidden": hidden, "dtype": dtype, "residual": residual},
        flops=flops,
        bytes_moved=bytes_moved,
        traffic_model="lower_bound_streams",
        extra_note="Elementwise-heavy fused norm.",
    )


def estimate_swiglu(
    *,
    rows: int,
    intermediate: int,
    dtype: str = "bfloat16",
) -> IntensityEstimate:
    """Elementwise SwiGLU on precomputed gate/up — never SoT."""
    n = int(rows) * int(intermediate)
    elem = dtype_nbytes(dtype)
    flops = 6.0 * n
    bytes_moved = 3.0 * n * elem
    return _pack(
        op="swiglu_elementwise",
        shape={"rows": rows, "intermediate": intermediate, "dtype": dtype},
        flops=flops,
        bytes_moved=bytes_moved,
        traffic_model="lower_bound_streams",
        extra_note="Pure elementwise silu×mul (no GEMM).",
    )


def estimate_fused_mlp_swiglu(
    *,
    rows: int,
    hidden: int,
    intermediate: int | None = None,
    dtype: str = "bfloat16",
) -> IntensityEstimate:
    """Fused gate/up + SwiGLU: two full classic GEMMs + epilogue out stream.

    Default intermediate is ``4 * hidden`` (textbook). Many SwiGLU LLMs use
    ~``8/3 * hidden`` — pass the real width for model-faithful intensity.
    """
    inter = int(intermediate if intermediate is not None else hidden * 4)
    r, h, mid = int(rows), int(hidden), inter
    elem = dtype_nbytes(dtype)
    f1, b1 = gemm_flops_bytes(m=r, n=mid, k=h, elem_bytes=elem)
    f2, b2 = gemm_flops_bytes(m=r, n=mid, k=h, elem_bytes=elem)
    elem_flops = 6.0 * r * mid
    flops = f1 + f2 + elem_flops
    # Full traffic per GEMM + final activation out stream.
    bytes_moved = b1 + b2 + float(r * mid) * elem
    return _pack(
        op="fused_mlp_swiglu",
        shape={
            "rows": r,
            "hidden": h,
            "intermediate": mid,
            "dtype": dtype,
        },
        flops=flops,
        bytes_moved=bytes_moved,
        traffic_model="classic_gemm_per_matmul",
        extra_note=(
            "Two dense GEMMs (full classic traffic each) + fused silu×mul. "
            "Default intermediate=4×hidden; real SwiGLU often ~8/3×hidden. "
            "Production uses cuBLAS GEMMs + fused SwiGLU epilogue."
        ),
    )


def estimate_lora_delta(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: str = "bfloat16",
) -> IntensityEstimate:
    """LoRA delta: two skinny classic GEMMs."""
    r, din, dout, k = int(rows), int(in_features), int(out_features), int(rank)
    elem = dtype_nbytes(dtype)
    f1, b1 = gemm_flops_bytes(m=r, n=k, k=din, elem_bytes=elem)
    f2, b2 = gemm_flops_bytes(m=r, n=dout, k=k, elem_bytes=elem)
    return _pack(
        op="lora_delta",
        shape={
            "rows": r,
            "in_features": din,
            "out_features": dout,
            "rank": k,
            "dtype": dtype,
        },
        flops=f1 + f2,
        bytes_moved=b1 + b2,
        traffic_model="classic_gemm_per_matmul",
        extra_note="Two skinny GEMMs (full traffic each).",
    )


def estimate_lora_qkv_delta(
    *,
    rows: int,
    hidden: int,
    rank: int,
    dtype: str = "bfloat16",
) -> IntensityEstimate:
    """Q/K/V LoRA as three independent LoRA pairs (pessimistic vs shared-x fuse).

    Intensity matches a single ``lora_delta`` (3f/3b); fusion reuse is *not*
    credited so SoT is not granted from optimistic shared-input traffic.
    """
    r, h, k = int(rows), int(hidden), int(rank)
    one = estimate_lora_delta(rows=r, in_features=h, out_features=h, rank=k, dtype=dtype)
    return _pack(
        op="lora_qkv_delta",
        shape={"rows": r, "hidden": h, "rank": k, "dtype": dtype},
        flops=3.0 * one.flops,
        bytes_moved=3.0 * one.bytes_moved,
        traffic_model="classic_gemm_per_matmul_x3_independent",
        extra_note=(
            "Modeled as three independent LoRA pairs (same I as one LoRA). "
            "Does not credit fused shared-x reuse."
        ),
    )


def estimate_cross_entropy(
    *,
    rows: int,
    vocab: int,
    dtype: str = "bfloat16",
) -> IntensityEstimate:
    """Fused CE: vocab logits — heuristic only (never SoT)."""
    r, v = int(rows), int(vocab)
    elem = dtype_nbytes(dtype)
    flops = 8.0 * r * v
    bytes_moved = 2.0 * r * v * elem + r * 8
    return _pack(
        op="fused_cross_entropy",
        shape={"rows": r, "vocab": v, "dtype": dtype},
        flops=flops,
        bytes_moved=bytes_moved,
        traffic_model="lower_bound_streams",
        extra_note="Vocab-wide logits dominate traffic.",
    )


def estimate_seiso_fused_ops(
    *,
    rows: int = 4096,
    hidden: int = 4096,
    vocab: int = 32000,
    intermediate: int | None = None,
    lora_rank: int = 16,
    dtype: str = "bfloat16",
) -> list[IntensityEstimate]:
    """Bundle intensity estimates for the main Seiso fused-op set."""
    inter = intermediate if intermediate is not None else hidden * 4
    return [
        estimate_rms_norm(rows=rows, hidden=hidden, dtype=dtype, residual=True),
        estimate_swiglu(rows=rows, intermediate=inter, dtype=dtype),
        estimate_fused_mlp_swiglu(rows=rows, hidden=hidden, intermediate=inter, dtype=dtype),
        estimate_lora_delta(
            rows=rows,
            in_features=hidden,
            out_features=hidden,
            rank=lora_rank,
            dtype=dtype,
        ),
        estimate_lora_qkv_delta(rows=rows, hidden=hidden, rank=lora_rank, dtype=dtype),
        estimate_cross_entropy(rows=rows, vocab=vocab, dtype=dtype),
    ]


def format_roofline_report(estimates: list[IntensityEstimate]) -> str:
    """Human-readable report for CLI."""
    ridge = REFERENCE_RIDGE_FLOP_PER_BYTE
    sot = [e for e in estimates if e.performance_truth]
    lines = [
        "Roofline-style estimates (Seiso fused ops only)",
        f"Shape-math SoT bar: FP16/BF16 GEMM-family ops with I ≥ {ridge:.0f} FLOP/byte "
        f"(H100-class dense TC/HBM reference ridge). That marks a strong compute-bound "
        f"*candidate* under efficient dense GEMM — not a measured device roofline.",
        "Elementwise/CE use lower-bound streams (heuristic). GEMMs use full classic "
        "traffic per matmul (conservative intensity for SoT).",
        "Training is never gated on these numbers.",
        "",
    ]
    for est in estimates:
        lines.append(est.summary_line())
        lines.append(
            f"  confidence={est.confidence} performance_truth={est.performance_truth} "
            f"traffic_model={est.traffic_model}"
        )
        lines.append(f"  note: {est.note}")
    lines.append("")
    if sot:
        lines.append(f"Shape-math SoT compute-bound candidates ({len(sot)}):")
        for est in sot:
            lines.append(f"  - {est.op}: I={est.intensity_flop_per_byte:.1f} FLOP/byte")
    else:
        lines.append(
            f"No op met the SoT bar (FP16/BF16 GEMM with I ≥ {ridge:.0f}). "
            "Increase rows/hidden/intermediate, or treat all labels as heuristic."
        )
    lines.append("")
    lines.append(
        "How to read: I ≥ 300 on FP16/BF16 fused MLP (or fat LoRA) → strong "
        "compute-bound candidate if GEMM runs efficiently; low I → fuse / cut "
        "traffic; between → shape-sensitive (heuristic)."
    )
    return "\n".join(lines)
