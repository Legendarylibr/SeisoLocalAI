# How to read kernel shape (Seiso fused ops)

Short guide for interpreting **shape → arithmetic intensity** estimates from:

```bash
seiso-bench-kernels --roofline-only
```

This is a **diagnostic for Seiso’s fused kernels**, not a full measured GPU roofline. Estimates **never gate training**.

---

## Two ceilings (mental model)

GPUs expose roughly two throughput limits:

1. **Math** (TFLOPS — Tensor Cores / CUDA cores)
2. **Memory bandwidth** (HBM TB/s)

**Arithmetic intensity** ≈ FLOPs performed per byte moved from/to memory.

- **Low intensity** → usually **bandwidth- or launch-limited**
- **High intensity** → **compute-bound candidate** if a large efficient GEMM actually runs

### Shape-math SoT bar (I ≥ 300, FP16/BF16 GEMM only)

Seiso sets `performance_truth=true` only when **all** hold:

1. Op is **GEMM-family** (`fused_mlp_swiglu`, `lora_delta`, `lora_qkv_delta`)
2. Dtype is **float16 or bfloat16** (TC ridge is defined for those paths)
3. Paper intensity **I ≥ 300 FLOP/byte**

**300** is the H100-class dense FP16/BF16 Tensor Core / HBM reference ridge (~989 TFLOPS / 3.35 TB/s ≈ 295, bar rounded to 300). It is the **bar for a strong compute-bound *candidate* under efficient dense GEMM** — not a measured device roofline, and not “every GPU’s ridge is 300.”

| Regime | Confidence |
|--------|------------|
| FP16/BF16 GEMM with I ≥ 300 | **`source_of_truth`** — strong compute-bound **candidate** (shape math only) |
| I &lt; 300, elementwise/CE, or **float32** | **`heuristic` only** |

---

## Traffic models (do not mix them up)

| `traffic_model` | Ops | Meaning |
|-----------------|-----|---------|
| `lower_bound_streams` | RMSNorm, SwiGLU elementwise, CE | Streams operands once (heuristic; never SoT) |
| `classic_gemm_per_matmul` | MLP, LoRA | Full classic GEMM traffic **per matmul** (conservative I for SoT) |
| `classic_gemm_per_matmul_x3_independent` | LoRA QKV | Three independent LoRA pairs — **same I as one LoRA**; does **not** credit fused shared-`x` |

GEMM SoT is deliberately **not** granted on optimistic reuse (shared activations counted once across matmuls).

Real kernels can still move more bytes (temps, multi-pass, autograd). Treat I as **directional**.

---

## How Seiso estimates intensity

| Field | Meaning |
|-------|---------|
| `flops` | Useful FLOPs for that shape (GEMM uses classic `2mnk`) |
| `bytes_moved` | Per `traffic_model` above |
| `intensity_flop_per_byte` | `flops / bytes_moved` |
| `likely_bound` | `bandwidth` \| `mixed` \| `compute` |
| `confidence` | `heuristic` \| `source_of_truth` |
| `performance_truth` | `true` only at the SoT bar |
| `traffic_model` | How bytes were counted |

---

## Seiso ops by shape

| Op | Dominant work | Typical intensity story |
|----|---------------|-------------------------|
| **RMSNorm** (+ residual) | Elementwise | Low → **bandwidth** (heuristic) |
| **SwiGLU** (gate/up already material) | Elementwise | Low → **bandwidth** (heuristic) |
| **Fused MLP SwiGLU** | Two large matmuls + silu×mul | Can meet SoT bar at large `rows`/dims (FP16/BF16) |
| **LoRA delta** | Skinny GEMMs | Modest; rank-sensitive; rarely SoT at small rank |
| **LoRA QKV** | Three LoRA pairs in the **estimate** | Reported **I = single LoRA I** (pessimistic vs shared-`x` fuse) |
| **Fused CE** | Vocab logits | Bandwidth-leaning (heuristic; never SoT) |

### Shape knobs

| Flag | Role |
|------|------|
| `--rows` | Token rows ≈ `batch × seq` |
| `--hidden` | Model hidden size |
| `--intermediate` | MLP width (**default `4×hidden` textbook**; many SwiGLU LLMs use **~8/3×hidden** — pass real width for model-faithful I) |
| `--vocab` | Vocab for CE |
| `--lora-rank` | LoRA rank |
| `--dtype` | `float16` / `bfloat16` (SoT-eligible) or `float32` (heuristic only) |

---

## Bound labels

| Label | When | Confidence |
|-------|------|------------|
| `bandwidth` | I ≲ 10 | Heuristic |
| `mixed` | 10 ≲ I &lt; 300, non-GEMM, or float32 | Heuristic |
| `compute` | I ≥ **300** on **FP16/BF16 GEMM** only | Shape-math SoT bar (candidate, not measured) |

---

## What Seiso already does with this logic

Without a roofline UI or training gates:

- **Fuse** low-intensity work (RMSNorm, SwiGLU, CE)
- **MLP** = torch/cuBLAS gate+up GEMMs + fused SwiGLU epilogue (default)
- **LoRA** = torch/cuBLAS skinny GEMMs by default (shared-`x` cat for QKV)
- Enable **TF32 / fast matmul** knobs on inference when safe

`seiso-bench-kernels --roofline` only **explains** shapes; it does not change training.
Naive scalar CUDA matmul/LoRA kernels are opt-in experiments only
(`SEISO_KERNEL_ALLOW_NAIVE_MLP`, `SEISO_KERNEL_ALLOW_NAIVE_LORA`).

---

## What this is not

| Not included | Why |
|--------------|-----|
| Forge “ridge point” UI | Easy to misread across GPUs |
| Claiming every GPU’s ridge is 300 | 300 is only the **shape-math SoT bar** |
| Measured HBM/TFLOPS sampling | Shape math only |
| Blocking train on intensity | Diagnostics only |

---

## Related

- [kernels.md](kernels.md) — enable fused kernels, auto-tune, low-VRAM modes
- [cli.md](../cli.md#seiso-bench-kernels) — CLI flags
- `seiso-bench-kernels --help`
