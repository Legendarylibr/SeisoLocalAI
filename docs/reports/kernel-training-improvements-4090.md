# GPU Training Kernel Improvements — RTX 4090 Benchmark

**Date:** 2026-06-25  
**Hardware:** NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89, 24 GB)  
**Branch:** `pr-74` (follow-up to PR #74 fused-kernel work)

## Summary

Follow-up fixes to the PR #74 fused training kernel stack. The main issue was that **LoRA QKV fusion was disabled during training** whenever activations had `requires_grad=True`, so the Q/K/V projection cache never ran and each attention layer paid for three separate base forwards.

Additional changes route LoRA QKV deltas through **cuBLAS-backed PyTorch matmuls** (with stacked `A @ x` when ranks match) instead of the naive serial CUDA kernel at training-scale tensor sizes.

## Changes

| Area | Fix |
|------|-----|
| `seiso/kernels/hooks.py` | Keep QKV fusion active during training; batched base Q/K/V `einsum` when weight shapes match; enable fused LoRA matmul path under grad |
| `seiso/kernels/dispatch.py` | Stacked `A @ x` for QKV deltas; prefer cuBLAS when `grad` is on or `rows × in_dim > 64 × 512` |
| `tests/test_fused_lora_qkv_training.py` | Regression tests for training-path QKV correctness |

## Benchmarks (RTX 4090)

### Per-op micro-benchmarks (native CUDA, bf16, 4096×4096)

| Operation | PyTorch | Fused | Speedup |
|-----------|---------|-------|---------|
| RMSNorm | 0.21 ms | 0.11 ms | 1.9× |
| SwiGLU | 0.20 ms | 0.11 ms | 1.8× |
| Cross-entropy | 2.65 ms | 2.04 ms | 1.3× |

*Note: RMSNorm/SwiGLU fusion still falls back to PyTorch during training forward when activations require grad.*

### End-to-end training step (Qwen2.5-0.5B, bf16 LoRA, batch 2, seq 256)

| Config | `train_runtime` | Steps/sec |
|--------|-----------------|-----------|
| Baseline (no fused kernels) | 1.49 s | 0.67 |
| Before this fix (PR #74 kernels, QKV gated off in training) | ~1.21 s | ~0.83 |
| **After this fix** | **1.11 s** | **0.90** |

**Net improvement vs baseline: ~34% faster train steps** (`1.49 / 1.11 ≈ 1.34×`).

Wall-clock for a single-epoch smoke run dropped from ~7.9 s to ~5.7 s (includes model load and checkpoint write).

### Ada profile applied at runtime

```json
{
  "arch_family": "ada",
  "arch_sm": 89,
  "use_wmma": true,
  "use_persistent_kernels": true,
  "prefer_flash_attn": "fa2"
}
```

## How to reproduce

```bash
export PATH="$VENV/bin:$PATH"   # venv with ninja on PATH for CUDA JIT

# Per-op kernels
python -m seiso.kernels.benchmark --rows 4096 --hidden 4096 --dtype bfloat16

# Training smoke (compare use_triton true vs false in config)
seiso train --config configs/smoke_train_gpu_e2e.yaml
```

## Remaining limits

- **QLoRA 4-bit:** CUDA graph capture is still skipped (bitsandbytes layers are not graph-safe).
- **RMSNorm / MLP fusion:** Native CUDA paths remain inference-oriented; training forward uses PyTorch when `requires_grad=True`.
- **GQA models:** Batched base QKV `einsum` falls back to three separate forwards when Q/K/V weight shapes differ.