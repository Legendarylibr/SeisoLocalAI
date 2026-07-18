# Fused training kernels

Seiso patches compatible model layers during training for lower memory bandwidth and VRAM use. Patches are **restored after training** to avoid memory leaks.

## What is fused

| Operation | Patch target | NVIDIA | AMD ROCm |
|-----------|--------------|--------|----------|
| RMSNorm | `LlamaRMSNorm`, `Qwen2RMSNorm`, … | CUDA stripe | Triton |
| SwiGLU MLP | `LlamaMLP`, `Qwen2MLP`, … | cuBLAS GEMM + fused SwiGLU | Triton / PyTorch |
| LoRA delta | PEFT `Linear` adapters | cuBLAS/torch skinny GEMMs | PyTorch |
| LoRA QKV | Attention `q_proj` / `k_proj` / `v_proj` adapters | cuBLAS (shared-`x` when ranks match) | PyTorch |
| Cross-entropy | `FusedSFTTrainer.compute_loss` | CUDA | Triton |

GEMM-heavy work uses library matmuls (Tensor Cores). Custom CUDA covers
bandwidth-bound elementwise/norm/CE — not hand-rolled dense GEMMs.

**MLP:** one stacked gate/up GEMM (`cat(W_gate,W_up)`) + fused SwiGLU when shapes match.
**LoRA QKV:** shared-`x` A matmul when ranks match; batched `bmm` for B when out dims and scales match.

## Enable

**YAML:**

```yaml
use_triton: true    # master switch for fused RMSNorm + MLP
use_fused_ce: true  # default on when CUDA fused stack is available
use_fused_lora: true   # cuBLAS skinny LoRA GEMMs
packing: true          # auto-recommended on CUDA (less pad waste)
padding_free: false    # true when flash-attn is installed (set by CUDA profile)
extra:
  use_fused_lora_qkv: true   # batched Q/K/V LoRA (cuBLAS; default follows use_fused_lora)
  use_cuda_graphs: true      # fixed-shape train steps; off with gradient checkpointing
```

**Attention (train load):** FA3 → FA2 → SDPA (never require eager). Override with
`SEISO_ATTN_IMPLEMENTATION=sdpa|flash_attention_2|eager`. Install FlashAttention via
`./scripts/install_flash_attn.sh`.

**Low VRAM:** free VRAM under 8 GB → lean profile (`SEISO_KERNEL_LOW_VRAM=1`, narrow
tiles, fused CE, gradient checkpointing). Forge Train tab shows a low-VRAM hint.

**Python:**

```python
from seiso.kernels import apply_training_kernels, release_training_memory

meta = apply_training_kernels(model)
# ... train ...
release_training_memory(model)
```

**Forge UI:** checkboxes *Fused kernels* and *Fused cross-entropy* (disabled when hardware has no GPU kernels).

## Backend selection

```
NVIDIA / WSL2  → native CUDA (.cu JIT, SM-targeted) → Triton → PyTorch
AMD            → Triton → PyTorch
macOS          → PyTorch only
```

Inspect at runtime:

```python
from seiso.kernels import kernel_metadata
print(kernel_metadata())
```

## Benchmark

```bash
seiso-bench-kernels --op all --rows 4096 --hidden 4096 --vocab 32000
```

### Roofline-style intensity (optional diagnostics)

**Shape → FLOP/byte** for **Seiso fused ops only**. Never blocks training.

**Shape-math SoT bar:** FP16/BF16 **GEMM-family** ops with **I ≥ 300 FLOP/byte** → `performance_truth=true` (strong compute-bound *candidate* under efficient dense GEMM / cuBLAS — which is the production MLP/LoRA path). Elementwise/CE and float32 stay heuristic. Default MLP intermediate is `4×hidden`; pass `--intermediate` for real SwiGLU widths (~8/3×hidden).

```bash
# CPU-ok
seiso-bench-kernels --roofline-only --rows 4096 --hidden 4096 --vocab 32000

# JSON only (no timed benches mixed on stdout)
seiso-bench-kernels --roofline-only --json

# Text roofline, then timed CUDA benches (omit --json)
seiso-bench-kernels --roofline --op all --rows 4096 --hidden 4096
```

See [kernel-shape.md](kernel-shape.md) and [cli.md](../cli.md#seiso-bench-kernels).

## Compile requirements (NVIDIA native)

- Linux or Windows
- `nvcc` (CUDA toolkit)
- Windows: MSVC Build Tools

First kernel use triggers JIT compile via `torch.utils.cpp_extension.load`.

## Leak safety

- `restore_kernel_patches()` restores original `forward` methods
- `release_training_memory()` — patches + `gc` + `empty_cache` + `synchronize`
- Called automatically at end of `SeisoTrainer.run()`

## Low VRAM mode

When free VRAM is under **8 GB** (or `SEISO_KERNEL_LOW_VRAM=1`), training uses **lean** mode:

| Behavior | Why |
|----------|-----|
| Fused cross-entropy | Avoids materializing full `[batch, vocab]` softmax |
| Gradient checkpointing | Recomputes activations instead of storing them |
| In-place fused LoRA | Writes adapter delta directly into the linear output |
| `narrow_opt` kernel profile | Smaller CUDA tiles, lower peak shared memory |

## CUDA auto-tune (speed + efficiency)

On NVIDIA with native CUDA kernels, Seiso picks a launch profile before training:

| Mode | When | Goal |
|------|------|------|
| **speed** | ≥16 GB free, model fits comfortably | Max throughput — **no** gradient checkpointing, `wide_throughput` kernels |
| **balanced** | 8–16 GB free | Auto-tuned profile + checkpointing only if the model is tight |
| **lean** | &lt;8 GB free | Min VRAM — in-place LoRA, fused CE, checkpointing |

Micro-benchmarks select the fastest kernel profile for your hidden size and batch×seq (cached). Disable with:

```bash
export SEISO_KERNEL_AUTO_TUNE=0
```

TF32 + cuDNN benchmark are enabled on CUDA when `deterministic: false` in your training config.

Inspect at runtime:

```python
from seiso.kernels.training_profile import last_cuda_training_profile
print(last_cuda_training_profile())
```
