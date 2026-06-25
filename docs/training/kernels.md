# Fused training kernels

Seiso patches compatible model layers during training for lower memory bandwidth and VRAM use. Patches are **restored after training** to avoid memory leaks.

## What is fused

| Operation | Patch target | NVIDIA | AMD ROCm |
|-----------|--------------|--------|----------|
| RMSNorm | `LlamaRMSNorm`, `Qwen2RMSNorm`, … | CUDA stripe | Triton |
| SwiGLU MLP | `LlamaMLP`, `Qwen2MLP`, … | CUDA | Triton |
| LoRA delta | PEFT `Linear` adapters (rank ≤ 64) | CUDA | PyTorch |
| LoRA QKV | Attention `q_proj` / `k_proj` / `v_proj` adapters (rank ≤ 64) | CUDA (cuBLAS at training scale) | PyTorch |
| Cross-entropy | `FusedSFTTrainer.compute_loss` | CUDA | Triton |

## Enable

**YAML:**

```yaml
use_triton: true    # master switch for fused RMSNorm + MLP
use_fused_ce: true
use_fused_lora: true   # CUDA / WSL2 only; fused low-rank delta
extra:
  use_fused_lora_qkv: true   # batched Q/K/V LoRA deltas (CUDA; default follows use_fused_lora)
```

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
