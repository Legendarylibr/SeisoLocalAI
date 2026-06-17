# Fused training kernels

Seiso patches compatible model layers during training for lower memory bandwidth and VRAM use. Patches are **restored after training** to avoid memory leaks.

## What is fused

| Operation | Patch target | NVIDIA | AMD ROCm |
|-----------|--------------|--------|----------|
| RMSNorm | `LlamaRMSNorm`, `Qwen2RMSNorm`, … | CUDA stripe | Triton |
| SwiGLU MLP | `LlamaMLP`, `Qwen2MLP`, … | CUDA | Triton |
| Cross-entropy | `FusedSFTTrainer.compute_loss` | CUDA | Triton |

## Enable

**YAML:**

```yaml
use_triton: true    # master switch for fused RMSNorm + MLP
use_fused_ce: true
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
NVIDIA  → native CUDA (.cu JIT) → Triton → PyTorch
AMD     → Triton → PyTorch
macOS   → PyTorch only
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
