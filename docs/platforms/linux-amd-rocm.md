# Linux + AMD ROCm

Training works with ROCm PyTorch. Fused kernels use **Triton** (native `.cu` kernels are NVIDIA-only).

## Install

1. Install AMD ROCm driver and [PyTorch ROCm wheel](https://pytorch.org/get-started/locally/) for your ROCm version.

2. Install Seiso (after ROCm PyTorch is installed):

```bash
SEISO_INSTALL_PROFILE=linux-rocm curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
source "$HOME/Seiso/.venv/bin/activate" && pip install triton
```

**Later sessions:** `cd "$HOME/Seiso" && start`

Or manually:

```bash
pip install -e ".[forge,train,dev]"
pip install triton
```

Do **not** expect `pip install -e ".[cuda]"` to pull Triton on non-Linux — install Triton manually as above.

## Verify GPU

```python
import torch
print(torch.cuda.is_available())   # True on ROCm builds
print(torch.version.hip)           # ROCm version string
```

Seiso detects AMD via `torch.version.hip` in `seiso.kernels.platform.detect_gpu()`.

## Start & train

```bash
cd "$HOME/Seiso" && start
# or from a clone:
cd "$HOME/Seiso" && source .venv/bin/activate && seiso forge
# or
seiso train --config configs/example_lora.yaml
```

Enable in config or Forge UI:
- **Fused kernels** — Triton RMSNorm + SwiGLU
- **Fused cross-entropy** — Triton path

Native CUDA extension is **not** loaded on AMD.

## Benchmark

```bash
seiso-bench-kernels --op all
```

Uses Triton backend when CUDA extension is unavailable.

## Limitations

- No flash-attn from `[cuda]` (optional `[flash-attn]` extra on Linux NVIDIA only)
- bitsandbytes support on ROCm varies by version — prefer 16-bit LoRA if quant fails
- Multi-GPU via Hugging Face Accelerate works when ROCm exposes multiple devices to PyTorch
