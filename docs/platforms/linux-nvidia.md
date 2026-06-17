# Linux + NVIDIA

Full Seiso support: fused CUDA kernels, QLoRA, multi-GPU, Forge UI.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,dev]"
```

Requirements:
- NVIDIA driver + `nvidia-smi` working
- CUDA toolkit (`nvcc`) for JIT-compiled fused kernels (first training run compiles)
- PyTorch CUDA wheel (installed automatically via `torch` dependency)

## Start Forge

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

Open **http://127.0.0.1:8765** — complete onboarding, then use **Training Studio** (`/train`).

## Train (CLI)

```bash
seiso train --config configs/example_lora.yaml
```

Config flags for kernels:

```yaml
use_triton: true      # enables fused RMSNorm + SwiGLU MLP (CUDA native on NVIDIA)
use_fused_ce: true    # fused cross-entropy in SFTTrainer
```

## Benchmark kernels

```bash
seiso-bench-kernels --rows 4096 --hidden 4096 --vocab 32000 --dtype bfloat16
```

## Multi-GPU

Enable **Multi-GPU** in Forge or set `multi_gpu: true` in config. Forge runs:

```bash
torchrun --nproc_per_node=N -m seiso.training.worker --config <yaml>
```

See [../training/multi-gpu.md](../training/multi-gpu.md).

## Kernel stack

| Layer | Backend |
|-------|---------|
| RMSNorm + MLP | Native CUDA (stripe) → Triton → PyTorch |
| Cross-entropy | Native CUDA → Triton → PyTorch |

See [../training/kernels.md](../training/kernels.md).
