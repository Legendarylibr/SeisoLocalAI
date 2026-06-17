# Windows (native)

Forge and training work on Windows 10/11 with NVIDIA GPUs. Use PowerShell 7+.

## Prerequisites

1. [Python 3.10+](https://www.python.org/downloads/) — check **Add to PATH**
2. [Git for Windows](https://git-scm.com/download/win)
3. NVIDIA driver + CUDA-enabled PyTorch
4. (For fused kernels) [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) + Visual Studio Build Tools (MSVC)

## Install

```bash
git clone https://github.com/seiso-ai/seiso.git Seiso
cd Seiso
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[forge,train,dev]"
```

Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel is CPU-only.

Triton and flash-attn from `[cuda]` extra are **Linux-only** in pyproject — on Windows, fused kernels use **native CUDA JIT** when `nvcc` is available.

## Start Forge

```powershell
cd forge-ui; npm install; npm run build; cd ..
seiso forge
```

Open **http://127.0.0.1:8765**

## Train

```powershell
seiso train --config configs\example_lora.yaml
```

Or use Training Studio in the web UI.

## Fused kernels on Windows

First training run JIT-compiles `seiso/kernels/cuda/*.cu` via PyTorch cpp_extension. Requires:
- `nvcc` on PATH
- MSVC cl.exe (Visual Studio Build Tools)

If compile fails, Seiso falls back to PyTorch (no fused ops).

## Multi-GPU

```powershell
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs\example_lora.yaml
```

Or enable Multi-GPU in Forge when multiple NVIDIA GPUs are visible.

## WSL alternative

For easier Triton/flash-attn support, use [WSL2 + NVIDIA](wsl.md).
