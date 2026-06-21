# Windows (native)

Forge and training work on Windows 10/11 with NVIDIA GPUs. Use **PowerShell 7+** (recommended) or Windows Terminal.

**Alternative:** [WSL2 + NVIDIA](wsl.md) is the recommended Windows path for full Triton and flash-attn support.

## Prerequisites

1. [Python 3.10+](https://www.python.org/downloads/) — check **Add to PATH** during install
2. [Node.js 18+ LTS](https://nodejs.org/) — includes npm
3. [Git for Windows](https://git-scm.com/download/win)
4. NVIDIA driver + CUDA-enabled PyTorch (for GPU training)
5. (For fused kernels) [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) + Visual Studio Build Tools (MSVC)

## Paths

| Purpose | Default path | Override |
|---------|--------------|----------|
| Repository | `%USERPROFILE%\Seiso` | clone anywhere |
| User data | `%USERPROFILE%\.seiso` | `SEISO_DATA_DIR` in `.env` |
| Virtualenv | `{repo}\.venv` | — |

## Install

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
cd forge-ui; npm ci; npm run build; cd ..
seiso doctor
seiso forge
```

Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel is CPU-only.

Triton from `[cuda]` is **Linux-only** in pyproject. Flash Attention is optional (`[flash-attn]`). On Windows, fused kernels use **native CUDA JIT** when `nvcc` is available.

## Start Forge

**Every session** — activate the venv, then launch:

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
seiso forge
```

Open **http://127.0.0.1:8765** in your browser. On first run, create your local admin password.

If the UI is blank, rebuild the frontend:

```powershell
cd "$env:USERPROFILE\Seiso\forge-ui"
npm ci; npm run build
```

## Train

```powershell
seiso train --config configs\example_lora.yaml
```

Or use Training Studio in the web UI.

## Fused kernels on Windows

First training run JIT-compiles `seiso/kernels/cuda/*.cu` via PyTorch cpp_extension. Requires:

- `nvcc` on PATH
- MSVC `cl.exe` (Visual Studio Build Tools)

If compile fails, Seiso falls back to PyTorch (no fused ops).

## Multi-GPU

```powershell
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs\example_lora.yaml
```

Or enable Multi-GPU in Forge when multiple NVIDIA GPUs are visible.

## Diagnose install

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
seiso doctor
```

There is no Windows `doctor.sh` — use the `seiso doctor` CLI command instead.

## WSL alternative

For easier Triton/flash-attn support, use [WSL2 + NVIDIA](wsl.md).

## Memory (CPU-only vs CUDA)

- **No NVIDIA GPU:** Forge sets `SEISO_LLAMA_GPU_LAYERS=0` — GGUF runs on CPU. Use smaller Q4 models.
- **CUDA + tight VRAM:** batch and GPU layers are capped automatically; use **Free memory** in Chat/Hub before loading a larger model.
- **Free memory** unloads the active inference model from VRAM/RAM; downloaded weights stay in `{SEISO_DATA_DIR}/hf_cache/`.
