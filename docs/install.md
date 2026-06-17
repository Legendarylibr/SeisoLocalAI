# Installation

Complete guide to installing Seiso on every supported platform.

**Repository:** [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)

---

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **Node.js** | 18 | 20 LTS |
| **git** | any recent | — |
| **RAM** | 16 GB | 32 GB+ |
| **Disk** | 20 GB free | 100 GB+ (models + checkpoints) |
| **GPU** | Optional | NVIDIA 12 GB+ VRAM for QLoRA training |

### GPU support matrix

| GPU / OS | Training | Inference | Fused kernels |
|----------|----------|-----------|---------------|
| NVIDIA + Linux | QLoRA 4-bit | GGUF, PyTorch | CUDA native |
| NVIDIA + Windows/WSL | QLoRA 4-bit | GGUF, PyTorch | CUDA / Triton |
| AMD ROCm | 4-bit* | PyTorch | Triton |
| Apple Silicon | 16-bit LoRA | MLX, PyTorch | — |
| CPU only | Tiny models | GGUF, Ollama | — |

\* bitsandbytes on ROCm depends on your PyTorch wheel.

---

## Linux & macOS — one command (recommended)

**Requirements:** Python 3.10+, [Node.js 18+](https://nodejs.org/), and git.

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh
```

### What the installer does

1. **Clones** Seiso to `~/Seiso` (override with `SEISO_INSTALL_DIR`)
2. **Creates** a Python virtualenv at `.venv`
3. **Installs** platform extras automatically:
   - **Linux + NVIDIA** (`nvidia-smi` detected) → `[forge,train,cuda,dev]`
   - **Linux (no NVIDIA)** → `[forge,train,dev]`
   - **macOS** → `[forge,train,mlx,dev]`
4. **Copies** `.env.example` → `.env` if missing
5. **Builds** the Forge UI (`forge-ui/dist`)

### Installer options

| Variable | Default | Effect |
|----------|---------|--------|
| `SEISO_INSTALL_DIR` | `~/Seiso` | Clone/install path |
| `SEISO_REPO_URL` | `https://github.com/Legendarylibr/SeisoLocalAI.git` | Git remote |
| `SEISO_BRANCH` | `main` | Branch to clone |
| `SEISO_SKIP_UI=1` | off | Skip `npm run build` |
| `SEISO_START=1` | off | Start Forge when install finishes |

```bash
# Custom location + immediate start
SEISO_INSTALL_DIR=~/code/Seiso SEISO_START=1 \
  curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
```

### Already cloned?

Run from the repository root:

```bash
./scripts/install.sh
./scripts/start.sh
```

First launch: open **http://127.0.0.1:8765** and create your local admin password.

---

## Manual install (all platforms)

All commands assume the **repository root** as working directory.

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git Seiso && cd Seiso
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
```

Optional environment file:

```bash
cp .env.example .env
# Edit SEISO_HF_TOKEN for gated models
```

Build the UI and launch:

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

---

## Platform-specific installs

### Linux + NVIDIA (recommended)

```bash
pip install -e ".[forge,train,cuda,dev]"
```

Requirements:
- NVIDIA driver (`nvidia-smi` works)
- CUDA toolkit (`nvcc`) for JIT-compiled fused kernels
- PyTorch CUDA wheel (installed via `torch` dependency)

Full guide: [platforms/linux-nvidia.md](platforms/linux-nvidia.md)

### Linux + AMD ROCm

1. Install [PyTorch ROCm wheel](https://pytorch.org/get-started/locally/) for your ROCm version
2. Install Seiso:

```bash
pip install -e ".[forge,train,dev]"
pip install triton
```

Full guide: [platforms/linux-amd-rocm.md](platforms/linux-amd-rocm.md)

### macOS Apple Silicon

```bash
pip install -e ".[forge,train,mlx,dev]"
```

MLX enables fast chat inference. Training uses PyTorch MPS with 16-bit LoRA.

Full guide: [platforms/macos.md](platforms/macos.md)

### Windows + NVIDIA

```powershell
pip install -e ".[forge,train,dev]"
# Install CUDA PyTorch from https://pytorch.org/get-started/locally/
cd forge-ui; npm install; npm run build; cd ..
seiso forge
```

Full guide: [platforms/windows.md](platforms/windows.md)

### WSL2 + NVIDIA

Same as Linux NVIDIA. Set `SEISO_NVIDIA_WSL_ACK=1` before GPU training.

Full guide: [platforms/wsl.md](platforms/wsl.md)

---

## Optional pip extras

| Extra | Packages / purpose | Platforms |
|-------|-------------------|-----------|
| `forge` | FastAPI, uvicorn, auth, SQLite, SSE | All |
| `train` | PyTorch, TRL, PEFT, bitsandbytes | All |
| `cuda` | Triton, flash-attn | **Linux NVIDIA only** |
| `mlx` | mlx-lm | **macOS only** |
| `llamacpp` | llama-cpp-python (GGUF inference) | All |
| `compress-quant` | auto-gptq, autoawq | CUDA recommended |
| `compress-eval` | lm-eval harness | All |
| `image-compress` | diffusers, torchvision, gradio | CUDA/MPS/CPU |
| `image-compress-onnx` | optimum, onnxruntime | All |
| `dev` | pytest, ruff, mypy, bandit | All |

Combine extras: `pip install -e ".[forge,train,cuda,dev,compress-quant]"`

---

## Frontend (Forge UI)

Forge serves the built UI from `forge-ui/dist`. Required before first launch:

```bash
cd forge-ui && npm install && npm run build && cd ..
```

Or use `./scripts/install.sh` / `./scripts/start.sh` (auto-builds if missing).

### UI development (hot reload)

```bash
# Terminal 1 — API
seiso forge

# Terminal 2 — Vite dev server (:5173, proxies /api → :8765)
cd forge-ui && npm run dev
```

See [forge.md](forge.md) for pages, API routes, and environment variables.

---

## First launch checklist

1. Start Forge: `seiso forge` or `./scripts/start.sh`
2. Open **http://127.0.0.1:8765**
3. Complete onboarding — create local admin password
4. (Optional) Add Hugging Face token in **Settings**
5. Download a model from **Model Hub**
6. Try **Chat** or **Training Studio**

Walkthrough: [getting-started.md](getting-started.md)

---

## Upgrade

From an existing clone:

```bash
cd ~/Seiso   # or your SEISO_INSTALL_DIR
git pull origin main
source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,dev]"   # adjust extras for your platform
cd forge-ui && npm install && npm run build && cd ..
```

Or re-run the installer (idempotent):

```bash
./scripts/install.sh
```

---

## Verify installation

```bash
source .venv/bin/activate
seiso forge                    # should bind 127.0.0.1:8765
pytest tests/ -q               # run test suite
make ci-fast                   # lint + types + test + security
```

Dev dependencies: `pip install -r requirements-dev.txt` ([CI_LOCAL.md](CI_LOCAL.md))

Hardware detection:

```bash
python -c "from seiso.training.platform_caps import training_capabilities; import json; print(json.dumps(training_capabilities(), indent=2))"
```

---

## Uninstall

```bash
# Remove install directory
rm -rf ~/Seiso

# Remove user data (models, checkpoints, exports)
rm -rf ~/.seiso
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| UI is blank | `cd forge-ui && npm run build` or `./scripts/start.sh` |
| `Missing required command: node` | Install Node.js 18+ from [nodejs.org](https://nodejs.org/) |
| Python too old | Requires 3.10+ (`python3 --version`) |
| CUDA kernels fail | Install CUDA toolkit; check `nvcc --version` |
| QLoRA fails on macOS | Use `quant: 16bit` in config |
| Install script can't find repo | Set `SEISO_INSTALL_DIR` or clone manually |

Full guide: [troubleshooting.md](troubleshooting.md)
