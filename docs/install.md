# Installation

Complete guide to installing Seiso on every supported platform.

**Repository:** [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)

---

## Paths on every OS

| Purpose | Linux / macOS / WSL | Windows (native) | Override |
|---------|---------------------|------------------|----------|
| **Repository** | `$HOME/Seiso` | `%USERPROFILE%\Seiso` | `SEISO_INSTALL_DIR` |
| **User data** | `$HOME/.seiso` | `%USERPROFILE%\.seiso` | `SEISO_DATA_DIR` |
| **Virtualenv** | `{repo}/.venv` | `{repo}\.venv` | — |
| **Activate venv** | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` | — |

Seiso config files accept `~/.seiso` and expand it correctly on all platforms. Shell one-liners must use the path style for your OS.

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

No manual prerequisites on most systems — the installer installs Python, Node, and git via Homebrew or your package manager when they are missing.

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
```

Forge starts when install finishes and your browser opens automatically at **http://127.0.0.1:8765**. You do **not** need to run `start.sh` immediately after a successful install. If anything fails, **doctor runs automatically** with a guided diagnosis.

**Start Forge on later sessions:**

```bash
"$HOME/Seiso/scripts/start.sh"
# or one-liner:
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/start.sh | bash
```

**Verify before running (recommended):**

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh -o install.sh
shasum -a 256 install.sh
bash install.sh
```

### What the installer does

1. **Clones** Seiso to `$HOME/Seiso` on Linux/macOS/WSL (override with `SEISO_INSTALL_DIR`; Windows uses manual clone — see below)
2. **Creates** a Python virtualenv at `.venv`
3. **Installs** platform extras automatically (includes GGUF chat via `llamacpp`):
   - **Linux + NVIDIA** (`nvidia-smi` detected) → `[forge,train,cuda,llamacpp,dev]`
   - **Linux (no NVIDIA)** → `[forge,train,llamacpp,dev]`
   - **macOS** → `[forge,train,mlx,llamacpp,dev]`
4. **Copies** `.env.example` → `.env` if missing
5. **Builds** the Forge UI (`forge-ui/dist`)
6. **Starts** Forge (unless `SEISO_START=0`)

### Installer options

| Variable | Default | Effect |
|----------|---------|--------|
| `SEISO_INSTALL_DIR` | `$HOME/Seiso` | Clone/install path (Linux/macOS/WSL) |
| `SEISO_REPO_URL` | `https://github.com/Legendarylibr/SeisoLocalAI.git` | Git remote |
| `SEISO_BRANCH` | `main` | Branch to clone |
| `SEISO_SKIP_UI=1` | off | Skip `npm run build` |
| `SEISO_START=0` | on (starts Forge) | Set to `0` to install without launching Forge |
| `SEISO_NO_OPEN=1` | off | Do not open the browser after Forge starts |
| `SEISO_NO_BANNER=1` | off | Skip install animation |
| `SEISO_VERBOSE=1` | off | Show full pip/npm output |

Custom location:

```bash
SEISO_INSTALL_DIR="$HOME/code/Seiso" curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
```

### Already cloned?

Run from the repository root:

```bash
./scripts/install.sh
```

First launch: open **http://127.0.0.1:8765** and create your local admin password.

---

## Manual install (all platforms)

Pick a repository path (`REPO`):

- Linux / macOS / WSL: `"$HOME/Seiso"` (or any directory)
- Windows: `"$env:USERPROFILE\Seiso"`

Base install (Linux / macOS / WSL):

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
```

Base install (Windows PowerShell):

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
```

Optional environment file:

```bash
cp -n .env.example .env    # Linux / macOS / WSL
```

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }   # Windows
```

Edit `.env` and set `SEISO_HF_TOKEN` only if you need gated Hugging Face models.

Build the UI and launch:

```bash
cd "$HOME/Seiso"    # or your REPO path
source .venv/bin/activate
cd forge-ui && npm ci && npm run build && cd ..
seiso doctor
seiso forge
```

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
cd forge-ui; npm ci; npm run build; cd ..
seiso doctor
seiso forge
```

Use `npm install` instead of `npm ci` only when you intentionally want to refresh the lockfile.

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

Optional Flash Attention 2 (Linux native filesystem only — not `/mnt/c/...` in WSL):

```bash
./scripts/install_flash_attn.sh
```

Alternative Flash Attention install:

```bash
pip install -e ".[flash-attn]" --no-build-isolation
```

Seiso works without flash-attn (PyTorch SDPA fallback).

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
cd "$HOME/Seiso"
source .venv/bin/activate
pip install -e ".[forge,train,mlx,dev]"
```

MLX enables fast chat inference. Training uses PyTorch MPS with 16-bit LoRA.

Full guide: [platforms/macos.md](platforms/macos.md)

### Windows + NVIDIA

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
pip install -e ".[forge,train,dev]"
cd forge-ui; npm ci; npm run build; cd ..
seiso forge
```

Install the CUDA PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) when you want GPU training.

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
| `cuda` | Triton fused-kernel support | **Linux NVIDIA only** |
| `flash-attn` | Flash Attention 2 (optional; build from source) | **Linux NVIDIA only** |
| `mlx` | mlx-lm | **macOS only** |
| `llamacpp` | llama-cpp-python (GGUF inference) | All |
| `compress-quant` | auto-gptq, autoawq (requires `torch`; Linux NVIDIA) | CUDA recommended |
| `compress-eval` | lm-eval harness | All |
| `image-compress` | diffusers, torchvision, gradio | CUDA/MPS/CPU |
| `image-compress-onnx` | optimum, onnxruntime | All |
| `rl-quant` | Integrated adaptive RL quant (stdlib; no extra deps) | All |
| `dev` | pytest, ruff, mypy, bandit | All |

Combine extras:

```bash
pip install -e ".[forge,train,cuda,dev,compress-quant]"
```

If `auto-gptq` fails to build, install the train stack first, then retry with build isolation disabled:

```bash
pip install -e ".[forge,train,compress-quant]"
pip install auto-gptq autoawq --no-build-isolation
```

---

## Frontend (Forge UI)

Forge serves the built UI from `forge-ui/dist`. Required before first launch:

```bash
cd forge-ui && npm ci && npm run build && cd ..
```

Or use `./scripts/install.sh` / `./scripts/start.sh` (auto-builds if missing).

### UI development (hot reload)

Terminal 1, API:

```bash
seiso forge
```

Terminal 2, Vite dev server on `:5173`:

```bash
cd forge-ui
npm run dev
```

See [forge.md](forge.md) for pages, API routes, and environment variables.

---

## First launch checklist

1. Start Forge:
   - Linux / macOS / WSL: `./scripts/start.sh` or `seiso forge` (with venv active)
   - Windows: activate venv → `seiso forge`
2. Open **http://127.0.0.1:8765**
3. Complete onboarding — create local admin password
4. (Optional) Add Hugging Face token in **Settings**
5. Download a model from **Model Hub**
6. Try **Chat** or **Training Studio**

If anything looks off, run the install doctor:

```bash
./scripts/doctor.sh
./scripts/doctor.sh --network
```

The `--network` option also checks `huggingface.co` reachability.

Model storage notes:

- Catalog chat downloads fetch a GGUF file into Seiso's Hugging Face cache (`$SEISO_DATA_DIR/hf_cache` by default) and register a local inventory link for llama.cpp.
- The Hub page shows the expected GGUF download size, usually 2-8 GB for small/medium Q4 models and 10-30+ GB for larger models.
- Ollama keeps its own model store. Seiso lists and chats with models already available in Ollama, but Hugging Face catalog downloads are not automatically imported into Ollama. Use `ollama pull` or `ollama create` for that path.

Walkthrough: [getting-started.md](getting-started.md)

---

## Upgrade

From an existing clone:

```bash
cd "$HOME/Seiso"
git pull origin main
source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,dev]"
cd forge-ui && npm ci && npm run build && cd ..
```

Adjust the extras for your platform, for example `.[forge,train,mlx,dev]` on Apple Silicon.

Or re-run the installer (idempotent):

```bash
./scripts/install.sh
```

---

## Verify installation

```bash
source .venv/bin/activate
seiso doctor
seiso forge
pytest tests/ -q
make ci-fast
```

`seiso forge` should bind `127.0.0.1:8765`; `make ci-fast` runs lint, type checks, tests, and security checks.

Dev dependencies: `pip install -r requirements-dev.txt` ([CI_LOCAL.md](CI_LOCAL.md))

Hardware detection:

```bash
python -c "from seiso.training.platform_caps import training_capabilities; import json; print(json.dumps(training_capabilities(), indent=2))"
```

---

## Uninstall

Remove the install directory:

```bash
rm -rf "$HOME/Seiso"      # repository
rm -rf "$HOME/.seiso"     # user data
```

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\Seiso", "$env:USERPROFILE\.seiso"
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
| Model downloads fail | Run `./scripts/doctor.sh --network`, then `source .venv/bin/activate && hf auth login` for gated models |

Full guide: [troubleshooting.md](troubleshooting.md)
