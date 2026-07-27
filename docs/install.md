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
| CPU only | Tiny models | GGUF | — |

\* bitsandbytes on ROCm depends on your PyTorch wheel.

---

## Linux & macOS — one command (recommended)

No manual prerequisites on most systems — the installer installs Python, Node, git, and native build tools (gcc, cmake, python dev headers) via your package manager when they are missing.

**Main one-liner** — auto-detects Linux, macOS, and WSL2, installs dependencies, builds Forge, and starts the app:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Quick installs** — use these when you already know the target platform. Each command is isolated so it can be copied individually.

Linux native + NVIDIA:

```bash
SEISO_INSTALL_PROFILE=linux-nvidia curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Linux native CPU:

```bash
SEISO_INSTALL_PROFILE=linux-cpu curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

WSL2 + NVIDIA:

```bash
SEISO_INSTALL_PROFILE=wsl-nvidia curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

macOS Apple Silicon:

```bash
SEISO_INSTALL_PROFILE=macos curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Windows (native)** — no bash installer; use PowerShell:

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"; cd "$env:USERPROFILE\Seiso"; python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -U pip wheel setuptools; pip install -e ".[forge,train,dev]"; if (-not (Test-Path .env)) { Copy-Item .env.example .env }; cd forge-ui; npm ci; npm run build; cd ..; seiso forge
```

Forge starts when install finishes and your browser opens automatically at **http://127.0.0.1:8765**. You do **not** need to run `start` again immediately after a successful install. If anything fails, **doctor runs automatically** with a guided diagnosis.

**Start Forge on later sessions:**

```bash
cd "$HOME/Seiso" && start
# or one-liner:
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Verify before running (recommended):**

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start -o start
shasum -a 256 start
bash start
```

### What the installer does

1. **Clones** Seiso to `$HOME/Seiso` on Linux/macOS/WSL (override with `SEISO_INSTALL_DIR`; Windows uses manual clone — see below)
2. **Creates** a Python virtualenv at `.venv`
3. **Installs** platform extras with `uv` when available, or pip as a fallback (includes GGUF support; native Linux NVIDIA uses a sidecar by default):
   - **Linux + NVIDIA** (`nvidia-smi` detected) → `[forge,train,cuda,llamacpp,dev]`
   - **Linux (no NVIDIA)** → `[forge,train,llamacpp,dev]`
   - **macOS** → `[forge,train,llamacpp,dev]` (optional: `[mlx]` for safetensors)
4. **Copies** `.env.example` → `.env` if missing
5. **Builds** the Forge UI with Bun when available, or npm as a fallback (`forge-ui/dist`)
6. **Installs sidecar stack** on native Linux NVIDIA (`linux-nvidia` profile: Ollama + health gate)
7. **Starts** Forge (unless `SEISO_START=0`)

### Installer options

| Variable | Default | Effect |
|----------|---------|--------|
| `SEISO_INSTALL_DIR` | `$HOME/Seiso` | Clone/install path (Linux/macOS/WSL) |
| `SEISO_REPO_URL` | `https://github.com/Legendarylibr/SeisoLocalAI.git` | Git remote |
| `SEISO_BRANCH` | `main` | Branch to clone |
| `SEISO_SKIP_UI=1` | off | Skip Forge UI build |
| `SEISO_START=0` | on (starts Forge) | Set to `0` to install without launching Forge |
| `SEISO_NO_OPEN=1` | off | Do not open the browser after Forge starts |
| `SEISO_NO_BANNER=1` | off | Skip install animation |
| `SEISO_VERBOSE=1` | off | Show full pip/Bun output |
| `SEISO_USE_NPM=1` | off | Use npm instead of Bun for `forge-ui` (Bun is default) |
| `SEISO_USE_UV=0` | on (use uv if installed) | Use pip instead of uv for Python deps |
| `SEISO_FAST_INSTALL=1` | off | Forge + GGUF chat only — skip PyTorch/training extras (same as `SEISO_INSTALL_PROFILE=chat`) |
| `SEISO_INSTALL_PROFILE` | auto | Target stack: `linux-nvidia`, `linux-cpu`, `linux-rocm`, `wsl-nvidia`, `macos`, `chat` |
| `SEISO_INSTALL_EXTRAS` | auto | Override pip extras directly (e.g. `forge,train,cuda,llamacpp`) |
| `SEISO_SIDECAR_AUTOSTART=0` | on | Do not auto-start Ollama/llama-swap before Forge |
| `SEISO_REQUIRE_SIDECAR=1` | on for linux-nvidia | Fail install/start when Ollama/llama-swap unavailable |
| `SEISO_SKIP_OLLAMA_INSTALL=1` | off | Skip official Ollama installer during bootstrap |
| `SEISO_SIDECAR_OPTIONAL=1` | off | Warn instead of hard-fail when sidecar missing |
| `SEISO_LLAMASWAP_ENGINE` | auto | Sidecar engine: `ollama` or `llamacpp` |
| `SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1` | off | Explicitly allow unsafe in-process llama.cpp on native Linux NVIDIA |

Custom location:

```bash
SEISO_INSTALL_DIR="$HOME/code/Seiso" curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Fast install (Forge + chat only — skips PyTorch/training download):

```bash
SEISO_FAST_INSTALL=1 curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

### Already cloned?

Run from the repository root:

```bash
start
```

Install registers `start` on your PATH (`~/.local/bin`). Open a new terminal if the command is not found yet.

First launch: open **http://127.0.0.1:8765**, **generate a key** (or import an `nsec`), write down the shown `nsec`, then Continue.

On native Linux + NVIDIA, `start` also tries to start the isolated GGUF chat
sidecar before Forge: Ollama when available/healthy, then `llama-swap`. If the
sidecar tools are not installed, Forge still starts and chat shows setup
guidance instead of falling back to unsafe in-process llama.cpp.

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
cd forge-ui && bun install --frozen-lockfile && bun run build && cd ..
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

Use `npm ci && npm run build` instead when Bun is unavailable. Use `bun install` or `npm install` only when you intentionally want to refresh lockfiles.

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
| `llamacpp` | llama-cpp-python (GGUF inference; unsafe native Linux NVIDIA fallback only by explicit opt-in) | All |
| `compress-quant` | auto-gptq, autoawq (requires `torch`; Linux NVIDIA) | CUDA recommended |
| `compress-eval` | lm-eval harness | All |
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

Or use `start` (auto-builds if missing).

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
   - Linux / macOS / WSL: `start` or `seiso forge` (with venv active)
   - Windows: activate venv → `seiso forge`
2. Open **http://127.0.0.1:8765**
3. Complete onboarding — generate a key (write down the one-time `nsec` → Continue; `npub` is public identity) or import an `nsec`
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

- Catalog chat downloads fetch a GGUF file into Seiso's Hugging Face cache (`$SEISO_DATA_DIR/hf_cache` by default) and register a local inventory link for GGUF chat. Native Linux NVIDIA serves GGUF through the llama-swap sidecar by default.
- The Hub page shows the expected GGUF download size, usually 2-8 GB for small/medium Q4 models and 10-30+ GB for larger models.

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
start
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

### Reproducible install (CI-equivalent)

GitHub Actions and `run_ci_local.py` install the hashed forge+train+dev resolve from `locks/python.lock`:

```bash
python scripts/install_locked_deps.py --editable
```

Platform extras (`cuda`, `mlx`, `llamacpp`, …) are not in that lock — add them after the locked install when needed, for example `pip install -e ".[cuda]"`.

### Refresh dependency locks

Python dependencies are declared in `pyproject.toml` and locked in `locks/python.lock` with hashes (universal resolve for Linux/macOS markers). The updater prefers `uv pip compile --upgrade --universal` when `uv` is installed, falls back to `pip-compile`, and refreshes `locks/digests.json`:

```bash
python scripts/update_dep_locks.py
```

CI fails if the lock is stale vs `pyproject.toml` (`scripts/check_python_lock_freshness.py`). Forge UI dependencies are declared in `forge-ui/package.json`; keep both npm and Bun locks in sync after changing frontend dependencies:

```bash
cd forge-ui && npm install && bun install && cd ..
python scripts/update_dep_locks.py --skip-python
```

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
| UI is blank | `cd forge-ui && npm run build` or `start` |
| `Missing required command: node` | Install Node.js 18+ from [nodejs.org](https://nodejs.org/) |
| Python too old | Requires 3.10+ (`python3 --version`) |
| CUDA kernels fail | Install CUDA toolkit; check `nvcc --version` |
| QLoRA fails on macOS | Use `quant: 16bit` in config |
| Install script can't find repo | Set `SEISO_INSTALL_DIR` or clone manually |
| Model downloads fail | Run `./scripts/doctor.sh --network`, then `source .venv/bin/activate && hf auth login` for gated models |

Full guide: [troubleshooting.md](troubleshooting.md)
