# Installation

## Linux & macOS — one command (recommended)

**Requirements:** Python 3.10+, [Node.js 18+](https://nodejs.org/), and git.

```bash
curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh
```

The installer:

1. Clones Seiso to `~/Seiso` (override with `SEISO_INSTALL_DIR`)
2. Creates `.venv` and installs platform extras automatically:
   - **Linux + NVIDIA** → `[forge,train,cuda,dev]`
   - **Linux (no NVIDIA)** → `[forge,train,dev]`
   - **macOS** → `[forge,train,mlx,dev]`
3. Copies `.env.example` → `.env` if missing
4. Builds the Forge UI (`forge-ui/dist`)

Useful options:

| Variable | Effect |
|----------|--------|
| `SEISO_INSTALL_DIR=~/code/Seiso` | Custom clone/install path |
| `SEISO_BRANCH=main` | Git branch to clone |
| `SEISO_SKIP_UI=1` | Skip `npm run build` |
| `SEISO_START=1` | Start Forge when install finishes |

Already cloned the repo? Run the same script from the tree:

```bash
./scripts/install.sh
./scripts/start.sh
```

First launch: open **http://127.0.0.1:8765** and create your local admin password.

---

## Manual install (all platforms)

All commands below assume the **repository root** as the working directory.

```bash
git clone https://github.com/seiso-ai/seiso.git Seiso && cd Seiso
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[forge,train,dev]"
```

Optional: copy environment defaults and set your Hugging Face token:

```bash
cp .env.example .env
```

## Optional extras

| Extra | Command | Platforms |
|-------|---------|-----------|
| Forge web server | `.[forge]` | All |
| Training (PyTorch, TRL, PEFT) | `.[train]` | All |
| NVIDIA fused kernels + Triton | `.[cuda]` | **Linux only** |
| MLX chat inference | `.[mlx]` | **macOS only** |
| GGUF via llama.cpp | `.[llamacpp]` | All (build may need CMake) |
| GPTQ/AWQ (LLM compress) | `.[compress-quant]` | CUDA recommended |
| lm-eval (LLM compress) | `.[compress-eval]` | All |
| SD image compression | `.[image-compress]` | CUDA/MPS/CPU |
| SD ONNX export | `.[image-compress-onnx]` | All |
| Dev tests / lint | `.[dev]` | All |

### Recommended per platform

```bash
# Linux NVIDIA
pip install -e ".[forge,train,cuda,dev]"

# Linux AMD ROCm (install ROCm PyTorch wheel first, then:)
pip install -e ".[forge,train,dev]"
pip install triton

# macOS Apple Silicon
pip install -e ".[forge,train,mlx,dev]"

# Windows NVIDIA
pip install -e ".[forge,train,dev]"
# Install CUDA PyTorch from https://pytorch.org/get-started/locally/
```

## Frontend (Forge UI)

Forge serves the built UI from `forge-ui/dist`. Build before your first launch:

```bash
cd forge-ui && npm install && npm run build && cd ..
```

Or use `./scripts/install.sh` / `./scripts/start.sh` (builds UI automatically if missing).

For UI development with hot reload (API must still be running):

```bash
# Terminal 1
seiso forge

# Terminal 2
cd forge-ui && npm run dev   # Vite on :5173, proxies /api to :8765
```

See [forge.md](forge.md) for pages, API routes, and environment variables.

## First launch

```bash
seiso forge
# or: ./scripts/start.sh
```

1. Open **http://127.0.0.1:8765**
2. Complete onboarding — create your local admin password
3. Use the Dashboard or sidebar to reach Chat, Training Studio, Export, etc.

If the UI is blank, rebuild: `cd forge-ui && npm run build` or run `./scripts/start.sh` (auto-builds).

## Verify

```bash
make ci-fast          # lint + types + test + security (see CI_LOCAL.md)
pytest tests/ -q
seiso forge           # should bind 127.0.0.1:8765
```

For local CI dev dependencies: `pip install -r requirements-dev.txt` (see [CI_LOCAL.md](CI_LOCAL.md)).
