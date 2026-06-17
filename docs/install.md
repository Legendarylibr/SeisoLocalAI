# Installation

## Base install

```bash
git clone <repo-url> Seiso && cd Seiso
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[forge,train,dev]"
```

## Optional extras

| Extra | Command | Platforms |
|-------|---------|-----------|
| Forge web server | `.[forge]` | All |
| Training (PyTorch, TRL, PEFT) | `.[train]` | All |
| NVIDIA fused kernels + Triton | `.[cuda]` | **Linux only** (in `pyproject.toml`) |
| MLX chat inference | `.[mlx]` | **macOS only** |
| GGUF via llama.cpp | `.[llamacpp]` | All (build may need CMake) |
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

## Frontend (Forge UI dev)

```bash
cd forge-ui && npm install && npm run build
```

Forge serves `forge-ui/dist` when present. For UI development:

```bash
cd forge-ui && npm run dev   # Vite on :5173, API proxy to Forge
```

## Verify

```bash
pytest tests/ -q
seiso forge   # should bind 127.0.0.1:8765
```
