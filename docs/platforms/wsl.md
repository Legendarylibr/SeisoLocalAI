# WSL2 + NVIDIA

Treat WSL2 as **Linux** for Seiso. This is the recommended Windows path for full kernel support.

## Setup

1. Install WSL2 + Ubuntu 22.04+
2. Install [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
3. Verify: `nvidia-smi` inside WSL

## Install (inside WSL)

**Use the Linux home directory**, not a Windows path. Building CUDA wheels on `/mnt/c/...` often fails (missing `pyproject.toml` / `setup.py` on C:).

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
# installs to $HOME/Seiso by default

# Or from an existing clone under $HOME:
cd "$HOME/Seiso"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,dev]"
./scripts/install_flash_attn.sh   # optional
```

Follow [linux-nvidia.md](linux-nvidia.md) for training, kernels, and multi-GPU.

## Secure GPU training

Before starting CUDA training inside WSL2, acknowledge the GPU boundary:

```bash
export SEISO_NVIDIA_WSL_ACK=1
seiso train --config configs/example_lora.yaml
```

This gates native kernel JIT and driver access. Prefer WSL2 over bare Windows for fused CUDA kernels (RMSNorm, SwiGLU, LoRA, cross-entropy).

## Access Forge from Windows browser

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

Browse to **http://127.0.0.1:8765** from Windows (WSL forwards localhost by default).

## Data paths

- Seiso data: `$HOME/.seiso` inside WSL (not on `/mnt/c/...`)
- Mount Windows datasets: `/mnt/c/Users/.../data/train.jsonl`

Use WSL paths in dataset fields, not `C:\...` strings.
