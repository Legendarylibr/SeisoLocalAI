# WSL2 + NVIDIA

Treat WSL2 as **Linux** for Seiso. This is the recommended Windows path for full kernel support.

## Setup

1. Install WSL2 + Ubuntu 22.04+
2. Install [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
3. Verify: `nvidia-smi` inside WSL

## Install (inside WSL)

```bash
cd /mnt/c/Users/<you>/Seiso   # or clone into ~/Seiso
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,dev]"
```

Follow [linux-nvidia.md](linux-nvidia.md) for training, kernels, and multi-GPU.

## Access Forge from Windows browser

```bash
seiso forge
```

Browse to **http://127.0.0.1:8765** from Windows (WSL forwards localhost by default).

## Data paths

- Seiso data: `~/.seiso` inside WSL
- Mount Windows datasets: `/mnt/c/Users/.../data/train.jsonl`

Use WSL paths in dataset fields, not `C:\...` strings.
