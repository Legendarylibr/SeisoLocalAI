# Linux + NVIDIA

Full Seiso support: fused CUDA kernels, QLoRA, multi-GPU, Forge UI.

## Install

**Recommended** — one command installs deps, builds the UI, and starts Forge:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Manual:

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,cuda,llamacpp,dev]"
```

Requirements:
- NVIDIA driver + `nvidia-smi` working
- CUDA toolkit (`nvcc`) for JIT-compiled fused kernels (first training run compiles)
- PyTorch CUDA wheel (installed automatically via `torch` dependency)
- `[llamacpp]` for local GGUF chat in Forge (included in `start` and the manual install above)

### Optional: Flash Attention 2

Not required. Install after the main editable install when the repo lives on the **Linux filesystem** (e.g. `$HOME/Seiso`), not on a Windows mount (`/mnt/c/...` in WSL):

```bash
./scripts/install_flash_attn.sh
```

Skip during install: `SEISO_SKIP_FLASH_ATTN=1 start`

## Start Forge

**Later sessions:**

```bash
cd "$HOME/Seiso" && start
```

Or from a clone with venv active:

```bash
cd "$HOME/Seiso"
source .venv/bin/activate
cd forge-ui && npm ci && npm run build && cd ..
seiso forge
```

Open **http://127.0.0.1:8765** — complete onboarding, then use **Training Studio** (`/train`).

## Train (CLI)

Approve GPU training on bare Linux hosts (gates native kernel JIT):

```bash
export SEISO_NVIDIA_HOST_VENV_ACK=1   # or use Docker / VM tier
seiso train --config configs/example_lora.yaml
```

```yaml
use_triton: true      # enables fused RMSNorm + SwiGLU MLP (CUDA native on NVIDIA)
use_fused_ce: true    # fused cross-entropy in SFTTrainer
use_fused_lora: true  # fused LoRA delta (CUDA / WSL2, rank ≤ 64)
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
| RMSNorm + MLP | Native CUDA (stripe, SM-targeted) → Triton → PyTorch |
| LoRA delta | Native CUDA (batched) → PyTorch |
| Cross-entropy | Native CUDA → Triton → PyTorch |

See [../training/kernels.md](../training/kernels.md).
