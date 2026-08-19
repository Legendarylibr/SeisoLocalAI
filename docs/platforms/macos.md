# macOS

## Summary

| Feature | Apple Silicon | Intel Mac |
|---------|---------------|-----------|
| Forge UI | ✓ | ✓ |
| MLX chat inference | ✓ (with `[mlx]`) | — |
| GGUF chat | ✓ | ✓ |
| QLoRA 4-bit training | ✗ (no bitsandbytes) | ✗ |
| 16-bit LoRA training | ✓ (MPS, small models) | ✓ (CPU, tiny models) |
| Fused GPU kernels | ✗ | ✗ |

## Install

**Recommended** — one command installs deps, builds the UI, and starts Forge (browser opens automatically):

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
# or explicitly for Apple Silicon (includes MLX):
SEISO_INSTALL_PROFILE=macos curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Manual:

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[forge,train,llamacpp,mlx,dev]"
cd forge-ui && npm ci && npm run build && cd ..
seiso doctor
seiso forge
```

## Start Forge

**Later sessions** (after the initial install):

```bash
cd "$HOME/Seiso" && start
```

Or from an existing clone:

```bash
cd "$HOME/Seiso"
source .venv/bin/activate
seiso forge
```

Rebuild the UI only when needed:

```bash
cd "$HOME/Seiso/forge-ui" && npm ci && npm run build && cd ..
```

## MLX inference (chat)

The `[mlx]` extra is included automatically by `start` on macOS. For manual installs or upgrades:

```bash
pip install -e ".[mlx]"
```

In Forge Chat, pick a model with MLX backend when you want safetensors on Apple Silicon.
On **≤24 GB** unified Macs, Forge prefers **llama.cpp + GGUF** (Metal GPU layers) over MLX so larger models can mmap and partially offload.

## Larger models / Metal offload

Apple Silicon GGUF chat uses **llama.cpp with Metal** (`SEISO_LLAMA_GPU_LAYERS=-1` by default). When a model does not fit the Metal working set, Seiso retries with fewer GPU layers and optional CPU-side KQV (`SEISO_LLAMA_MAC_CPU_OFFLOAD=true`, default on), plus `mmap` so weights can exceed free RAM. Prefer Q4/Q5 GGUF; use **Free memory** before loading the next large model.

| Goal | Path |
|------|------|
| Bigger chat models on Mac | GGUF + llama.cpp Metal / partial CPU offload |
| Fast small–mid safetensors | MLX (`.[mlx]`) when RAM headroom is ample (typically 32 GB+) |
| Training | PyTorch MPS 16-bit LoRA only (small models) — not Metal/MLX chat offload |

## Training on macOS

Training **always uses PyTorch** (never MLX), even when MLX is installed for chat.

- Use **16-bit LoRA** — Forge pre-fills `quant: 16bit` when bitsandbytes is unavailable
- **MPS** is used automatically when available (`torch.backends.mps`)
- Fused kernel checkboxes are **disabled** in Training Studio (no CUDA GPU)
- Keep models small (1–3B), `max_seq_length` 1024–2048, gradient checkpointing on

```bash
seiso train --config configs/example_lora.yaml
```

Set in YAML:

```yaml
quant: 16bit
use_triton: false
use_fused_ce: false
```

## Build Forge UI

```bash
cd forge-ui && npm install && npm run build && cd ..
```

## Low-RAM Mac guide (Apple Silicon + Intel)

Seiso detects **installed RAM and free headroom**, not chip model (M1, M2, Intel, etc.).

| RAM | Suggested models | Backend |
|-----|------------------|---------|
| 16 GB | Phi-4 Mini, Gemma 3 4B, Qwen 3.6 4B–9B | llama.cpp GGUF |
| 24 GB | Gemma 3 12B, Mistral Small 24B, GPT-OSS 20B | Free memory before load |
| 32 GB+ | 27B class when Hub shows ideal/good fit | llama.cpp; MLX optional for safetensors |

**Intel Mac:** CPU-only — stick to ≤3B Q4 for responsive chat.

**Free memory:** Use the button in Chat or Model Hub before switching to a larger model. Weights stay on disk under `{SEISO_DATA_DIR}/hf_cache/`.

Optional `.env` knobs (see `.env.example`): `SEISO_MEMORY_PROFILE=low`, `SEISO_LLAMA_USE_MMAP=true`, `SEISO_LLAMA_USE_MLOCK=false`, `SEISO_LLAMA_CACHE_MB=0`, `SEISO_SKIP_MLX_PROBE=1`.
