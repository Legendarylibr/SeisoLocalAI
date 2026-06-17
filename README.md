# Seiso Local AI

WIP (only been three hours of work so far)

**Seiso** is a free, local-first AI workspace for running, fine-tuning, quantizing, compressing, and deploying open models on your own hardware. No cloud required — download models, chat, train, export, and ship to Hugging Face Hub from a single interface.

**Repository:** [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)  
**License:** [GPL-3.0](LICENSE)  
**Docs:** [docs/README.md](docs/README.md)

---

## Table of contents

- [What Seiso does](#what-seiso-does)
- [Quick start](#quick-start)
- [Forge UI walkthrough](#forge-ui-walkthrough)
- [CLI reference](#cli)
- [Architecture](#architecture)
- [Platform support](#platform-support)
- [Features in depth](#features-in-depth)
- [Data & storage](#data--storage)
- [OpenAI-compatible API](#openai-compatible-api)
- [Development](#development)
- [Security](#security)
- [Documentation index](#documentation-index)

---

## What Seiso does

Seiso combines a **web workspace (Forge)** and a **Python core (CLI + library)** so you can work entirely on your machine:

| Workflow | Forge UI | CLI |
|----------|----------|-----|
| Download & chat with open models | Model Hub, Chat | `seiso chat` |
| QLoRA / LoRA / full fine-tune | Training Studio | `seiso train` |
| Merge, GGUF, Hub publish | Export | `seiso export` |
| LLM distill → prune → quant | Compress | `seiso compress` |
| SD image compression | Image Compress | — |
| RL adaptive GGUF quantization | RL Quant | — |
| Visual data/recipe pipelines | Recipe Studio | — |
| RAG knowledge bases | API | — |

**Why local-first?**

- **Privacy** — prompts, datasets, and checkpoints never leave your machine unless you publish them
- **Control** — pick quant method, fused kernels, multi-GPU layout, and export formats
- **Offline-capable** — chat and train on downloaded weights without ongoing API costs
- **Secure by default** — localhost binding, encrypted sensitive fields, path sandboxing, opt-in tools

Two surfaces share the same core:

| Surface | Path | Role |
|---------|------|------|
| **Seiso Forge** | `forge/` + `forge-ui/` | React UI + FastAPI backend |
| **Seiso Core** | `seiso/` + `seiso_cli/` | Training, inference, export, compression library + CLI |

Runs on **Windows**, **Linux**, **WSL2**, and **macOS**. See [docs/platforms/](docs/README.md#platform-guides) for per-OS guides.

---

## Quick start

### Linux & macOS (recommended)

Install **Python 3.10+**, **[Node.js 18+](https://nodejs.org/)**, and **git**, then:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh
```

Open **http://127.0.0.1:8765** and complete onboarding (create your local admin password).

**Options:**

```bash
# Custom install location
SEISO_INSTALL_DIR=~/code/Seiso curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash

# Install and start Forge immediately
SEISO_START=1 curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
```

The installer clones to `~/Seiso`, creates a venv, installs platform extras (CUDA on Linux + NVIDIA, MLX on macOS), builds the Forge UI, and copies `.env.example` → `.env`.

### Manual install (all platforms)

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git Seiso && cd Seiso
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# Pick your platform extras:
pip install -e ".[forge,train,dev]"              # base
pip install -e ".[forge,train,cuda,dev]"         # Linux NVIDIA
pip install -e ".[forge,train,mlx,dev]"          # macOS Apple Silicon

cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

Full install guide: **[docs/install.md](docs/install.md)**  
First-run walkthrough: **[docs/getting-started.md](docs/getting-started.md)**

### Windows

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git Seiso
cd Seiso
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[forge,train,dev]"
cd forge-ui; npm install; npm run build; cd ..
seiso forge
```

See [docs/platforms/windows.md](docs/platforms/windows.md) for CUDA PyTorch and build tools.

---

## Forge UI walkthrough

After `seiso forge` (or `./scripts/start.sh`), browse to **http://127.0.0.1:8765**:

| Page | Path | What it does |
|------|------|--------------|
| Dashboard | `/` | Workspace overview |
| Model Hub | `/hub` | Browse & download ~46 curated HF models |
| Chat | `/chat` | Local inference (GGUF, MLX, PyTorch, Ollama) |
| Training Studio | `/train` | LoRA / QLoRA fine-tune with live SSE logs |
| Export | `/export` | Merge LoRA, GGUF quant, Hugging Face publish |
| Compress | `/compress` | Code Llama distill → prune → finetune → quant |
| Image Compress | `/image-compress` | Stable Diffusion compression pipeline |
| RL Quant | `/rl-quant` | Adaptive GGUF quantization via RL |
| Recipe Studio | `/recipes` | Visual `@xyflow/react` graph editor |
| Integrations | `/integrations` | Route to OpenAI, Anthropic, Ollama, vLLM |
| Settings | `/settings` | HF token, hardware info, security toggles |

Knowledge-base ingest/retrieve is **API-only** (`/api/knowledge/...`).

Forge details: **[docs/forge.md](docs/forge.md)**

---

## CLI

| Command | Purpose |
|---------|---------|
| `seiso forge` | Launch Forge web server |
| `seiso train` | Train from YAML config |
| `seiso chat` | Terminal chat with local models |
| `seiso export` | Export merged / GGUF / LoRA + Hub push |
| `seiso compress run` | Code Llama compression pipeline |
| `seiso compress manifest-verify` | Verify hash-chained run manifest |
| `seiso compress speculative` | Speculative decoding (draft + target) |
| `seiso inference` | One-shot inference |
| `seiso-bench-kernels` | Benchmark fused GPU kernels |

```bash
# Example: fine-tune Llama 3.2 3B on sample data
seiso train --config configs/example_lora.yaml

# Example: export checkpoint to GGUF
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --formats merged,gguf
```

Full reference: **[docs/cli.md](docs/cli.md)**

---

## Architecture

```
Seiso/
├── seiso/              # Core library — training, inference, export, kernels, compression
├── seiso_cli/          # CLI entrypoints (seiso, seiso-bench-kernels, seiso-train-worker)
├── forge/              # FastAPI backend, auth, orchestrators, SSE job streaming
├── forge-ui/           # React 19 + TypeScript + Vite frontend
├── third_party/        # Vendored compression pipelines (codellama, SD, RL quant)
├── configs/            # Example YAML/JSON configs
├── deploy/             # Caddy, nginx, systemd, HTTPS env templates
├── docs/               # Documentation
└── scripts/            # install.sh, start.sh, CI runner
```

Backend orchestrators spawn isolated workers with **SSE log streaming**:

| Module | Role |
|--------|------|
| `forge/orchestrators/inference` | Local inference, tools, provider routing |
| `forge/orchestrators/training` | QLoRA jobs, multi-GPU via `torchrun` |
| `forge/orchestrators/export` | Merge LoRA, GGUF conversion, Hub upload |
| `forge/orchestrators/recipes` | Recipe graph jobs, HF dataset ops |
| `forge/orchestrators/knowledge` | RAG ingest and retrieve |
| `forge/orchestrators/compress` | LLM distillation, pruning, quant |
| `forge/orchestrators/image_compress` | Stable Diffusion compression |
| `forge/orchestrators/rl_quant` | Adaptive RL GGUF quant policy |

Training stack: **TRL `SFTTrainer`** + **PEFT** (LoRA/QLoRA) + optional **fused CUDA/Triton kernels**.

---

## Platform support

| Platform | Chat | Train | Fused kernels | Guide |
|----------|------|-------|---------------|-------|
| Linux + NVIDIA | ✓ | ✓ QLoRA 4-bit | ✓ CUDA native | [linux-nvidia](docs/platforms/linux-nvidia.md) |
| Linux + AMD ROCm | ✓ | ✓ 4-bit* | ✓ Triton | [linux-amd-rocm](docs/platforms/linux-amd-rocm.md) |
| Windows + NVIDIA | ✓ | ✓ QLoRA | ✓ CUDA JIT | [windows](docs/platforms/windows.md) |
| WSL2 + NVIDIA | ✓ | ✓ QLoRA | ✓ CUDA + Triton | [wsl](docs/platforms/wsl.md) |
| macOS Apple Silicon | ✓ MLX | ✓ 16-bit LoRA | — | [macos](docs/platforms/macos.md) |
| CPU | ✓ GGUF/Ollama | limited | — | — |

\* bitsandbytes on ROCm depends on your PyTorch build.

**macOS note:** MLX is inference-only. Training always uses PyTorch (MPS or CPU) with 16-bit LoRA.

---

## Features in depth

### Training

- **Methods:** LoRA, QLoRA (4-bit), full fine-tune, embedding training
- **Formats:** JSONL chat datasets with auto format detection
- **Optimizations:** gradient checkpointing, packing, RSLoRA, train-on-responses-only
- **Multi-GPU:** `torchrun` distributed workers; rank-0 checkpoint writes ([multi-gpu](docs/training/multi-gpu.md))
- **Fused kernels:** RMSNorm, SwiGLU MLP, cross-entropy, fused LoRA delta ([kernels](docs/training/kernels.md))

### Inference

- **Backends:** llama.cpp (GGUF), MLX (macOS), PyTorch (4-bit/16-bit), Ollama ([backends](docs/inference/backends.md))
- **Tool calling:** web search, sandboxed code execution, artifact writes (opt-in)
- **Providers:** OpenAI, Anthropic, Ollama, vLLM with SSRF hardening

### Export

- **Formats:** merged safetensors, LoRA adapter, full fine-tune, GGUF (Q4_K_M, Q8_0, etc.)
- **Hub publish:** metadata preflight, model card generation, repo creation
- **Profiles:** inference-optimized export presets

### Compression

Three integrated pipelines ([compression.md](docs/compression.md)):

1. **Code Llama** — distill → MLP prune → recovery finetune → GPTQ/AWQ → speculative decoding
2. **Stable Diffusion** — progressive distill, prune, quantize, ONNX export
3. **RL quantization** — train a policy for adaptive GGUF quant levels

---

## Data & storage

All user data lives under **`~/.seiso`** by default (`SEISO_DATA_DIR`):

```
~/.seiso/
├── models/           # Downloaded model weights
├── checkpoints/      # Training outputs (per user)
├── exports/          # Merged / GGUF / LoRA exports
├── compress/         # LLM compression artifacts
├── image_compress/   # SD compression outputs
├── rl_quant/         # RL quant outputs
├── uploads/          # Datasets and user files
├── knowledge/        # RAG stores
├── hf_cache/         # Hugging Face cache
└── sandbox/          # Sandboxed code-exec workspace
```

Database defaults to **ephemeral in-memory SQLite** — chat history and job metadata are wiped on restart unless you opt into persistence (`SEISO_DB_EPHEMERAL=false`).

---

## OpenAI-compatible API

Point Cursor, Continue, or any OpenAI client at Forge while it is running:

```text
Base URL: http://127.0.0.1:8765/v1
```

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Explain QLoRA in one paragraph."}],
    "stream": true
  }'
```

Set `SEISO_ALLOW_OPENAI_TOOLS=true` for tool calling on this endpoint.

---

## Development

```bash
make ci-fast          # lint + types + test + security
make ci               # full gate (+ frontend build)
pytest tests/ -q
cd forge-ui && npm run typecheck
```

Quality gate details: **[docs/CI_LOCAL.md](docs/CI_LOCAL.md)**  
UI dev workflow: **[docs/forge.md](docs/forge.md)**

---

## Security

Seiso is **secure by default** for single-user localhost use. Review every flag before remote or multi-user deployment.

### Network binding

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEISO_ALLOW_REMOTE=false` | off | Binds Forge to `127.0.0.1` only |
| `SEISO_ALLOW_REMOTE=true` | — | Allows LAN/WAN binding; enables rate limits |
| `SEISO_TRUST_PROXY=true` | — | Honor `X-Forwarded-*` from reverse proxy |
| `SEISO_SECURE_COOKIES=true` | — | Secure cookies when TLS is terminated upstream |

Deploy configs: [`deploy/`](deploy/) · Guide: [docs/deployment/reverse-proxy.md](docs/deployment/reverse-proxy.md)

### Opt-in capabilities (all default **off**)

| Variable | Enables |
|----------|---------|
| `SEISO_ALLOW_TOOLS=true` | Web search, artifact writes |
| `SEISO_ALLOW_CODE_EXEC=true` | Sandboxed `execute_code` tool |
| `SEISO_ALLOW_OPENAI_TOOLS=true` | Tool calling on `/v1/chat/completions` |

### Path sandbox & tenant isolation

All filesystem access is scoped under `SEISO_DATA_DIR`. Per-user directories (`models/`, `checkpoints/`, `exports/`, etc.) are namespaced by user ID. Cross-user access is rejected with 403.

### Provider SSRF protection

Outbound provider calls block private/metadata hosts, require HTTPS for remote endpoints, and use **DNS pinning** to prevent rebinding attacks.

### Auth & database

- Signed session tokens; login throttling (10 attempts/min per IP)
- Rate limiting when `SEISO_ALLOW_REMOTE=true` (default 120 req/min per IP)
- AES-256-GCM encryption for sensitive DB columns
- Ephemeral SQLite by default — zero retention on restart

### Production checklist

```bash
export SEISO_ALLOW_REMOTE=false
export SEISO_ALLOW_TOOLS=false
export SEISO_ALLOW_CODE_EXEC=false
export SEISO_ALLOW_OPENAI_TOOLS=false
export SEISO_SECRET_KEY="$(openssl rand -hex 32)"
```

---

## Licensing

Seiso is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE).

---

## Documentation index

| Topic | Guide |
|-------|-------|
| **Getting started (full walkthrough)** | [docs/getting-started.md](docs/getting-started.md) |
| Install & extras | [docs/install.md](docs/install.md) |
| Documentation hub | [docs/README.md](docs/README.md) |
| Forge UI & API | [docs/forge.md](docs/forge.md) |
| CLI commands | [docs/cli.md](docs/cli.md) |
| Training | [docs/training/quickstart.md](docs/training/quickstart.md) |
| GPU kernels | [docs/training/kernels.md](docs/training/kernels.md) |
| Multi-GPU | [docs/training/multi-gpu.md](docs/training/multi-gpu.md) |
| Inference backends | [docs/inference/backends.md](docs/inference/backends.md) |
| Compression | [docs/compression.md](docs/compression.md) |
| HTTPS deployment | [docs/deployment/reverse-proxy.md](docs/deployment/reverse-proxy.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Local CI | [docs/CI_LOCAL.md](docs/CI_LOCAL.md) |
