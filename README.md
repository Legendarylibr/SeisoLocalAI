# Seiso

**Seiso** is a local-first AI platform for running, training, and fine-tuning open models. Two surfaces:

| Surface | Description |
|---------|-------------|
| **Seiso Forge** | Web UI + backend API for chat, training, export, recipes, and knowledge bases |
| **Seiso Core** | Python library + CLI for programmatic training, export, and inference |

Runs on Windows, Linux, WSL, and macOS. See **[docs/](docs/README.md)** for per-platform install and startup.

## Core value

- QLoRA / LoRA / full fine-tune and embedding training with TRL SFTTrainer
- Curated catalog of popular Hugging Face models (~46 entries, expandable)
- Local-first: download models, chat, train, export, deploy — mostly offline
- **Secure by default**: localhost binding, auth-guarded APIs, path sandboxing, signed tokens
- OpenAI-compatible `/v1/chat/completions` for Cursor, Continue, and other clients

## Quick start

### Linux & macOS (recommended)

Install Python 3.10+, [Node.js 18+](https://nodejs.org/), and git, then run:

```bash
curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh
```

Open **http://127.0.0.1:8765** — complete onboarding to create your local admin account.

Options:

```bash
# Custom install location
SEISO_INSTALL_DIR=~/code/Seiso curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash

# Install and start Forge immediately
SEISO_START=1 curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash
```

The installer clones to `~/Seiso` by default, creates a venv, installs platform extras (CUDA on Linux + NVIDIA, MLX on macOS), builds the Forge UI, and copies `.env.example` → `.env`. See [docs/install.md](docs/install.md) for manual install and Windows.

### Manual install (from a clone)

```bash
# Install — see docs/install.md for platform extras
pip install -e ".[forge,train,dev]"

# Linux NVIDIA (fused CUDA kernels)
pip install -e ".[forge,train,cuda,dev]"

# macOS (MLX chat)
pip install -e ".[forge,train,mlx,dev]"

# Build UI (first time or after UI changes)
cd forge-ui && npm install && npm run build && cd ..

# Launch Forge (web UI + API)
seiso forge

# Or train from CLI
seiso train --config configs/example_lora.yaml
```

For UI hot reload during development, see [docs/forge.md](docs/forge.md).

### Local CI

Before opening PRs, run the quality gate:

```bash
make ci-fast    # lint + types + test + security
make ci         # full gate (+ frontend build + optional import smokes)
```

See **[docs/CI_LOCAL.md](docs/CI_LOCAL.md)** for all jobs (Ruff, Pylint, Mypy, Bandit, detect-secrets, pip-audit, pytest, forge-ui build).

## CLI

| Command | Purpose |
|---------|---------|
| `seiso forge` | Launch Forge web server |
| `seiso train` | Train from config/checkpoint |
| `seiso chat` | Terminal chat with local models |
| `seiso export` | Export merged/GGUF/LoRA + Hub push |
| `seiso compress run` | Code Llama compression pipeline (distill → prune → finetune → export) |
| `seiso compress manifest-verify` | Verify hash-chained compression run manifest |
| `seiso compress speculative` | Speculative decoding with draft + target models |
| `seiso inference` | One-shot inference |
| `seiso-bench-kernels` | Benchmark fused GPU kernels (NVIDIA / ROCm) |

Full command reference: [docs/cli.md](docs/cli.md) · Documentation index: [docs/README.md](docs/README.md)

## Architecture

```
Seiso/
├── seiso/           # Core library (Apache-2.0)
├── seiso_cli/       # CLI entrypoints
├── forge/           # FastAPI backend + orchestrators
└── forge-ui/        # React/TypeScript frontend (AGPL-3.0)
```

Backend orchestrators spawn isolated workers with SSE log streaming:

| Module | Role |
|--------|------|
| `forge/orchestrators/inference` | Local inference, tools, providers |
| `forge/orchestrators/training` | QLoRA training jobs, multi-GPU via torchrun |
| `forge/orchestrators/export` | Merge LoRA, GGUF, Hub upload |
| `forge/orchestrators/recipes` | Recipe jobs, HF dataset ops |
| `forge/orchestrators/knowledge` | RAG ingest and retrieve (API only) |
| `forge/orchestrators/compress` | Distillation, pruning, fine-tune, export bundle jobs |
| `forge/orchestrators/rl_quant` | Adaptive RL quantization policy training |

## Platform support

| Platform | Chat | Train | Fused kernels | Notes |
|----------|------|-------|---------------|-------|
| Linux + NVIDIA | ✓ | ✓ QLoRA | ✓ CUDA | [docs](docs/platforms/linux-nvidia.md) |
| Linux + AMD ROCm | ✓ | ✓ | ✓ Triton | [docs](docs/platforms/linux-amd-rocm.md) |
| Windows + NVIDIA | ✓ | ✓ QLoRA | ✓ CUDA JIT | [docs](docs/platforms/windows.md) |
| WSL2 + NVIDIA | ✓ | ✓ | ✓ CUDA + Triton | [docs](docs/platforms/wsl.md) |
| macOS Apple Silicon | ✓ MLX | ✓ 16-bit LoRA | — | [docs](docs/platforms/macos.md) |
| CPU | ✓ GGUF/Ollama | limited | — | |

Inference vs training differ on macOS — MLX is inference-only; training uses PyTorch MPS/CPU.

## Licensing

- **Apache-2.0** — `seiso/` core package and CLI
- **AGPL-3.0** — `forge-ui/` web components

## Features

- **Fused GPU kernels** — RMSNorm, SwiGLU MLP, cross-entropy ([docs/training/kernels.md](docs/training/kernels.md); `pip install -e ".[train,cuda]"` on Linux NVIDIA)
- **Multi-GPU** — torchrun distributed workers; rank-0 checkpoint writes
- **Tool calling** — web search, sandboxed code execution, artifact writes
- **Providers** — OpenAI, Anthropic, Ollama, vLLM routing
- **Recipe Studio** — visual `@xyflow/react` canvas → backend graph executor
- **Model compression** — distill, MLP prune, recovery fine-tune, GPTQ/AWQ, speculative decoding ([codellama-compress](https://github.com/Legendarylibrorg/codellama-compress))
- **RL quantization** — adaptive GGUF quant policy via reinforcement learning

## Development

```bash
make ci-fast    # lint + types + test + security
pytest tests/
cd forge-ui && npm install && npm run typecheck
```

See [docs/CI_LOCAL.md](docs/CI_LOCAL.md) and [docs/forge.md](docs/forge.md) for the full quality gate and UI dev workflow.

## Security

Seiso is **secure by default** for single-user localhost use. Multi-user or remote deployments should review every flag below.

### Network binding

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEISO_ALLOW_REMOTE=false` | off | Binds Forge to `127.0.0.1` only |
| `SEISO_ALLOW_REMOTE=true` | — | Allows LAN/WAN binding; sets secure session cookies |
| `SEISO_TRUST_PROXY=true` | — | Trust `X-Forwarded-*` from reverse proxy (rate limits, client IP) |
| `SEISO_SECURE_COOKIES=true` | — | `Secure` cookies when TLS is terminated by a reverse proxy |

Deploy configs: [`deploy/`](deploy/) · Guide: [docs/deployment/reverse-proxy.md](docs/deployment/reverse-proxy.md)

### Opt-in capabilities (all default **off**)

| Variable | Enables |
|----------|---------|
| `SEISO_ALLOW_TOOLS=true` | Web search, artifact writes |
| `SEISO_ALLOW_CODE_EXEC=true` | Sandboxed `execute_code` tool (also requires per-request flag) |
| `SEISO_ALLOW_OPENAI_TOOLS=true` | OpenAI-compatible `/v1/chat/completions` tool calling |

### Path sandbox & tenant isolation

All filesystem access is scoped under `SEISO_DATA_DIR` (default `~/.seiso`):

- **Per-user dirs** — `models/`, `checkpoints/`, `exports/`, `artifacts/`, `uploads/` are namespaced by user ID
- **Knowledge bases** — ingest only from `uploads/{user_id}/`; retrieve only from `knowledge/{user_id}/{kb_id}`
- **Cross-user access** — rejected at the API layer with 403

### Provider SSRF protection

Outbound calls to OpenAI, Anthropic, Ollama, and vLLM providers are hardened:

- HTTPS required for remote hosts (HTTP only for local Ollama/vLLM on loopback)
- Private, link-local, metadata, and unresolvable hosts blocked at config time
- **DNS pinning** — hostname is resolved and validated immediately before connect; the socket layer is forced to the validated IP to close DNS-rebinding windows
- Local Ollama/vLLM limited to ports `11434` and `8000`/`8001`

### Auth & rate limits

- Signed session tokens; login throttling (10 attempts/minute per IP) always enabled
- Global API rate limiting when `SEISO_ALLOW_REMOTE=true` (`SEISO_RATE_LIMIT`, default 120/min per IP)
- Job and resource ownership enforced on all streaming endpoints

### Database (local-only, zero retention)

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEISO_DB_EPHEMERAL=true` | on | In-memory SQLite — all metadata wiped on restart |
| `SEISO_DB_EPHEMERAL=false` | — | Opt-in persistence to `{SEISO_DATA_DIR}/forge.db` |
| `SEISO_DB_ENCRYPTION_KEY` | auto | AES-256-GCM key for sensitive fields (chat, provider configs) |

Sensitive columns are encrypted at the application layer (same AES-256-GCM pattern as [Web-app-practice](https://github.com/Legendarylibr/Web-app-practice)). When ephemeral, a fresh session key is generated each run; the legacy `forge.db` file is removed on startup.

### Recommended production checklist

```bash
# Keep defaults unless you explicitly need remote/multi-user access
export SEISO_ALLOW_REMOTE=false
export SEISO_ALLOW_TOOLS=false
export SEISO_ALLOW_CODE_EXEC=false
export SEISO_ALLOW_OPENAI_TOOLS=false

# Use a strong secret (auto-generated on first run if unset)
export SEISO_SECRET_KEY="$(openssl rand -hex 32)"
```
