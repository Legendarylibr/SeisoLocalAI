# Seiso

**Seiso** is a local-first AI platform for running, training, and fine-tuning open models. Two surfaces:

| Surface | Description |
|---------|-------------|
| **Seiso Forge** | Web UI + backend API for chat, training, export, recipes, and knowledge bases |
| **Seiso Core** | Python library + CLI for programmatic training, export, and inference |

Runs on Windows, Linux, WSL, and macOS (NVIDIA GPU, Apple MLX, CPU for select flows).

## Core value

- QLoRA / LoRA / full fine-tune and embedding training with TRL SFTTrainer
- Curated catalog of popular Hugging Face models (~46 entries, expandable)
- Local-first: download models, chat, train, export, deploy — mostly offline
- **Secure by default**: localhost binding, auth-guarded APIs, path sandboxing, signed tokens
- OpenAI-compatible `/v1/chat/completions` for Cursor, Continue, and other clients

## Quick start

```bash
# Install core + forge server
pip install -e ".[forge,train,dev]"

# Launch Forge (web UI + API)
seiso forge

# Or train from CLI
seiso train --config configs/example_lora.yaml
```

Open **http://127.0.0.1:8765** — complete onboarding to create your local admin account.

## CLI

| Command | Purpose |
|---------|---------|
| `seiso forge` | Launch Forge web server |
| `seiso train` | Train from config/checkpoint |
| `seiso chat` | Terminal chat with local models |
| `seiso export` | Export merged/GGUF/LoRA + Hub push |
| `seiso inference` | One-shot inference |

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
| `forge/orchestrators/inference` | Local inference, tools, MCP, providers |
| `forge/orchestrators/training` | QLoRA training jobs, multi-GPU via torchrun |
| `forge/orchestrators/export` | Merge LoRA, GGUF, Hub upload |
| `forge/orchestrators/recipes` | Recipe jobs, HF dataset ops |
| `forge/orchestrators/knowledge` | RAG ingest and retrieve (API only) |

## Platform support

| Platform | Chat | Train | GGUF/MLX inference |
|----------|------|-------|--------------------|
| NVIDIA GPU | ✓ | ✓ | ✓ |
| macOS (MLX) | ✓ | ✓ | ✓ |
| CPU | ✓ | limited | GGUF via llama.cpp |

## Licensing

- **Apache-2.0** — `seiso/` core package and CLI
- **AGPL-3.0** — `forge-ui/` web components

## Features

- **Triton kernels** — fused RMSNorm patching during training (`pip install -e ".[train,cuda]"`)
- **Multi-GPU** — torchrun distributed workers; rank-0 checkpoint writes
- **Tool calling** — web search, sandboxed code execution, artifact writes
- **Providers** — OpenAI, Anthropic, Ollama, vLLM routing
- **MCP** — connect stdio MCP servers; tools auto-register in chat
- **Recipe Studio** — visual `@xyflow/react` canvas → backend graph executor

## Development

```bash
pytest tests/
cd forge-ui && npm install && npm run typecheck
```

## Security

Seiso is **secure by default** for single-user localhost use. Multi-user or remote deployments should review every flag below.

### Network binding

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEISO_ALLOW_REMOTE=false` | off | Binds Forge to `127.0.0.1` only |
| `SEISO_ALLOW_REMOTE=true` | — | Allows LAN/WAN binding; sets secure session cookies |

### Opt-in capabilities (all default **off**)

| Variable | Enables |
|----------|---------|
| `SEISO_ALLOW_TOOLS=true` | Web search, artifact writes, MCP server create/connect |
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

### MCP hardening

When `SEISO_ALLOW_TOOLS=true`:

- Per-user server pools (max 8 servers per user)
- Ownership validated on inference connect
- Blocked env keys: `PATH`, proxies, `PYTHON*`, `LD_*`, `DYLD_*`, `NODE_*`, `SSL_*`
- Inline exec args blocked (`python3 -c`, `-e`, `-m`); unpinned `npx -y` rejected

### Auth & rate limits

- Signed session tokens; login rate-limited to 10 attempts/minute per IP
- Job and resource ownership enforced on all streaming endpoints

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
