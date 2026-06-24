# Seiso Local AI

[![CI](https://github.com/Legendarylibr/SeisoLocalAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Legendarylibr/SeisoLocalAI/actions/workflows/ci.yml)

**Seiso** is a local-first AI workspace for running, fine-tuning, quantizing, compressing, and deploying open models on your own hardware. No cloud required — download models, chat, train, export, and ship to Hugging Face Hub from a single interface.

**Repository:** [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)  
**License:** [GPL-3.0](LICENSE) · **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)  
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
| LLM distill → prune → quant | Compress | `seiso compress run` |
| Teacher distill + DPO alignment | Distill-RL | `seiso distill-rl run` |
| RL quant + CUDA kernel policy | RL Quant | `seiso rl-quant run` |
| Visual data/recipe pipelines | Recipe Studio | — |
| RAG knowledge bases | Knowledge | — |

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

**Forge URL (all platforms):** [http://127.0.0.1:8765](http://127.0.0.1:8765)  
On first launch, create your local admin password in the browser tab that opens.

### Paths on your system

Seiso uses two directories. Override either with environment variables when needed.

| Purpose | Linux / macOS / WSL | Windows (native) | Override |
|---------|---------------------|------------------|----------|
| **Repository** (clone, venv, scripts) | `$HOME/Seiso` | `%USERPROFILE%\Seiso` | `SEISO_INSTALL_DIR` |
| **User data** (models, checkpoints, exports) | `$HOME/.seiso` | `%USERPROFILE%\.seiso` | `SEISO_DATA_DIR` |

In config files and Python, `~/.seiso` expands correctly on every OS. In shell commands, use the column for your platform — do not paste Unix `~/` paths into PowerShell.

### Linux, macOS, and WSL2 (recommended)

One command installs missing system tools (Python, Node, git when possible), clones the repo, builds the UI, and **starts Forge automatically** (browser opens when ready):

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

What the installer does:

1. Clones to `$HOME/Seiso` (or `SEISO_INSTALL_DIR`)
2. Creates `.venv` and installs platform extras (CUDA on Linux + NVIDIA, MLX on macOS, GGUF via `llamacpp`)
3. Copies `.env.example` → `.env` if missing
4. Builds `forge-ui/dist`
5. Runs `seiso forge` and opens the browser

**Install only** (no auto-start):

```bash
SEISO_START=0 curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Custom install location:**

```bash
SEISO_INSTALL_DIR="$HOME/code/Seiso" curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Already cloned?** From the repository root:

```bash
start                    # install / upgrade deps + build UI; starts Forge by default
SEISO_START=0 start      # install only
```

Install registers `start` on your PATH (`~/.local/bin`). Open a new terminal if the command is not found yet.

### Starting Forge after install

| Situation | Linux / macOS / WSL | Notes |
|-----------|---------------------|-------|
| **First install** | Handled by `start` — no extra step | Browser opens at `http://127.0.0.1:8765` |
| **Later sessions** | `start` | Re-checks deps, builds UI if missing, opens browser |
| **One-liner restart** | `curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start \| bash` | Bootstraps install if repo is missing |
| **Manual** | `cd "$HOME/Seiso" && source .venv/bin/activate && seiso forge` | Add `--open` to launch the browser |
| **Custom port** | `SEISO_PORT=8766 seiso forge` | Or set in `.env` |

Stop Forge with `Ctrl+C` in the terminal where it is running.

### Windows (native PowerShell)

There is no Windows installer script — use manual steps. **WSL2 + NVIDIA** is recommended on Windows for full CUDA/Triton support ([docs/platforms/wsl.md](docs/platforms/wsl.md)).

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
Copy-Item .env.example .env -ErrorAction SilentlyContinue
cd forge-ui; npm ci; npm run build; cd ..
seiso doctor
seiso forge
```

**Start Forge on later sessions:**

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
seiso forge
```

Open **http://127.0.0.1:8765** in your browser. See [docs/platforms/windows.md](docs/platforms/windows.md) for CUDA PyTorch and MSVC build tools.

### Manual install (all platforms)

**Requirements:** Python 3.10+, [Node.js 18+](https://nodejs.org/) (20 LTS recommended), and git.

Replace `REPO` with your clone path: `$HOME/Seiso` (Unix), `%USERPROFILE%\Seiso` (Windows), or any directory you prefer.

<details>
<summary><strong>Linux / macOS / WSL — base install</strong></summary>

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
cp -n .env.example .env
cd forge-ui && npm ci && npm run build && cd ..
seiso doctor
seiso forge
```

**macOS Apple Silicon** — add MLX for fast chat:

```bash
pip install -e ".[forge,train,mlx,dev]"
```

**Linux + NVIDIA** — add Triton fused kernels:

```bash
pip install -e ".[forge,train,cuda,dev]"
```

</details>

<details>
<summary><strong>Windows — base install</strong></summary>

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
cd forge-ui; npm ci; npm run build; cd ..
seiso doctor
seiso forge
```

</details>

Optional compression pipelines (install on top of the base stack, from an activated venv):

```bash
pip install -e ".[compress-quant,compress-eval]"       # LLM compression GPTQ/AWQ (Linux NVIDIA)
```

### Diagnose problems

If install or start fails, **doctor runs automatically**. Run it manually anytime:

```bash
# Linux / macOS / WSL — adjust path if SEISO_INSTALL_DIR differs
"$HOME/Seiso/scripts/doctor.sh"
"$HOME/Seiso/scripts/doctor.sh" --network
```

```powershell
# Windows — from repo root with venv active
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
seiso doctor
```

### Model download sizes

Catalog chat downloads are local GGUF files for llama.cpp and usually need 2–8 GB each; larger models can need 10–30+ GB. Weights land in your Hugging Face cache (`SEISO_DATA_DIR/hf_cache` by default).

Full install guide: **[docs/install.md](docs/install.md)**  
First-run walkthrough: **[docs/getting-started.md](docs/getting-started.md)**

---

## Forge UI walkthrough

After `seiso forge` (or `start`), browse to **http://127.0.0.1:8765**:

| Page | Path | What it does |
|------|------|--------------|
| Dashboard | `/` | Workspace overview |
| Model Hub | `/hub` | Live Hugging Face Hub search for GGUF models |
| Chat | `/chat` | Local inference (GGUF, MLX, PyTorch) |
| Training Studio | `/train` | LoRA / QLoRA fine-tune with live SSE logs |
| Export | `/export` | Merge LoRA, GGUF quant, Hugging Face publish |
| Compress | `/compress` | LLM distill → prune (Llama-family) → finetune → quant |
| Distill-RL | `/distill-rl` | Teacher → student distillation + DPO (auto-sweep) |
| RL Quant | `/rl-quant` | Adaptive GGUF quantization via RL (auto-sweep) |
| Recipe Studio | `/recipes` | Visual `@xyflow/react` graph editor |
| Integrations | `/integrations` | Route to OpenAI, Anthropic, vLLM |
| Knowledge | `/knowledge` | RAG corpus ingest and retrieval |
| Settings | `/settings` | HF token, hardware info, security toggles |

Knowledge-base ingest/retrieve is also available via API (`/api/knowledge/...`).

Forge details: **[docs/forge.md](docs/forge.md)**

---

## CLI

| Command | Purpose |
|---------|---------|
| `seiso forge` | Launch Forge web server |
| `seiso doctor` | Diagnose install / HF / GPU stack |
| `seiso train` | Train from YAML config |
| `seiso chat` | Terminal chat with local models |
| `seiso export` | Export merged / GGUF / LoRA + Hub push |
| `seiso compress run` | LLM compression pipeline |
| `seiso compress manifest-verify` | Verify hash-chained run manifest |
| `seiso compress speculative` | Speculative decoding (draft + target) |
| `seiso distill-rl run` | Teacher distill → preference rollouts → DPO |
| `seiso distill-rl presets` | List distill-RL presets and stages |
| `seiso rl-quant run` | RL quantization (+ optional `--kernel-rl`) |
| `seiso rl-quant profiles` | List CUDA kernel RL launch profiles |
| `seiso inference` | One-shot inference |
| `seiso bench-inference` | Inference load / TTFT / tok/s benchmark |
| `seiso-bench-kernels` | Benchmark fused GPU training kernels |
| `seiso-train-worker` | Multi-GPU worker (via `torchrun`, see docs) |

```bash
# Example: fine-tune Llama 3.2 3B on sample data (CLI → ./outputs/lora-run/)
seiso train --config configs/example_lora.yaml

# Example: export CLI checkpoint to GGUF
seiso export --checkpoint ./outputs/lora-run/checkpoint-<timestamp> --formats merged,gguf

# Example: RL quant with CUDA kernel co-training
seiso rl-quant run --preset minimal --kernel-rl --training-episodes 256

# Example: distill-RL smoke (teacher → DPO)
seiso distill-rl run --preset smoke
```

Full reference: **[docs/cli.md](docs/cli.md)**

---

## Architecture

```
Seiso/
├── seiso/              # Core library — training, inference, export, kernels, compression
├── seiso_cli/          # CLI: seiso, seiso-bench-kernels, seiso-train-worker
├── forge/              # FastAPI backend, auth, orchestrators, SSE job streaming
├── forge-ui/           # React 19 + TypeScript + Vite frontend
├── third_party/        # Vendored compression pipelines (codellama, RL quant)
├── configs/            # Example YAML/JSON configs
├── deploy/             # Caddy, nginx, systemd, HTTPS env templates
├── docs/               # Documentation
├── start               # install or launch Forge (primary entry point)
└── scripts/            # install.sh, doctor.sh, CI runner
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
| `forge/orchestrators/distill_rl` | Teacher distill + DPO preference alignment |
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
| CPU | ✓ GGUF | limited | — | — |

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

- **Backends:** llama.cpp (GGUF), MLX (macOS), PyTorch (4-bit/16-bit) ([backends](docs/inference/backends.md))
- **Tool calling:** web search, sandboxed code execution, artifact writes (opt-in)
- **Providers:** OpenAI, Anthropic, vLLM with SSRF hardening

### Export

- **Formats:** merged safetensors, LoRA adapter, full fine-tune, GGUF (Q4_K_M, Q8_0, etc.)
- **Hub publish:** metadata preflight, model card generation, repo creation
- **Profiles:** inference-optimized export presets

### Compression

Three integrated pipelines ([compression.md](docs/compression.md)):

1. **LLM compression** — distill → MLP prune (Llama-family) → recovery finetune → GPTQ/AWQ → speculative decoding. Any HF causal LM; override teacher/student via CLI or Forge.
2. **Distill-RL** — teacher KL distillation → preference rollouts → DPO fine-tuning with auto hyperparameter sweep (`seiso distill-rl run`)
3. **RL quantization** — train a policy for adaptive GGUF quant levels with auto sweep (`seiso rl-quant run`, optional `--kernel-rl`)

---

## Data & storage

All user data lives under **`SEISO_DATA_DIR`** (default below):

| OS | Default data directory |
|----|------------------------|
| Linux / macOS / WSL | `$HOME/.seiso` |
| Windows | `%USERPROFILE%\.seiso` |

```
{SEISO_DATA_DIR}/
├── hf_cache/         # Hugging Face hub cache (downloaded weights)
├── hf_home/          # HF_HOME mirror (when configured by Seiso)
├── hf_xet_cache/     # hf-xet transfer cache
├── hf_tokens/        # Encrypted Hugging Face tokens (per user)
├── models/           # Per-user inventory links to cached weights
├── checkpoints/      # Training outputs (per user)
├── exports/          # Merged / GGUF / LoRA exports
├── compress/         # LLM compression artifacts
├── rl_quant/         # RL quant outputs
├── distill_rl/       # Distillation / RL artifacts
├── recipes/          # Recipe Studio job data
├── uploads/          # Datasets and user files
├── knowledge/        # RAG stores
├── sandbox/          # Sandboxed code-exec workspace
├── artifacts/        # General job artifacts
├── forge.db          # SQLite database (when persistent)
├── .secret_key       # Session signing key
├── .db_encryption_key # DB encryption key (when persistent)
├── .inference_api_key # Optional inference API key
├── .forge.lock       # Single-instance lock
└── runtime.json      # Runtime metadata
```

**Free memory:** In Chat or Model Hub, use **Free memory** to unload the active llama.cpp / MLX / PyTorch model from RAM/VRAM. This does **not** delete downloaded files in `hf_cache/`. On Mac (Apple Silicon or Intel), free memory before loading a larger model — Seiso sizes loads as file size + ~0.8 GB overhead.

Database defaults to **ephemeral in-memory SQLite** — chat history and job metadata are wiped on restart unless you opt into persistence (`SEISO_DB_EPHEMERAL=false`).

---

## OpenAI-compatible API

Point Cursor, Continue, or any OpenAI client at Forge while it is running:

```text
Base URL: http://127.0.0.1:8765/v1
API key:  Inference key from SEISO_DATA_DIR/.inference_api_key
          (Linux/macOS/WSL: $HOME/.seiso/.inference_api_key
           Windows: %USERPROFILE%\.seiso\.inference_api_key)
```

```bash
# Linux / macOS / WSL
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $(cat "$HOME/.seiso/.inference_api_key")" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Explain QLoRA in one paragraph."}],
    "stream": true
  }'
```

```powershell
# Windows PowerShell
$key = Get-Content "$env:USERPROFILE\.seiso\.inference_api_key" -Raw
curl.exe http://127.0.0.1:8765/v1/chat/completions `
  -H "Authorization: Bearer $($key.Trim())" `
  -H "Content-Type: application/json" `
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Explain QLoRA in one paragraph."}],"stream":true}'
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
| `SEISO_ALLOW_REMOTE=true` | — | Allows LAN/WAN binding; requires `SEISO_REMOTE_ACK=1` |
| `SEISO_REMOTE_ACK=1` | — | Required acknowledgement to bind beyond localhost |
| `SEISO_REMOTE_DANGEROUS_ACK=1` | — | Required for remote + tools/code-exec/openai-tools |
| `SEISO_TRUST_PROXY=true` | — | Honor `X-Forwarded-*` only from `SEISO_TRUSTED_PROXY_IPS` |
| `SEISO_TRUSTED_PROXY_IPS` | — | Comma-separated proxy IPs/CIDRs (e.g. `127.0.0.1,::1`) |
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
