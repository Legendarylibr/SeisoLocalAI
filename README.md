# Seiso Local AI

Going to work on a self hosted mirror then merge large changes (maybe waiting till it's done) to reduce ci/github actions

[![CI](https://github.com/Legendarylibr/SeisoLocalAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Legendarylibr/SeisoLocalAI/actions/workflows/ci.yml)

**Seiso** is a **local-first AI platform** that runs entirely on your machine: chat with open models, fine-tune them (LoRA / QLoRA), post-train, quantize, compress, and export or publish to the Hugging Face Hub — through one Forge web UI and a matching CLI. Weights, prompts, and datasets stay on your hardware unless you choose to share them; no cloud account is required for day-to-day work.

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
- [Compat API](#compat-api)
- [Development](#development)
- [Security](#security)
- [Opt-in marketplace & Buzz mesh](#opt-in-marketplace--buzz-mesh)
- [Reporting vulnerabilities](SECURITY.md)
- [Documentation index](#documentation-index)
- [Inference stack](#inference-stack)

---

## What Seiso does

Seiso combines a **web workspace (Forge)** and a **Python core (CLI + library)** so you can work entirely on your machine:

| Workflow | Forge UI | CLI |
|----------|----------|-----|
| Download & chat with open models | Model Hub, Chat | `seiso chat` / `seiso tui` |
| QLoRA / LoRA / full fine-tune | Training Studio | `seiso train` |
| Single-GPU slime post-training | CLI | `seiso train --config configs/example_training_slime.yaml` |
| Multi-GPU slime (vLLM rollouts) | CLI | `scripts/run_slime_vllm_ddp.sh 2 configs/example_training_slime_vllm.yaml` |
| External NVIDIA NeMo RL | CLI | `seiso nemo-rl --config configs/example_training_nemo_rl.yaml` |
| Merge, GGUF, Hub publish | Export | `seiso export` |
| LLM distill → prune → quant | Compress | `seiso compress run` |
| Teacher distill + DPO alignment | Distill-RL | `seiso distill-rl run` |
| Visual data/recipe pipelines | Recipe Studio | — |
| RAG knowledge bases | Knowledge | — |
| Opt-in remote sats marketplace (Ark + L402) — **not functional, do not use yet** | — | `seiso pay` ([docs](docs/pay/marketplace.md)) |
| Experimental Buzz shared / multi-node train — **not functional, do not use yet** | — | `seiso mesh` ([docs](docs/training/mesh.md)) |

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
On first launch, **create a local account** in the browser (save the one-time recovery key → continue; public ID is safe to share), or restore from a saved recovery key.

### Paths on your system

Seiso uses two directories. Override either with environment variables when needed.

| Purpose | Linux / macOS / WSL | Windows (native) | Override |
|---------|---------------------|------------------|----------|
| **Repository** (clone, venv, scripts) | `$HOME/Seiso` | `%USERPROFILE%\Seiso` | `SEISO_INSTALL_DIR` |
| **User data** (models, checkpoints, exports) | `$HOME/.seiso` | `%USERPROFILE%\.seiso` | `SEISO_DATA_DIR` |

In config files and Python, `~/.seiso` expands correctly on every OS. In shell commands, use the column for your platform — do not paste Unix `~/` paths into PowerShell.

### Linux, macOS, and WSL2 (recommended)

One command installs missing system tools (Python, Node, git, build deps when possible), clones the repo, builds the UI, and **starts Forge automatically** (browser opens when ready):

#### Quick install

**Main one-liner** — auto-detects Linux, macOS, and WSL2, installs dependencies, builds Forge, and starts the app:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Quick installs** — use these when you already know the target platform:

Linux native + NVIDIA:

```bash
SEISO_INSTALL_PROFILE=linux-nvidia curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Linux native CPU:

```bash
SEISO_INSTALL_PROFILE=linux-cpu curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

WSL2 + NVIDIA:

```bash
SEISO_INSTALL_PROFILE=wsl-nvidia curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

macOS Apple Silicon:

```bash
SEISO_INSTALL_PROFILE=macos curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

What the installer does:

1. Clones to `$HOME/Seiso` (or `SEISO_INSTALL_DIR`)
2. Creates `.venv` and installs platform extras (CUDA on Linux + NVIDIA, MLX on macOS, GGUF support)
3. Copies `.env.example` → `.env` if missing
4. Builds `forge-ui/dist`
5. Runs `seiso tui` (terminal UI; no browser)

On native Linux + NVIDIA, the `linux-nvidia` profile installs **Ollama**
(Ollama-first isolated GGUF chat), seeds sidecar `.env` defaults, and verifies
the stack before Forge opens. `start` also auto-starts Ollama when installed;
llama-swap is an optional fallback when Ollama is down.

See [docs/install.md](docs/install.md) for custom paths, install-only mode, Windows PowerShell, and advanced installer options.

**Already cloned?** From the repository root:

```bash
start                    # install / upgrade deps + build UI; starts Forge by default
SEISO_START=0 start      # install only
```

Install registers `start` on your PATH (`~/.local/bin`). Open a new terminal if the command is not found yet.

### Manual Linux setup

Use this when you already manage Python/Node yourself, want a custom clone path, or prefer not to run the one-liner installer.

**Prerequisites**

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10+ (3.11+ recommended) | `python3 --version` |
| Bun or Node.js | Bun auto-installed; or Node 18+ | For `forge-ui` build (Bun is default) |
| git | any recent | — |
| NVIDIA driver | optional | `nvidia-smi` must work for CUDA training |

**1. Clone and create a virtualenv**

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
cp -n .env.example .env
```

**2. Install Python extras (pick your hardware)**

Linux with **NVIDIA GPU** (matches what `start` installs):

```bash
pip install -e ".[forge,train,cuda,llamacpp,dev]"
```

Linux **without NVIDIA** (CPU or non-CUDA inference):

```bash
pip install -e ".[forge,train,llamacpp,dev]"
```

Optional add-ons (from an activated venv):

```bash
pip install -e ".[compress-quant,compress-eval]"      # LLM compression pipelines (NVIDIA)
./scripts/install_flash_attn.sh                         # Flash Attention 2 (NVIDIA, optional)
```

**3. Build the Forge UI and verify**

```bash
cd forge-ui && bun install --frozen-lockfile && bun run build && cd ..
seiso doctor
seiso doctor --network    # optional: Hugging Face + download readiness
```

**4. Start Forge**

```bash
seiso forge               # http://127.0.0.1:8765
seiso forge --open        # same, and open the browser
```

On first launch, **create a local account** (save the one-time recovery key → continue; public ID is safe to share) or restore a recovery key in the browser. User data (models, checkpoints, exports) lives in `~/.seiso` unless you set `SEISO_DATA_DIR`.

**Later sessions:** `cd "$HOME/Seiso" && source .venv/bin/activate && seiso forge`, or use `start` from the repo root after `scripts/install.sh` has registered it.

More detail: [docs/platforms/linux-nvidia.md](docs/platforms/linux-nvidia.md) (CUDA kernels, multi-GPU), [docs/install.md](docs/install.md) (full install reference).

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

There is no `curl | bash` installer on Windows — use this one-liner (clone, venv, Forge UI, start) or follow the manual steps below. **WSL2 + NVIDIA** is recommended on Windows for full CUDA/Triton support ([docs/platforms/wsl.md](docs/platforms/wsl.md)).

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"; cd "$env:USERPROFILE\Seiso"; python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -U pip wheel setuptools; pip install -e ".[forge,train,dev]"; if (-not (Test-Path .env)) { Copy-Item .env.example .env }; cd forge-ui; npm ci; npm run build; cd ..; seiso doctor; seiso forge
```

Manual steps (same result, easier to read):

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
| `seiso slime` | Single-process slime GRPO post-train (also `seiso train -c … method: slime`) |
| `seiso nemo-rl` | Launch external [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) (`method: nemo_rl`; requires `SEISO_NEMO_RL_ROOT`) |
| `seiso chat` | Terminal chat with local models |
| `seiso tui` | Lightweight offline terminal UI that mimics Forge (no browser) |
| `seiso export` | Export merged / GGUF / LoRA + Hub push |
| `seiso compress run` | LLM compression pipeline |
| `seiso compress manifest-verify` | Verify hash-chained run manifest |
| `seiso compress speculative` | Speculative decoding (draft + target) |
| `seiso distill-rl run` | Teacher distill → preference rollouts → DPO |
| `seiso distill-rl presets` | List distill-RL presets and stages |
| `seiso inference` | One-shot inference |
| `seiso bench-inference` | Inference load / TTFT / tok/s benchmark |
| `seiso-bench-kernels` | Benchmark fused GPU training kernels |
| `seiso-train-worker` | Multi-GPU worker (via `torchrun`, see docs) |
| `seiso pay` | Opt-in sats marketplace client / operator sidecar (`SEISO_ALLOW_PAY=1`) |
| `seiso mesh` | Experimental Buzz-coordinated multi-node mesh (`SEISO_ALLOW_MESH=1`) |
| `seiso provenance` | Nostr digest attestation / membership proofs |

```bash
# Example: fine-tune Llama 3.2 3B on sample data (CLI → ./outputs/lora-run/)
seiso train --config configs/example_lora.yaml
# Example: export CLI checkpoint to GGUF
seiso export --checkpoint ./outputs/lora-run/checkpoint-<timestamp> --formats merged,gguf

# Example: distill-RL smoke (teacher → DPO)
seiso distill-rl run --preset smoke

# Example: quant regression study (train → export → eval)
```

Full reference: **[docs/cli.md](docs/cli.md)**

---

## Architecture

```
Seiso/
├── seiso/              # Core library — training, inference, export, kernels, compression, experiments
├── seiso_cli/          # CLI: seiso, seiso-bench-kernels, seiso-train-worker
├── forge/              # FastAPI backend, auth, orchestrators, SSE job streaming
├── forge-ui/           # React 19 + TypeScript + Vite frontend
├── seiso/codellama_compress/ # Bundled LLM compression implementation
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
| `` | Adaptive RL GGUF quant policy |
| `forge/orchestrators/hub_publish` | Hugging Face Hub publish jobs (via Export) |

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

## Features

> Adaptive RL quantization research: [Adaptive-RL-Quantization](https://github.com/Legendarylibr/Adaptive-RL-Quantization).
 in depth

### Training

- **Methods:** LoRA, QLoRA (4-bit), full fine-tune, embedding training, slime GRPO, external NVIDIA NeMo RL
- **Formats:** JSONL chat datasets with auto format detection
- **Optimizations:** gradient checkpointing, packing, RSLoRA, train-on-responses-only
- **Multi-GPU:** `torchrun` distributed workers; rank-0 checkpoint writes ([multi-gpu](docs/training/multi-gpu.md))
- **Buzz mesh (experimental) — not functional, do not use yet:** opt-in peer coordination for shared / multi-node jobs ([mesh](docs/training/mesh.md))
- **Opt-in sats marketplace — not functional, do not use yet:** remote finetune/RL/inference with Ark + L402 settlement + protocol fee ([marketplace](docs/pay/marketplace.md))
- **Fused kernels:** RMSNorm, SwiGLU MLP, cross-entropy, fused LoRA delta ([kernels](docs/training/kernels.md))
- **Release-style post-training:** `method: slime` adds rollout rewards, verifier data, best/final checkpoints, and plateau auto-stop; multi-GPU rollouts can use **vLLM** (`rollout_backend: vllm`) or SGLang ([training](docs/training/quickstart.md#slime-post-training))
- **External NeMo RL:** `method: nemo_rl` shells out to a local [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) checkout via `uv run` (not vendored); see [NeMo RL](docs/training/quickstart.md#nemo-rl)

### Inference

- **Backends:** llama.cpp (GGUF), MLX (macOS), PyTorch (4-bit/16-bit) ([backends](docs/inference/backends.md))
- **Smart Router:** optional connection to an external router service such as [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter)
- **Tool calling:** web search, sandboxed code execution, artifact writes (opt-in)
- **Providers:** OpenAI, Anthropic, vLLM with SSRF hardening

#### External Smart Router (Just to keep open source after removing it, no longer intend on serving inference). Feel free to use other external routers

Run [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter) or a compatible local router service separately, then enable it in Forge (`.env` or environment) so Chat shows **Smart Router (auto-route)**:

```bash
SEISO_MODEL_ROUTER_ENABLED=true
SEISO_MODEL_ROUTER_URL=http://127.0.0.1:8780
```

Router endpoint expected by Forge: `http://127.0.0.1:8780/v1/chat/completions`.

### Export

- **Formats:** merged safetensors, LoRA adapter, full fine-tune, GGUF (Q4_K_M, Q8_0, etc.)
- **Hub publish:** metadata preflight, model card generation, repo creation
- **Profiles:** inference-optimized export presets

### Compression

Three integrated pipelines ([compression.md](docs/compression.md)):

1. **LLM compression** — distill → MLP prune (Llama-family) → recovery finetune → GPTQ/AWQ → speculative decoding. Any HF causal LM; override teacher/student via CLI or Forge.
2. **Distill-RL** — teacher KL distillation → preference rollouts → DPO fine-tuning with auto hyperparameter sweep (`seiso distill-rl run`)

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
├── hf_home/          # HF_HOME mirror (created on first Hub configure)
├── hf_xet_cache/     # hf-xet transfer cache (created on first Hub configure)
├── hf_tokens/        # Encrypted Hugging Face tokens (per user)
├── nostr_keys/       # Encrypted Nostr keys (auth + provenance attest)
├── models/           # Per-user inventory links to cached weights
├── checkpoints/      # Training outputs (per user)
├── exports/          # Merged / GGUF / LoRA exports
├── compress/         # LLM compression artifacts
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

## Compat API

Point Cursor, Continue, or any chat-completions client at Forge while it is running:

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

Set `SEISO_ALLOW_COMPAT_TOOLS=true` for tool calling on this endpoint (legacy alias: `SEISO_ALLOW_OPENAI_TOOLS`).

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

Found a vulnerability? Report it privately via [GitHub private vulnerability reporting](https://github.com/Legendarylibr/SeisoLocalAI/security/advisories/new) — see [SECURITY.md](SECURITY.md). Please don't open public issues for security problems.

### Network binding

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEISO_ALLOW_REMOTE=false` | off | Binds Forge to `127.0.0.1` only |
| `SEISO_ALLOW_REMOTE=true` | — | Allows LAN/WAN binding; requires `SEISO_REMOTE_ACK=1` |
| `SEISO_REMOTE_ACK=1` | — | Required acknowledgement to bind beyond localhost |
| `SEISO_REMOTE_DANGEROUS_ACK=1` | — | Required for remote + tools/compat-tools |
| `SEISO_ALLOW_CODE_EXEC` + remote | blocked | Remote + code-exec is refused (AST sandbox is not OS isolation) |
| `SEISO_TRUST_PROXY=true` | — | Honor `X-Forwarded-*` only from `SEISO_TRUSTED_PROXY_IPS` |
| `SEISO_TRUSTED_PROXY_IPS` | — | Comma-separated proxy IPs/CIDRs (e.g. `127.0.0.1,::1`) |
| `SEISO_SECURE_COOKIES=true` | — | Secure cookies when TLS is terminated upstream |

Deploy configs: [`deploy/`](deploy/) · Guide: [docs/deployment/reverse-proxy.md](docs/deployment/reverse-proxy.md)

### Opt-in capabilities (all default **off**)

| Variable | Enables |
|----------|---------|
| `SEISO_ALLOW_TOOLS=true` | Web search, artifact writes |
| `SEISO_ALLOW_CODE_EXEC=true` | Sandboxed `execute_code` tool |
| `SEISO_ALLOW_COMPAT_TOOLS=true` | Tool calling on Compat API `/v1` (session JWT only; inference API key stays chat-only) |
| `SEISO_ALLOW_PAY=1` | Opt-in sats marketplace sidecar (remote buyers; self-hosted stays free) |
| `SEISO_ALLOW_MESH=1` | Experimental Buzz mesh coordination (trusted peers; no protocol fee) |

### Path sandbox & tenant isolation

All filesystem access is scoped under `SEISO_DATA_DIR`. Per-user directories (`models/`, `checkpoints/`, `exports/`, etc.) are namespaced by user ID. Cross-user access is rejected with 403.

### Provider SSRF protection

Outbound provider calls block private/metadata hosts, require HTTPS for remote endpoints, and use **DNS pinning** to prevent rebinding attacks.

### Auth & database

- Signed session tokens; login throttling (10 attempts/min per IP)
- Rate limiting on all bindings (default 120 req/min per IP when remote; ≥240 on localhost)
- AES-256-GCM encryption for sensitive DB columns
- Ephemeral SQLite by default — zero retention on restart

### Production checklist

```bash
export SEISO_ALLOW_REMOTE=false
export SEISO_ALLOW_TOOLS=false
export SEISO_ALLOW_CODE_EXEC=false
export SEISO_ALLOW_COMPAT_TOOLS=false
export SEISO_SECRET_KEY="$(openssl rand -hex 32)"
# If you run a public pay sidecar: leave SEISO_PAY_FAUCET unset/off;
# require SEISO_PROTOCOL_TREASURY_ARK + TLS in front of seiso pay serve.
# Leave SEISO_ALLOW_PAY / SEISO_ALLOW_MESH unset unless you intentionally opt in.
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
| Quant regression study | [docs/cli.md § seiso experiment](docs/cli.md#seiso-experiment) |
| Training | [docs/training/quickstart.md](docs/training/quickstart.md) |
| NeMo RL (external) | [docs/training/quickstart.md § NeMo RL](docs/training/quickstart.md#nemo-rl) |
| GPU kernels | [docs/training/kernels.md](docs/training/kernels.md) |
| Multi-GPU | [docs/training/multi-gpu.md](docs/training/multi-gpu.md) |
| Opt-in sats marketplace (Ark + L402) | [docs/pay/marketplace.md](docs/pay/marketplace.md) |
| Buzz mesh shared training | [docs/training/mesh.md](docs/training/mesh.md) |
| Buzz agent orchestration skill | [`.agents/skills/seiso-orchestrate/`](.agents/skills/seiso-orchestrate/SKILL.md) |
| Inference backends | [docs/inference/backends.md](docs/inference/backends.md) |
| Compression | [docs/compression.md](docs/compression.md) |
| HTTPS deployment | [docs/deployment/reverse-proxy.md](docs/deployment/reverse-proxy.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Local CI | [docs/CI_LOCAL.md](docs/CI_LOCAL.md) |
| Security policy / reporting | [SECURITY.md](SECURITY.md) |
| External Smart Router | [Legendarylibr/SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter) |

---

## Opt-in marketplace & Buzz mesh

Self-hosted Forge/CLI remain **free** and unchanged unless you opt in. Two optional surfaces:

> **Pay not functional yet — do not use for real funds.** Opt-in **Ark / L402 marketplace** (`seiso pay`) is scaffolding / faucet-sim only. **Buzz mesh** (`seiso mesh`) is an **opt-in secondary** multi-node path (Buzz-agent-only; local Forge/CLI stays primary) — coordination, plan import, rank claim, config materialize, and optional `--launch` are wired; real multi-host still needs reachable peers + GPUs. Live Ark pay-in / Bark–Second settlement and live L402 (Lightning HTTP 402) are **not wired**. Use `SEISO_PAY_FAUCET=1` for local pay experiments only — never with real money or a public market.

| Mode | Flag | Settlement | Protocol fee | Docs |
|------|------|------------|--------------|------|
| **Self-hosted** (default) | — | None | None | this README |
| **Sats marketplace** | `SEISO_ALLOW_PAY=1` | Opt-in **Ark** + **L402** (**not functional — do not use yet**; faucet/sim only) | Default **5%** on top of compute | [pay/marketplace.md](docs/pay/marketplace.md) |
| **Buzz mesh** (experimental secondary) | `SEISO_ALLOW_MESH=1` | Reciprocal peers; `SEISO_MESH_TOKEN` out-of-band | **None** | [training/mesh.md](docs/training/mesh.md) |

```bash
# Marketplace operator (Forge stays on localhost; expose pay sidecar + TLS only)
export SEISO_ALLOW_PAY=1
export SEISO_PROTOCOL_TREASURY_ARK=ark1…   # required for real settles (fail-closed)
export SEISO_OPERATOR_ARK=ark1…
# export SEISO_PAY_FAUCET=1               # dev only — never on a public market
seiso pay serve --host 127.0.0.1 --port 8787

# Buzz shared / multi-node coordination (trusted peers)
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN='…'               # never post to Buzz
seiso mesh announce --channel "$CHANNEL" --gpus 2
seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2
```

Buzz agents should follow [`.agents/skills/seiso-orchestrate/`](.agents/skills/seiso-orchestrate/SKILL.md): prefer local free compute → mesh peers → paid marketplace → ask a human. Post receipts (job ids, fee split, mesh plan ids) to the channel; never post `SEISO_PAY_TOKEN`, `SEISO_MESH_TOKEN`, or `nsec`.

`SEISO_ARK_BACKEND=bark|second` is **not functional currently** (reserved for a future Bark/Second client wire). L402 is advertised in discovery (`SEISO_PAY_L402`, default on) but **not functional currently** — see [L402 payments explained](https://lightningfaucet.com/learn/l402-payments-explained/). Until either rail is wired, use faucet/simulated settlement or leave backends unset.

---
## RL Stack

Seiso exposes two post-training RL paths:

| Path | Upstream | Role in Seiso |
|------|----------|----------------|
| **slime** (`method: slime`) | [THUDM/slime](https://github.com/THUDM/slime) | Seiso-native GRPO: colocated HF generate on single-GPU; multi-GPU can drive rollouts through **vLLM** (`rollout_backend: vllm`, including Seiso-managed multi-GPU) or SGLang |
| **NeMo RL** (`method: nemo_rl`) | [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | External launcher only — Seiso does **not** vendor NeMo RL. Point `SEISO_NEMO_RL_ROOT` at a recursive clone; Seiso projects YAML knobs into Hydra overrides and runs `uv run python examples/run_*.py` inside that checkout |

Prefer **slime** for local `rl_verify` loops and Forge SSE metrics. Prefer **NeMo RL** for Ray-scale multi-node jobs, Megatron/DTensor backends, DAPO/GDPO recipes, or NeMo-Gym environments. Details: [docs/training/quickstart.md § NeMo RL](docs/training/quickstart.md#nemo-rl).

If you use NeMo RL in research, cite NVIDIA’s BibTeX:

```bibtex
@misc{nemo-rl,
title = {NeMo RL: A Scalable and Efficient Post-Training Library},
howpublished = {\url{https://github.com/NVIDIA-NeMo/RL}},
year = {2025},
note = {GitHub repository},
}
```

## Inference Stack

Seiso’s local chat builds on these inference projects:

| Project | Role in Seiso |
|---------|----------------|
| [**llama.cpp**](https://github.com/ggml-org/llama.cpp) | Default GGUF chat backend (`llama-cpp-python`) |
| [**vLLM**](https://github.com/vllm-project/vllm) | Optional provider endpoint for OpenAI-compatible local serving |

@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}

## Distributed Training

Distributed training integrates https://github.com/huggingface/accelerate to extend training configurations to multi-gpu distributed training. See [docs/training/multi-gpu.md](docs/training/multi-gpu.md).

For **trusted peers** coordinating multi-node jobs over Buzz (opt-in, no marketplace fee), see [docs/training/mesh.md](docs/training/mesh.md) (`SEISO_ALLOW_MESH=1`). Remote paid capacity is separate: [docs/pay/marketplace.md](docs/pay/marketplace.md).

For **Buzz-coordinated multi-node** (trusted peers, no marketplace fee), use experimental [`seiso mesh`](docs/training/mesh.md) with `SEISO_ALLOW_MESH=1`.

Smart Router backend orchestration now lives in [SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter).
