# Seiso documentation

Complete guide to installing, running, and extending **Seiso Local AI** — a free local workspace for fine-tuning, quantization, distillation, compression, and reinforcement learning workflows.

**Repository:** [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)

---

## New here?

Start with **[getting-started.md](getting-started.md)** — a step-by-step walkthrough from install → onboarding → download model → chat → train → export.

---

## I want to…

| Goal | Start here |
|------|------------|
| Install on Linux or macOS in one command | [install.md](install.md#linux--macos--one-command-recommended) |
| Install on Windows or AMD ROCm | [platforms/windows.md](platforms/windows.md) · [platforms/linux-amd-rocm.md](platforms/linux-amd-rocm.md) |
| Launch the web UI | `start` or `seiso forge` → [forge.md](forge.md) |
| Diagnose install / HF / GPU | `seiso doctor` or `./scripts/doctor.sh` → [cli.md](cli.md) |
| Chat with a local model | [getting-started.md § Step 4](getting-started.md#step-4--chat-with-a-local-model) |
| Fine-tune with QLoRA / LoRA | [training/quickstart.md](training/quickstart.md) |
| Run each training pipeline step by step | [training/pipelines.md](training/pipelines.md) |
| Single-GPU slime post-training | [training/quickstart.md § Slime Post-Training](training/quickstart.md#slime-post-training) |
| Multi-reward coding RL (contest + packages + functions) | [training/multi_reward_coding.md](training/multi_reward_coding.md) |
| RL quant from CLI | [cli.md § seiso rl-quant](cli.md#seiso-rl-quant) · [compression.md](compression.md) |
| Use Cursor / Continue with local models | [getting-started.md § Connect external tools](getting-started.md#connect-external-tools-cursor-continue-etc) |
| Export to GGUF or Hugging Face Hub | [getting-started.md § Step 6](getting-started.md#step-6--export-and-deploy) · [cli.md](cli.md) |
| Compress / distill / quantize models | [compression.md](compression.md) |
| Teacher distill + DPO alignment | [cli.md § seiso distill-rl](cli.md#seiso-distill-rl) · [compression.md](compression.md) |
| Quant regression study (train → export → eval) | [cli.md § seiso experiment](cli.md#seiso-experiment) |
| Build a local RAG corpus | [forge.md](forge.md) · `/knowledge` |
| Enable fused GPU kernels | [training/kernels.md](training/kernels.md) |
| Train on multiple GPUs | [training/multi-gpu.md](training/multi-gpu.md) |
| Deploy with HTTPS | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |
| Fix a problem | [troubleshooting.md](troubleshooting.md) |
| Run tests before a PR | [CI_LOCAL.md](CI_LOCAL.md) |

---

## Startup paths

**Forge URL (all platforms):** http://127.0.0.1:8765

### Paths

| Purpose | Linux / macOS / WSL | Windows | Override |
|---------|---------------------|---------|----------|
| Repository | `$HOME/Seiso` | `%USERPROFILE%\Seiso` | `SEISO_INSTALL_DIR` |
| User data | `$HOME/.seiso` | `%USERPROFILE%\.seiso` | `SEISO_DATA_DIR` |

### How to start

| Mode | Linux / macOS / WSL | Windows |
|------|---------------------|---------|
| **Install + start** | `curl -fsSL …/start \| bash` — starts Forge when done | Manual install → `seiso forge` ([install.md](install.md)) |
| **Later sessions** | `start` or `seiso forge` from repo | `cd "$env:USERPROFILE\Seiso"` → activate venv → `seiso forge` |
| **From a clone** | `start` (starts by default) or `SEISO_START=0 start` | Build UI + `seiso forge` ([platforms/windows.md](platforms/windows.md)) |
| **Forge (UI dev)** | Terminal 1: `seiso forge` · Terminal 2: `cd forge-ui && npm run dev` | Same |
| **CLI training** | `seiso train --config configs/example_lora.yaml` | `seiso train --config configs\example_lora.yaml` |
| **OpenAI-compatible API** | `seiso forge` then POST `/v1/chat/completions` | Same |

First launch opens onboarding — create your local admin password. Copy `.env.example` to `.env` to override host, port, or data directory.

On native Linux + NVIDIA, `start` also prepares the isolated GGUF chat sidecar:
healthy Ollama first, otherwise llama-swap's `llamacpp` engine. See
[inference/backends.md](inference/backends.md#llama-swap-setup) for overrides
and the explicit unsafe in-process llama.cpp opt-in.

See [install.md](install.md) and [forge.md](forge.md) for full details.

---

## Learning paths

### Path A — End user (Forge UI)

1. [getting-started.md](getting-started.md) — install through first export
2. [forge.md](forge.md) — pages, API routes, environment variables
3. [training/quickstart.md](training/quickstart.md) — Training Studio details
4. [training/pipelines.md](training/pipelines.md) — step-by-step pipeline runs
5. [compression.md](compression.md) — compression pipelines
6. [troubleshooting.md](troubleshooting.md) — common fixes

### Path B — CLI / automation

1. [install.md](install.md) — pip extras per platform
2. [cli.md](cli.md) — full command reference (`seiso experiment` for research studies)
3. [training/quickstart.md](training/quickstart.md) — YAML config fields
4. [training/pipelines.md](training/pipelines.md) — CLI pipeline runbooks
5. [inference/backends.md](inference/backends.md) — backend selection and [memory management](inference/backends.md#memory-management)

### Path C — Developer / contributor

1. [install.md](install.md) — dev extras (`.[dev]`)
2. [CI_LOCAL.md](CI_LOCAL.md) — quality gate (`make ci-fast`)
3. [forge.md](forge.md) — UI dev with Vite hot reload
4. [training/kernels.md](training/kernels.md) — CUDA/Triton kernel stack

### Path D — Production deployment

1. [deployment/reverse-proxy.md](deployment/reverse-proxy.md) — Caddy/nginx TLS
2. [deploy/README.md](../deploy/README.md) — systemd, env templates
3. [README.md § Security](../README.md#security) — hardening checklist

---

## Platform guides

| Platform | Chat | Train | Kernels | Guide |
|----------|------|-------|---------|-------|
| Linux + NVIDIA | ✓ | QLoRA 4-bit | CUDA | [linux-nvidia.md](platforms/linux-nvidia.md) |
| Linux + AMD ROCm | ✓ | 4-bit* | Triton | [linux-amd-rocm.md](platforms/linux-amd-rocm.md) |
| macOS | ✓ MLX | 16-bit LoRA | — | [macos.md](platforms/macos.md) |
| Windows | ✓ | QLoRA | CUDA JIT | [windows.md](platforms/windows.md) |
| WSL2 + NVIDIA | ✓ | QLoRA | CUDA + Triton | [wsl.md](platforms/wsl.md) |

---

## Task reference

| Task | Guide |
|------|-------|
| Install & dependencies | [install.md](install.md) |
| First-run walkthrough | [getting-started.md](getting-started.md) |
| Forge UI & API | [forge.md](forge.md) |
| CLI commands | [cli.md](cli.md) |
| Local CI / quality gate | [CI_LOCAL.md](CI_LOCAL.md) |
| Training | [training/quickstart.md](training/quickstart.md) |
| Training pipelines | [training/pipelines.md](training/pipelines.md) |
| Fused GPU kernels | [training/kernels.md](training/kernels.md) |
| Multi-GPU | [training/multi-gpu.md](training/multi-gpu.md) |
| Inference backends | [inference/backends.md](inference/backends.md) |
| Model compression | [compression.md](compression.md) |
| HTTPS / reverse proxy | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |
| Deployment configs | [deploy/README.md](../deploy/README.md) |
| Troubleshooting | [troubleshooting.md](troubleshooting.md) |

---

## Quick commands

```bash
# One-shot install + start (Linux / macOS / WSL — Forge starts automatically)
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash

# Start Forge on later sessions (Linux / macOS / WSL)
cd "$HOME/Seiso" && start

# From a clone (Linux / macOS / WSL)
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso" && cd "$HOME/Seiso"
start

# Linux NVIDIA (manual pip; use start for sidecar autostart)
pip install -e ".[forge,train,cuda,llamacpp,dev]"

# Build UI (first launch or after UI changes)
cd forge-ui && npm ci && npm run build && cd ..

# Start Forge manually
seiso forge    # → http://127.0.0.1:8765

# Train (CLI → ./outputs/lora-run/ per example_lora.yaml)
seiso train --config configs/example_lora.yaml

# Slime post-training (CLI → ./outputs/slime-train-method/)
seiso train --config configs/example_training_slime.yaml

# RL quant (CLI → $SEISO_DATA_DIR/rl_quant/cli/<job_id>/)
seiso rl-quant run --preset minimal --kernel-rl

# Distill-RL (CLI → $SEISO_DATA_DIR/distill_rl/cli/<job_id>/; smoke preset uses gpt2)
seiso distill-rl run --preset smoke

# Quant regression study (CLI → study output_dir in YAML)
seiso experiment quant-regression -c configs/examples/quant_regression_study.yaml

# LLM compression (CLI → $SEISO_DATA_DIR/compress/local/cli/runs/<run_id>/)
seiso compress run --preset smoke

# Diagnose install
seiso doctor
```

```powershell
# Windows — install + start (manual)
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[forge,train,dev]"
cd forge-ui; npm ci; npm run build; cd ..
seiso forge
```

---

## Repository layout

```
Seiso/
├── seiso/              # Core library (GPL-3.0)
│   ├── training/       # TRL trainer, dataset analysis, practices, platform caps
│   ├── kernels/        # Fused CUDA/Triton ops
│   ├── export/         # Merge, GGUF, Hub publish
│   ├── compress/       # LLM compression bootstrap
│   ├── distill_rl/     # Teacher distill + DPO pipeline
│   ├── rl_quant/       # Adaptive RL quant + kernel policy bridge
│   ├── experiments/    # Research studies (quant regression, HF deploy eval)
│   └── security/       # NVIDIA boundary gates
├── seiso_cli/          # CLI entrypoints
├── forge/              # FastAPI backend
│   ├── api/routes/     # REST + OpenAI-compatible endpoints
│   ├── orchestrators/  # Job workers (train, export, compress, …)
│   └── security/       # Auth, CSRF, path sandbox
├── forge-ui/           # React + TypeScript + Vite (GPL-3.0)
├── seiso/codellama_compress/ # Bundled LLM compression implementation
├── seiso/adaptive_quant/     # Bundled adaptive RL quant implementation
├── seiso/analysis/           # RL quant analysis CLI/helpers
├── configs/            # Example YAML/JSON configs
├── data/               # Sample training JSONL
├── deploy/             # Caddy, nginx, systemd templates
├── docs/               # This documentation
├── start               # install or launch Forge (primary entry point)
├── scripts/            # install.sh, doctor.sh, CI runner
├── locks/              # python.lock, dependency digests
└── tests/              # pytest suite
```

---

## FAQ

**Is Seiso free?**  
Yes. GPL-3.0 licensed. You run it on your own hardware with no subscription.

**Does it work offline?**  
After models are downloaded, chat and training work without internet. Hub publish and provider routing need network access.

**Where is my data stored?**  
`$HOME/.seiso` on Linux/macOS/WSL, `%USERPROFILE%\.seiso` on Windows. Override with `SEISO_DATA_DIR`. See [getting-started.md § Data directory](getting-started.md#data-directory-layout).

**Why is the UI blank?**  
The Forge UI must be built: `cd forge-ui && npm run build`, or run `start` (auto-builds).

**Can I use QLoRA on macOS?**  
No — macOS uses 16-bit LoRA via PyTorch MPS/CPU. QLoRA 4-bit requires NVIDIA CUDA or Linux with bitsandbytes.

**What's the difference between Forge and the CLI?**  
Same core library. Forge adds a web UI, job orchestration, SSE logs, and multi-user auth. The CLI is for scripting and headless workflows.

**How do I connect Cursor or VS Code extensions?**  
Point them at `http://127.0.0.1:8765/v1` while Forge is running. See [getting-started.md](getting-started.md).

---

## Glossary

| Term | Meaning |
|------|---------|
| **Forge** | Seiso's web UI + FastAPI backend |
| **QLoRA** | 4-bit quantized LoRA fine-tuning |
| **GGUF** | llama.cpp model format for efficient CPU/GPU inference |
| **MLX** | Apple's ML framework for fast inference on Apple Silicon |
| **SSE** | Server-Sent Events — live log streaming in the UI |
| **Hub** | Hugging Face Hub — model hosting and download |
| **Fused kernels** | Custom CUDA/Triton ops for faster training (RMSNorm, SwiGLU, CE) |
