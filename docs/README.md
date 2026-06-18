# Seiso documentation

Complete guide to installing, running, and extending **Seiso Local AI** — a free local workspace for fine-tuning, quantization, distillation, compression, reinforcement learning, and image model workflows.

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
| Launch the web UI | `./scripts/start.sh` or `seiso forge` → [forge.md](forge.md) |
| Diagnose install / HF / GPU | `seiso doctor` or `./scripts/doctor.sh` → [cli.md](cli.md) |
| Chat with a local model | [getting-started.md § Step 4](getting-started.md#step-4--chat-with-a-local-model) |
| Fine-tune with QLoRA / LoRA | [training/quickstart.md](training/quickstart.md) |
| RL quant from CLI | [cli.md § seiso rl-quant](cli.md#seiso-rl-quant) · [compression.md](compression.md) |
| Use Cursor / Continue with local models | [getting-started.md § Connect external tools](getting-started.md#connect-external-tools-cursor-continue-etc) |
| Export to GGUF or Hugging Face Hub | [getting-started.md § Step 6](getting-started.md#step-6--export-and-deploy) · [cli.md](cli.md) |
| Compress / distill / quantize models | [compression.md](compression.md) |
| Enable fused GPU kernels | [training/kernels.md](training/kernels.md) |
| Train on multiple GPUs | [training/multi-gpu.md](training/multi-gpu.md) |
| Deploy with HTTPS | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |
| Fix a problem | [troubleshooting.md](troubleshooting.md) |
| Run tests before a PR | [CI_LOCAL.md](CI_LOCAL.md) |

---

## Startup paths

| Mode | Commands | URL |
|------|----------|-----|
| **Install + start (Linux / macOS)** | `curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh \| bash` then `~/Seiso/scripts/start.sh` | http://127.0.0.1:8765 |
| **From a clone** | `./scripts/install.sh && ./scripts/start.sh` | http://127.0.0.1:8765 |
| **Forge (production)** | `cd forge-ui && npm install && npm run build` then `seiso forge` | http://127.0.0.1:8765 |
| **Forge (UI dev)** | Terminal 1: `seiso forge` · Terminal 2: `cd forge-ui && npm run dev` | http://127.0.0.1:5173 |
| **CLI training** | `seiso train --config configs/example_lora.yaml` | — |
| **CLI chat** | `seiso chat --model <id-or-path> --prompt "..."` | — |
| **CLI RL quant** | `seiso rl-quant run --preset minimal` | — |
| **OpenAI-compatible API** | `seiso forge` then POST `/v1/chat/completions` | http://127.0.0.1:8765/v1/... |

First launch opens onboarding — create your local admin password. Copy `.env.example` to `.env` to override host, port, or data directory.

Data directory defaults to `~/.seiso` (`SEISO_DATA_DIR`). See [install.md](install.md) and [forge.md](forge.md).

---

## Learning paths

### Path A — End user (Forge UI)

1. [getting-started.md](getting-started.md) — install through first export
2. [forge.md](forge.md) — pages, API routes, environment variables
3. [training/quickstart.md](training/quickstart.md) — Training Studio details
4. [compression.md](compression.md) — compression pipelines
5. [troubleshooting.md](troubleshooting.md) — common fixes

### Path B — CLI / automation

1. [install.md](install.md) — pip extras per platform
2. [cli.md](cli.md) — full command reference
3. [training/quickstart.md](training/quickstart.md) — YAML config fields
4. [inference/backends.md](inference/backends.md) — backend selection

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
# One-shot install + start (Linux / macOS)
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh

# From a clone
git clone https://github.com/Legendarylibr/SeisoLocalAI.git Seiso && cd Seiso
./scripts/install.sh && ./scripts/start.sh

# Linux NVIDIA (manual pip)
pip install -e ".[forge,train,cuda,dev]"

# Build UI (first launch or after UI changes)
cd forge-ui && npm install && npm run build && cd ..

# Start Forge
seiso forge    # → http://127.0.0.1:8765

# Train (CLI → ./outputs/lora-run/ per example_lora.yaml)
seiso train --config configs/example_lora.yaml

# RL quant (CLI → ~/.seiso/rl_quant/cli/<job_id>/)
seiso rl-quant run --preset minimal --kernel-rl

# Benchmark fused kernels
seiso-bench-kernels --op all

# Diagnose install
seiso doctor
```

---

## Repository layout

```
Seiso/
├── seiso/              # Core library (GPL-3.0)
│   ├── training/       # TRL trainer, config, platform caps
│   ├── kernels/        # Fused CUDA/Triton ops
│   ├── export/         # Merge, GGUF, Hub publish
│   ├── compress/       # LLM compression bootstrap
│   └── security/       # NVIDIA boundary gates
├── seiso_cli/          # CLI entrypoints
├── forge/              # FastAPI backend
│   ├── api/routes/     # REST + OpenAI-compatible endpoints
│   ├── orchestrators/  # Job workers (train, export, compress, …)
│   └── security/       # Auth, CSRF, path sandbox
├── forge-ui/           # React + TypeScript + Vite (GPL-3.0)
├── third_party/        # Vendored compression pipelines
│   ├── codellama-compress/
│   ├── sd-distill-prune-quant/
│   └── adaptive-rl-quant/
├── configs/            # Example YAML/JSON configs
├── data/               # Sample training JSONL
├── deploy/             # Caddy, nginx, systemd templates
├── docs/               # This documentation
├── scripts/            # install.sh, start.sh, CI runner
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
`~/.seiso` by default. Override with `SEISO_DATA_DIR`. See [getting-started.md § Data directory](getting-started.md#data-directory-layout).

**Why is the UI blank?**  
The Forge UI must be built: `cd forge-ui && npm run build`, or run `./scripts/start.sh` (auto-builds).

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
