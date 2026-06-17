# Seiso documentation

Choose your platform, install, then run Forge or the CLI.

## Startup paths

| Mode | Commands | URL |
|------|----------|-----|
| **Install + start (Linux / macOS)** | `curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh \| bash` then `~/Seiso/scripts/start.sh` | http://127.0.0.1:8765 |
| **Forge (production)** | `./scripts/install.sh` then `./scripts/start.sh` — or `cd forge-ui && npm install && npm run build` then `seiso forge` | http://127.0.0.1:8765 |
| **Forge (UI dev)** | Terminal 1: `seiso forge` · Terminal 2: `cd forge-ui && npm run dev` | http://127.0.0.1:5173 (API proxied to :8765) |
| **CLI training** | `seiso train --config configs/example_lora.yaml` | — |
| **CLI chat** | `seiso chat --model <id-or-path> --prompt "..."` | — |
| **OpenAI-compatible API** | `seiso forge` then POST `/v1/chat/completions` | http://127.0.0.1:8765/v1/... |

First launch opens onboarding at http://127.0.0.1:8765 — create your local admin password. Copy `.env.example` to `.env` to override host, port, or data directory.

Data directory defaults to `~/.seiso` (override with `SEISO_DATA_DIR`). See [install.md](install.md) for install extras and [forge.md](forge.md) for Forge UI pages and environment variables.

## Platform guides

| Platform | Guide |
|----------|--------|
| Linux + NVIDIA (recommended) | [platforms/linux-nvidia.md](platforms/linux-nvidia.md) |
| Linux + AMD ROCm | [platforms/linux-amd-rocm.md](platforms/linux-amd-rocm.md) |
| macOS (Apple Silicon / Intel) | [platforms/macos.md](platforms/macos.md) |
| Windows (native) | [platforms/windows.md](platforms/windows.md) |
| WSL2 + NVIDIA | [platforms/wsl.md](platforms/wsl.md) |

## Tasks

| Task | Guide |
|------|--------|
| Install extras & dependencies | [install.md](install.md) |
| Forge UI, pages, and dev workflow | [forge.md](forge.md) |
| CLI command reference | [cli.md](cli.md) |
| Local CI / quality gate | [CI_LOCAL.md](CI_LOCAL.md) |
| Train (CLI + Forge) | [training/quickstart.md](training/quickstart.md) |
| Fused GPU kernels | [training/kernels.md](training/kernels.md) |
| Multi-GPU | [training/multi-gpu.md](training/multi-gpu.md) |
| Inference backends | [inference/backends.md](inference/backends.md) |
| Model compression (LLM, image, RL quant) | [compression.md](compression.md) |
| HTTPS / reverse proxy | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |
| Deployment configs | [deploy/README.md](../deploy/README.md) |
| Problems & fixes | [troubleshooting.md](troubleshooting.md) |

## Quick commands

```bash
# One-shot install + start (Linux / macOS)
curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash
~/Seiso/scripts/start.sh

# From a clone
./scripts/install.sh && ./scripts/start.sh

# Full stack (Linux NVIDIA, manual)
pip install -e ".[forge,train,cuda,dev]"

# Build UI (required before first Forge launch, or when UI changes)
cd forge-ui && npm install && npm run build && cd ..

# Start Forge (API + web UI)
seiso forge
# → http://127.0.0.1:8765

# Train from config
seiso train --config configs/example_lora.yaml

# Benchmark fused kernels (NVIDIA / ROCm GPU)
seiso-bench-kernels --op all
```

## Repository layout

```
Seiso/
├── seiso/           # Core library (GPL-3.0)
├── seiso_cli/       # CLI entrypoints (`seiso`, `seiso-bench-kernels`)
├── forge/           # FastAPI backend + orchestrators
├── forge-ui/        # React/TypeScript frontend (GPL-3.0)
├── configs/         # Example training/compression configs
├── data/            # Sample dataset (example_lora.yaml)
├── deploy/          # Caddy, nginx, systemd, HTTPS env example
├── docs/            # This documentation
└── scripts/         # install.sh, start.sh, CI runner, dependency locks
```
