# Getting started

This guide walks you from a fresh machine to your first chat, training run, and export in Seiso Forge.

**Time:** ~30–60 minutes (excluding model downloads).

## What you need

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| Node.js | 18+ | 20 LTS |
| RAM | 16 GB | 32 GB+ |
| GPU | Optional (CPU works for small models) | NVIDIA 12 GB+ VRAM for QLoRA |
| Disk | 20 GB free | 100 GB+ for multiple models |
| OS | Linux, macOS, WSL2, Windows | Linux + NVIDIA |

## Step 1 — Install

### Linux, macOS, and WSL2 (fastest)

One command installs dependencies (including native Linux build tools), builds the UI, and **starts Forge** (browser opens automatically):

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Quick installs** — use one of these four when you already know the target platform:

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

See [install.md](install.md) for Windows PowerShell, custom paths, and all installer options.

No separate start step is needed after a successful install.

**Start Forge on later sessions:**

```bash
cd "$HOME/Seiso" && start
# or: curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Custom clone location: set `SEISO_INSTALL_DIR` before running the installer (for example `SEISO_INSTALL_DIR="$HOME/code/Seiso"`).

### From a git clone (any platform)

**Linux / macOS / WSL:**

```bash
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$HOME/Seiso"
cd "$HOME/Seiso"
start
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/Legendarylibr/SeisoLocalAI.git "$env:USERPROFILE\Seiso"
cd "$env:USERPROFILE\Seiso"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install -e ".[forge,train,dev]"
cd forge-ui; npm ci; npm run build; cd ..
seiso doctor
seiso forge
```

See [install.md](install.md) for AMD ROCm, pip extras, and upgrade steps.

## Step 2 — Onboarding

1. Open **http://127.0.0.1:8765**
2. Choose **Create account and continue** (default). To restore an existing account, open **Already have a recovery key?** and paste it
3. If you created an account: **save the recovery key** shown on screen (store it in a password manager), optionally **Download encrypted .txt** with a passphrase, then press **I saved my recovery key — continue**. The public ID shown is safe to share
4. Later sessions: sign in by pasting that recovery key (lost key → start a new local session)
5. Optionally paste a [Hugging Face token](https://huggingface.co/settings/tokens) in **Settings** for gated models and faster downloads

| What the UI says | Technical name | Role |
|------------------|----------------|------|
| **Recovery key** | `nsec` | Private — save on create; paste to sign in later |
| **Public ID** | `npub` | Public owner identity (safe to share; cannot unlock alone) |

You do not need a Nostr app. Same key formats under the hood. Seiso binds to `127.0.0.1` by default — the owner pair is the instance identity (encrypted under `nostr_keys/`; Compat `/v1` key is bound to the same public ID / npub).

## Step 3 — Download a model

1. Go to **Model Hub** (`/hub`)
2. Search live Hugging Face Hub results for GGUF models (Llama, Mistral, Qwen, Gemma, and more)
3. Click **Download** on a model sized for your hardware (start with 1–3B for training, 7B+ for chat if you have VRAM)
4. Watch live download progress in the UI

Models are cached under `{SEISO_DATA_DIR}/hf_cache` (default `$HOME/.seiso/hf_cache` on Linux/macOS/WSL, `%USERPROFILE%\.seiso\hf_cache` on Windows). The Hub UI also registers inventory links under `{SEISO_DATA_DIR}/models/{user_id}/`.

Before loading a larger model, use **Free memory** in Chat or Model Hub to unload the active inference model from RAM/VRAM (downloaded files in `hf_cache/` are kept).

## Step 4 — Chat with a local model

1. Open **Chat** (`/chat`)
2. Select a downloaded model or a GGUF backend
3. Send a message — responses stream in real time

**Backend auto-selection:**

| Your hardware | Typical backend |
|---------------|-----------------|
| macOS Apple Silicon | MLX (with `[mlx]` extra) |
| Native Linux NVIDIA + GGUF on disk | llama-swap sidecar (Ollama if healthy, else sidecar llama.cpp) |
| CPU / macOS + GGUF on disk | llama.cpp |
| NVIDIA GPU + safetensors | PyTorch 4-bit |
| CPU only | GGUF |

Details: [inference/backends.md](inference/backends.md).

### Connect external tools (Cursor, Continue, etc.)

With Forge running, point any chat-completions client at:

```text
Base URL: http://127.0.0.1:8765/v1
API key:  Inference key from `{SEISO_DATA_DIR}/.inference_api_key` (default `$HOME/.seiso` or `%USERPROFILE%\.seiso`; scoped to /v1 only)
```

```bash
# Linux / macOS / WSL
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $(cat "$HOME/.seiso/.inference_api_key")" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello from Seiso"}]
  }'
```

```powershell
# Windows PowerShell
$key = Get-Content "$env:USERPROFILE\.seiso\.inference_api_key" -Raw
Invoke-RestMethod http://127.0.0.1:8765/v1/chat/completions `
  -Headers @{ Authorization = "Bearer $($key.Trim())" } `
  -ContentType "application/json" `
  -Method Post `
  -Body '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Hello from Seiso"}]}'
```

## Step 5 — Fine-tune with Training Studio

1. Open **Training Studio** (`/train`)
2. Pick a **safetensors** base model (from Model Hub — not a GGUF-only mirror)
3. Pick a dataset (Hugging Face hub ID such as `HuggingFaceH4/no_robots`, or upload JSONL). `data/sample.jsonl` is a 4-row CI format smoke only.
4. Wait for **dataset analysis** — Seiso scans the full corpus and suggests format, epochs, and seq length from the schema
5. Review hardware-tuned settings (quant, batch size, fused kernels) and click **Start training** — logs stream over SSE

**CLI equivalent:**

```bash
# Linux / macOS / WSL — activate venv first
source .venv/bin/activate
seiso train --config configs/example_lora.yaml
```

On Linux NVIDIA bare metal, approve native CUDA kernel JIT before training:

```bash
export SEISO_NVIDIA_HOST_VENV_ACK=1
seiso train --config configs/example_lora.yaml
```

Checkpoints:

- **Forge UI:** `{SEISO_DATA_DIR}/checkpoints/{user_id}/{job_id}/checkpoint-*` (with `seiso_manifest.json`)
- **CLI (`seiso train`):** YAML `output_dir` — see `configs/example_lora.yaml` (`./outputs/lora-run/`)

Platform notes: [training/quickstart.md](training/quickstart.md) · [platforms/](platforms/).

## Step 6 — Export and deploy

1. Open **Export** (`/export`)
2. Select a training checkpoint
3. Choose formats: merged safetensors, LoRA adapter, GGUF quantizations
4. Optionally publish to Hugging Face Hub with model card preflight

**CLI equivalent:**

```bash
# Linux / macOS / WSL
seiso export --checkpoint "$HOME/.seiso/checkpoints/<user>/<job_id>/checkpoint-<timestamp>" \
  --formats merged,gguf --profile inference
```

```powershell
# Windows
seiso export --checkpoint "$env:USERPROFILE\.seiso\checkpoints\<user>\<job_id>\checkpoint-<timestamp>" `
  --formats merged,gguf --profile inference
```

GGUF export requires `llama.cpp` (set `LLAMA_CPP_DIR` or install system `convert_hf_to_gguf`).

## Step 7 — Explore advanced features

| Feature | Where | Guide |
|---------|-------|-------|
| Model compression (any HF causal LM; Llama-family prune) | `/compress` | [compression.md](compression.md) · `seiso compress run` |
| Teacher distill + DPO alignment | `/distill-rl` | [compression.md](compression.md) · `seiso distill-rl run` |
| RL adaptive GGUF quantization | `/rl-quant` | [compression.md](compression.md) · `seiso rl-quant run` |
| Local RAG corpus | `/knowledge` | [forge.md](forge.md) |
| Visual recipe graphs | `/recipes` | [forge.md](forge.md) |
| External providers (OpenAI, vLLM) | `/integrations` | [forge.md](forge.md) |
| Multi-GPU training | Training Studio checkbox | [training/multi-gpu.md](training/multi-gpu.md) |
| Fused GPU kernels | Training config / Studio | [training/kernels.md](training/kernels.md) |
| Quant regression study (CLI) | `seiso experiment quant-regression` | [cli.md](cli.md#seiso-experiment) |
| HTTPS / LAN access | `deploy/` + `.env` | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |

## Data directory layout

Default data directory (override with `SEISO_DATA_DIR`):

| OS | Default path |
|----|--------------|
| Linux / macOS / WSL | `$HOME/.seiso` |
| Windows | `%USERPROFILE%\.seiso` |

```
{SEISO_DATA_DIR}/
├── hf_cache/         # Hugging Face hub cache (GGUF / safetensors downloads)
├── hf_home/          # HF_HOME mirror (created on first Hub configure)
├── hf_xet_cache/     # hf-xet transfer cache (created on first Hub configure)
├── hf_tokens/        # Encrypted Hugging Face tokens
├── models/           # Per-user inventory links to cached weights
├── checkpoints/      # Training outputs (per user)
├── exports/          # Merged / GGUF / LoRA exports
├── compress/         # LLM compression artifacts ({user_id}/{job_id}/runs/<run_id>/)
├── rl_quant/         # RL quant policy outputs
├── distill_rl/       # Distillation / RL artifacts
├── recipes/          # Recipe Studio job data
├── knowledge/        # RAG vector stores
├── uploads/          # User-uploaded datasets and files
├── artifacts/        # Tool-generated files
├── sandbox/          # Sandboxed code-exec workspace
├── forge.db          # SQLite database (persistent mode)
├── .secret_key       # Session signing key
├── .db_encryption_key
├── .inference_api_key
├── .forge.lock
└── runtime.json
```

See also the canonical tree in [README.md](../README.md#data--storage).

## Next steps

- **Developers:** [CI_LOCAL.md](CI_LOCAL.md) · run `make ci-fast`
- **Problems:** [troubleshooting.md](troubleshooting.md)
- **Full CLI:** [cli.md](cli.md)
- **Security hardening:** [README.md](../README.md#security) and [forge.md](forge.md)
- **Opt-in remote marketplace (Ark + L402) — not functional yet, do not use:** [pay/marketplace.md](pay/marketplace.md) — self-hosted stays free
- **Buzz shared / multi-node training — not functional yet, do not use:** [training/mesh.md](training/mesh.md) · skill [seiso-orchestrate](../.agents/skills/seiso-orchestrate/SKILL.md)
