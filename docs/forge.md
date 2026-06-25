# Forge (web UI + API)

Forge is Seiso's local web server: React UI, REST API under `/api`, and an OpenAI-compatible endpoint at `/v1/chat/completions`.

## Start Forge

**URL (all platforms):** http://127.0.0.1:8765

### Linux / macOS / WSL

From an existing clone (recommended helpers):

```bash
cd "$HOME/Seiso" && start     # checks deps, builds UI if needed
# or after manual install:
cd "$HOME/Seiso" && source .venv/bin/activate && seiso forge
```

First time or after UI changes, build the frontend:

```bash
cd "$HOME/Seiso/forge-ui" && npm ci && npm run build && cd ..
seiso forge
```

Add `--open` to launch your browser automatically.

### Windows (PowerShell)

```powershell
cd "$env:USERPROFILE\Seiso"
.\.venv\Scripts\Activate.ps1
cd forge-ui; npm ci; npm run build; cd ..
seiso forge
```

On later sessions, skip the UI build unless `forge-ui/dist` is missing or you changed frontend code.

Open **http://127.0.0.1:8765**. On first run, complete onboarding to create your local admin password.

### UI development (hot reload)

Run the API and Vite dev server in two terminals:

```bash
# Terminal 1 — API (default :8765)
seiso forge

# Terminal 2 — Vite dev server (:5173, proxies /api and /health to Forge)
cd forge-ui && npm run dev
```

Browse **http://127.0.0.1:5173** (or `localhost:5173`) — CORS is pre-configured; no `.env` change needed.

Optional API auto-reload during backend work:

```bash
seiso forge --reload
```

### Custom host / port

```bash
seiso forge --host 127.0.0.1 --port 8766
# or via environment (see .env.example)
SEISO_PORT=8766 seiso forge
```

### Single instance

Forge binds exclusively to `SEISO_HOST:SEISO_PORT` (default `127.0.0.1:8765`). Starting a second `seiso forge` on the same address exits immediately with an error. Two locks enforce this:

1. **Port slot lock** — a file lock under your temp directory (`seiso-forge-locks/<user>/`) held for the entire `seiso forge` process, even when `SEISO_DATA_DIR` differs.
2. **Data-dir lock** — `{SEISO_DATA_DIR}/.forge.lock` prevents two processes from sharing the same data directory on different ports.

To run a second Forge intentionally, change **both** `SEISO_PORT` and `SEISO_DATA_DIR`.

### Process model

Forge runs as a **single uvicorn worker** by default. Job orchestrators (training, export, compress, distill-RL, RL quant), live SSE log streams, in-memory rate limiting, and loaded inference models all live in that one process.

| Constraint | Why |
|------------|-----|
| Do not use `--workers N` with `N > 1` | Each worker gets its own job store and model pool — jobs started on worker A are invisible to worker B |
| Restart clears in-flight SSE | Log buffers are in-memory; reconnect after restart shows DB-persisted job status only |
| One Forge per data directory | The `.forge.lock` file prevents two processes from sharing checkpoints and the HF cache |

For production behind a reverse proxy, terminate TLS upstream and run **one** Forge process per machine (or per isolated `SEISO_DATA_DIR`). See [deployment/reverse-proxy.md](deployment/reverse-proxy.md).

## UI pages

| Path | Page | Purpose |
|------|------|---------|
| `/` | Dashboard | Workspace overview and quick links |
| `/hub` | Model Hub | Browse and download catalog models |
| `/chat` | Chat | Local inference (GGUF, MLX, PyTorch); **Free memory** unloads models from RAM/VRAM without changing selection |
| `/train` | Training Studio | LoRA / QLoRA fine-tuning with live SSE logs |
| `/export` | Export | Merge LoRA, GGUF, Hub publish from checkpoints |
| `/compress` | Compress | LLM distillation / prune (Llama-family) / quant pipeline |
| `/distill-rl` | Distill-RL | Teacher → student distillation + DPO alignment |
| `/rl-quant` | RL Quant | Adaptive GGUF quantization policy training |
| `/recipes` | Recipe Studio | Visual graph editor for data/recipe jobs |
| `/knowledge` | Knowledge | RAG corpus ingest and retrieval |
| `/integrations` | Integrations | External providers (OpenAI, Anthropic, vLLM) |
| `/settings` | Settings | HF token, hardware info, security toggles |

Knowledge-base ingest and retrieve are also available via API (`/api/knowledge/...`).

## API surface

| Prefix | Purpose |
|--------|---------|
| `/api/auth` | Login, register (onboarding), session |
| `/api/models` | Catalog, downloads, VRAM management (`GET /vram`, `POST /vram/unload`) |
| `/api/inference` | Chat completions, streaming |
| `/api/training` | Training jobs, dataset search/analysis, recommendations, SSE logs |
| `/api/export` | Export jobs, Hub publish |
| `/api/compress` | LLM compression jobs |
| `/api/distill-rl` | Distill → rollout → DPO jobs |
| `/api/rl-quant` | RL quantization jobs |
| `/api/recipes` | Recipe graph execution |
| `/api/knowledge` | RAG ingest and retrieve |
| `/api/providers` | External LLM provider configs |
| `/api/system` | Hardware detection, metrics |
| `/health` | Liveness check |
| `/v1/chat/completions` | OpenAI-compatible chat (no `/api` prefix) |

Set `SEISO_DEBUG=true` to expose interactive API docs at `/api/docs`.

### Training API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/training/datasets` | Search Hugging Face datasets |
| `GET` | `/api/training/models` | List locally cached trainable snapshots |
| `GET` | `/api/training/recommendations` | Hardware + dataset-aware hyperparameter hints |
| `POST` | `/api/training/analyze-dataset` | Full-corpus schema analysis and training suggestions |
| `POST` | `/api/training/validate-dataset` | Preflight validation (same full-corpus pass) |
| `POST` | `/api/training/jobs` | Start a training job |
| `GET` | `/api/training/jobs` | List jobs for the signed-in user |
| `GET` | `/api/training/jobs/{id}/stream` | SSE logs and metrics |
| `POST` | `/api/training/jobs/{id}/cancel` | Cancel a running job |

See [training/quickstart.md](training/quickstart.md) for dataset formats and config fields.

### VRAM / RAM management

| Endpoint | Purpose |
|----------|---------|
| `GET /api/models/vram` | Loaded local model, headroom, recommended largest fit |
| `POST /api/models/vram/unload` | Free memory — unload local pool + refresh headroom |
| `POST /api/inference/cancel` | Backward-compatible alias of `vram/unload` |
| `POST /api/inference/cancel-generation` | Stop streaming only; keeps model warm |

Model Hub shows headroom, loaded model name, **Free memory**, and largest catalog model that fits this machine.

## Environment variables

Copy `.env.example` to `.env` in the repo root. Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SEISO_HOST` | `127.0.0.1` | Bind address (forced to localhost unless remote allowed) |
| `SEISO_PORT` | `8765` | HTTP port |
| `SEISO_DATA_DIR` | `~/.seiso` (expands on all OSes) | Models, checkpoints, exports, uploads |
| `SEISO_SECRET_KEY` | auto | Session signing key |
| `SEISO_ALLOW_REMOTE` | `false` | Bind `0.0.0.0` (requires `SEISO_REMOTE_ACK=1`) |
| `SEISO_TRUST_PROXY` | `false` | Honor `X-Forwarded-*` from `SEISO_TRUSTED_PROXY_IPS` only |
| `SEISO_TRUSTED_PROXY_IPS` | — | Comma-separated proxy IPs (e.g. `127.0.0.1,::1`) |
| `SEISO_INFERENCE_API_KEY` | auto | Scoped key for `/v1` only (file: `{SEISO_DATA_DIR}/.inference_api_key`) |
| `SEISO_SECURE_COOKIES` | `false` | `Secure` cookies when TLS is terminated upstream |
| `SEISO_CORS_ORIGINS` | *(local defaults)* | Only set for HTTPS reverse proxy |
| `SEISO_HF_TOKEN` | — | Hugging Face token for gated models |
| `SEISO_DB_EPHEMERAL` | `true` | In-memory SQLite (wiped on restart) |
| `SEISO_ALLOW_TOOLS` | `false` | Web search, artifacts |
| `SEISO_ALLOW_CODE_EXEC` | `false` | Sandboxed `execute_code` tool |
| `SEISO_ALLOW_OPENAI_TOOLS` | `false` | Tool calling on `/v1/chat/completions` |

HTTPS reverse-proxy deployment: [deployment/reverse-proxy.md](deployment/reverse-proxy.md) and [deploy/README.md](../deploy/README.md).
