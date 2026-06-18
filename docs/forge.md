# Forge (web UI + API)

Forge is Seiso's local web server: React UI, REST API under `/api`, and an OpenAI-compatible endpoint at `/v1/chat/completions`.

## Start Forge

**URL (all platforms):** http://127.0.0.1:8765

### Linux / macOS / WSL

From an existing clone (recommended helpers):

```bash
"$HOME/Seiso/scripts/start.sh"     # later sessions — checks deps, builds UI if needed
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

Browse **http://127.0.0.1:5173**. CORS for this origin is pre-configured in `.env.example`.

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

## UI pages

| Path | Page | Purpose |
|------|------|---------|
| `/` | Dashboard | Workspace overview and quick links |
| `/hub` | Model Hub | Browse and download catalog models |
| `/chat` | Chat | Local inference (GGUF, MLX, PyTorch, Ollama) |
| `/train` | Training Studio | LoRA / QLoRA fine-tuning with live SSE logs |
| `/export` | Export | Merge LoRA, GGUF, Hub publish from checkpoints |
| `/compress` | Compress | Code Llama distillation / prune / quant pipeline |
| `/image-compress` | Image Compress | Stable Diffusion distill / prune / quant pipeline |
| `/rl-quant` | RL Quant | Adaptive GGUF quantization policy training |
| `/recipes` | Recipe Studio | Visual graph editor for data/recipe jobs |
| `/integrations` | Integrations | External providers (OpenAI, Anthropic, Ollama, vLLM) |
| `/settings` | Settings | HF token, hardware info, security toggles |

Knowledge-base ingest and retrieve are **API-only** (`/api/knowledge/...`); there is no dedicated UI page.

## API surface

| Prefix | Purpose |
|--------|---------|
| `/api/auth` | Login, register (onboarding), session |
| `/api/models` | Catalog, downloads, VRAM management |
| `/api/inference` | Chat completions, streaming |
| `/api/training` | Training jobs, datasets, SSE logs |
| `/api/export` | Export jobs, Hub publish |
| `/api/compress` | LLM compression jobs |
| `/api/image-compress` | Image compression jobs |
| `/api/rl-quant` | RL quantization jobs |
| `/api/recipes` | Recipe graph execution |
| `/api/knowledge` | RAG ingest and retrieve |
| `/api/providers` | External LLM provider configs |
| `/api/system` | Hardware detection, metrics |
| `/health` | Liveness check |
| `/v1/chat/completions` | OpenAI-compatible chat (no `/api` prefix) |

Set `SEISO_DEBUG=true` to expose interactive API docs at `/api/docs`.

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
| `SEISO_CORS_ORIGINS` | `http://127.0.0.1:8765,http://localhost:5173` | Allowed browser origins |
| `SEISO_HF_TOKEN` | — | Hugging Face token for gated models |
| `SEISO_DB_EPHEMERAL` | `true` | In-memory SQLite (wiped on restart) |
| `SEISO_ALLOW_TOOLS` | `false` | Web search, artifacts |
| `SEISO_ALLOW_CODE_EXEC` | `false` | Sandboxed `execute_code` tool |
| `SEISO_ALLOW_OPENAI_TOOLS` | `false` | Tool calling on `/v1/chat/completions` |

HTTPS reverse-proxy deployment: [deployment/reverse-proxy.md](deployment/reverse-proxy.md) and [deploy/README.md](../deploy/README.md).
