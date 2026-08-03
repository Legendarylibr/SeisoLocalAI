# Forge (web UI + API)

Forge is Seiso's local web server: React UI, REST API under `/api`, and a Compat API endpoint at `/v1/chat/completions`.

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

Open **http://127.0.0.1:8765**. On first run:

1. **Create account and continue** (default), or open **Already have a recovery key?** to restore
2. **Save the recovery key** shown once (optional encrypted `.txt` download with passphrase), then **I saved my recovery key — continue**. The public ID is safe to share
3. Later: unlock by pasting that recovery key (or an encrypted backup + passphrase)

You do not need a Nostr app. See [Auth (local account / Nostr keys)](#auth-local-account--nostr-keys) for the technical mapping.

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

Forge runs as a **single uvicorn worker** by default. Job orchestrators (training, export, compress, distill-RL), live SSE log streams, in-memory rate limiting, and loaded inference models all live in that one process.

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
| `/chat` | Chat | Local inference (GGUF, MLX, PyTorch); native Linux NVIDIA GGUF uses Ollama first (llama-swap fallback); **Free memory** unloads models from RAM/VRAM without changing selection |
| `/train` | Training Studio | LoRA / QLoRA fine-tuning with live SSE logs |
| `/export` | Export | Merge LoRA, GGUF, Hub publish from checkpoints |
| `/compress` | Compress | LLM distillation / prune (Llama-family) / quant pipeline |
| `/distill-rl` | Distill-RL | Teacher → student distillation + DPO alignment |
| `/knowledge` | Knowledge | RAG corpus ingest and retrieval |
| `/integrations` | Integrations | External providers + Nostr provenance |
| `/settings` | Settings | HF token, hardware info, security toggles |

Knowledge-base ingest and retrieve are also available via API (`/api/knowledge/...`).

## Auth (local account / Nostr keys)

Forge is single-tenant: one **owner public ID** per instance. The matching **recovery key** proves ownership. Browser sessions are HttpOnly cookies + CSRF (no Bearer JWT in the JSON body). Compat `/v1` uses a file-backed inference key that is **bound to that same owner**.

The UI speaks in everyday terms; crypto is unchanged (open Nostr key formats). You do **not** need a Nostr client or relay to sign in.

| UI label | Technical name | Meaning in Seiso |
|----------|----------------|------------------|
| **Public ID** | `npub` | Public owner identity (safe to share / show in UI) |
| **Recovery key** | `nsec` | Private — save on create; paste to sign in later |
| **Encrypted backup** | `ncryptsec` (NIP-49) | Passphrase-locked file backup of the recovery key |
| **Compat key** | file under data dir | `{SEISO_DATA_DIR}/.inference_api_key` for `/v1` only; owner in `.inference_api_key.owner` |
| **Relays** | allowlisted `wss://` | Digests-only provenance prefs next to the public ID — not on the recovery key |

| Step | What happens |
|------|----------------|
| First launch | **Create account and continue** (default), or restore a recovery key / encrypted backup — that public ID becomes the instance owner |
| After create | UI shows the recovery key once — save it, optionally **Download encrypted .txt**, then **I saved my recovery key — continue**. Public ID is shown for reference. |
| Later sessions | Paste the recovery key, or encrypted backup plus passphrase (decrypted in the browser before login). The public ID alone cannot unlock. |
| Lost recovery key | **Start a new session** clears the local account, owner binding, Compat key, `job_events`, and `nostr_keys/` (downloaded model files remain) |
| Ephemeral DB | In-memory SQLite (`SEISO_DB_EPHEMERAL`); signing keys are **not** written under `nostr_keys/` |
| Settings key rotate | Import/keygen updates the account public ID (`npub`), attest key, and Compat owner binding together (keygen returns `nsec` once; Compat key rotates) |

There is no password path. Generated secrets are shown once in the browser; an encrypted signing key is kept under `{SEISO_DATA_DIR}/nostr_keys/` for provenance attest (skipped in ephemeral mode). Non-browser clients that need a Bearer JWT can send `X-Seiso-Return-Token: 1` on login/register. See also [provenance-nostr.md](provenance-nostr.md).

### Auth crypto (what is / is not guaranteed)

| Layer | Mechanism |
|-------|-----------|
| Keygen | `os.urandom(32)` → BIP-340 x-only secp256k1; bech32 `nsec` / `npub` |
| At-rest signing key | AES-256-GCM (`enc:v1:`) under `{SEISO_DATA_DIR}/nostr_keys/`; plaintext files are refused on load |
| Master AES key | `os.urandom(32)` in `.nostr_key_encryption_key` (mode `0600`) — same-machine trust |
| Optional backup | NIP-49 scrypt (encrypt `log_n` ≥ 16) + XChaCha20-Poly1305 → `ncryptsec` |
| Session | HS256 JWT (`secrets.token_urlsafe` ≥ 32 bytes), HttpOnly cookie + CSRF; login pubkey check is constant-time |

**Residual risks (local-first model):** anyone with OS access to the data dir can use the AES key + ciphertext; XSS in the Forge UI can read the one-time recovery key from `sessionStorage` until Continue; default bind is localhost — remote exposure requires explicit `SEISO_ALLOW_REMOTE` acknowledgements.

**Keep it simple:** Nostr owns **identity and Buzz-facing signatures** (login npub, attest, mesh/agent events). Do **not** replace local JWT sessions with NIP-42, or put HF/mesh/pay tokens in Nostr events. At-rest AES (`enc:v1`) is one shared helper for DB columns and signing-key files — not a second crypto religion.

## API surface

| Prefix | Purpose |
|--------|---------|
| `/api/auth` | Nostr register (generate/import), nsec login, session, reset |
| `/api/models` | Catalog, downloads, VRAM management (`GET /vram`, `POST /vram/unload`) |
| `/api/inference` | Chat completions, streaming |
| `/api/training` | Training jobs, dataset search/analysis, recommendations, SSE logs |
| `/api/export` | Export jobs, Hub publish |
| `/api/compress` | LLM compression jobs |
| `/api/distill-rl` | Distill → rollout → DPO jobs |
| `/api/knowledge` | RAG ingest and retrieve |
| `/api/providers` | External LLM provider configs |
| `/api/system` | Hardware detection, metrics |
| `/api/settings` | App settings, HF token, security posture |
| `/health` | Liveness check |
| `/v1/chat/completions` | Compat API chat (no `/api` prefix) |

Set `SEISO_DEBUG=true` to expose interactive API docs at `/api/docs`.

### Settings API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/settings` | Host, port, data dir, inference backends, HF auth summary, security posture |
| `PUT` | `/api/settings/hf-token` | Save and validate per-user Hugging Face token |
| `DELETE` | `/api/settings/hf-token` | Clear saved per-user HF token |
| `GET` | `/api/settings/hf-status` | Hub connectivity, transfer stack, inference runtime probe |

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
| `SEISO_INFERENCE_API_KEY` | auto | Scoped `/v1` key bound to the owner npub (files: `.inference_api_key` + `.inference_api_key.owner`; rotated on reset / npub rotate unless env-bound) |
| `SEISO_SECURE_COOKIES` | `false` | `Secure` cookies when TLS is terminated upstream |
| `SEISO_CORS_ORIGINS` | *(local defaults)* | Only set for HTTPS reverse proxy |
| `SEISO_HF_TOKEN` | — | Hugging Face token for gated models |
| `SEISO_DB_EPHEMERAL` | `true` | In-memory SQLite (wiped on restart); also skips durable `nostr_keys/` writes |
| `SEISO_ALLOW_TOOLS` | `false` | Web search, artifacts |
| `SEISO_ALLOW_CODE_EXEC` | `false` | AST-limited `execute_code` tool (not OS isolation); **refused** when `SEISO_ALLOW_REMOTE=true` |
| `SEISO_ALLOW_COMPAT_TOOLS` | `false` | Tool calling on `/v1/chat/completions` for session JWT only (inference API key stays chat-only; alias: `SEISO_ALLOW_OPENAI_TOOLS`) |
| `SEISO_RATE_LIMIT` | `120` | Requests/minute per IP (≥240 on localhost) |
| `SEISO_SESSION_HOURS` | `24` | Signed session lifetime |
| `SEISO_MEMORY_PROFILE` | — | Set to `low` for lean RAM / llama.cpp tuning (see `.env.example`) |
| `SEISO_SIDECAR_AUTOSTART` | `1` | `start` auto-starts Ollama/llama-swap before Forge when needed |
| `SEISO_LLAMASWAP_ENGINE` | auto | Sidecar engine override: `ollama` or `llamacpp` |
| `SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX` | off | Explicitly allow unsafe in-process llama.cpp on native Linux NVIDIA |
| `SEISO_MODEL_ROUTER_ENABLED` | `false` | Enable Smart Router model in Chat |
| `SEISO_MODEL_ROUTER_URL` | — | Router base URL (e.g. `http://127.0.0.1:8780`) |

HTTPS reverse-proxy deployment: [deployment/reverse-proxy.md](deployment/reverse-proxy.md) and [deploy/README.md](../deploy/README.md).

Adaptive RL quantization research: [Adaptive-RL-Quantization](https://github.com/Legendarylibr/Adaptive-RL-Quantization).
