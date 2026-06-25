# Seiso Model Router

Classifier + RL policy gateway over **llama.cpp** or **vLLM** specialists. **[Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B)** routing is supported **only** on **vLLM stacks with sleep mode** (`--enable-sleep-mode`). **llama-swap** proxies specialists.

## Architecture

```
Client → Router → [Nemotron-Orchestrator-8B] → specialist pick → LiteLLM → llama-swap / cloud API / cloud vLLM
              │    (vLLM + sleep mode only)
              ├─ heuristic classifier + RouteBandit (llama.cpp → httpx)
              └─ lifecycle (local vLLM sleep/wake) + GPU/RAM metrics
```

- **Router** (`seiso/model_router/`): domain classifier, contextual UCB bandit, optional Nemotron orchestrator (vLLM sleep stacks only), **LiteLLM execution on all vLLM stacks**, httpx for llama.cpp, fallback chain, GPU/RAM metrics.
- **Nemotron-Orchestrator-8B**: ToolOrchestra model served via vLLM with sleep mode; picks specialists via the `answer` tool.
- **LiteLLM**: Always executes vLLM-stack completions — local `hosted_vllm/*` via llama-swap, plus optional `cloud_vllm` and `cloud_api` catalog routes.
- **llama-swap**: YAML-defined model registry, TTL, proxy to backends.
- **Local default**: **llama.cpp** (`llama-server` containers, GGUF).
- **Local vLLM / Production**: **vLLM** with `--enable-sleep-mode` + HTTP sleep/wake (level 1).

### VRAM policy (vLLM routes)

| Tier | Behavior |
|------|----------|
| **VRAM hot** (`vram_hot: true`, max 2) | No idle sleep — stays loaded in GPU |
| **Other specialists** | Sleep level 1 after `idle_sleep_sec` (weights in CPU RAM) |
| **Fallback** | If primary unreachable, try routes by `fallback_priority` |

llama.cpp routes rely on llama-swap TTL for unloading cold models; the router does not call a sleep API on llama-server.

## Quick start (local — llama.cpp, default)

```bash
cd SeisoLocalAI
pip install -e ".[router]"

# Router only (point router.local.yaml at your llama.cpp / llama-swap URLs)
seiso router --config deploy/model-router/config/router.local.yaml

# Full stack (GPU + GGUF files under models/)
cd deploy/model-router
docker compose -f docker-compose.local.yml up --build
```

Set GGUF paths (defaults assume files in `../../models/`):

```bash
export SEISO_GENERAL_GGUF=/models/your-general.gguf
export SEISO_CODE_GGUF=/models/your-code.gguf
export SEISO_REASONING_GGUF=/models/your-reasoning.gguf
```

## Quick start (local — vLLM + Nemotron orchestrator)

Requires vLLM specialists and the orchestrator all started with `--enable-sleep-mode` (see `docker-compose.local.vllm.yml`).

```bash
cd deploy/model-router
docker compose -f docker-compose.local.vllm.yml up --build
```

This starts **Nemotron-Orchestrator-8B** (`vllm-orchestrator`) plus three vLLM specialists. The router config sets `inference_backend: vllm`, `vllm_sleep_mode: true`, and `routing_mode: nemotron`.

Override the orchestrator checkpoint:

```bash
export SEISO_ORCHESTRATOR_MODEL=nvidia/Nemotron-Orchestrator-8B
```

Router only (orchestrator already running on vLLM with sleep mode):

```bash
seiso router --config deploy/model-router/config/router.local.vllm.yaml
```

Nemotron routing is rejected at startup unless all three are set: `inference_backend: vllm`, `vllm_sleep_mode: true`, and `orchestrator_url`.

## Production (vLLM)

```bash
export SEISO_ROUTER_API_KEYS=prod-key-one,prod-key-two
cd deploy/model-router
docker compose -f docker-compose.prod.yml up --build -d
```

- API key: `Authorization: Bearer <key>` or `X-API-Key`
- Rate limit: `SEISO_ROUTER_RATE_LIMIT_RPM` (default 120)
- Metrics: `GET /metrics` (Prometheus); prod compose includes Prometheus on `:9090`

## Example request

```bash
curl http://127.0.0.1:8780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"write a python function to merge two dicts"}],
    "max_tokens": 128
  }'
```

## Configuration

| File | Purpose |
|------|---------|
| `config/router.local.yaml` | Local router — **llama.cpp** (default) |
| `config/router.local.vllm.yaml` | Local router — vLLM + Nemotron orchestrator |
| `config/router.prod.yaml` | Prod auth, rate limits, vLLM |
| `config/specialists.local.llamacpp.json` | llama.cpp specialist catalog |
| `config/specialists.local.vllm.json` | Local vLLM specialist catalog |
| `config/specialists.prod.vllm.json` | Production vLLM catalog |
| `config/specialists.cloud.example.json` | Example cloud API + cloud vLLM routes |
| `config/llama-swap.local.llamacpp.yaml` | llama-swap → llama.cpp containers |
| `config/litellm.local.vllm.yaml` | Standalone LiteLLM proxy + Nemotron callback |
| `config/llama-swap.local.vllm.yaml` | llama-swap → local vLLM containers |
| `config/llama-swap.prod.yaml` | llama-swap spawns vLLM on demand |

| Compose file | Stack |
|--------------|-------|
| `docker-compose.local.yml` | Router + llama.cpp (default) |
| `docker-compose.local.vllm.yml` | Router + Nemotron + vLLM specialists |
| `docker-compose.prod.yml` | Prod router + on-demand vLLM |

## Nemotron orchestrator (vLLM sleep mode only)

Nemotron routing activates only when **all** of the following are true:

| Requirement | Config |
|-------------|--------|
| vLLM backend | `inference_backend: vllm` |
| Sleep mode stack | `vllm_sleep_mode: true` (all vLLM containers use `--enable-sleep-mode`) |
| Orchestrator endpoint | `orchestrator_url` pointing at Nemotron on vLLM |
| Routing mode | `routing_mode: nemotron` |

The orchestrator issues an `answer` tool call with a specialist alias (`seiso-general-1`, `seiso-code-1`, `seiso-reasoning-1`) from `orchestrator_alias` in `specialists.local.vllm.json`.

**llama.cpp** stacks and vLLM without sleep mode always use heuristic classifier + RL bandit routing, even if `routing_mode: nemotron` is set (startup validation rejects invalid Nemotron configs).

| Setting | Purpose |
|---------|---------|
| `routing_mode` | `nemotron` (vLLM sleep only) or `heuristic` |
| `vllm_sleep_mode` | Must be `true` for Nemotron |
| `orchestrator_url` | vLLM OpenAI base URL for Nemotron |
| `orchestrator_model` | Served model name (default `seiso-orchestrator`) |

Response metadata includes `seiso_router.orchestrator` and `orchestrator_alias` when Nemotron routing is active.

## LiteLLM execution (vLLM stacks)

When `inference_backend: vllm`, the router **always** executes completions through LiteLLM (no direct httpx to vLLM). llama.cpp stacks use httpx to llama-swap / llama-server.

| Setting | Purpose |
|---------|---------|
| `litellm_routing_strategy` | LiteLLM Router strategy (default `simple-shuffle`) |

Flow: **Nemotron or heuristic picks route** → **LiteLLM Router** → llama-swap, direct vLLM URL, cloud vLLM, or managed API (`openai/*`, `anthropic/*`, …).

Cloud routes use `backend_type: cloud_api` or `cloud_vllm` in the specialist catalog — see `specialists.cloud.example.json`. Set `litellm_model` and `api_key_env` for API providers.

Install router extras locally:

```bash
pip install -e ".[router]"
```

vLLM startup fails fast if LiteLLM is not installed.

### Standalone LiteLLM proxy (optional)

Run the LiteLLM proxy with the Nemotron pre-call hook instead of the Seiso router:

```bash
pip install -e ".[router]"
export SEISO_ROUTER_CONFIG_PATH=deploy/model-router/config/router.local.vllm.yaml
litellm --config deploy/model-router/config/litellm.local.vllm.yaml --port 4000
```

The callback in `seiso/model_router/litellm_callback.py` calls Nemotron before LiteLLM forwards to vLLM.

## Endpoints

| Path | Description |
|------|-------------|
| `POST /v1/chat/completions` | Routed chat (OpenAI-compatible) |
| `GET /v1/models` | List specialist model IDs |
| `GET /router/status` | Lifecycle + policy stats (`inference_backend` in status) |
| `GET /health` | Liveness |
| `GET /ready` | Backend readiness |
| `GET /metrics` | Prometheus metrics |

Response includes `seiso_router` metadata (route_id, domain, latency, reward).

## RL policy

The bandit learns from latency-based rewards and persists to `data/router/policy_state.json`. It reuses feature extraction from `adaptive_quant` (entropy, complexity buckets) compatible with the existing RL quant route learner.

## Sleep / wake hooks (vLLM)

```bash
deploy/model-router/scripts/wake_hook.sh http://vllm-code:8000
deploy/model-router/scripts/sleep_hook.sh http://vllm-reasoning:8000
```

vLLM sleep API: [vLLM Sleep Mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode/)

## Forge UI integration

Enable the router in Forge so Chat shows **Smart Router (auto-route)** in the model picker:

```bash
# .env or environment
SEISO_MODEL_ROUTER_ENABLED=true
SEISO_MODEL_ROUTER_URL=http://127.0.0.1:8780
# Optional if router runs in prod mode with API keys:
SEISO_MODEL_ROUTER_API_KEY=your-router-key
```

Start the router (`seiso router` or docker compose), then `seiso forge`. Chat skips local VRAM preload when Smart Router is selected and streams via `/api/inference/chat`. Status: `GET /api/inference/router/status`.
