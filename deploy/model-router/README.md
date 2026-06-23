# Seiso Model Router

Classifier + RL policy gateway over **llama.cpp** or **vLLM** specialists, with **llama-swap** as the unified OpenAI-compatible orchestrator.

## Architecture

```
Client → Router (classifier + RouteBandit) → llama-swap → specialist backend
              │                                    │
              └─ lifecycle (vLLM sleep/wake) ─────┘
              └─ llama.cpp: health + llama-swap TTL unload
```

- **Router** (`seiso/model_router/`): domain classifier, contextual UCB bandit, fallback chain, GPU/RAM metrics.
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
pip install -e ".[forge]"

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

## Quick start (local — vLLM)

```bash
cd deploy/model-router
docker compose -f docker-compose.local.vllm.yml up --build
```

Or router only:

```bash
seiso router --config deploy/model-router/config/router.local.vllm.yaml
```

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
| `config/router.local.vllm.yaml` | Local router — vLLM |
| `config/router.prod.yaml` | Prod auth, rate limits, vLLM |
| `config/specialists.local.llamacpp.json` | llama.cpp specialist catalog |
| `config/specialists.local.vllm.json` | Local vLLM specialist catalog |
| `config/specialists.prod.vllm.json` | Production vLLM catalog |
| `config/llama-swap.local.yaml` | llama-swap → llama.cpp containers |
| `config/llama-swap.local.vllm.yaml` | llama-swap → local vLLM containers |
| `config/llama-swap.prod.yaml` | llama-swap spawns vLLM on demand |

| Compose file | Stack |
|--------------|-------|
| `docker-compose.local.yml` | Router + llama.cpp (default) |
| `docker-compose.local.vllm.yml` | Router + vLLM |
| `docker-compose.prod.yml` | Prod router + on-demand vLLM |

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
