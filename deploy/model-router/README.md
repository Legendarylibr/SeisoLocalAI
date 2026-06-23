# Seiso Model Router

Classifier + RL policy gateway over **vLLM** specialists, with **llama-swap** as the unified OpenAI-compatible orchestrator.

## Architecture

```
Client → Router (classifier + RouteBandit) → llama-swap → vLLM specialist
              │                                    │
              └─ wake/sleep hooks (level 1) ───────┘
```

- **Router** (`seiso/model_router/`): domain classifier, contextual UCB bandit (extends adaptive-rl-quant features), fallback chain, GPU/RAM metrics.
- **llama-swap**: YAML-defined model registry, TTL, proxy to vLLM backends.
- **vLLM**: `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1` for HTTP sleep/wake (level 1 = weights in RAM).

### VRAM policy

| Tier | Behavior |
|------|----------|
| **VRAM hot** (`vram_hot: true`, max 2) | No idle sleep — stays loaded in GPU |
| **Other specialists** | Sleep level 1 after `idle_sleep_sec` (weights in CPU RAM) |
| **Fallback** | If primary unreachable, try routes by `fallback_priority` |

## Quick start (local)

```bash
cd SeisoLocalAI
pip install -e ".[forge]"

# Without Docker — router only (point specialists.json at your vLLM URLs)
seiso router --config deploy/model-router/config/router.local.yaml

# Full stack (GPU required)
cd deploy/model-router
docker compose -f docker-compose.local.yml up --build
```

```bash
curl http://127.0.0.1:8780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"write a python function to merge two dicts"}],
    "max_tokens": 128
  }'
```

## Production

```bash
export SEISO_ROUTER_API_KEYS=prod-key-one,prod-key-two
cd deploy/model-router
docker compose -f docker-compose.prod.yml up --build -d
```

- API key: `Authorization: Bearer <key>` or `X-API-Key`
- Rate limit: `SEISO_ROUTER_RATE_LIMIT_RPM` (default 120)
- Metrics: `GET /metrics` (Prometheus); prod compose includes Prometheus on `:9090`

## Configuration

| File | Purpose |
|------|---------|
| `config/specialists.json` | Specialist routes, VRAM hot flags, vLLM URLs |
| `config/router.local.yaml` | Local router settings |
| `config/router.prod.yaml` | Prod auth, rate limits, logging |
| `config/llama-swap.local.yaml` | llama-swap → pre-running vLLM containers |
| `config/llama-swap.prod.yaml` | llama-swap spawns vLLM with sleep flags |

## Endpoints

| Path | Description |
|------|-------------|
| `POST /v1/chat/completions` | Routed chat (OpenAI-compatible) |
| `GET /v1/models` | List specialist model IDs |
| `GET /router/status` | Lifecycle + policy stats |
| `GET /health` | Liveness |
| `GET /ready` | Backend readiness |
| `GET /metrics` | Prometheus metrics |

Response includes `seiso_router` metadata (route_id, domain, latency, reward).

## RL policy

The bandit learns from latency-based rewards and persists to `data/router/policy_state.json`. It reuses feature extraction from `adaptive_quant` (entropy, complexity buckets) compatible with the existing RL quant route learner.

## Sleep / wake hooks

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
