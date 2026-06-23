#!/usr/bin/env sh
# Wake hook for llama-swap / router — call vLLM wake_up on routed backend.
set -eu
PROXY_URL="${PROXY_URL:-${1:-http://127.0.0.1:8000}}"
curl -sf -X POST "${PROXY_URL}/wake_up" || curl -sf -X POST "${PROXY_URL}/wake_up?tags=weights" || true
