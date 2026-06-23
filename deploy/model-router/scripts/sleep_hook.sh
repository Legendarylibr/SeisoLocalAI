#!/usr/bin/env sh
# Sleep hook — vLLM level 1 (weights to CPU RAM, KV cache discarded).
set -eu
PROXY_URL="${PROXY_URL:-${1:-http://127.0.0.1:8000}}"
LEVEL="${SLEEP_LEVEL:-1}"
curl -sf -X POST "${PROXY_URL}/sleep?level=${LEVEL}" \
  || curl -sf -X POST "${PROXY_URL}/sleep" -H "Content-Type: application/json" -d "{\"level\":\"${LEVEL}\"}" \
  || true
