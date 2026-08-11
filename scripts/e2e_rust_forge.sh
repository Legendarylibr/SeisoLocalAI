#!/usr/bin/env bash
# End-to-end smoke: build seiso-forge, run train+export jobs, crypto, sandbox.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER:-}"
if ! command -v gcc >/dev/null 2>&1 || ! gcc -v >/dev/null 2>&1; then
  if command -v clang >/dev/null 2>&1; then
    export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang
    export CC=clang
    export CXX=clang++
  fi
fi

echo "== cargo test (workspace + e2e) =="
cargo test --workspace -- --nocapture

echo "== python worker direct smoke =="
echo '{"v":1,"op":"train.start","job_id":"sh","config":{"smoke_only":true},"paths":{}}' \
  | PYTHONPATH=python python3 -m seiso_ml_worker | tee /tmp/seiso-worker-out.txt
grep -q 'smoke train complete' /tmp/seiso-worker-out.txt

echo "== live forge binary =="
DATA="$(mktemp -d)"
export SEISO_DATA_DIR="$DATA"
export SEISO_HOST=127.0.0.1
export SEISO_PORT=18765
cargo build -p seiso-forge
./target/debug/seiso-forge &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; rm -rf "$DATA"; }
trap cleanup EXIT
for i in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${SEISO_PORT}/api/health" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl -sf "http://127.0.0.1:${SEISO_PORT}/api/health" | tee /tmp/seiso-health.json
grep -q '"status":"ok"' /tmp/seiso-health.json

JOB=$(curl -sf -X POST "http://127.0.0.1:${SEISO_PORT}/api/jobs" \
  -H 'content-type: application/json' \
  -d '{"kind":"train","config":{"smoke_only":true}}')
echo "$JOB" | tee /tmp/seiso-job.json
ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$JOB")
for i in $(seq 1 100); do
  ST=$(curl -sf "http://127.0.0.1:${SEISO_PORT}/api/jobs/${ID}")
  STATUS=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$ST")
  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" || "$STATUS" == "cancelled" ]]; then
    echo "$ST" | tee /tmp/seiso-job-final.json
    break
  fi
  sleep 0.05
done
grep -q '"status":"succeeded"' /tmp/seiso-job-final.json

curl -sf -X POST "http://127.0.0.1:${SEISO_PORT}/api/crypto/roundtrip" \
  -H 'content-type: application/json' \
  -d '{"plaintext":"shell-e2e"}' | tee /tmp/seiso-crypto.json
grep -q '"ok":true' /tmp/seiso-crypto.json

echo "e2e_rust_forge: OK"
