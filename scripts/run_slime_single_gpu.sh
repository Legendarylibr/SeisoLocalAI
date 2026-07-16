#!/usr/bin/env bash
# Single-GPU slime-style GRPO (HF colocated generate + local data_gen corpus).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/example_slime_single_gpu.yaml}"
export PYTHONUNBUFFERED=1
exec python -m seiso_cli slime --config "$CONFIG"
