#!/usr/bin/env bash
# Multi-GPU slime-style GRPO: SGLang generate + Accelerate DDP policy + weight sync.
#
# Prerequisites:
#   1) Shared filesystem so SGLang can read output_dir/sglang_weight_sync/
#   2) SGLang server(s) already running, e.g.:
#        python -m sglang.launch_server \
#          --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000
#
# Usage:
#   scripts/run_slime_ddp.sh [num_processes] [config_yaml]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NPROC="${1:-2}"
CONFIG="${2:-configs/example_training_slime_ddp.yaml}"
export PYTHONUNBUFFERED=1
export SEISO_EMIT_METRICS_STDOUT="${SEISO_EMIT_METRICS_STDOUT:-1}"

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate not found; install training extras (pip install -e '.[train]')" >&2
  exit 1
fi

exec accelerate launch \
  --multi_gpu \
  --num_processes="$NPROC" \
  --module seiso.training.worker \
  --config "$CONFIG"
