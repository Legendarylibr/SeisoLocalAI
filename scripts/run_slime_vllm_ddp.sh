#!/usr/bin/env bash
# Multi-GPU slime-style GRPO: vLLM generate + Accelerate DDP policy + weight sync.
#
# Prerequisites:
#   1) Shared filesystem so vLLM can read output_dir/vllm_weight_sync/
#   2) vLLM server already running with LoRA enabled (recommended), e.g.:
#        python -m vllm.entrypoints.openai.api_server \
#          --model Qwen/Qwen2.5-0.5B-Instruct \
#          --host 127.0.0.1 --port 8000 \
#          --tensor-parallel-size 2 \
#          --enable-lora
#      Or Seiso managed multi-GPU:
#        SEISO_MANAGED_VLLM_ENABLED=true SEISO_MANAGED_VLLM_ENABLE_LORA=true \
#        SEISO_MANAGED_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
#   3) Optional synth data via NVIDIA NeMo Data Designer (multi-GPU vLLM only):
#        pip install -e '.[data-designer]'
#      With data_gen: true and data_designer: auto, rank0 materializes numeric/choice
#      prompts through Data Designer → local vLLM before DDP training starts.
#
# Usage:
#   scripts/run_slime_vllm_ddp.sh [num_processes] [config_yaml]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NPROC="${1:-2}"
CONFIG="${2:-configs/example_training_slime_vllm.yaml}"
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
