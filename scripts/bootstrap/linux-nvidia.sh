#!/usr/bin/env bash
# Native Linux + NVIDIA — full stack with Ollama-first sidecar for GGUF chat.
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/bootstrap/linux-nvidia.sh | bash
set -euo pipefail

export SEISO_INSTALL_PROFILE=linux-nvidia
# Soft-warn if Ollama/llama-swap is missing (set SEISO_REQUIRE_SIDECAR=1 to hard-fail).
export SEISO_REQUIRE_SIDECAR="${SEISO_REQUIRE_SIDECAR:-0}"

RAW_BASE="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"

if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  exec bash "$REPO_ROOT/scripts/install.sh" "$@"
fi

exec bash -c "$(curl -fsSL "${RAW_BASE}/scripts/install.sh")" bash "$@"
