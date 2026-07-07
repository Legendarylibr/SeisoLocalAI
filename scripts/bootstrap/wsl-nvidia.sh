#!/usr/bin/env bash
# WSL2 + NVIDIA — CUDA stack; sidecar optional (no hard gate).
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/bootstrap/wsl-nvidia.sh | bash
set -euo pipefail

export SEISO_INSTALL_PROFILE=wsl-nvidia
export SEISO_REQUIRE_SIDECAR=0
export SEISO_NVIDIA_WSL_ACK="${SEISO_NVIDIA_WSL_ACK:-1}"

RAW_BASE="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"

if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  exec bash "$REPO_ROOT/scripts/install.sh" "$@"
fi

exec bash -c "$(curl -fsSL "${RAW_BASE}/scripts/install.sh")" bash "$@"
