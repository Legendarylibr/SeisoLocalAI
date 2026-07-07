#!/usr/bin/env bash
# Linux AMD ROCm — install Seiso; install PyTorch ROCm wheel separately after.
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/bootstrap/linux-rocm.sh | bash
set -euo pipefail

export SEISO_INSTALL_PROFILE=linux-rocm
export SEISO_REQUIRE_SIDECAR=0

RAW_BASE="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"

if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  exec bash "$REPO_ROOT/scripts/install.sh" "$@"
fi

exec bash -c "$(curl -fsSL "${RAW_BASE}/scripts/install.sh")" bash "$@"
