#!/usr/bin/env bash
# Optional Flash Attention 2 install for Linux + NVIDIA.
#
# Seiso works without flash-attn (PyTorch SDPA fallback). Run this after the main
# installer if you want Flash Attention 2 for training/inference.
#
# Usage (from repo root, venv active):
#   ./scripts/install_flash_attn.sh
#
# Env:
#   SEISO_SKIP_FLASH_ATTN=1     Skip (no-op)
#   MAX_JOBS=4                  Parallel compile jobs (default: nproc)
#   CUDA_HOME                   CUDA toolkit root (auto-detected when possible)
set -euo pipefail

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

on_windows_mount() {
  [[ "${1:-}" == /mnt/* ]]
}

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
    printf '%s\n' "$CUDA_HOME"
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    dirname "$(dirname "$(command -v nvcc)")"
    return
  fi
  for candidate in /usr/local/cuda /opt/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

torch_cuda_ok() {
  python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(2)
print(torch.__version__)
PY
}

main() {
  [[ "${SEISO_SKIP_FLASH_ATTN:-0}" == "1" ]] && { log "SEISO_SKIP_FLASH_ATTN=1 — skipping"; exit 0; }

  uname -s | grep -q '^Linux$' || die "Flash Attention install is supported on Linux only"

  local root
  root="$(repo_root)"
  [[ -f "$root/pyproject.toml" ]] || die "Run from the Seiso repo (missing $root/pyproject.toml)"

  if on_windows_mount "$root"; then
    die "Refusing to build flash-attn on a Windows mount ($root).

Clone into the Linux filesystem instead, e.g.:
  SEISO_INSTALL_DIR=~/Seiso curl -fsSL .../start | bash

Building CUDA wheels on /mnt/c/... often fails with missing pyproject.toml/setup.py errors."
  fi

  if ! command -v python >/dev/null 2>&1; then
    die "Activate the Seiso venv first: source $root/.venv/bin/activate"
  fi

  if python -c "import flash_attn" 2>/dev/null; then
    log "flash-attn already installed"
    exit 0
  fi

  local torch_ver cuda_home
  if ! torch_ver="$(torch_cuda_ok)"; then
    die "PyTorch with CUDA is required before installing flash-attn.
Install Seiso first: pip install -e \".[forge,train,cuda,dev]\""
  fi
  log "PyTorch $torch_ver with CUDA detected"

  if ! cuda_home="$(detect_cuda_home)"; then
    warn "CUDA toolkit (nvcc) not found — flash-attn build may fail"
    warn "Install the CUDA toolkit or set CUDA_HOME, then re-run this script"
  else
    export CUDA_HOME="$cuda_home"
    export PATH="$CUDA_HOME/bin:$PATH"
    log "Using CUDA_HOME=$CUDA_HOME"
  fi

  export MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
  log "Building flash-attn (MAX_JOBS=$MAX_JOBS) — this can take several minutes"

  python -m pip install -U pip wheel setuptools ninja packaging

  if ! python -m pip install "flash-attn>=2.5" --no-build-isolation; then
    cat <<EOF >&2

Flash Attention build failed (optional). Seiso will use PyTorch SDPA instead.

Common fixes on Linux + NVIDIA (4090):
  1. Clone/install on the Linux filesystem (~/Seiso), not /mnt/c/...
  2. Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit  (or NVIDIA .run / cuda-nvcc)
  3. Match PyTorch CUDA to your driver: https://pytorch.org/get-started/locally/
  4. Retry: MAX_JOBS=4 ./scripts/install_flash_attn.sh

To skip permanently: export SEISO_SKIP_FLASH_ATTN=1 before start
EOF
    exit 1
  fi

  python -c "import flash_attn; print('flash-attn OK:', flash_attn.__version__)"
}

main "$@"
