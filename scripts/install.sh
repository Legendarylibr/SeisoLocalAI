#!/usr/bin/env bash
# Install Seiso on Linux or macOS — clone (if needed), venv, pip extras, Forge UI build.
#
# One-liner (Linux / macOS):
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
#
# Options (env vars):
#   SEISO_INSTALL_DIR   Install location (default: ~/Seiso)
#   SEISO_REPO_URL      Git remote (default: https://github.com/Legendarylibr/SeisoLocalAI.git)
#   SEISO_BRANCH        Branch to clone (default: main)
#   SEISO_SKIP_UI=1     Skip forge-ui npm build
#   SEISO_START=1       Run scripts/start.sh when install finishes
#   SEISO_NO_BANNER=1   Skip glitch install TUI
#   SEISO_VERBOSE=1     Show full pip/npm output (no TUI overlay)
#
# Recommended (verify before run):
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh -o install.sh
#   shasum -a 256 install.sh    # compare with published hash in docs/install.md
#   bash install.sh
set -euo pipefail

REPO_URL="${SEISO_REPO_URL:-https://github.com/Legendarylibr/SeisoLocalAI.git}"
INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
BRANCH="${SEISO_BRANCH:-main}"

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

install_tui_enabled() {
  local root="$1"
  [[ "${SEISO_NO_BANNER:-0}" == "1" ]] && return 1
  [[ "${SEISO_VERBOSE:-0}" == "1" ]] && return 1
  [[ -t 1 ]] || return 1
  [[ -f "$root/scripts/install_tui.py" ]] || return 1
  return 0
}

install_tui_intro() {
  local root="$1"
  install_tui_enabled "$root" || return 0
  python3 "$root/scripts/install_tui.py" intro
}

install_tui_outro() {
  local root="$1"
  install_tui_enabled "$root" || return 0
  python3 "$root/scripts/install_tui.py" outro
}

run_with_install_tui() {
  local root="$1" label="$2" logfile="$3"
  shift 3
  if install_tui_enabled "$root"; then
    "$@" >"$logfile" 2>&1 &
    local job_pid=$!
    if ! python3 "$root/scripts/install_tui.py" during --wait-pid "$job_pid" --log "$logfile" --label "$label"; then
      warn "$label failed — see $logfile"
      tail -30 "$logfile" >&2 || true
      return 1
    fi
    return 0
  fi
  "$@"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

python_version_ok() {
  python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

detect_platform_extras() {
  local os
  os="$(uname -s)"
  case "$os" in
    Darwin)
      echo "forge,train,mlx,dev"
      ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "forge,train,cuda,dev"
      else
        echo "forge,train,dev"
      fi
      ;;
    *)
      die "Unsupported OS: $os (use docs/platforms/windows.md on Windows)"
      ;;
  esac
}

resolve_root() {
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    local candidate
    candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ -f "$candidate/pyproject.toml" && -d "$candidate/seiso_cli" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  if [[ -d "$INSTALL_DIR/seiso_cli" && -f "$INSTALL_DIR/pyproject.toml" ]]; then
    printf '%s\n' "$INSTALL_DIR"
    return
  fi
  need_cmd git
  if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    log "Cloning Seiso into $INSTALL_DIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  else
    log "Updating existing clone in $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$INSTALL_DIR" checkout "$BRANCH"
    git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" || true
  fi
  printf '%s\n' "$INSTALL_DIR"
}

warn_windows_mount() {
  local root="$1"
  [[ "$root" == /mnt/* ]] || return 0
  cat <<EOF >&2

warning: Seiso is on a Windows drive mount ($root).
CUDA wheel builds (flash-attn) often fail there with pyproject.toml / setup.py errors.

Recommended: install on the Linux filesystem instead:
  SEISO_INSTALL_DIR=~/Seiso curl -fsSL .../scripts/install.sh | bash

EOF
}

maybe_install_flash_attn() {
  local root="$1"
  [[ "${SEISO_SKIP_FLASH_ATTN:-0}" == "1" ]] && {
    log "Skipping Flash Attention (SEISO_SKIP_FLASH_ATTN=1)"
    return 0
  }
  if [[ "$root" == /mnt/* ]]; then
    log "Skipping Flash Attention on Windows mount ($root)"
    log "Optional later: clone to ~/Seiso and run $root/scripts/install_flash_attn.sh"
    return 0
  fi
  if [[ -x "$root/scripts/install_flash_attn.sh" ]]; then
    log "Installing optional Flash Attention 2 (safe to skip on failure)"
    if ! bash "$root/scripts/install_flash_attn.sh"; then
      warn "Flash Attention install failed — Seiso will use PyTorch SDPA instead"
    fi
  fi
}

main() {
  local root extras venv_py py_ver install_log ui_log
  uname -s | grep -Eq '^(Linux|Darwin)$' || die "This installer supports Linux and macOS only"

  need_cmd python3
  python_version_ok || die "Python 3.10+ is required ($(python3 --version 2>&1 || echo unknown))"
  py_ver="$(python3 --version 2>&1 || true)"
  log "Using $py_ver"
  need_cmd git

  root="$(resolve_root)"
  install_tui_intro "$root"
  if ! install_tui_enabled "$root"; then
    log "Using repository at $root"
  fi
  warn_windows_mount "$root"

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    die "Node.js and npm are required to build the Forge UI. Install Node 18+ from https://nodejs.org/"
  fi

  extras="$(detect_platform_extras)"
  if ! install_tui_enabled "$root"; then
    log "Installing Python extras: [$extras]"
  fi

  venv_py="$root/.venv/bin/python"
  if [[ ! -x "$venv_py" ]]; then
    log "Creating virtualenv at $root/.venv"
    python3 -m venv "$root/.venv"
  fi

  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  python -m pip install -U pip wheel setuptools

  install_log="$root/.seiso-install.log"
  if ! run_with_install_tui "$root" "pip · seiso core" "$install_log" \
    pip install -e "${root}[${extras}]"; then
    die "Python install failed"
  fi

  if [[ "$extras" == *cuda* ]]; then
    maybe_install_flash_attn "$root"
  fi

  if [[ ! -f "$root/.env" && -f "$root/.env.example" ]]; then
    cp "$root/.env.example" "$root/.env"
    log "Created $root/.env from .env.example"
  fi

  if [[ "${SEISO_SKIP_UI:-0}" != "1" ]]; then
    ui_log="$root/.seiso-install-ui.log"
    if ! run_with_install_tui "$root" "npm · forge ui" "$ui_log" \
      bash -c "cd \"$root/forge-ui\" && npm install && npm run build"; then
      die "Forge UI build failed"
    fi
  elif ! install_tui_enabled "$root"; then
    log "Skipping Forge UI build (SEISO_SKIP_UI=1)"
  fi

  install_tui_outro "$root"

  cat <<EOF

Seiso is installed at: $root

Start Forge:
  $root/scripts/start.sh

Or manually:
  source $root/.venv/bin/activate
  seiso forge

Then open http://127.0.0.1:8765 and complete onboarding.

Platform guide: $root/docs/platforms/
EOF

  if [[ "${SEISO_START:-0}" == "1" ]]; then
    log "Starting Forge (SEISO_START=1)"
    exec "$root/scripts/start.sh"
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    cat <<EOF

Linux NVIDIA detected:
  - CUDA extras were installed ([cuda] — Triton; flash-attn is optional).
  - Optional Flash Attention: $root/scripts/install_flash_attn.sh
  - Before GPU training, set ONE of these in $root/.env:
      SEISO_NVIDIA_HOST_VENV_ACK=1   # bare-metal Linux
      SEISO_NVIDIA_WSL_ACK=1         # WSL2 only
  - See $root/docs/platforms/linux-nvidia.md

EOF
  fi
}

main "$@"
