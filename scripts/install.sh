#!/usr/bin/env bash
# Install Seiso on Linux or macOS — clone (if needed), venv, pip extras, Forge UI build.
#
# One-liner (Linux / macOS) — installs and starts Forge:
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash
#
# Options (env vars):
#   SEISO_INSTALL_DIR   Install location (default: ~/Seiso)
#   SEISO_REPO_URL      Git remote (default: https://github.com/Legendarylibr/SeisoLocalAI.git)
#   SEISO_BRANCH        Branch to clone (default: main)
#   SEISO_START=0       Install only — do not launch Forge when finished (default: start)
#   SEISO_SKIP_UI=1     Skip forge-ui npm build
#   SEISO_NO_BANNER=1   Skip glitch install TUI
#   SEISO_VERBOSE=1     Show full pip/npm output (no TUI overlay)
#   SEISO_SKIP_FLASH_ATTN=0  Try optional Flash Attention during install (NVIDIA Linux)
set -euo pipefail

REPO_URL="${SEISO_REPO_URL:-https://github.com/Legendarylibr/SeisoLocalAI.git}"
INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
BRANCH="${SEISO_BRANCH:-main}"
SEISO_START="${SEISO_START:-1}"
SEISO_SKIP_FLASH_ATTN="${SEISO_SKIP_FLASH_ATTN:-1}"
FORGE_URL="http://127.0.0.1:8765"

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

quiet_install_output() {
  [[ "${SEISO_VERBOSE:-0}" == "1" ]] && return 1
  [[ "${SEISO_NO_BANNER:-0}" == "1" ]] && return 1
  [[ -t 1 ]] || return 1
  return 0
}

log_unless_quiet() {
  quiet_install_output || log "$@"
}

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
  local root="$1" logfile="$2"
  shift 2
  if install_tui_enabled "$root"; then
    : >"$logfile"
    "$@" >>"$logfile" 2>&1 &
    local job_pid=$!
    if ! python3 "$root/scripts/install_tui.py" during --wait-pid "$job_pid"; then
      warn "Install failed — see $logfile"
      tail -30 "$logfile" >&2 || true
      return 1
    fi
    return 0
  fi
  "$@"
}

pre_clone_hint() {
  quiet_install_output || return 0
  printf '\n\033[1mSeisoLocalAI\033[0m · fetching repository...\n\n'
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
    pre_clone_hint
    if quiet_install_output; then
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" >/dev/null 2>&1
    else
      log "Cloning Seiso into $INSTALL_DIR"
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi
  else
    log_unless_quiet "Updating existing clone in $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1
    git -C "$INSTALL_DIR" checkout "$BRANCH" >/dev/null 2>&1
    git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" >/dev/null 2>&1 || true
  fi
  printf '%s\n' "$INSTALL_DIR"
}

warn_windows_mount() {
  local root="$1"
  [[ "$root" == /mnt/* ]] || return 0
  warn "Install path is on a Windows mount ($root). Use SEISO_INSTALL_DIR=~/Seiso on WSL."
}

run_install_worker() {
  local root="$1" extras="$2"
  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  python -m pip install -U pip wheel setuptools
  pip install -e "${root}[${extras}]"
  if [[ "${SEISO_SKIP_FLASH_ATTN}" != "1" && "$extras" == *cuda* && "$root" != /mnt/* ]]; then
    if [[ -x "$root/scripts/install_flash_attn.sh" ]]; then
      bash "$root/scripts/install_flash_attn.sh" || true
    fi
  fi
  if [[ "${SEISO_SKIP_UI:-0}" != "1" ]]; then
    (cd "$root/forge-ui" && npm ci && npm run build)
  fi
}

main() {
  local root extras install_log
  uname -s | grep -Eq '^(Linux|Darwin)$' || die "This installer supports Linux and macOS only"

  need_cmd python3
  python_version_ok || die "Python 3.10+ is required ($(python3 --version 2>&1 || echo unknown))"
  need_cmd git

  root="$(resolve_root)"
  install_tui_intro "$root"
  log_unless_quiet "Using repository at $root"
  warn_windows_mount "$root"

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    die "Node.js 18+ is required — install from https://nodejs.org/ then re-run this script"
  fi

  extras="$(detect_platform_extras)"
  log_unless_quiet "Installing Python extras: [$extras]"

  install_log="$root/.seiso-install.log"
  if [[ ! -x "$root/.venv/bin/python" ]]; then
    log_unless_quiet "Creating virtualenv at $root/.venv"
    python3 -m venv "$root/.venv"
  fi

  if [[ ! -f "$root/.env" && -f "$root/.env.example" ]]; then
    cp "$root/.env.example" "$root/.env"
    log_unless_quiet "Created $root/.env from .env.example"
  fi

  export SEISO_SKIP_UI="${SEISO_SKIP_UI:-0}"
  export SEISO_SKIP_FLASH_ATTN

  if ! run_with_install_tui "$root" "$install_log" \
    bash -c "$(declare -f run_install_worker); run_install_worker \"$root\" \"$extras\""; then
    die "Install failed. Run $root/scripts/doctor.sh for a guided diagnosis."
  fi

  install_tui_outro "$root"

  if [[ "$SEISO_START" == "1" ]]; then
    export SEISO_INSTALL_JUST_RAN=1
    exec "$root/scripts/start.sh"
  fi

  if install_tui_enabled "$root"; then
    printf '\n%s\nDoctor: %s/scripts/doctor.sh\n\n' "$FORGE_URL" "$root"
  else
    printf '\nOpen %s and complete onboarding.\n\n' "$FORGE_URL"
    printf 'Model storage: GGUF downloads go to ~/.seiso/hf_cache and load with llama.cpp.\n'
    printf 'Typical GGUF downloads are 2-8 GB each; larger models can be 10-30+ GB.\n'
    printf 'Ollama uses its own store — use ollama pull/create for Ollama models.\n'
    printf 'Need help? Run: %s/scripts/doctor.sh\n\n' "$root"
  fi
}

main "$@"
