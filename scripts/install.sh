#!/usr/bin/env bash
# Install Seiso on Linux or macOS — system deps, clone, venv, pip extras, Forge UI build.
#
# Prefer start (on PATH after install, or curl …/start | bash). This script is the lower-level installer.
#
# One-liner (Linux / macOS) — installs and starts Forge (opens browser when ready):
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
#
# Options (env vars):
#   SEISO_INSTALL_DIR   Install location (default: ~/Seiso)
#   SEISO_REPO_URL      Git remote (default: https://github.com/Legendarylibr/SeisoLocalAI.git)
#   SEISO_BRANCH        Branch to clone (default: main)
#   SEISO_START=0       Install only — do not launch Forge when finished (default: start)
#   SEISO_SKIP_UI=1     Skip forge-ui npm build
#   SEISO_NO_BANNER=1   Skip glitch install TUI
#   SEISO_VERBOSE=1     Show full pip/npm output (no TUI overlay)
#   SEISO_NO_OPEN=1     Do not open the browser after Forge starts
#   SEISO_SKIP_FLASH_ATTN=0  Try optional Flash Attention during install (NVIDIA Linux)
set -euo pipefail

REPO_URL="${SEISO_REPO_URL:-https://github.com/Legendarylibr/SeisoLocalAI.git}"
INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
BRANCH="${SEISO_BRANCH:-main}"
SEISO_START="${SEISO_START:-1}"
SEISO_SKIP_FLASH_ATTN="${SEISO_SKIP_FLASH_ATTN:-1}"

load_seiso_common() {
  local lib_path=""
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    lib_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
  fi
  if [[ -f "$lib_path" ]]; then
    # shellcheck source=lib/common.sh
    source "$lib_path"
    return 0
  fi
  local raw_base="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/${SEISO_BRANCH:-main}}"
  local tmp
  tmp="$(mktemp)"
  if ! curl -fsSL "${raw_base}/scripts/lib/common.sh" -o "$tmp"; then
    rm -f "$tmp"
    printf 'error: could not load install helpers from %s\n' "$raw_base" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$tmp"
  rm -f "$tmp"
}

load_seiso_common
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" 2>/dev/null && pwd || echo "")"

log() { seiso_log "$@"; }
warn() { seiso_warn "$@"; }
die() { seiso_die "$@"; }

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
  python3 "$root/scripts/install_tui.py" outro --url "$FORGE_URL"
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
  printf '\n\033[1mSeisoLocalAI\033[0m · fetching repository...\n\n' >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

detect_platform_extras() {
  seiso_detect_platform_extras
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
      log "Cloning Seiso into $INSTALL_DIR" >&2
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi
  else
    log_unless_quiet "Updating existing clone in $INSTALL_DIR" >&2
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
  seiso_run_install_worker "$1" "$2"
}

install_failed() {
  local root="$1"
  warn "Install failed."
  seiso_run_doctor "$root"
  exit 1
}

main() {
  local root extras install_log
  uname -s | grep -Eq '^(Linux|Darwin)$' || die "This installer supports Linux and macOS only"

  log_unless_quiet "Checking system dependencies"
  seiso_require_system_deps

  root="$(resolve_root)"
  install_tui_intro "$root"
  log_unless_quiet "Using repository at $root"
  warn_windows_mount "$root"

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
    bash -c "source \"$root/scripts/lib/common.sh\"; seiso_run_install_worker \"$root\" \"$extras\""; then
    install_failed "$root"
  fi

  seiso_install_start_command "$root"

  install_tui_outro "$root"

  if [[ "$SEISO_START" == "1" ]]; then
    export SEISO_INSTALL_JUST_RAN=1
    export SEISO_OPEN_BROWSER=1
    exec "$root/scripts/start.sh"
  fi

  local forge_url
  forge_url="$(seiso_forge_url)"
  if install_tui_enabled "$root"; then
    printf '\n%s\nDoctor: %s/scripts/doctor.sh\n\n' "$forge_url" "$root"
  else
    printf '\nStart Forge: start\n'
    printf 'Doctor: %s/scripts/doctor.sh\n\n' "$root"
  fi
}

main "$@"
