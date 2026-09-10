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
#   SEISO_SKIP_UI=1     Skip forge-ui build
#   SEISO_USE_NPM=1     Force npm for forge-ui (Linux prefers npm when Node 18+ is present)
#   SEISO_USE_BUN=1     On Linux, prefer Bun even when npm is available
#   SEISO_USE_UV=0      Use pip instead of uv for Python deps (uv is default when available)
#   SEISO_NO_BANNER=1   Skip install banner / progress output
#   SEISO_VERBOSE=1     Show full pip/UI package manager output
#   SEISO_NO_OPEN=1     Do not open the browser after Forge starts
#   SEISO_SKIP_FLASH_ATTN=1 Skip Flash Attention during install (default: skip; set 0 to try)
#   SEISO_FAST_INSTALL=1    Skip PyTorch/training extras (Forge + GGUF chat only)
#   SEISO_INSTALL_PROFILE=… Target install: linux-nvidia, linux-cpu, linux-rocm, wsl-nvidia, macos, chat
#   SEISO_INSTALL_DEV=1     Include dev extras (pytest, ruff, mypy, …)
#   SEISO_INSTALL_EXTRAS=…  Override auto-detected pip extras (e.g. forge,train,cuda)
#   SEISO_REQUIRE_SIDECAR=1 Hard-fail install/start if Ollama/llama-swap missing (default: soft warn)
#   SEISO_GIT_PULL=1        On start/reinstall of an existing clone, git pull --ff-only
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

load_seiso_sidecar_install() {
  local root="${1:-}"
  local lib_path=""
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    lib_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/sidecar_install.sh"
  fi
  if [[ -f "$lib_path" ]]; then
    # shellcheck source=lib/sidecar_install.sh
    source "$lib_path"
    return 0
  fi
  if [[ -n "$root" && -f "$root/scripts/lib/sidecar_install.sh" ]]; then
    # shellcheck source=lib/sidecar_install.sh
    source "$root/scripts/lib/sidecar_install.sh"
    return 0
  fi
  local raw_base="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/${SEISO_BRANCH:-main}}"
  local tmp
  tmp="$(mktemp)"
  if ! curl -fsSL "${raw_base}/scripts/lib/sidecar_install.sh" -o "$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  # shellcheck source=/dev/null
  source "$tmp"
  rm -f "$tmp"
}

load_seiso_sidecar_install

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

run_install_prep_and_worker() {
  quiet_install_output || return 0
  printf '\n\033[1mSeisoLocalAI\033[0m · fetching repository...\n\n' >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

detect_platform_extras() {
  seiso_detect_platform_extras
}

repo_layout_complete() {
  local root="$1"
  [[ -f "$root/pyproject.toml" && -d "$root/seiso_cli" && -f "$root/scripts/lib/common.sh" ]]
}

assert_repo_layout() {
  local root="$1"
  repo_layout_complete "$root" || {
    die "Seiso repository incomplete at $root. Remove the directory or set SEISO_INSTALL_DIR elsewhere, then re-run start."
  }
}

sync_install_clone() {
  git -C "$INSTALL_DIR" fetch --depth 1 origin "$BRANCH" >/dev/null 2>&1 || true
  git -C "$INSTALL_DIR" checkout -f "$BRANCH" >/dev/null 2>&1 || true
  git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH" >/dev/null 2>&1 \
    || git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" >/dev/null 2>&1 || true
}

resolve_root() {
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    local candidate
    candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if repo_layout_complete "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  if repo_layout_complete "$INSTALL_DIR"; then
    printf '%s\n' "$INSTALL_DIR"
    return
  fi
  need_cmd git
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log_unless_quiet "Repairing existing clone in $INSTALL_DIR" >&2
    sync_install_clone
  elif [[ -e "$INSTALL_DIR" ]]; then
    die "Seiso repository incomplete at $INSTALL_DIR. Remove the directory or set SEISO_INSTALL_DIR elsewhere, then re-run start."
  else
    pre_clone_hint
    if quiet_install_output; then
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" >/dev/null 2>&1 \
        || die "git clone failed into $INSTALL_DIR. Check network access and that the directory is empty or missing."
    else
      log "Cloning Seiso into $INSTALL_DIR" >&2
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" \
        || die "git clone failed into $INSTALL_DIR. Check network access and that the directory is empty or missing."
    fi
  fi
  assert_repo_layout "$INSTALL_DIR"
  printf '%s\n' "$INSTALL_DIR"
}

warn_windows_mount() {
  local root="$1"
  if seiso_is_wsl && [[ "$root" == /mnt/* ]]; then
    warn "WSL install path is on a Windows mount ($root). Use SEISO_INSTALL_DIR=~/Seiso on the Linux filesystem for CUDA and native builds."
    return 0
  fi
  [[ "$root" == /mnt/* ]] || return 0
  warn "Install path is on a Windows mount ($root). Use SEISO_INSTALL_DIR=~/Seiso on WSL."
}

run_install_worker() {
  seiso_run_install_worker "$1" "$2"
}

run_install_prep_and_worker() {
  local root="$1" extras="$2"
  if [[ ! -x "$root/.venv/bin/python" ]]; then
    log_unless_quiet "Creating virtualenv at $root/.venv"
    python3 -m venv "$root/.venv"
  fi

  if [[ ! -f "$root/.env" && -f "$root/.env.example" ]]; then
    cp "$root/.env.example" "$root/.env"
    log_unless_quiet "Created $root/.env from .env.example"
  fi

  run_install_worker "$root" "$extras"
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
  load_seiso_sidecar_install "$root"
  if declare -F seiso_warn_native_linux_nvidia_banner >/dev/null 2>&1; then
    seiso_warn_native_linux_nvidia_banner
  fi

  log_unless_quiet "Using repository at $root"
  warn_windows_mount "$root"

  extras="$(detect_platform_extras)"
  if [[ -n "${SEISO_INSTALL_PROFILE:-}" ]]; then
    log_unless_quiet "Install profile: ${SEISO_INSTALL_PROFILE} → [$extras]"
  else
    log_unless_quiet "Installing Python extras: [$extras]"
  fi
  # Sidecar hard-fail is opt-in (SEISO_REQUIRE_SIDECAR=1). Default is soft-warn so a
  # missing Ollama install cannot abort a completed Python/UI install.

  install_log="$root/.seiso-install.log"

  export SEISO_SKIP_UI="${SEISO_SKIP_UI:-0}"
  export SEISO_SKIP_FLASH_ATTN

  if ! run_install_prep_and_worker "$root" "$extras" >"$install_log" 2>&1; then
    install_failed "$root"
  fi

  if ! seiso_verify_cli "$root"; then
    warn "Install finished but Seiso CLI is missing — see $install_log"
    tail -30 "$install_log" >&2 || true
    install_failed "$root"
  fi

  if [[ "$extras" == *cuda* || "$extras" == *llamacpp* ]]; then
    seiso_repair_linux_cuda_stack "$root" || true
    if [[ "$extras" == *llamacpp* ]]; then
      seiso_verify_cuda_inference_stack "$root" || true
    fi
  fi

  seiso_install_start_command "$root"

  if declare -F seiso_run_sidecar_install_phase >/dev/null 2>&1; then
    seiso_run_sidecar_install_phase "$root"
  fi

  if [[ "$SEISO_START" == "1" ]]; then
    export SEISO_INSTALL_JUST_RAN=1
    exec "$root/scripts/start.sh"
  fi

  if [[ "$SEISO_START" != "1" ]]; then
    printf '\nInstall complete.\nStart Seiso: start\nDoctor: %s/scripts/doctor.sh\n\n' "$root"
  fi
}

main "$@"
