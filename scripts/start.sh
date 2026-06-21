#!/usr/bin/env bash
# Start Seiso Forge (API + web UI). Installs missing deps, runs doctor on failure,
# and opens the browser when Forge is ready.
#
# Usage (prefer start on PATH after install):
#   start
#   SEISO_INSTALL_DIR=~/Seiso ./scripts/start.sh
#
# One-liner (installs if needed, then starts):
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
set -euo pipefail

INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
INSTALL_URL="${SEISO_INSTALL_URL:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh}"

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
  local raw_base="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"
  local tmp
  tmp="$(mktemp)"
  if ! curl -fsSL "${raw_base}/scripts/lib/common.sh" -o "$tmp"; then
    rm -f "$tmp"
    printf 'error: could not load start helpers from %s\n' "$raw_base" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$tmp"
  rm -f "$tmp"
}

load_seiso_common
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" 2>/dev/null && pwd || echo "")"

log() { seiso_log "$@"; }
die() { seiso_die "$@"; }

resolve_root() {
  if root="$(seiso_resolve_repo_for_start "${BASH_SOURCE[0]:-}")"; then
    printf '%s\n' "$root"
    return 0
  fi
  return 1
}

bootstrap_install() {
  log "Seiso not found — running full installer"
  if [[ -f "${SCRIPT_DIR}/install.sh" ]]; then
    SEISO_INSTALL_DIR="$INSTALL_DIR" SEISO_START=0 bash "${SCRIPT_DIR}/install.sh"
    return
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to bootstrap install"
  SEISO_INSTALL_DIR="$INSTALL_DIR" SEISO_START=0 bash -c "$(curl -fsSL "$INSTALL_URL")"
}

main() {
  local root forge_url open_flag seiso_bin

  if ! root="$(resolve_root)"; then
    bootstrap_install
    root="$(resolve_root)" || die "Install did not complete. Run: curl -fsSL $INSTALL_URL | bash"
  fi

  if ! seiso_ensure_installed "$root"; then
    die "Could not complete install. See doctor output above."
  fi

  seiso_install_start_command "$root"

  export PATH="$root/.venv/bin:${PATH}"

  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"

  if [[ -f "$root/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$root/.env"
    set +a
    export PATH="$root/.venv/bin:${PATH}"
  fi

  seiso_bin="$(seiso_require_cli "$root")"

  forge_url="$(seiso_forge_url)"
  open_flag=""
  if [[ "${SEISO_NO_OPEN:-0}" != "1" ]]; then
    open_flag="--open"
  fi

  if [[ "${SEISO_INSTALL_JUST_RAN:-0}" == "1" ]]; then
    log "Starting Forge — opening $forge_url when ready"
  else
    log "Starting Forge at $forge_url"
  fi

  if [[ -n "$open_flag" ]]; then
    "$seiso_bin" forge $open_flag
  else
    exec "$seiso_bin" forge
  fi
}

main "$@"
