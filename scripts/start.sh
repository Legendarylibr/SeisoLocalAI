#!/usr/bin/env bash
# Start Seiso Forge (API + web UI).
#
# Usage:
#   ~/Seiso/scripts/start.sh
#   SEISO_INSTALL_DIR=~/Seiso ./scripts/start.sh
#
# One-liner after install:
#   curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/start.sh | bash
set -euo pipefail

INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

resolve_root() {
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
    return
  fi
  if [[ -d "$INSTALL_DIR/seiso_cli" && -f "$INSTALL_DIR/pyproject.toml" ]]; then
    printf '%s\n' "$INSTALL_DIR"
    return
  fi
  die "Seiso not found. Run the installer first:
  curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh | bash"
}

main() {
  local root ui_dist
  root="$(resolve_root)"

  [[ -x "$root/.venv/bin/seiso" ]] || die "Virtualenv missing at $root/.venv — run $root/scripts/install.sh"

  ui_dist="$root/forge-ui/dist/index.html"
  if [[ ! -f "$ui_dist" ]]; then
    log "Forge UI not built — building now"
    if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
      (cd "$root/forge-ui" && npm install && npm run build)
    else
      die "Forge UI is not built and Node.js/npm are unavailable. Run: $root/scripts/install.sh"
    fi
  fi

  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"

  if [[ -f "$root/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$root/.env"
    set +a
  fi

  log "Starting Seiso Forge at http://${SEISO_HOST:-127.0.0.1}:${SEISO_PORT:-8765}"
  exec seiso forge
}

main "$@"
