#!/usr/bin/env bash
# Install Seiso on Linux or macOS — clone (if needed), venv, pip extras, Forge UI build.
#
# One-liner (Linux / macOS):
#   curl -fsSL https://raw.githubusercontent.com/seiso-ai/seiso/main/scripts/install.sh | bash
#
# Options (env vars):
#   SEISO_INSTALL_DIR   Install location (default: ~/Seiso)
#   SEISO_REPO_URL      Git remote (default: https://github.com/seiso-ai/seiso.git)
#   SEISO_BRANCH        Branch to clone (default: main)
#   SEISO_SKIP_UI=1     Skip forge-ui npm build
#   SEISO_START=1       Run scripts/start.sh when install finishes
set -euo pipefail

REPO_URL="${SEISO_REPO_URL:-https://github.com/seiso-ai/seiso.git}"
INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
BRANCH="${SEISO_BRANCH:-main}"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

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
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
    return
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

main() {
  local root extras venv_py
  uname -s | grep -Eq '^(Linux|Darwin)$' || die "This installer supports Linux and macOS only"

  need_cmd python3
  python_version_ok || die "Python 3.10+ is required ($(python3 --version 2>&1 || echo unknown))"
  need_cmd git

  root="$(resolve_root)"
  log "Using repository at $root"

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    die "Node.js and npm are required to build the Forge UI. Install Node 18+ from https://nodejs.org/"
  fi

  extras="$(detect_platform_extras)"
  log "Installing Python extras: [$extras]"

  venv_py="$root/.venv/bin/python"
  if [[ ! -x "$venv_py" ]]; then
    log "Creating virtualenv at $root/.venv"
    python3 -m venv "$root/.venv"
  fi

  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  python -m pip install -U pip wheel setuptools

  log "Installing Seiso (editable) — this may take several minutes"
  pip install -e "$root/.[$extras]"

  if [[ ! -f "$root/.env" && -f "$root/.env.example" ]]; then
    cp "$root/.env.example" "$root/.env"
    log "Created $root/.env from .env.example"
  fi

  if [[ "${SEISO_SKIP_UI:-0}" != "1" ]]; then
    log "Building Forge UI"
    (cd "$root/forge-ui" && npm install && npm run build)
  else
    log "Skipping Forge UI build (SEISO_SKIP_UI=1)"
  fi

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
}

main "$@"
