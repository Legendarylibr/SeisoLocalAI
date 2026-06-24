#!/usr/bin/env bash
# Start the vLLM Smart Router stack: Nemotron orchestrator + vLLM specialists +
# llama-swap + LiteLLM-routed Seiso router (Docker).
#
# Usage (from repo root or on PATH after install):
#   start-router-vllm
#   start-router-vllm -d          # detached
#   SEISO_ROUTER_STACK_DETACHED=1 start-router-vllm
#
# Router endpoint: http://127.0.0.1:8780/v1/chat/completions
# Enable in Forge Chat: SEISO_MODEL_ROUTER_ENABLED=true (see .env.example)
set -euo pipefail

INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
COMPOSE_FILE="docker-compose.local.vllm.yml"

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
  return 1
}

resolve_root() {
  if load_seiso_common && root="$(seiso_resolve_repo_for_start "${BASH_SOURCE[0]:-}")"; then
    printf '%s\n' "$root"
    return 0
  fi
  if [[ -d "$INSTALL_DIR/deploy/model-router" && -f "$INSTALL_DIR/pyproject.toml" ]]; then
    printf '%s\n' "$INSTALL_DIR"
    return 0
  fi
  return 1
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

main() {
  local root router_dir detach_args=()

  command -v docker >/dev/null 2>&1 || die "docker is required for the vLLM router stack"

  root="$(resolve_root)" || die "Seiso repo not found. Clone to \$HOME/Seiso or set SEISO_INSTALL_DIR."

  router_dir="$root/deploy/model-router"
  [[ -f "$router_dir/$COMPOSE_FILE" ]] || die "missing $router_dir/$COMPOSE_FILE"

  if [[ "${SEISO_ROUTER_STACK_DETACHED:-0}" == "1" ]]; then
    detach_args=(-d)
  fi
  for arg in "$@"; do
    if [[ "$arg" == "-d" || "$arg" == "--detach" ]]; then
      detach_args=(-d)
    fi
  done

  log "Starting vLLM Smart Router stack (Nemotron + LiteLLM + vLLM specialists)"
  log "Router will listen on http://127.0.0.1:8780 when healthy"
  log "Compose: $router_dir/$COMPOSE_FILE"
  printf '\n' >&2
  printf 'Forge integration (.env):\n' >&2
  printf '  SEISO_MODEL_ROUTER_ENABLED=true\n' >&2
  printf '  SEISO_MODEL_ROUTER_URL=http://127.0.0.1:8780\n' >&2
  printf '\n' >&2

  exec docker compose -f "$COMPOSE_FILE" up --build "${detach_args[@]}" "$@"
}

main "$@"