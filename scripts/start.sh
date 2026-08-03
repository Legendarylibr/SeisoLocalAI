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
  local raw_base="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" 2>/dev/null && pwd || echo "")"

log() { seiso_log "$@"; }
die() { seiso_die "$@"; }

sidecar_url_base() {
  local raw="$1"
  printf '%s\n' "${raw%/}"
}

sidecar_health_ok() {
  local url="$1" path="${2:-/health}"
  curl -fsS --max-time 1 "$(sidecar_url_base "$url")$path" >/dev/null 2>&1
}

wait_sidecar_health() {
  local url="$1" path="${2:-/health}" attempts="${3:-10}"
  local i
  for ((i = 0; i < attempts; i++)); do
    sidecar_health_ok "$url" "$path" && return 0
    sleep 0.5
  done
  return 1
}

native_linux_nvidia() {
  [[ "$(uname -s)" == "Linux" ]] || return 1
  seiso_is_wsl && return 1
  seiso_nvidia_gpu_detected
}

start_background_sidecar() {
  local name="$1" log_file="$2"
  shift 2
  mkdir -p "$(dirname "$log_file")"
  nohup "$@" >>"$log_file" 2>&1 &
  printf '%s\n' "$!" >"${log_file%.log}.pid"
}

preferred_sidecar_engine() {
  local override
  override="$(printf '%s' "${SEISO_LLAMASWAP_ENGINE:-auto}" | tr '[:upper:]' '[:lower:]')"
  if [[ -n "$override" && "$override" != "auto" ]]; then
    printf '%s\n' "$override"
    return
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    printf 'llamacpp\n'
    return
  fi
  if native_linux_nvidia && sidecar_health_ok "${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}" "/api/tags"; then
    printf 'ollama\n'
    return
  fi
  printf 'llamacpp\n'
}

ensure_inference_sidecars() {
  [[ "${SEISO_SIDECAR_AUTOSTART:-1}" == "1" ]] || return 0

  local run_dir="${SEISO_DATA_DIR:-$HOME/.seiso}/run"
  local ollama_url="${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}"
  local swap_url="${SEISO_LLAMASWAP_URL:-http://127.0.0.1:8080}"
  local needs_swap=0

  if native_linux_nvidia && [[ "${SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX:-0}" != "1" ]]; then
    needs_swap=1
    export SEISO_LLAMASWAP_ENABLED="${SEISO_LLAMASWAP_ENABLED:-true}"
    export SEISO_LLAMASWAP_URL="$swap_url"
  elif [[ "${SEISO_LLAMASWAP_ENABLED:-}" == "true" || -n "${SEISO_LLAMASWAP_URL:-}" ]]; then
    needs_swap=1
    export SEISO_LLAMASWAP_URL="$swap_url"
  fi

  [[ "$needs_swap" == "1" ]] || return 0

  if native_linux_nvidia && [[ "${SEISO_LLAMASWAP_ENGINE:-auto}" =~ ^(auto|ollama)?$ ]]; then
    if ! sidecar_health_ok "$ollama_url" "/api/tags"; then
      if command -v ollama >/dev/null 2>&1; then
        log "Starting Ollama sidecar at $ollama_url"
        if declare -F seiso_export_ollama_cli_env >/dev/null 2>&1; then
          seiso_export_ollama_cli_env "$ollama_url"
        fi
        if declare -F seiso_export_ollama_server_env >/dev/null 2>&1; then
          seiso_export_ollama_server_env
        else
          export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
          export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
          export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
          export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
        fi
        start_background_sidecar "ollama" "$run_dir/ollama.log" ollama serve
        wait_sidecar_health "$ollama_url" "/api/tags" 60 || true
      else
        log "Ollama not found; llama-swap will use llama.cpp engine if configured"
      fi
    fi
  fi

  local engine
  engine="$(preferred_sidecar_engine)"
  export SEISO_LLAMASWAP_ENGINE="${SEISO_LLAMASWAP_ENGINE:-$engine}"

  if sidecar_health_ok "$ollama_url" "/api/tags"; then
    log "Ollama sidecar is ready at $ollama_url (engine=${SEISO_LLAMASWAP_ENGINE})"
    return 0
  fi

  if sidecar_health_ok "$swap_url" "/health"; then
    log "llama-swap fallback is ready at $swap_url"
    return 0
  fi

  if ! command -v llama-swap >/dev/null 2>&1; then
    log "llama-swap is not installed; install Ollama or llama-swap for native Linux NVIDIA GGUF chat"
    return 0
  fi

  local -a cmd=(llama-swap)
  if [[ -n "${SEISO_LLAMASWAP_CONFIG:-}" ]]; then
    cmd+=(--config "$SEISO_LLAMASWAP_CONFIG")
  fi
  log "Starting llama-swap sidecar at $swap_url (engine=${SEISO_LLAMASWAP_ENGINE})"
  start_background_sidecar "llama-swap" "$run_dir/llama-swap.log" "${cmd[@]}"
  if ! wait_sidecar_health "$swap_url" "/health" 12; then
    log "llama-swap did not become ready yet; check $run_dir/llama-swap.log or set SEISO_LLAMASWAP_URL/SEISO_LLAMASWAP_CONFIG"
  fi
}

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

  load_seiso_sidecar_install "$root"

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

  # Default path: Nostr provenance gate on + public digests-only relays.
  # Override with SEISO_ALLOW_NOSTR=0 or SEISO_NOSTR_RELAYS=... Auto-attest stays off.
  export SEISO_ALLOW_NOSTR="${SEISO_ALLOW_NOSTR:-1}"
  export SEISO_NOSTR_RELAYS="${SEISO_NOSTR_RELAYS:-wss://nos.lol,wss://relay.damus.io}"

  seiso_bin="$(seiso_require_cli "$root")"
  ensure_inference_sidecars
  if declare -F seiso_verify_sidecar_stack >/dev/null 2>&1; then
    seiso_verify_sidecar_stack
  fi

  # Optional multi-GPU path (off by default). When enabled, Forge may autostart
  # managed vLLM after boot — never replaces Ollama/llama-swap GGUF sidecars.
  if [[ "${SEISO_MANAGED_VLLM_ENABLED:-0}" == "1" || "${SEISO_MANAGED_VLLM_ENABLED:-}" == "true" ]]; then
    if [[ "${SEISO_MANAGED_VLLM_AUTOSTART:-0}" == "1" || "${SEISO_MANAGED_VLLM_AUTOSTART:-}" == "true" ]]; then
      if [[ -n "${SEISO_MANAGED_VLLM_MODEL:-}" ]]; then
        log "Optional managed multi-GPU vLLM autostart enabled (model=${SEISO_MANAGED_VLLM_MODEL})"
      else
        log "SEISO_MANAGED_VLLM_AUTOSTART set but SEISO_MANAGED_VLLM_MODEL empty — skipping"
      fi
    else
      log "Managed multi-GPU vLLM enabled (start from Integrations or API; not auto-started)"
    fi
  fi

  forge_url="$(seiso_forge_url)"

  # Already running (healthy or still in lifespan / holding the data-dir lock):
  # open browser and exit instead of failing on the instance lock.
  if seiso_forge_instance_active "$forge_url"; then
    log "Forge is already running at $forge_url"
    if [[ "${SEISO_NO_OPEN:-0}" != "1" ]]; then
      seiso_open_browser "$forge_url" || true
    fi
    return 0
  fi

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
