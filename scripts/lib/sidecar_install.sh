#!/usr/bin/env bash
# Sidecar stack install + health checks for native Linux NVIDIA GGUF chat.
# shellcheck shell=bash

seiso_sidecar_url_base() {
  local raw="$1"
  printf '%s\n' "${raw%/}"
}

seiso_ollama_health_ok() {
  local url="${1:-${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}}"
  curl -fsS --max-time 1 "$(seiso_sidecar_url_base "$url")/api/tags" >/dev/null 2>&1
}

seiso_llamaswap_health_ok() {
  local url="${1:-${SEISO_LLAMASWAP_URL:-http://127.0.0.1:8080}}"
  curl -fsS --max-time 1 "$(seiso_sidecar_url_base "$url")/health" >/dev/null 2>&1
}

seiso_wait_ollama_health() {
  local url="${1:-${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}}" attempts="${2:-60}"
  local i
  for ((i = 0; i < attempts; i++)); do
    seiso_ollama_health_ok "$url" && return 0
    sleep 0.5
  done
  return 1
}

seiso_ollama_cli_host() {
  local url="${1:-${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}}"
  url="$(seiso_sidecar_url_base "$url")"
  url="${url#http://}"
  url="${url#https://}"
  url="${url%%/*}"
  printf '%s\n' "$url"
}

seiso_export_ollama_cli_env() {
  local url="${1:-${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}}"
  export OLLAMA_HOST="$(seiso_ollama_cli_host "$url")"
}

# Long-context defaults for the Ollama server (user overrides win):
# flash attention keeps large contexts efficient, q8_0 halves KV memory vs f16,
# one parallel slot keeps the whole context budget for one chat, and one loaded
# model prevents multiple resident GGUFs from stacking VRAM on consumer GPUs.
seiso_export_ollama_server_env() {
  export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
  export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
}

seiso_native_linux_nvidia() {
  [[ "$(uname -s)" == "Linux" ]] || return 1
  seiso_is_wsl && return 1
  seiso_nvidia_gpu_detected
}

seiso_should_require_sidecar() {
  # Hard-fail only when the user explicitly sets SEISO_REQUIRE_SIDECAR=1.
  # Profiles no longer force a hard fail — missing Ollama/llama-swap should not
  # abort a long successful pip install; chat shows setup guidance instead.
  [[ "${SEISO_SIDECAR_OPTIONAL:-0}" == "1" ]] && return 1
  [[ "${SEISO_REQUIRE_SIDECAR:-0}" == "1" ]] && return 0
  return 1
}

seiso_start_background_sidecar() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  nohup "$@" >>"$log_file" 2>&1 &
  printf '%s\n' "$!" >"${log_file%.log}.pid"
}

seiso_install_ollama() {
  [[ "${SEISO_SKIP_OLLAMA_INSTALL:-0}" == "1" ]] && return 0

  local ollama_url="${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}"
  local run_dir="${SEISO_DATA_DIR:-$HOME/.seiso}/run"

  if seiso_ollama_health_ok "$ollama_url"; then
    seiso_log "Ollama is already running at $ollama_url"
    return 0
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    if [[ "$(uname -s)" != "Linux" ]]; then
      seiso_warn "Ollama is not installed — install from https://ollama.com/download"
      return 1
    fi
    seiso_log "Installing Ollama (official installer)"
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  command -v ollama >/dev/null 2>&1 || {
    seiso_warn "Ollama install did not place ollama on PATH"
    return 1
  }

  if ! seiso_ollama_health_ok "$ollama_url"; then
    seiso_log "Starting Ollama at $ollama_url"
    seiso_export_ollama_cli_env "$ollama_url"
    seiso_export_ollama_server_env
    seiso_start_background_sidecar "$run_dir/ollama.log" ollama serve
  fi

  if seiso_wait_ollama_health "$ollama_url" 60; then
    seiso_log "Ollama is ready at $ollama_url"
    return 0
  fi

  seiso_warn "Ollama did not become ready — check $run_dir/ollama.log"
  return 1
}

seiso_seed_sidecar_env() {
  local root="$1"
  local env_file="$root/.env"
  [[ -f "$env_file" ]] || touch "$env_file"

  seiso_env_set_default() {
    local key="$1" value="$2"
    if grep -q "^${key}=" "$env_file" 2>/dev/null; then
      return 0
    fi
    printf '%s=%s\n' "$key" "$value" >>"$env_file"
  }

  seiso_env_set_default "SEISO_SIDECAR_AUTOSTART" "1"
  seiso_env_set_default "SEISO_LLAMASWAP_ENABLED" "true"
  seiso_env_set_default "SEISO_LLAMASWAP_URL" "http://127.0.0.1:8080"
  seiso_env_set_default "SEISO_OLLAMA_URL" "http://127.0.0.1:11434"
  seiso_env_set_default "SEISO_LLAMASWAP_ENGINE" "auto"
  # Soft by default: Forge starts even if Ollama install fails (no sudo / offline).
  # Set SEISO_REQUIRE_SIDECAR=1 explicitly to hard-fail install/start.
  seiso_env_set_default "OLLAMA_FLASH_ATTENTION" "1"
  seiso_env_set_default "OLLAMA_KV_CACHE_TYPE" "q8_0"
  seiso_env_set_default "OLLAMA_NUM_PARALLEL" "1"
  seiso_env_set_default "OLLAMA_MAX_LOADED_MODELS" "1"
}

seiso_sidecar_fallback_config_path() {
  local data_dir="${SEISO_DATA_DIR:-$HOME/.seiso}"
  printf '%s\n' "$data_dir/llama-swap/config.yaml"
}

seiso_write_sidecar_fallback_config() {
  local config_path
  config_path="$(seiso_sidecar_fallback_config_path)"
  mkdir -p "$(dirname "$config_path")"
  cat >"$config_path" <<'EOF'
# Seiso fallback llama-swap config (used when Ollama is unavailable).
# Model IDs from Forge are absolute GGUF paths; add entries via seiso after download.
globalTTL: 600
models: {}
EOF
  printf '%s\n' "$config_path"
}

seiso_install_sidecar_fallback() {
  local bin_dir asset arch url tmp config_path
  bin_dir="$(seiso_start_bin_dir)"
  mkdir -p "$bin_dir"
  seiso_ensure_bin_on_path "$bin_dir"

  config_path="$(seiso_write_sidecar_fallback_config)"
  seiso_log "Wrote llama-swap fallback config at $config_path"

  if command -v llama-swap >/dev/null 2>&1; then
    return 0
  fi

  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) asset="llama-swap_Linux_x86_64.tar.gz" ;;
    aarch64|arm64) asset="llama-swap_Linux_arm64.tar.gz" ;;
    *)
      seiso_warn "Skipping llama-swap auto-install on unsupported arch: $arch"
      return 0
      ;;
  esac

  url="https://github.com/mostlygeek/llama-swap/releases/latest/download/${asset}"
  tmp="$(mktemp -d)"
  if curl -fsSL "$url" -o "$tmp/archive.tar.gz" 2>/dev/null; then
    tar -xzf "$tmp/archive.tar.gz" -C "$tmp" 2>/dev/null || true
    if [[ -f "$tmp/llama-swap" ]]; then
      install -m 0755 "$tmp/llama-swap" "$bin_dir/llama-swap"
      seiso_log "Installed llama-swap to $bin_dir/llama-swap"
    else
      seiso_warn "llama-swap archive did not contain a binary — install manually from https://github.com/mostlygeek/llama-swap"
    fi
  else
    seiso_warn "Could not download llama-swap — install manually for Ollama-down fallback"
  fi
  rm -rf "$tmp"
}

seiso_sidecar_stack_ready() {
  local ollama_url="${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}"
  local swap_url="${SEISO_LLAMASWAP_URL:-http://127.0.0.1:8080}"

  if seiso_ollama_health_ok "$ollama_url"; then
    return 0
  fi
  if seiso_llamaswap_health_ok "$swap_url"; then
    return 0
  fi
  return 1
}

seiso_verify_sidecar_stack() {
  local ollama_url="${SEISO_OLLAMA_URL:-http://127.0.0.1:11434}"
  local swap_url="${SEISO_LLAMASWAP_URL:-http://127.0.0.1:8080}"
  local msg

  if seiso_sidecar_stack_ready; then
    if seiso_ollama_health_ok "$ollama_url"; then
      seiso_log "Sidecar ready: Ollama at $ollama_url"
    else
      seiso_log "Sidecar ready: llama-swap fallback at $swap_url"
    fi
    return 0
  fi

  msg="Ollama/llama-swap not reachable (GGUF chat needs a sidecar on native Linux NVIDIA). \
Install Ollama or run: curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/bootstrap/linux-nvidia.sh | bash \
(or start Ollama at $ollama_url). Forge can still start; chat will show setup guidance."

  if seiso_should_require_sidecar; then
    seiso_die "Native Linux NVIDIA GGUF chat requires Ollama or llama-swap. $msg Set SEISO_REQUIRE_SIDECAR=0 or SEISO_SIDECAR_OPTIONAL=1 to continue without a sidecar."
  fi

  seiso_warn "$msg"
  return 0
}

seiso_run_sidecar_install_phase() {
  local root="$1"
  seiso_native_linux_nvidia || return 0
  [[ "${SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX:-0}" == "1" ]] && return 0

  seiso_log "Setting up Ollama-first sidecar stack for native Linux NVIDIA"
  seiso_seed_sidecar_env "$root"
  seiso_install_ollama || true
  seiso_install_sidecar_fallback || true

  if [[ -f "$root/.env" ]]; then
    config_path="$(seiso_sidecar_fallback_config_path)"
    if ! grep -q "^SEISO_LLAMASWAP_CONFIG=" "$root/.env" 2>/dev/null; then
      printf 'SEISO_LLAMASWAP_CONFIG=%s\n' "$config_path" >>"$root/.env"
    fi
  fi

  seiso_verify_sidecar_stack
}

seiso_warn_native_linux_nvidia_banner() {
  seiso_native_linux_nvidia || return 0
  [[ -n "${SEISO_INSTALL_PROFILE:-}" ]] && return 0
  [[ "${SEISO_NO_PROFILE_BANNER:-0}" == "1" ]] && return 0
  local raw_base="${SEISO_RAW_BASE:-https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main}"
  seiso_warn \
    "Detected native Linux + NVIDIA. For reliable GGUF chat, use: \
curl -fsSL ${raw_base}/scripts/bootstrap/linux-nvidia.sh | bash"
}
