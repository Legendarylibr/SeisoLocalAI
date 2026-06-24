#!/usr/bin/env bash
# Shared helpers for install.sh, start.sh, and doctor.sh.
# shellcheck shell=bash

seiso_log() { printf '==> %s\n' "$*"; }
seiso_warn() { printf 'warning: %s\n' "$*" >&2; }
seiso_die() { printf 'error: %s\n' "$*" >&2; exit 1; }

seiso_forge_url() {
  printf 'http://%s:%s' "${SEISO_HOST:-127.0.0.1}" "${SEISO_PORT:-8765}"
}

seiso_follow_symlinks() {
  local path="$1"
  [[ -n "$path" && -e "$path" ]] || return 1
  while [[ -L "$path" ]]; do
    local target
    target="$(readlink "$path")"
    if [[ "$target" != /* ]]; then
      path="$(cd "$(dirname "$path")" && pwd)/$target"
    else
      path="$target"
    fi
  done
  cd "$(dirname "$path")" && pwd
}

seiso_resolve_repo_for_start() {
  local install_dir="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
  local src="${1:-}"

  if [[ -n "$src" && -e "$src" ]]; then
    local root
    root="$(seiso_follow_symlinks "$src")" || return 1
    if [[ -f "$root/pyproject.toml" && -d "$root/seiso_cli" ]]; then
      printf '%s\n' "$root"
      return 0
    fi
  fi

  if [[ -d "$install_dir/seiso_cli" && -f "$install_dir/pyproject.toml" ]]; then
    printf '%s\n' "$install_dir"
    return 0
  fi

  return 1
}

seiso_start_bin_dir() {
  printf '%s\n' "${SEISO_BIN_DIR:-$HOME/.local/bin}"
}

seiso_install_start_command() {
  local root="$1"
  local bin_dir start_script link_path

  bin_dir="$(seiso_start_bin_dir)"
  start_script="$root/start"
  link_path="$bin_dir/start"

  [[ -f "$start_script" ]] || return 0
  chmod +x "$start_script" 2>/dev/null || true

  mkdir -p "$bin_dir"
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    seiso_warn "$link_path exists and is not a symlink — leaving it unchanged"
    return 0
  fi
  ln -sf "$start_script" "$link_path"

  seiso_ensure_bin_on_path "$bin_dir"
}

seiso_install_router_start_command() {
  local root="$1"
  local bin_dir script link_path

  bin_dir="$(seiso_start_bin_dir)"
  script="$root/start-router-vllm"
  link_path="$bin_dir/start-router-vllm"

  [[ -f "$script" ]] || return 0
  chmod +x "$script" 2>/dev/null || true
  chmod +x "$root/scripts/start-router-vllm.sh" 2>/dev/null || true

  mkdir -p "$bin_dir"
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    seiso_warn "$link_path exists and is not a symlink — leaving it unchanged"
    return 0
  fi
  ln -sf "$script" "$link_path"

  seiso_ensure_bin_on_path "$bin_dir"
}

seiso_ensure_bin_on_path() {
  local bin_dir="$1"
  local marker line profile

  [[ -n "$bin_dir" ]] || return 0
  [[ "${SEISO_SKIP_PATH_SETUP:-0}" == "1" ]] && return 0

  case ":${PATH}:" in
    *":${bin_dir}:"*) ;;
    *)
      export PATH="${bin_dir}:${PATH}"
      ;;
  esac

  marker='# seiso-start-path'
  line="export PATH=\"${bin_dir}:\$PATH\" ${marker}"

  if [[ -n "${ZSH_VERSION:-}" ]]; then
    profile="${ZDOTDIR:-$HOME}/.zshrc"
  elif [[ -f "$HOME/.bashrc" ]]; then
    profile="$HOME/.bashrc"
  elif [[ -f "$HOME/.profile" ]]; then
    profile="$HOME/.profile"
  else
    return 0
  fi

  if grep -qF "$marker" "$profile" 2>/dev/null; then
    return 0
  fi

  {
    printf '\n%s\n' "$line"
  } >>"$profile"
}

seiso_cli_bin() {
  local root="$1"
  printf '%s\n' "$root/.venv/bin/seiso"
}

seiso_verify_cli() {
  local root="$1"
  [[ -x "$(seiso_cli_bin "$root")" ]]
}

seiso_require_cli() {
  local root="$1" cli log
  cli="$(seiso_cli_bin "$root")"
  if [[ -x "$cli" ]]; then
    printf '%s\n' "$cli"
    return 0
  fi
  log="$root/.seiso-install.log"
  seiso_die "Seiso CLI missing at $cli. Re-run: SEISO_NO_BANNER=1 start${log:+ — see $log}"
}

seiso_needs_install() {
  local root="$1"
  [[ -x "$root/.venv/bin/seiso" && -f "$root/forge-ui/dist/index.html" ]] || return 0
  return 1
}

seiso_detect_platform_extras() {
  local os
  os="$(uname -s)"
  case "$os" in
    Darwin)
      echo "forge,train,llamacpp,mlx,dev"
      ;;
    Linux)
      if seiso_nvidia_gpu_detected; then
        echo "forge,train,cuda,llamacpp,dev"
      else
        echo "forge,train,llamacpp,dev"
      fi
      ;;
    *)
      seiso_die "Unsupported OS: $os (use docs/platforms/windows.md on Windows)"
      ;;
  esac
}

seiso_resolve_nvidia_smi() {
  local candidate env_path
  for env_path in "${SEISO_NVIDIA_SMI_PATH:-}" "${NVIDIA_SMI_PATH:-}"; do
    [[ -n "$env_path" && -x "$env_path" ]] || continue
    printf '%s\n' "$env_path"
    return 0
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    command -v nvidia-smi
    return 0
  fi
  for candidate in \
    /usr/bin/nvidia-smi \
    /usr/lib/nvidia/bin/nvidia-smi \
    /usr/local/nvidia/bin/nvidia-smi \
    /usr/local/cuda/bin/nvidia-smi \
    /usr/lib/wsl/lib/nvidia-smi; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

seiso_nvidia_gpu_detected() {
  local smi
  smi="$(seiso_resolve_nvidia_smi)" || return 1
  if "$smi" --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q '[^[:space:]]'; then
    return 0
  fi
  if "$smi" -L 2>/dev/null | grep -q '^GPU '; then
    return 0
  fi
  return 1
}

seiso_python_version_ok() {
  python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

seiso_ensure_system_deps() {
  local missing=()
  command -v python3 >/dev/null 2>&1 || missing+=(python3)
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v node >/dev/null 2>&1 || missing+=(node)
  command -v npm >/dev/null 2>&1 || missing+=(npm)
  [[ ${#missing[@]} -eq 0 ]] && return 0

  seiso_log "Installing missing system tools: ${missing[*]}"

  if command -v brew >/dev/null 2>&1; then
    local brew_pkgs=() dep
    for dep in "${missing[@]}"; do
      case "$dep" in
        python3) brew_pkgs+=(python@3.12) ;;
        git) brew_pkgs+=(git) ;;
        node|npm) [[ " ${brew_pkgs[*]} " == *" node "* ]] || brew_pkgs+=(node) ;;
      esac
    done
    if ((${#brew_pkgs[@]} > 0)); then
      brew install "${brew_pkgs[@]}" || return 1
    fi
    if command -v python3.12 >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
      export PATH="$(brew --prefix python@3.12)/bin:$PATH"
    fi
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    local apt_cmd=(apt-get)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      apt_cmd=(sudo apt-get)
    fi
    "${apt_cmd[@]}" update -qq
    "${apt_cmd[@]}" install -y \
      python3 python3-venv python3-pip git curl ca-certificates nodejs npm \
      || return 1
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    local dnf_cmd=(dnf)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      dnf_cmd=(sudo dnf)
    fi
    "${dnf_cmd[@]}" install -y python3 python3-pip git curl nodejs npm \
      || return 1
    return 0
  fi

  if command -v pacman >/dev/null 2>&1; then
    local pacman_cmd=(pacman)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      pacman_cmd=(sudo pacman)
    fi
    "${pacman_cmd[@]}" -Sy --noconfirm python python-pip git curl nodejs npm \
      || return 1
    return 0
  fi

  return 1
}

seiso_require_system_deps() {
  if ! seiso_ensure_system_deps; then
    seiso_warn "Could not auto-install system dependencies."
    command -v python3 >/dev/null 2>&1 || seiso_die "Python 3.10+ is required — install from https://www.python.org/downloads/"
    command -v git >/dev/null 2>&1 || seiso_die "git is required — install git, then re-run this script"
    command -v node >/dev/null 2>&1 || seiso_die "Node.js 18+ is required — install from https://nodejs.org/"
    command -v npm >/dev/null 2>&1 || seiso_die "npm is required — install Node.js 18+ from https://nodejs.org/"
  fi
  seiso_python_version_ok || seiso_die "Python 3.10+ is required ($(python3 --version 2>&1 || echo unknown))"
}

seiso_run_doctor() {
  local root="$1"
  shift || true
  if [[ -x "$root/scripts/doctor.sh" ]]; then
    seiso_warn "Running Seiso doctor..."
    printf '\n'
    bash "$root/scripts/doctor.sh" "$@" || true
    printf '\n'
  else
    seiso_warn "Doctor script not found at $root/scripts/doctor.sh"
  fi
}

seiso_wait_for_health() {
  local url="$1" attempts="${2:-60}" i
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS --max-time 2 "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

seiso_open_browser() {
  local url="$1"
  [[ "${SEISO_NO_OPEN:-0}" == "1" ]] && return 0
  [[ "${CI:-}" =~ ^(1|true|yes)$ ]] && return 0

  case "$(uname -s)" in
    Darwin)
      open "$url" >/dev/null 2>&1 && return 0
      ;;
    Linux)
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 && return 0
      fi
      ;;
  esac

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$url" <<'PY'
import sys
import webbrowser
webbrowser.open(sys.argv[1], new=2)
PY
    return 0
  fi
  return 1
}

seiso_open_forge_when_ready() {
  local url="$1"
  if seiso_wait_for_health "$url"; then
    if seiso_open_browser "$url"; then
      seiso_log "Opened Forge in your browser"
    else
      seiso_log "Forge is ready — open $url"
    fi
  else
    seiso_warn "Forge is starting — open $url when ready"
  fi
}

seiso_llamacpp_import_ok() {
  local root="$1"
  [[ -x "$root/.venv/bin/python" ]] || return 1
  "$root/.venv/bin/python" -c "from seiso.platform import ensure_cuda_library_path; ensure_cuda_library_path(); import llama_cpp" >/dev/null 2>&1
}

seiso_ensure_llamacpp() {
  local root="$1"
  [[ -x "$root/.venv/bin/python" ]] || return 1
  seiso_log "Ensuring llama.cpp (GGUF chat) runtime..."
  if "$root/.venv/bin/python" -m seiso.inference.llamacpp_install --quiet; then
    return 0
  fi
  seiso_warn "GGUF chat requires llama-cpp-python with GPU support when NVIDIA is present."
  seiso_warn "Try: pip install -U \"llama-cpp-python>=0.3\" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
  return 1
}

seiso_run_install_worker() {
  local root="$1" extras="$2"
  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  python -m pip install -U pip wheel setuptools hatchling
  pip install -e "${root}[forge,dev]"
  seiso_verify_cli "$root" || return 1
  if [[ "$extras" != "forge,dev" ]]; then
    pip install -e "${root}[${extras}]" || {
      seiso_warn "Optional extras failed (${extras}). Forge can still start — see $root/.seiso-install.log"
      seiso_verify_cli "$root" || return 1
    }
  fi
  if [[ "$extras" == *llamacpp* ]]; then
    seiso_ensure_llamacpp "$root" || true
  fi
  if [[ "${SEISO_SKIP_FLASH_ATTN:-1}" != "1" && "$extras" == *cuda* && "$root" != /mnt/* ]]; then
    if [[ -x "$root/scripts/install_flash_attn.sh" ]]; then
      bash "$root/scripts/install_flash_attn.sh" || true
    fi
  fi
  if [[ "${SEISO_SKIP_UI:-0}" != "1" ]]; then
    (cd "$root/forge-ui" && npm ci && npm run build)
  fi
  seiso_verify_cli "$root"
}

seiso_required_python_modules() {
  local extras="$1"
  printf '%s\n' seiso forge fastapi huggingface_hub
  if [[ "$extras" == *llamacpp* ]]; then
    printf '%s\n' llama_cpp
  fi
  if [[ "$extras" == *mlx* ]]; then
    printf '%s\n' mlx_lm
  fi
  if [[ "$extras" == *train* ]]; then
    printf '%s\n' torch transformers
  fi
}

seiso_python_modules_available() {
  local root="$1" extras="$2" module
  [[ -x "$root/.venv/bin/python" ]] || return 1
  while IFS= read -r module; do
    [[ -n "$module" ]] || continue
    "$root/.venv/bin/python" - "$module" <<'PY' >/dev/null 2>&1 || return 1
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
  done < <(seiso_required_python_modules "$extras")
  return 0
}

seiso_ensure_installed() {
  local root="$1"
  local extras install_log

  extras="$(seiso_detect_platform_extras)"

  if [[ "$extras" == *cuda* ]]; then
    seiso_log "NVIDIA GPU detected — installing with CUDA extras"
  fi

  if [[ -x "$root/.venv/bin/python" && "$extras" == *llamacpp* ]]; then
    seiso_ensure_llamacpp "$root" || true
  fi

  if [[ -x "$root/.venv/bin/seiso" && -f "$root/forge-ui/dist/index.html" ]] \
    && seiso_python_modules_available "$root" "$extras"; then
    return 0
  fi

  seiso_log "Completing Seiso install..."

  if [[ ! -x "$root/.venv/bin/python" ]]; then
    seiso_log "Creating virtualenv at $root/.venv"
    python3 -m venv "$root/.venv"
  fi

  if [[ ! -f "$root/.env" && -f "$root/.env.example" ]]; then
    cp "$root/.env.example" "$root/.env"
    seiso_log "Created $root/.env from .env.example"
  fi

  install_log="$root/.seiso-install.log"
  if ! seiso_run_install_worker "$root" "$extras" >"$install_log" 2>&1; then
    seiso_warn "Install failed — see $install_log"
    tail -30 "$install_log" >&2 || true
    seiso_run_doctor "$root"
    return 1
  fi
  if ! seiso_verify_cli "$root"; then
    seiso_warn "Install finished but Seiso CLI is missing — see $install_log"
    tail -30 "$install_log" >&2 || true
    seiso_run_doctor "$root"
    return 1
  fi
  seiso_install_start_command "$root"
  return 0
}
