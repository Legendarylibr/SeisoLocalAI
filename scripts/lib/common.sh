#!/usr/bin/env bash
# Shared helpers for install.sh, start.sh, and doctor.sh.
# shellcheck shell=bash

seiso_log() { printf '==> %s\n' "$*"; }
seiso_warn() { printf 'warning: %s\n' "$*" >&2; }
seiso_die() { printf 'error: %s\n' "$*" >&2; exit 1; }

seiso_forge_url() {
  printf 'http://%s:%s' "${SEISO_HOST:-127.0.0.1}" "${SEISO_PORT:-8765}"
}

seiso_forge_instance_active() {
  # True when /health is up, or a live process still holds SEISO_DATA_DIR/.forge.lock
  # (lifespan may still be starting — port not listening yet).
  local url="${1:-$(seiso_forge_url)}"
  local lock pid
  if curl -fsS --max-time 2 "${url}/health" >/dev/null 2>&1; then
    return 0
  fi
  lock="${SEISO_DATA_DIR:-$HOME/.seiso}/.forge.lock"
  [[ -f "$lock" ]] || return 1
  pid="$(
    python3 - "$lock" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(int(json.load(open(sys.argv[1])).get("pid") or 0))
except Exception:
    print(0)
PY
  )"
  [[ -n "$pid" && "$pid" != "0" && -d "/proc/$pid" ]] || return 1
  return 0
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

seiso_repo_layout_ok() {
  local root="$1"
  [[ -n "$root" && -f "$root/pyproject.toml" && -d "$root/seiso_cli" ]]
}

seiso_resolve_repo_for_start() {
  # Prefer the repository that owns the invoked start/start.sh script, walking
  # parents (scripts/start.sh → repo root). Only then fall back to SEISO_INSTALL_DIR.
  local install_dir="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
  local src="${1:-}"
  local root candidate

  if [[ -n "$src" && -e "$src" ]]; then
    root="$(seiso_follow_symlinks "$src")" || true
    candidate="$root"
    # Walk up a few levels: start (repo root), scripts/start.sh → scripts → repo.
    local i
    for ((i = 0; i < 5; i++)); do
      if seiso_repo_layout_ok "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
      [[ -n "$candidate" && "$candidate" != "/" ]] || break
      candidate="$(cd "$candidate/.." 2>/dev/null && pwd)" || break
    done
  fi

  if seiso_repo_layout_ok "$install_dir"; then
    printf '%s\n' "$install_dir"
    return 0
  fi

  return 1
}

seiso_start_bin_dir() {
  printf '%s\n' "${SEISO_BIN_DIR:-$HOME/.local/bin}"
}

seiso_link_start_command() {
  # Install a symlink into SEISO_BIN_DIR. Skip non-symlink collisions so we never
  # clobber an unrelated tool (especially the generic name "start").
  local start_script="$1" link_path="$2"
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    seiso_warn "$link_path exists and is not a symlink — leaving it unchanged"
    return 1
  fi
  ln -sf "$start_script" "$link_path"
}

seiso_install_start_command() {
  local root="$1"
  local bin_dir start_script

  bin_dir="$(seiso_start_bin_dir)"
  start_script="$root/start"

  [[ -f "$start_script" ]] || return 0
  chmod +x "$start_script" 2>/dev/null || true

  mkdir -p "$bin_dir"
  # Prefer the unambiguous name; keep "start" for backward compatibility when free.
  seiso_link_start_command "$start_script" "$bin_dir/seiso-start" || true
  seiso_link_start_command "$start_script" "$bin_dir/start" || true

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

seiso_is_wsl() {
  [[ -f /proc/version ]] && grep -qiE 'microsoft|WSL' /proc/version 2>/dev/null
}

# Bash 3.2 (macOS /bin/bash) has no ${var,,}; keep install helpers portable.
seiso_tolower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

seiso_install_profile_extras() {
  local profile
  profile="$(seiso_tolower "$1")"
  case "$profile" in
    linux-nvidia|linux-nvidia-native)
      printf '%s\n' "forge,train,cuda,llamacpp"
      ;;
    linux-cpu|linux)
      printf '%s\n' "forge,train,llamacpp"
      ;;
    linux-rocm|rocm)
      printf '%s\n' "forge,train,llamacpp"
      ;;
    wsl-nvidia|wsl)
      printf '%s\n' "forge,train,cuda,llamacpp"
      ;;
    macos|darwin|apple-silicon)
      printf '%s\n' "forge,train,llamacpp,mlx"
      ;;
    chat|fast|chat-only)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        printf '%s\n' "forge,llamacpp,mlx"
      else
        printf '%s\n' "forge,llamacpp"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

seiso_detect_platform_extras() {
  if [[ -n "${SEISO_INSTALL_EXTRAS:-}" ]]; then
    printf '%s\n' "$SEISO_INSTALL_EXTRAS"
    return 0
  fi

  local profile extras os
  if [[ -n "${SEISO_INSTALL_PROFILE:-}" ]]; then
    extras="$(seiso_install_profile_extras "$SEISO_INSTALL_PROFILE")" \
      || seiso_die "Unknown SEISO_INSTALL_PROFILE=${SEISO_INSTALL_PROFILE}. Use: linux-nvidia, linux-cpu, linux-rocm, wsl-nvidia, macos, chat"
    case "$(seiso_tolower "$SEISO_INSTALL_PROFILE")" in
      wsl|wsl-nvidia)
        export SEISO_NVIDIA_WSL_ACK="${SEISO_NVIDIA_WSL_ACK:-1}"
        ;;
    esac
  elif [[ "${SEISO_FAST_INSTALL:-0}" == "1" ]]; then
    extras="$(seiso_install_profile_extras chat)"
  else
    os="$(uname -s)"
    case "$os" in
      Darwin)
        extras="forge,train,llamacpp,mlx"
        ;;
      Linux)
        if seiso_nvidia_gpu_detected; then
          extras="forge,train,cuda,llamacpp"
        else
          extras="forge,train,llamacpp"
        fi
        ;;
      *)
        seiso_die "Unsupported OS: $os (use docs/platforms/windows.md on Windows)"
        ;;
    esac
  fi
  if [[ "${SEISO_INSTALL_DEV:-0}" == "1" ]]; then
    extras="${extras},dev"
  fi
  printf '%s\n' "$extras"
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

seiso_bun_bin_dir() {
  printf '%s\n' "${BUN_INSTALL:-$HOME/.bun}/bin"
}

seiso_ensure_bun_on_path() {
  local bun_bin
  bun_bin="$(seiso_bun_bin_dir)"
  if [[ -d "$bun_bin" ]]; then
    export PATH="$bun_bin:$PATH"
  fi
}

seiso_ensure_bun() {
  seiso_ensure_bun_on_path
  command -v bun >/dev/null 2>&1 && return 0
  [[ "${SEISO_USE_NPM:-0}" == "1" ]] && return 1

  seiso_log "Installing Bun (fast JS package manager)..."
  if ! curl -fsSL https://bun.sh/install | bash; then
    seiso_warn "Bun install failed — will fall back to npm for Forge UI"
    return 1
  fi
  seiso_ensure_bun_on_path
  seiso_ensure_bin_on_path "$(seiso_bun_bin_dir)"
  command -v bun >/dev/null 2>&1
}

seiso_ui_pkg_manager() {
  seiso_ensure_bun_on_path
  if [[ "${SEISO_USE_NPM:-0}" != "1" ]] && command -v bun >/dev/null 2>&1; then
    printf 'bun\n'
  elif command -v npm >/dev/null 2>&1; then
    printf 'npm\n'
  else
    return 1
  fi
}

seiso_ui_bun_timeout_sec() {
  # Bun can hang with 0% CPU on some hosts; never block install forever.
  local t="${SEISO_BUN_INSTALL_TIMEOUT_SEC:-180}"
  case "$t" in
    ''|*[!0-9]*) t=180 ;;
  esac
  if [[ "$t" -lt 30 ]]; then
    t=30
  fi
  printf '%s\n' "$t"
}

seiso_run_with_timeout() {
  # Run command with a wall-clock timeout when `timeout` exists.
  local sec="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10s "${sec}s" "$@"
  else
    "$@"
  fi
}

seiso_ui_install_deps() {
  local ui_dir="$1" pm bun_timeout
  pm="$(seiso_ui_pkg_manager)" || seiso_die "Bun or npm is required for Forge UI — install Bun (https://bun.sh) or Node.js 18+"
  if [[ "$pm" == "bun" ]]; then
    # Prefer frozen installs for reproducibility. If Dependabot (or a human)
    # bumped package-lock.json / package.json without regenerating bun.lock,
    # frozen fails and macOS/Linux `start` never builds Forge UI — fall back
    # once so local installs keep working, then warn to commit bun.lock.
    # Wall-clock timeout: bun has been observed to hang after printing its version.
    bun_timeout="$(seiso_ui_bun_timeout_sec)"
    if seiso_run_with_timeout "$bun_timeout" bash -c "cd \"\$1\" && bun install --frozen-lockfile" _ "$ui_dir"; then
      return 0
    fi
    seiso_warn "bun install --frozen-lockfile failed or timed out after ${bun_timeout}s — retrying without freeze"
    if seiso_run_with_timeout "$bun_timeout" bash -c "cd \"\$1\" && bun install" _ "$ui_dir"; then
      seiso_warn "forge-ui/bun.lock may be out of sync — commit an updated bun.lock if package.json changed"
      return 0
    fi
    if command -v npm >/dev/null 2>&1; then
      seiso_warn "bun install hung or failed — falling back to npm ci for forge-ui"
      (cd "$ui_dir" && npm ci --no-audit --no-fund) || return 1
      return 0
    fi
    return 1
  else
    (cd "$ui_dir" && npm ci --no-audit --no-fund)
  fi
}

seiso_ui_run_script() {
  local ui_dir="$1" script="$2" pm
  pm="$(seiso_ui_pkg_manager)" || return 1
  if [[ "$pm" == "bun" ]]; then
    (cd "$ui_dir" && bun run "$script")
  else
    (cd "$ui_dir" && npm run "$script")
  fi
}

seiso_forge_ui_dist_ready() {
  local root="$1"
  [[ -f "$root/forge-ui/dist/index.html" ]]
}

seiso_build_forge_ui() {
  local root="$1"
  # Skip dependency install + rebuild when a production UI already exists unless forced.
  if seiso_forge_ui_dist_ready "$root" && [[ "${SEISO_FORCE_UI:-0}" != "1" ]]; then
    seiso_log "Forge UI dist present — skipping rebuild (set SEISO_FORCE_UI=1 to rebuild)"
    return 0
  fi
  seiso_ensure_bun || true
  seiso_ui_install_deps "$root/forge-ui" || return 1
  seiso_ui_run_script "$root/forge-ui" build || return 1
}

seiso_python_venv_ok() {
  # Debian/Ubuntu often ship python3 without python3-venv / ensurepip.
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY' >/dev/null 2>&1
import ensurepip
import venv
raise SystemExit(0)
PY
}

seiso_build_tools_ok() {
  # Native extensions (bitsandbytes, llama-cpp source builds, fused kernels)
  # need a C/C++ toolchain. cmake helps llama-cpp and similar.
  command -v gcc >/dev/null 2>&1 || return 1
  command -v g++ >/dev/null 2>&1 || return 1
  command -v make >/dev/null 2>&1 || return 1
  return 0
}

seiso_ensure_system_deps() {
  local missing=() need_pkgs=0
  command -v python3 >/dev/null 2>&1 || missing+=(python3)
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v curl >/dev/null 2>&1 || missing+=(curl)
  if [[ "${SEISO_USE_NPM:-0}" == "1" ]]; then
    command -v node >/dev/null 2>&1 || missing+=(node)
    command -v npm >/dev/null 2>&1 || missing+=(npm)
  fi

  # Even when python3/git/curl already exist, install venv headers + compilers
  # when missing — minimal images commonly hit this gap.
  if ((${#missing[@]} > 0)); then
    need_pkgs=1
  elif ! seiso_python_venv_ok; then
    need_pkgs=1
    seiso_log "Python venv support missing (python3-venv / ensurepip) — installing system packages"
  elif ! seiso_build_tools_ok; then
    need_pkgs=1
    seiso_log "Build tools missing (gcc/g++/make) — installing system packages"
  fi
  [[ "$need_pkgs" -eq 1 ]] || return 0

  if ((${#missing[@]} > 0)); then
    seiso_log "Installing missing system tools: ${missing[*]}"
  fi

  if command -v brew >/dev/null 2>&1; then
    local brew_pkgs=() dep
    for dep in "${missing[@]}"; do
      case "$dep" in
        python3) brew_pkgs+=(python@3.12) ;;
        git) brew_pkgs+=(git) ;;
        node|npm) [[ " ${brew_pkgs[*]} " == *" node "* ]] || brew_pkgs+=(node) ;;
      esac
    done
    # Always ensure a compiler toolchain on macOS when Xcode CLT is absent.
    if ! seiso_build_tools_ok; then
      [[ " ${brew_pkgs[*]} " == *" gcc "* ]] || brew_pkgs+=(gcc)
    fi
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
      python3 python3-venv python3-dev python3-pip git curl ca-certificates \
      build-essential pkg-config cmake \
      || return 1
    return 0
  fi

  if command -v dnf >/dev/null 2>&1; then
    local dnf_cmd=(dnf)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      dnf_cmd=(sudo dnf)
    fi
    "${dnf_cmd[@]}" install -y \
      python3 python3-pip python3-devel git curl ca-certificates \
      gcc gcc-c++ make cmake pkgconfig \
      || return 1
    return 0
  fi

  if command -v zypper >/dev/null 2>&1; then
    local zypper_cmd=(zypper)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      zypper_cmd=(sudo zypper)
    fi
    "${zypper_cmd[@]}" --non-interactive install \
      python3 python3-pip python3-devel git curl ca-certificates \
      gcc gcc-c++ make cmake pkg-config \
      || return 1
    return 0
  fi

  if command -v pacman >/dev/null 2>&1; then
    local pacman_cmd=(pacman)
    if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      pacman_cmd=(sudo pacman)
    fi
    "${pacman_cmd[@]}" -Sy --noconfirm \
      python python-pip base-devel git curl ca-certificates cmake pkgconf \
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
    command -v curl >/dev/null 2>&1 || seiso_die "curl is required — install curl, then re-run this script"
  fi
  seiso_python_version_ok || seiso_die "Python 3.10+ is required ($(python3 --version 2>&1 || echo unknown))"
  if [[ "${SEISO_USE_NPM:-0}" == "1" ]]; then
    command -v node >/dev/null 2>&1 || seiso_die "Node.js 18+ is required — install from https://nodejs.org/ or unset SEISO_USE_NPM"
    command -v npm >/dev/null 2>&1 || seiso_die "npm is required — install Node.js 18+ from https://nodejs.org/ or unset SEISO_USE_NPM"
  else
    seiso_ensure_bun || command -v npm >/dev/null 2>&1 \
      || seiso_die "Bun or npm is required for Forge UI — install Bun (https://bun.sh) or Node.js 18+"
  fi
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

seiso_ensure_cu12_runtime() {
  local root="$1"
  [[ "$(uname -s)" == "Linux" ]] || return 0
  seiso_nvidia_gpu_detected || return 0
  [[ -x "$root/.venv/bin/python" ]] || return 0
  seiso_log "Ensuring CUDA 12 runtime (llama.cpp GPU offload)..."
  seiso_pip_install_for_venv "$root/.venv/bin/python" \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 --prefer-binary \
    || seiso_warn "CUDA 12 runtime install failed — GGUF GPU chat may be CPU-only"
}

seiso_llamacpp_import_ok() {
  local root="$1"
  [[ -x "$root/.venv/bin/python" ]] || return 1
  "$root/.venv/bin/python" -c "from seiso.platform import ensure_cuda_library_path; ensure_cuda_library_path(); import llama_cpp" >/dev/null 2>&1
}

# cuda-toolkit 13.0.2 pip wheels ship ptxas capped at PTX 9.0 while nvcc emits 9.3.
seiso_repair_cuda_ptxas() {
  local root="$1"
  local py="$root/.venv/bin/python"
  [[ -x "$py" ]] || return 0
  if ! "$py" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    return 0
  fi
  if "$py" -c "from seiso.kernels.cuda_env import cuda_toolkit_status; s=cuda_toolkit_status(); raise SystemExit(0 if s.get('ptxas_compatible') else 1)" 2>/dev/null; then
    return 0
  fi
  seiso_log "Repairing CUDA toolkit ptxas (PTX 9.3 required for RTX 4090 kernel JIT)..."
  seiso_pip_install_for_venv "$py" 'cuda-toolkit[nvcc]>=13.1.0' || return 1
  rm -rf "${HOME:-/tmp}/.cache/torch_extensions"/*/seiso_cuda_kernels 2>/dev/null || true
}

seiso_repair_linux_cuda_stack() {
  local root="$1"
  local py="$root/.venv/bin/python"
  [[ "$(uname -s)" == "Linux" ]] || return 0
  [[ -x "$py" ]] || return 0
  seiso_nvidia_gpu_detected || return 0

  seiso_ensure_cu12_runtime "$root"
  seiso_repair_cuda_ptxas "$root" \
    || seiso_warn "CUDA ptxas repair skipped — fused kernels may fall back to PyTorch"

  if "$py" -c "from seiso.platform import repair_linux_cuda_stack; repair_linux_cuda_stack(auto_install=False)" >/dev/null 2>&1; then
    return 0
  fi
  seiso_warn "CUDA stack repair incomplete — try: pip install 'cuda-toolkit[nvcc]>=13.1.0' nvidia-cuda-runtime-cu12"
  return 1
}

seiso_verify_cuda_inference_stack() {
  local root="$1"
  local py="$root/.venv/bin/python"
  [[ -x "$py" ]] || return 0
  seiso_nvidia_gpu_detected || return 0
  if "$py" -c "
from seiso.platform import ensure_cuda_library_path, repair_linux_cuda_stack
report = repair_linux_cuda_stack(auto_install=False)
ensure_cuda_library_path()
import llama_cpp
gpu = llama_cpp.llama_supports_gpu_offload()
if not report.get('cu12_runtime'):
    raise SystemExit('missing cu12 runtime')
if not report.get('ptxas_compatible'):
    raise SystemExit('ptxas incompatible with nvcc (need cuda-toolkit>=13.1.0)')
if not gpu:
    raise SystemExit('llama-cpp-python lacks GPU offload')
print('cuda stack ok')
" 2>/dev/null; then
    seiso_log "CUDA inference stack verified (llama.cpp GPU + fused kernel JIT)"
    return 0
  fi
  seiso_warn "CUDA inference stack not fully ready — GGUF GPU chat or fused kernels may fail"
  return 1
}

seiso_ensure_llamacpp() {
  local root="$1"
  [[ -x "$root/.venv/bin/python" ]] || return 1
  seiso_ensure_cu12_runtime "$root"
  seiso_log "Ensuring llama.cpp (GGUF chat) runtime..."
  if "$root/.venv/bin/python" -m seiso.inference.llamacpp_install --quiet; then
    return 0
  fi
  seiso_warn "GGUF chat requires llama-cpp-python with GPU support when NVIDIA is present."
  seiso_warn "Try: pip install -U \"llama-cpp-python>=0.3\" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
  return 1
}

seiso_use_uv() {
  [[ "${SEISO_USE_UV:-1}" == "0" ]] && return 1
  command -v uv >/dev/null 2>&1
}

seiso_pip_quiet_args() {
  [[ "${SEISO_VERBOSE:-0}" != "1" ]] && printf '%s\n' -q
}

seiso_pip_filter_uv_args() {
  local arg
  for arg in "$@"; do
    [[ "$arg" == "--prefer-binary" ]] && continue
    printf '%s\0' "$arg"
  done
}

seiso_pip_install() {
  if seiso_use_uv; then
    local -a filtered=()
    local arg
    for arg in "$@"; do
      [[ "$arg" == "--prefer-binary" ]] && continue
      filtered+=("$arg")
    done
    if [[ "${SEISO_VERBOSE:-0}" != "1" ]]; then
      uv pip install -q "${filtered[@]}"
    else
      uv pip install "${filtered[@]}"
    fi
  else
    if [[ "${SEISO_VERBOSE:-0}" != "1" ]]; then
      python -m pip install -q "$@"
    else
      python -m pip install "$@"
    fi
  fi
}

seiso_pip_install_for_venv() {
  local venv_python="$1"
  shift
  if seiso_use_uv; then
    local -a filtered=()
    local arg
    for arg in "$@"; do
      [[ "$arg" == "--prefer-binary" ]] && continue
      filtered+=("$arg")
    done
    if [[ "${SEISO_VERBOSE:-0}" != "1" ]]; then
      uv pip install --python "$venv_python" -q "${filtered[@]}"
    else
      uv pip install --python "$venv_python" "${filtered[@]}"
    fi
  else
    if [[ "${SEISO_VERBOSE:-0}" != "1" ]]; then
      "$venv_python" -m pip install -q "$@"
    else
      "$venv_python" -m pip install "$@"
    fi
  fi
}

seiso_pip_bootstrap() {
  # Match pyproject build-system / [dev] pin (setuptools>=83, PYSEC-2026-3447).
  # setuptools is the build backend; hatchling is not required.
  seiso_pip_install -U pip wheel "setuptools>=83"
}

seiso_extras_without_llamacpp() {
  local extras="$1" part
  local -a kept=()
  IFS=',' read -r -a parts <<<"$extras"
  for part in "${parts[@]}"; do
    [[ "$part" == "llamacpp" ]] && continue
    [[ -n "$part" ]] && kept+=("$part")
  done
  (IFS=','; printf '%s' "${kept[*]}")
}

seiso_pip_install_extras() {
  local root="$1" extras="$2"
  local pip_extras="$extras"

  # llama-cpp-python often compiles from source when bundled with other extras;
  # install it separately via seiso_ensure_llamacpp (prebuilt CUDA wheels first).
  if [[ "$extras" == *llamacpp* ]]; then
    pip_extras="$(seiso_extras_without_llamacpp "$extras")"
  fi

  if seiso_pip_install -e "${root}[${pip_extras}]" --prefer-binary; then
    return 0
  fi
  seiso_warn "Full install failed — installing core [forge] then retrying optional extras"
  seiso_pip_install -e "${root}[forge]" --prefer-binary || return 1
  seiso_verify_cli "$root" || return 1
  seiso_pip_install -e "${root}[${pip_extras}]" --prefer-binary || {
    seiso_warn "Optional extras failed (${pip_extras}). Forge can still start — see $root/.seiso-install.log"
    seiso_verify_cli "$root" || return 1
  }
  return 0
}

seiso_run_install_worker() {
  local root="$1" extras="$2"
  local ui_pid=0 ui_status=0 llamacpp_pid=0

  if [[ "${SEISO_SKIP_UI:-0}" != "1" ]]; then
    (seiso_build_forge_ui "$root") &
    ui_pid=$!
  fi

  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  seiso_pip_bootstrap
  if [[ "$extras" == *cuda* || "$extras" == *llamacpp* ]]; then
    seiso_ensure_cu12_runtime "$root"
  fi
  seiso_pip_install_extras "$root" "$extras" || {
    if [[ "$ui_pid" -ne 0 ]]; then
      kill "$ui_pid" 2>/dev/null || true
      wait "$ui_pid" 2>/dev/null || true
    fi
    return 1
  }
  seiso_verify_cli "$root" || return 1

  if [[ "$extras" == *llamacpp* ]]; then
    seiso_ensure_llamacpp "$root" &
    llamacpp_pid=$!
  fi
  if [[ "${SEISO_SKIP_FLASH_ATTN:-1}" != "1" && "$extras" == *cuda* && "$root" != /mnt/* ]]; then
    if [[ -x "$root/scripts/install_flash_attn.sh" ]]; then
      bash "$root/scripts/install_flash_attn.sh" || true
    fi
  fi
  if [[ "$ui_pid" -ne 0 ]]; then
    wait "$ui_pid" || ui_status=$?
  fi
  if [[ "$llamacpp_pid" -ne 0 ]]; then
    wait "$llamacpp_pid" || true
  fi
  [[ "$ui_status" -eq 0 ]] || return 1
  if [[ "$extras" == *cuda* || "$extras" == *llamacpp* ]]; then
    seiso_repair_linux_cuda_stack "$root" || true
    if [[ "$extras" == *llamacpp* ]]; then
      seiso_verify_cuda_inference_stack "$root" || true
    fi
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
  # CUDA-linked wheels (llama-cpp-python) need venv nvidia/* on LD_LIBRARY_PATH
  # before import. Bare importlib checks falsely fail on native Linux NVIDIA and
  # force a full reinstall on every `start` even when the stack is healthy.
  local root="$1" extras="$2" module
  [[ -x "$root/.venv/bin/python" ]] || return 1
  while IFS= read -r module; do
    [[ -n "$module" ]] || continue
    if [[ "$module" == "llama_cpp" ]]; then
      seiso_llamacpp_import_ok "$root" || return 1
      continue
    fi
    "$root/.venv/bin/python" - "$module" <<'PY' >/dev/null 2>&1 || return 1
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
  done < <(seiso_required_python_modules "$extras")
  return 0
}

seiso_maybe_git_pull() {
  # Opt-in code upgrade for complete clones. Never force-resets local work.
  # Repair of incomplete clones still uses install.sh sync_install_clone.
  local root="$1"
  [[ "${SEISO_GIT_PULL:-0}" == "1" ]] || return 0
  [[ -d "$root/.git" ]] || return 0
  local branch
  branch="${SEISO_BRANCH:-main}"
  seiso_log "SEISO_GIT_PULL=1 — fast-forwarding $root to origin/$branch"
  git -C "$root" fetch --depth 1 origin "$branch" >/dev/null 2>&1 || {
    seiso_warn "git fetch failed — continuing with local tree"
    return 0
  }
  if git -C "$root" merge-base --is-ancestor HEAD "origin/$branch" 2>/dev/null \
    && ! git -C "$root" merge-base --is-ancestor "origin/$branch" HEAD 2>/dev/null; then
    git -C "$root" pull --ff-only origin "$branch" >/dev/null 2>&1 \
      || seiso_warn "git pull --ff-only failed (local commits or dirty tree?) — continuing"
  fi
}

seiso_ensure_installed() {
  local root="$1"
  local extras install_log

  seiso_maybe_git_pull "$root"

  extras="$(seiso_detect_platform_extras)"

  # Fast path: CLI + UI dist + CUDA-aware module imports — do not reinstall.
  if [[ -x "$root/.venv/bin/seiso" ]] && seiso_forge_ui_dist_ready "$root" \
    && seiso_python_modules_available "$root" "$extras"; then
    if [[ "$extras" == *cuda* || "$extras" == *llamacpp* ]]; then
      seiso_repair_linux_cuda_stack "$root" || true
    fi
    return 0
  fi

  if [[ "$extras" == *cuda* ]]; then
    seiso_log "NVIDIA GPU detected — installing with CUDA extras"
    seiso_ensure_cu12_runtime "$root"
  fi

  if [[ -x "$root/.venv/bin/python" && "$extras" == *llamacpp* ]]; then
    seiso_ensure_llamacpp "$root" || true
  fi

  # Re-check after CUDA runtime / llamacpp ensure (may have fixed import path only).
  if [[ -x "$root/.venv/bin/seiso" ]] && seiso_forge_ui_dist_ready "$root" \
    && seiso_python_modules_available "$root" "$extras"; then
    if [[ "$extras" == *cuda* || "$extras" == *llamacpp* ]]; then
      seiso_repair_linux_cuda_stack "$root" || true
    fi
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
