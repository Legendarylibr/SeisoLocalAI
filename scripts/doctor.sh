#!/usr/bin/env bash
# Diagnose a Seiso install without requiring Forge to start.
set -euo pipefail

INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
NETWORK_CHECK=0
DOCTOR_CACHE=""

cleanup() {
  if [[ -n "$DOCTOR_CACHE" ]]; then
    rm -rf "$DOCTOR_CACHE"
  fi
}
trap cleanup EXIT

for arg in "$@"; do
  case "$arg" in
    --network) NETWORK_CHECK=1 ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/doctor.sh [--network]

Checks Python, Node, the Seiso virtualenv, Forge UI build, Hugging Face tooling,
and common install/runtime gaps. Add --network to probe huggingface.co.
EOF
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

ok() { printf 'OK   %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; }
info() { printf 'INFO %s\n' "$*"; }

free_gb() {
  local path="$1"
  mkdir -p "$path" 2>/dev/null || true
  df -Pk "$path" 2>/dev/null | awk 'NR==2 { printf "%.1f", ($4 * 1024) / (1024^3) }'
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
  printf '%s\n' "$PWD"
}

check_cmd() {
  local name="$1" hint="$2"
  if command -v "$name" >/dev/null 2>&1; then
    ok "$name: $(command -v "$name")"
  else
    fail "$name missing. $hint"
  fi
}

root="$(resolve_root)"
printf '\nSeiso Doctor\n'
printf '============\n'
info "repo: $root"

if [[ ! -f "$root/pyproject.toml" || ! -d "$root/seiso_cli" ]]; then
  fail "Seiso repository not found. Set SEISO_INSTALL_DIR or run from the repo root."
  exit 1
fi

check_cmd git "Install git, then rerun start."
check_cmd python3 "Install Python 3.10+."

if [[ -f "$root/scripts/lib/common.sh" ]]; then
  # shellcheck source=lib/common.sh
  source "$root/scripts/lib/common.sh"
  seiso_ensure_bun_on_path
fi

if command -v bun >/dev/null 2>&1; then
  ok "bun version: $(bun --version 2>&1)"
elif command -v npm >/dev/null 2>&1; then
  ok "npm version: $(npm --version 2>&1)"
else
  fail "bun or npm missing. Run start to auto-install Bun, or install Node.js 18+ from https://nodejs.org/."
fi

if command -v node >/dev/null 2>&1; then
  ok "node version: $(node --version 2>&1)"
elif command -v bun >/dev/null 2>&1; then
  ok "node not installed (Bun provides the JS runtime for Forge UI)"
else
  fail "node missing. Install Node.js 18+ from https://nodejs.org/ or run start to install Bun."
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    ok "python3 version: $(python3 --version 2>&1)"
  else
    fail "python3 is too old: $(python3 --version 2>&1). Install Python 3.10+."
  fi
fi

if [[ -f "$root/.env" ]]; then
  ok ".env exists"
else
  warn ".env missing. Copy .env.example to .env or rerun start."
fi

data_dir="${SEISO_DATA_DIR:-$HOME/.seiso}"
hf_cache_dir="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}"
hf_cache_dir="${hf_cache_dir:-$data_dir/hf_cache}"
ok "install disk free: $(free_gb "$root") GB at $root"
ok "data disk free: $(free_gb "$data_dir") GB at $data_dir"
ok "HF cache disk free: $(free_gb "$hf_cache_dir") GB at $hf_cache_dir"
info "catalog GGUF downloads are usually 2-8 GB each; larger models can be 10-30+ GB."
info "Hugging Face GGUFs download to Seiso's cache/inventory and load with llama.cpp."

if [[ -x "$root/.venv/bin/python" ]]; then
  ok "virtualenv: $root/.venv"
else
  fail "virtualenv missing. Run: start"
fi

if [[ -x "$root/.venv/bin/seiso" ]]; then
  ok "seiso CLI: $root/.venv/bin/seiso"
else
  warn "seiso CLI missing from venv. Run: source $root/.venv/bin/activate && pip install -e '$root[forge,train,dev]'"
fi

if [[ -f "$root/forge-ui/dist/index.html" ]]; then
  ok "Forge UI build exists"
else
  if command -v bun >/dev/null 2>&1; then
    warn "Forge UI build missing. Run: cd $root/forge-ui && bun install --frozen-lockfile && bun run build"
  else
    warn "Forge UI build missing. Run: cd $root/forge-ui && npm ci && npm run build"
  fi
fi

if [[ -x "$root/.venv/bin/python" ]]; then
  DOCTOR_CACHE="$(mktemp -d 2>/dev/null || mktemp -d -t seiso-doctor)"
  SEISO_DOCTOR_NETWORK="$NETWORK_CHECK" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DOCTOR_CACHE}" \
  "$root/.venv/bin/python" - <<'PY'
import importlib.util
import os
import shutil
import sys
from pathlib import Path

def line(level: str, message: str) -> None:
    print(f"{level:<4} {message}")

def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False

checks = {
    "huggingface_hub": "required for model downloads",
    "hf_xet": "recommended for faster Hugging Face downloads",
    "fastapi": "required for Forge",
    "uvicorn": "required for Forge",
    "torch": "required for training / safetensors workflows",
    "llama_cpp": "required for GGUF chat",
    "mlx_lm": "recommended on macOS Apple Silicon",
}
for module, hint in checks.items():
    level = "OK" if has(module) else ("WARN" if module in {"hf_xet", "llama_cpp", "mlx_lm", "torch"} else "FAIL")
    line(level, f"python package {module}: {'found' if has(module) else 'missing'} ({hint})")

python_bin = Path(sys.executable).parent
hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")
if not hf_cli:
    for name in ("hf", "huggingface-cli"):
        candidate = python_bin / name
        if candidate.exists():
            hf_cli = str(candidate)
            break
if hf_cli:
    line("OK", f"Hugging Face CLI: {hf_cli}")
else:
    line("WARN", "Hugging Face CLI not on PATH. Activate the venv, then run: hf auth login")

try:
    from forge.services.hf_connectivity import build_hf_status
    from seiso.models.hf_env import configure_hf_hub_cache

    configured_cache = configure_hf_hub_cache(Path(os.environ.get("SEISO_DATA_DIR", "~/.seiso")).expanduser())
    status = build_hf_status(probe=os.environ.get("SEISO_DOCTOR_NETWORK") == "1")
    transfer = status["transfer"]
    line("OK", f"HF cache configured: {configured_cache}")
    line(
        "OK" if transfer["xet_available"] else "WARN",
        f"HF transfer backend: {transfer['backend']} (threads={transfer['num_threads']}, timeout={transfer['download_timeout_s']}s)",
    )
    if transfer.get("hint"):
        line("INFO", transfer["hint"])
    line("OK" if status["runtime"]["huggingface_hub"] else "FAIL", "Forge can inspect Hugging Face runtime")
    auth = status["auth"]
    conn = status["connectivity"]
    if auth.get("token_configured") and conn.get("token_invalid"):
        line("WARN", "Configured Hugging Face token was rejected — public downloads still work")
    elif auth.get("token_configured") and conn.get("token_valid"):
        line("OK", f"Hugging Face token valid for {conn.get('token_username') or 'user'}")
    line("OK" if status.get("ready_for_download") else "FAIL", f"Hub ready for download: {status.get('ready_for_download')}")
    line("OK" if status.get("ready_for_local_chat") else "WARN", f"Local chat runtime ready: {status.get('ready_for_local_chat')}")
    line("OK" if status["auth"]["cli_available"] else "WARN", f"HF CLI visible to Forge: {status['auth']['cli_binary'] or 'no'}")
    if os.environ.get("SEISO_DOCTOR_NETWORK") == "1":
        conn = status["connectivity"]
        line("OK" if conn["reachable"] else "FAIL", f"huggingface.co reachable: {conn['reachable']} ({conn.get('error') or 'ok'})")
    else:
        line("INFO", "network probe skipped. Run scripts/doctor.sh --network to test huggingface.co")
except Exception as exc:
    line("FAIL", f"Forge status check failed: {exc}")
PY
fi

printf '\nNext steps:\n'
printf '  Start Forge: start\n'
printf '  If gated models fail: source %s/.venv/bin/activate && hf auth login\n' "$root"
printf '  Full network check: %s/scripts/doctor.sh --network\n\n' "$root"
