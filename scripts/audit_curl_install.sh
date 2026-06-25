#!/usr/bin/env bash
# Read-only audit for the curl | bash install one-liner.
# Usage: ./scripts/audit_curl_install.sh
set -euo pipefail

INSTALL_URL="https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh"
START_URL="https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start"
COMMON_URL="https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/lib/common.sh"
REPO_URL="https://github.com/Legendarylibr/SeisoLocalAI.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pass=0
fail=0

ok() { printf 'PASS: %s\n' "$*"; pass=$((pass + 1)); }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=$((fail + 1)); }

log() { printf '==> %s\n' "$*"; }

log "Fetching install script"
tmp="$(mktemp)"
code="$(curl -fsSL -w '%{http_code}' -o "$tmp" "$INSTALL_URL" || true)"
if [[ "$code" != "200" ]]; then
  bad "curl returned HTTP $code for $INSTALL_URL"
else
  ok "curl HTTP 200 ($INSTALL_URL)"
fi

start_tmp="$(mktemp)"
start_code="$(curl -fsSL -w '%{http_code}' -o "$start_tmp" "$START_URL" || true)"
if [[ "$start_code" != "200" ]]; then
  bad "curl returned HTTP $start_code for $START_URL"
else
  ok "curl HTTP 200 ($START_URL)"
fi
if rg -q 'scripts/install\.sh' "$start_tmp" "$ROOT/start" 2>/dev/null; then
  ok "start script delegates to scripts/install.sh"
else
  bad "start script missing install.sh delegation"
fi

if bash -n "$tmp" 2>/dev/null; then
  ok "install.sh bash syntax valid"
else
  bad "install.sh bash syntax invalid"
fi

common_tmp="$(mktemp)"
common_code="$(curl -fsSL -w '%{http_code}' -o "$common_tmp" "$COMMON_URL" || true)"
if [[ "$common_code" != "200" ]]; then
  bad "curl returned HTTP $common_code for $COMMON_URL"
else
  ok "curl HTTP 200 ($COMMON_URL)"
fi

if rg -q 'seiso_pip_install.*-e "\$\{root\}\[\$\{pip_extras\}\]"' "$common_tmp" "$ROOT/scripts/lib/common.sh" 2>/dev/null; then
  ok "pip extras syntax uses \${root}[\${pip_extras}] via seiso_pip_install"
else
  bad "pip extras syntax missing in scripts/lib/common.sh (expected seiso_pip_install -e \"\${root}[\${pip_extras}]\")"
fi

if rg -q 'pip install -e "\$root/\.\[' "$tmp" "$common_tmp" "$ROOT/scripts/lib/common.sh" 2>/dev/null; then
  bad "broken pip path still present (\$root/.[extras])"
else
  ok "no broken \$root/.[extras] pip path"
fi

log "Checking git remote clone"
clone_dir="$(mktemp -d)/Seiso"
if git clone --depth 1 --branch main "$REPO_URL" "$clone_dir" >/dev/null 2>&1; then
  ok "git clone succeeds ($REPO_URL)"
else
  bad "git clone failed ($REPO_URL)"
fi

if [[ -f "$clone_dir/pyproject.toml" && -d "$clone_dir/seiso_cli" ]]; then
  ok "cloned repo has pyproject.toml and seiso_cli/"
else
  bad "cloned repo missing expected layout"
fi

log "Simulating curl|bash resolve_root (stdin, no BASH_SOURCE file)"
sim_root=""
if bash -c '
  set -euo pipefail
  INSTALL_DIR="'"$clone_dir"'"
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  elif [[ -d "$INSTALL_DIR/seiso_cli" && -f "$INSTALL_DIR/pyproject.toml" ]]; then
    printf "%s\n" "$INSTALL_DIR"
  else
    exit 1
  fi
' <<< ""; then
  sim_root="$clone_dir"
  ok "resolve_root falls through to INSTALL_DIR on piped bash"
else
  bad "resolve_root simulation failed for piped bash"
fi

if rg -q 'seiso_pip_bootstrap' "$ROOT/scripts/lib/common.sh" 2>/dev/null \
  && rg -q 'hatchling' "$ROOT/scripts/lib/common.sh" 2>/dev/null; then
  ok "install worker bootstraps hatchling before editable install"
else
  bad "install worker missing hatchling bootstrap (editable install may fail on fresh venvs)"
fi

if rg -q 'cuda-toolkit\[nvcc\]==13\.0\.2' "$ROOT/pyproject.toml" 2>/dev/null; then
  ok "cuda-toolkit pin matches torch 2.12 (13.0.2)"
else
  bad "cuda-toolkit pin should be 13.0.2 to match torch 2.12"
fi

if rg -q 'seiso_use_uv' "$ROOT/scripts/lib/common.sh" 2>/dev/null; then
  ok "install path supports uv for Python deps"
else
  bad "install path missing uv support"
fi

if rg -q 'wait "\$job_pid"' "$ROOT/scripts/install.sh" 2>/dev/null; then
  ok "install TUI uses bash wait on background install job"
else
  bad "install.sh does not bash-wait on background install job (Python waitpid cannot reap bash siblings)"
fi

if rg -q 'seiso_verify_cli' "$ROOT/scripts/lib/common.sh" "$ROOT/scripts/install.sh" 2>/dev/null; then
  ok "install path verifies seiso CLI after pip install"
else
  bad "install path missing post-install seiso CLI verification"
fi

log "Dry-run pip editable install syntax"
audit_venv="$(mktemp -d)/venv"
python3 -m venv "$audit_venv"
# shellcheck disable=SC1091
source "$audit_venv/bin/activate"
python -m pip install -U pip wheel setuptools hatchling -q
if python -m pip install -e "${clone_dir}[forge,train,dev]" --dry-run >/dev/null 2>&1; then
  ok "pip dry-run [forge,train,dev] succeeds"
else
  bad "pip dry-run [forge,train,dev] failed"
fi
if python -m pip install -e "${ROOT}[forge,train,cuda,dev]" --dry-run >/dev/null 2>&1; then
  ok "pip dry-run [forge,train,cuda,dev] succeeds"
else
  bad "pip dry-run [forge,train,cuda,dev] failed"
fi
deactivate 2>/dev/null || true
rm -rf "$(dirname "$audit_venv")"

if rg -A3 '^cuda = \[' "$clone_dir/pyproject.toml" | rg -q 'flash-attn'; then
  bad "pyproject [cuda] still requires flash-attn (NVIDIA curl installs may fail building wheels)"
else
  ok "flash-attn not required in [cuda] extra"
fi

log "Local vs published install.sh"
if [[ -f "$ROOT/scripts/install.sh" ]]; then
  if diff -q "$ROOT/scripts/install.sh" "$tmp" >/dev/null 2>&1; then
    ok "local install.sh matches published main"
  else
    bad "local install.sh differs from published main (unpushed changes)"
  fi
fi

if [[ -f "$ROOT/scripts/lib/common.sh" ]]; then
  if diff -q "$ROOT/scripts/lib/common.sh" "$common_tmp" >/dev/null 2>&1; then
    ok "local common.sh matches published main"
  else
    bad "local common.sh differs from published main (unpushed changes)"
  fi
fi

rm -f "$tmp" "$common_tmp" "$start_tmp"
rm -rf "$(dirname "$clone_dir")"

printf '\nAudit complete: %d passed, %d failed\n' "$pass" "$fail"
exit $(( fail > 0 ? 1 : 0 ))
