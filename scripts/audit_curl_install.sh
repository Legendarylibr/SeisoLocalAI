#!/usr/bin/env bash
# Read-only audit for the curl | bash install one-liner.
# Usage: ./scripts/audit_curl_install.sh
set -euo pipefail

INSTALL_URL="https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/install.sh"
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

if bash -n "$tmp" 2>/dev/null; then
  ok "install.sh bash syntax valid"
else
  bad "install.sh bash syntax invalid"
fi

if rg -q 'pip install -e "\$\{root\}\[\$\{extras\}\]"' "$tmp"; then
  ok "pip extras syntax uses \${root}[\${extras}]"
else
  bad "pip extras syntax missing or wrong (expected pip install -e \"\${root}[\${extras}]\")"
fi

if rg -q 'pip install -e "\$root/\.\[' "$tmp"; then
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
if python -m pip install -e "${clone_dir}[forge,train,cuda,dev]" --dry-run >/dev/null 2>&1; then
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

rm -f "$tmp"
rm -rf "$(dirname "$clone_dir")"

printf '\nAudit complete: %d passed, %d failed\n' "$pass" "$fail"
exit $(( fail > 0 ? 1 : 0 ))
