#!/usr/bin/env bash
# Quick pre-PR gate: lint + types + tests + security.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/run_ci_local.py" --fast "$@"
