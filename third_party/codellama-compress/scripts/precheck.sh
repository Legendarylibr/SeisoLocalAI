#!/usr/bin/env bash
# Run the local quality gate before pushing or opening a PR (see docs/CI_LOCAL.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/run_ci_local.py" --fast "$@"
