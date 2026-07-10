# Local CI / quality gate

Run the full local quality gate before opening PRs or cutting releases.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,dev]"

make ci-fast      # daily loop: lint + types + test + security
make ci           # full gate (+ frontend build + optional import smokes)
make check-changed # changed Python files + directly changed test modules
make test-parallel # CPU tests with two pytest-xdist workers
make test-hardware # opt-in tests that require a working GPU/toolkit
make ci-list      # show jobs and recommended matrix
```

Or use the shell wrapper:

```bash
./scripts/precheck.sh
```

## Jobs

| Job | What it runs | Skipped by `--fast` |
|-----|--------------|---------------------|
| **deps** | dependency lockfile digest verification | no |
| **lint** | `ruff check`, `ruff format --check`, `pylint` (E/F only) | no |
| **types** | `mypy seiso forge seiso_cli` | no |
| **test** | smoke imports + `pytest -m "not slow"` | no |
| **security** | `bandit`, `detect-secrets`, `pip check`, `pip-audit` | no |
| **frontend** | `npm run typecheck` + `npm run build` in `forge-ui/` | yes |
| **imports** | optional-extra import smokes (`train`, `mlx` on macOS) | yes |

### Lint detail

1. **Ruff check** — style, imports, pyupgrade, bugbear, simplify rules (`pyproject.toml`)
2. **Pylint** — fatal/error class only (`--enable=E,F`), optional deps ignored

Ruff uses a **baseline** (`scripts/ruff-baseline.txt`): CI fails only on *new* issues. Refresh after intentional cleanup:

```bash
python3 scripts/run_ci_local.py --job lint --update-ruff-baseline --skip-install
make fix   # auto-fix Ruff issues + format, then refresh baseline
```

### Types detail

Mypy uses gradual typing settings in `pyproject.toml`. A **baseline** (`scripts/mypy-baseline.txt`) tracks known errors; CI fails only on *new* type errors.

The baseline is calibrated for Python 3.10. When `scripts/run_ci_local.py` is
run with another interpreter, the types job is skipped with a warning instead
of comparing third-party stubs against the wrong Python version. Use
`--python-bin` with Python 3.10 for an authoritative local type check.

Refresh after fixing types:

```bash
python3 scripts/run_ci_local.py --job types --update-mypy-baseline --skip-install
```

### Test detail

- Installs `.[forge,train,dev]` once unless `--skip-install`
- `tests/test_docs_accuracy.py` — doc links, example configs, and training API references stay aligned with the codebase
- Runs CPU unit/integration tests excluding `@pytest.mark.slow` and `@pytest.mark.gpu`
- `--pytest-workers N` enables pytest-xdist with scope-aware distribution; CI uses two workers
- `--hardware-tests` selects non-slow `@pytest.mark.gpu` tests on hosts with a matching runtime/toolkit
- Slow tests: `pytest -m slow`

### Security detail

| Tool | Purpose |
|------|---------|
| **Bandit** | Python SAST (`-l` medium+, skips for known ML/subprocess patterns) |
| **detect-secrets** | committed secret scan vs `.secrets.baseline` |
| **pip check** | dependency consistency |
| **pip-audit** | known CVEs in installed packages |

Update the secrets baseline after reviewing new findings:

```bash
detect-secrets scan seiso forge seiso_cli tests forge-ui/src docs scripts \
  .env.example README.md pyproject.toml Makefile > .secrets.baseline
```

### Frontend detail

Requires Node.js/npm. Runs in `forge-ui/`:

1. `npm ci` (if `node_modules/` missing)
2. `npm run typecheck`
3. `npm run build`

## Pre-commit hooks

Optional but recommended for faster feedback on changed files:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # first-time baseline
```

Hooks: detect-secrets and Ruff run on staged files. The pre-push hook runs the
changed-file quality path. Full Mypy, security, and test coverage remain
authoritative CI checks.

## Changed-file checks

For a quick local loop, compare against `origin/main`, lint changed Python
files, and run directly changed test modules:

```bash
make check-changed
python3 scripts/run_ci_local.py --changed --changed-base main --skip-install
```

This mode deliberately does not claim full transitive coverage. Run
`make ci-fast` before merging when CI is unavailable.

## GitHub Actions

CI runs dependency locks, lint, Mypy, CPU tests, security, and frontend checks
as independent parallel jobs. Each Python job installs its tailored dependency
set once and calls the local runner with `--skip-install`. GitHub's pip and npm
caches reuse downloaded packages; virtualenv artifacts are not shared because
the training environment is large and environment paths are not portable. An
aggregate `Quality gate` job preserves the single branch-protection check.

## Environment variables

| Variable | Effect |
|----------|--------|
| `PYTHON_BIN` | Force interpreter (default: `.venv/bin/python` or `venv/bin/python`) |
| `CHANGED_BASE` | Override the default `origin/main` base for `--changed` |

## Recommended matrix

Run `make ci-fast` on each row before merging significant changes:

| OS | Python | Notes |
|----|--------|-------|
| Linux | 3.10+ | primary CI target |
| macOS | 3.10+ | MLX import smoke in full `make ci` |

## Individual jobs

```bash
make lint
make deps
make types
make test
make test-parallel
make test-hardware
make security
make frontend
make imports

python3 scripts/run_ci_local.py --job test --skip-install
python3 scripts/run_ci_local.py --list
```

## Live Forge integration check

After starting Forge (`seiso forge`), verify API routes the UI depends on:

```bash
make live-check
# or: python3 scripts/live_frontend_backend_check.py
```

This script hits authenticated Forge endpoints (health, catalog, training models, etc.) and is **not** part of `make ci-fast` — it requires a running server on `http://127.0.0.1:8765`.

## Scope

Lint/type/test jobs target first-party code only:

- `seiso/`, `forge/`, `seiso_cli/`, `tests/`

`forge-ui/dist/` build output is excluded.
