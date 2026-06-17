# Local CI / quality gate

Run the full local quality gate before opening PRs or cutting releases.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,dev]"
pip install -r requirements-dev.txt

make ci-fast      # daily loop: lint + types + test + security
make ci           # full gate (+ frontend build + optional import smokes)
make ci-list      # show jobs and recommended matrix
```

Or use the shell wrapper:

```bash
./scripts/precheck.sh
```

## Jobs

| Job | What it runs | Skipped by `--fast` |
|-----|--------------|---------------------|
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

Refresh after fixing types:

```bash
python3 scripts/run_ci_local.py --job types --update-mypy-baseline --skip-install
```

### Test detail

- Installs `.[forge,dev]` unless `--skip-install`
- Runs unit/integration tests excluding `@pytest.mark.slow`
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

Hooks: detect-secrets, ruff (+ format), mypy, bandit, pytest (pre-push only).

## Environment variables

| Variable | Effect |
|----------|--------|
| `PYTHON_BIN` | Force interpreter (default: `.venv/bin/python` or `venv/bin/python`) |

## Recommended matrix

Run `make ci-fast` on each row before merging significant changes:

| OS | Python | Notes |
|----|--------|-------|
| Linux | 3.10+ | primary CI target |
| macOS | 3.10+ | MLX import smoke in full `make ci` |

## Individual jobs

```bash
make lint
make types
make test
make security
make frontend
make imports

python3 scripts/run_ci_local.py --job test --skip-install
python3 scripts/run_ci_local.py --list
```

## Scope

Lint/type/test jobs target first-party code only:

- `seiso/`, `forge/`, `seiso_cli/`, `tests/`

`third_party/` vendored trees have their own CI. `forge-ui/dist/` build output is excluded.
