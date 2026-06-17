# Local quality gate

This repository does **not** run GitHub Actions CI workflows on push/PR. Run the same checks **locally** before opening or updating a pull request.

**Entrypoints:**

```bash
python3 scripts/run_ci_local.py          # all jobs
python3 scripts/run_ci_local.py --fast   # test + security (daily loop)
python3 scripts/run_ci_local.py --list   # show jobs and recommended matrix
make ci
make ci-fast
```

Requires **Python 3.11+** and a venv with the project installed (see README quickstart).

---

## Jobs

| Job | Steps (matches former GitHub workflow) |
| --- | --- |
| **test** | `pip install .` + `requirements-dev.txt` → `ruff check .` → `black --check .` → smoke import → `pytest -q` |
| **imports** | `pip install ".[eval]"` → benchmarks import smoke → `pip install .` → code_eval/code_exec import smoke |
| **security** | `detect-secrets-hook` → `pip install .` → `pip-audit` |

### Test detail (former `ci.yml` → `lint-and-smoke`)

1. **Install:** `pip install .` and `pip install -r requirements-dev.txt`
2. **Lint:** `ruff check .` and `black --check .`
3. **Smoke:** import `codellama_compress` and `codellama_compress.cli`
4. **Tests:** `pytest -q`

### Imports detail (former `ci.yml` → `eval-extra-import` + `code-eval-import`)

Optional extra dependency checks. Skipped by `--fast` for quicker iteration; run **`make ci`** before release merges.

### Security detail (former `security.yml`)

1. **Secrets:** `detect-secrets-hook` with `.secrets.baseline`
2. **Audit:** install the package, then `pip-audit` (audits resolved project deps from `pyproject.toml`)

---

## Recommended matrix

Run **`python3 scripts/run_ci_local.py --fast`** on each row before merging:

| OS | Python |
| --- | --- |
| Linux | 3.11 |
| macOS | 3.11 |

---

## Lighter checks

```bash
make ci-fast
# or
./scripts/precheck.sh
```

Quick lint + unit tests only:

```bash
pip install -r requirements-dev.txt
ruff check .
black --check .
pytest -q
```

---

## Environment

`run_ci_local.py` sets:

| Variable | Value |
| --- | --- |
| `PIP_DISABLE_PIP_VERSION_CHECK` | `1` |
| `PYTHONUTF8` | `1` |
| `PYTHONIOENCODING` | `utf-8` |

Use **`PYTHON_BIN=/path/to/python`** to force a specific interpreter (otherwise `venv/` or `.venv/` is preferred when present).

See also [CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md).
