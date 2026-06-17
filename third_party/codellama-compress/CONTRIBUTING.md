# Contributing

## Development
1. Create a feature branch from `main`.
2. Make focused changes with clear commit messages.
3. Run the local quality gate before opening a PR.

## Local quality gate

From the repository root:

```bash
make ci-fast    # test + security (daily loop)
make ci         # full gate including eval import smokes
make ci-list    # show jobs and recommended matrix
```

See [docs/CI_LOCAL.md](docs/CI_LOCAL.md) for job details.

## Required Checks
- **Test job:** Ruff, Black, smoke import, `pytest -q`
- **Imports job:** eval-extra and code-eval import smokes (included in `make ci`)
- **Security job:** `detect-secrets` and `pip-audit`

## Dependency management

- `pyproject.toml` is the **source of truth** for dependencies.
- `requirements.txt` is **auto-generated** for tooling compatibility.
  Regenerate it with:

```bash
python scripts/export_requirements.py
```

## Pull Requests
- Keep PRs small and reviewable.
- Explain purpose, scope, and test results.
- Ensure no credentials or secrets are committed.
- Confirm `make ci-fast` (or `make ci` when changing optional extras) passed locally.
