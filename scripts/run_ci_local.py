#!/usr/bin/env python3
"""Run the Seiso local quality gate (see docs/CI_LOCAL.md).

Jobs:

  lint      — Ruff check + format, Pylint (E/F)
  types     — Mypy on seiso, forge, seiso_cli
  test      — smoke imports + pytest
  security  — Bandit, detect-secrets, pip-audit, pip check
  deps      — dependency lockfile integrity
  frontend  — bun/npm typecheck + production build (forge-ui)
  imports   — optional-extra import smokes (train, compress, mlx)

Usage:
  python3 scripts/run_ci_local.py              # all jobs
  python3 scripts/run_ci_local.py --list       # show jobs and matrix
  python3 scripts/run_ci_local.py --job test
  python3 scripts/run_ci_local.py --fast       # lint + types + test + security
  python3 scripts/run_ci_local.py --fix        # auto-fix Ruff issues before lint
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CI_DOC = "docs/CI_LOCAL.md"
CI_PYTHON = "3.10"

CI_ENV: dict[str, str] = {
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

PY_PACKAGES = ("seiso", "forge", "seiso_cli", "tests")
PY_TYPE_PACKAGES = ("seiso", "forge", "seiso_cli")

ALL_JOBS = ("deps", "lint", "types", "test", "security", "frontend", "imports")
FAST_JOBS = ("deps", "lint", "types", "test", "security")

SECRET_SCAN_SHELL = r"""
set -euo pipefail
: "${DETECT_SECRETS_CMD:=detect-secrets}"
paths=(seiso forge seiso_cli tests forge-ui/src docs scripts .env.example README.md pyproject.toml Makefile)
if [ -f .secrets.baseline ]; then
  ${DETECT_SECRETS_CMD} scan --baseline .secrets.baseline "${paths[@]}"
else
  ${DETECT_SECRETS_CMD} scan "${paths[@]}" > .secrets.baseline
  git diff --exit-code .secrets.baseline
fi
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_python_bin(root: Path) -> str:
    requested = os.environ.get("PYTHON_BIN")
    if requested:
        return requested
    if os.name == "nt":
        candidates = (
            root / "venv" / "Scripts" / "python.exe",
            root / ".venv" / "Scripts" / "python.exe",
        )
    else:
        candidates = (
            root / "venv" / "bin" / "python",
            root / ".venv" / "bin" / "python",
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _ci_env(python: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CI_ENV)
    venv_bin = str(Path(python).resolve().parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def _banner(title: str) -> None:
    width = max(len(title) + 4, 72)
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _step(title: str, cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def _shell_step(title: str, script: str, *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    print(script.strip())
    subprocess.run(["bash", "-c", script], cwd=str(cwd), env=env, check=True)


def _pip_upgrade(python: str, *packages: str) -> list[str]:
    cmd = [python, "-m", "pip", "install", "--upgrade", "pip"]
    if packages:
        cmd.extend(packages)
    return cmd


def _ensure_dev_tools(root: Path, python: str, env: dict[str, str]) -> None:
    missing = []
    for module in ("ruff", "mypy", "pylint", "bandit", "pytest"):
        probe = subprocess.run(
            [python, "-c", f"import {module}"],
            cwd=str(root),
            env=env,
            capture_output=True,
        )
        if probe.returncode != 0:
            missing.append(module)
    if missing:
        print(f"Installing dev tools (missing: {', '.join(missing)})")
        _step(
            "Install dev requirements",
            [python, "-m", "pip", "install", "-r", "requirements-dev.txt"],
            cwd=root,
            env=env,
        )


def _install_project(
    root: Path, python: str, env: dict[str, str], *, extra: str = "forge,train,dev"
) -> None:
    _step(
        "Install project",
        [python, "-m", "pip", "install", "-e", f".[{extra}]"],
        cwd=root,
        env=env,
    )
    _step(
        "Install dev tools",
        [python, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        cwd=root,
        env=env,
    )


def _baseline_check(
    *,
    label: str,
    baseline_path: Path,
    current_lines: list[str],
    update_baseline: bool,
) -> None:
    if update_baseline:
        baseline_path.write_text(
            "\n".join(current_lines) + ("\n" if current_lines else ""),
            encoding="utf-8",
        )
        print(f"Updated {label} baseline ({len(current_lines)} items): {baseline_path}")
        return

    if not baseline_path.is_file():
        raise SystemExit(
            f"Missing {baseline_path}. Run with --update-{label}-baseline to create it."
        )

    baseline_set = set(baseline_path.read_text(encoding="utf-8").splitlines())
    current_set = set(current_lines)
    new_items = sorted(current_set - baseline_set)
    fixed_items = sorted(baseline_set - current_set)

    if new_items:
        print(f"\n--- New {label} issues (not in baseline) ---", file=sys.stderr)
        for line in new_items:
            print(line, file=sys.stderr)
        raise subprocess.CalledProcessError(1, label)

    print(
        f"{label}: {len(current_lines)} known issues (baseline), "
        f"{len(fixed_items)} fixed since baseline."
    )
    if fixed_items:
        print(f"Consider refreshing baseline with --update-{label}-baseline")


def job_lint(
    root: Path,
    python: str,
    env: dict[str, str],
    *,
    fix: bool,
    update_baseline: bool,
) -> None:
    _banner(f"Job: lint (Python {CI_PYTHON}, host {platform.system()})")

    if fix:
        _step(
            "Ruff auto-fix",
            [python, "-m", "ruff", "check", *PY_PACKAGES, "--fix", "--unsafe-fixes"],
            cwd=root,
            env=env,
        )
        _step(
            "Ruff format",
            [python, "-m", "ruff", "format", *PY_PACKAGES],
            cwd=root,
            env=env,
        )

    result = subprocess.run(
        [python, "-m", "ruff", "check", *PY_PACKAGES, "--output-format=json"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    ruff_data = json.loads(result.stdout or "[]")
    ruff_lines = sorted(
        f"{Path(item['filename']).resolve().relative_to(root)}:"
        f"{item['location']['row']}:{item['location']['column']}: "
        f"{item['code']} {item['message']}"
        for item in ruff_data
    )
    if ruff_lines:
        print("\n--- Ruff findings ---")
        for line in ruff_lines:
            print(line)
    _baseline_check(
        label="ruff",
        baseline_path=root / "scripts" / "ruff-baseline.txt",
        current_lines=ruff_lines,
        update_baseline=update_baseline,
    )

    _step(
        "Pylint (errors/fatals)",
        [
            python,
            "-m",
            "pylint",
            "seiso",
            "forge",
            "seiso_cli",
            "--jobs=1",
            "--disable=all",
            "--enable=E,F",
            "--disable=possibly-used-before-assignment",
            "--disable=invalid-enum-extension",
            "--score=n",
        ],
        cwd=root,
        env=env,
    )


def job_deps(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: deps (lock verification)")
    _step(
        "Verify dependency lock digests",
        [python, "scripts/verify_dep_locks.py"],
        cwd=root,
        env=env,
    )


def job_types(
    root: Path, python: str, env: dict[str, str], *, update_baseline: bool
) -> None:
    _banner("Job: types (mypy)")
    version = subprocess.run(
        [
            python,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if version != CI_PYTHON:
        print(
            f"WARNING: mypy baseline is calibrated for Python {CI_PYTHON}; "
            f"local interpreter is {version}. Use --python-bin with Python {CI_PYTHON} for authoritative results.",
            file=sys.stderr,
        )

    baseline_path = root / "scripts" / "mypy-baseline.txt"
    result = subprocess.run(
        [python, "-m", "mypy", *PY_TYPE_PACKAGES],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    print(combined, end="")

    errors = sorted(line for line in combined.splitlines() if ": error:" in line)
    _baseline_check(
        label="mypy",
        baseline_path=baseline_path,
        current_lines=errors,
        update_baseline=update_baseline,
    )


def job_test(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: test (smoke imports + pytest)")

    _step(
        "Smoke import core",
        [
            python,
            "-c",
            "import seiso; import forge; import seiso_cli; "
            "print('seiso', seiso.__version__ if hasattr(seiso, '__version__') else 'ok')",
        ],
        cwd=root,
        env=env,
    )
    _step(
        "Pytest",
        [python, "-m", "pytest", "tests/", "-q", "-m", "not slow"],
        cwd=root,
        env=env,
    )


def job_security(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: security (Bandit, secrets, pip-audit)")

    _step(
        "Bandit (medium+, production paths)",
        [
            python,
            "-m",
            "bandit",
            "-r",
            "seiso",
            "forge",
            "seiso_cli",
            "-l",
            "-c",
            "pyproject.toml",
        ],
        cwd=root,
        env=env,
    )
    secret_env = dict(env)
    secret_env["DETECT_SECRETS_CMD"] = f"{python} -m detect_secrets"
    _shell_step(
        "Secret scan (detect-secrets)", SECRET_SCAN_SHELL, cwd=root, env=secret_env
    )
    _step("pip check", [python, "-m", "pip", "check"], cwd=root, env=env)
    _step(
        "Dependency vulnerability audit",
        [
            python,
            "-m",
            "pip_audit",
            "--cache-dir",
            str(root / ".cache" / "pip-audit"),
            "--progress-spinner=off",
            "--ignore-vuln",
            "CVE-2025-3000",
            "--ignore-vuln",
            "PYSEC-2025-194",
        ],
        cwd=root,
        env=env,
    )


def _ui_pkg_manager(env: dict[str, str]) -> tuple[str, list[str]]:
    bun_install = Path(os.environ.get("BUN_INSTALL", Path.home() / ".bun"))
    bun_bin = bun_install / "bin" / "bun"
    if env.get("SEISO_USE_NPM") != "1" and bun_bin.is_file():
        return "bun", [str(bun_bin)]
    bun = shutil.which("bun", path=env.get("PATH"))
    if env.get("SEISO_USE_NPM") != "1" and bun:
        return "bun", [bun]
    npm = shutil.which("npm", path=env.get("PATH"))
    if npm:
        return "npm", [npm]
    raise SystemExit(
        "bun or npm not found; install Bun (https://bun.sh) or Node.js to run the frontend job"
    )


def job_frontend(root: Path, env: dict[str, str]) -> None:
    _banner("Job: frontend (forge-ui)")

    ui = root / "forge-ui"
    pm_name, pm_cmd = _ui_pkg_manager(env)

    if not (ui / "node_modules").is_dir():
        if pm_name == "bun":
            _step(
                "bun install --frozen-lockfile",
                [*pm_cmd, "install", "--frozen-lockfile"],
                cwd=ui,
                env=env,
            )
        else:
            _step("npm ci", [*pm_cmd, "ci"], cwd=ui, env=env)
    _step("TypeScript check", [*pm_cmd, "run", "typecheck"], cwd=ui, env=env)
    _step("Production build", [*pm_cmd, "run", "build"], cwd=ui, env=env)


def job_imports(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: imports (optional extras smoke)")

    _step(
        "Install train extra",
        [python, "-m", "pip", "install", "-e", ".[train]"],
        cwd=root,
        env=env,
    )
    _step(
        "Smoke import training stack",
        [
            python,
            "-c",
            "import transformers; import peft; import datasets; print('train ok')",
        ],
        cwd=root,
        env=env,
    )

    if platform.system() == "Darwin":
        _step(
            "Install mlx extra",
            [python, "-m", "pip", "install", "-e", ".[mlx]"],
            cwd=root,
            env=env,
        )
        _step(
            "Smoke import MLX",
            [
                python,
                "-c",
                "import importlib.util; "
                "print('mlx ok' if importlib.util.find_spec('mlx_lm') else 'mlx skipped')",
            ],
            cwd=root,
            env=env,
        )


def _print_plan(*, python_bin: str, jobs: Sequence[str], fix: bool) -> None:
    host = platform.system()
    print(f"Quality gate: {CI_DOC}")
    print(
        f"Local host: {host}  |  interpreter: {python_bin}  |  CI Python: {CI_PYTHON}"
    )
    if fix:
        print("Ruff auto-fix: enabled (--fix)")
    print("\nRecommended matrix (run --fast before merges):")
    print(f"  - Linux, Python {CI_PYTHON}")
    print(f"  - macOS, Python {CI_PYTHON}")
    print("\nAll local jobs:")
    for job in ALL_JOBS:
        print(f"  - {job}")
    print("\nLocal run plan:")
    for job in jobs:
        print(f"  - {job}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the local quality gate (docs/CI_LOCAL.md)."
    )
    parser.add_argument(
        "--job",
        choices=ALL_JOBS,
        action="append",
        help="Run one job (repeatable). Default: all jobs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run lint + types + test + security (skip frontend build and optional imports).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix Ruff issues before lint checks.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print recommended matrix vs local plan and exit.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python interpreter (default: venv/.venv or current python).",
    )
    parser.add_argument(
        "--update-mypy-baseline",
        action="store_true",
        help="Refresh scripts/mypy-baseline.txt from current mypy output.",
    )
    parser.add_argument(
        "--update-ruff-baseline",
        action="store_true",
        help="Refresh scripts/ruff-baseline.txt from current Ruff output.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install steps (assume deps already installed).",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    python_bin = args.python_bin or resolve_python_bin(root)

    if args.fast:
        jobs = list(FAST_JOBS)
    elif args.job:
        jobs = list(dict.fromkeys(args.job))
    else:
        jobs = list(ALL_JOBS)

    if args.list:
        _print_plan(python_bin=python_bin, jobs=jobs, fix=args.fix)
        return 0

    _print_plan(python_bin=python_bin, jobs=jobs, fix=args.fix)
    env = _ci_env(python_bin)

    try:
        if not args.skip_install:
            _banner("Setup")
            _step("Upgrade pip", _pip_upgrade(python_bin), cwd=root, env=env)
            _install_project(root, python_bin, env)
        else:
            _ensure_dev_tools(root, python_bin, env)

        for job in jobs:
            if job == "deps":
                job_deps(root, python_bin, env)
            elif job == "lint":
                job_lint(
                    root,
                    python_bin,
                    env,
                    fix=args.fix,
                    update_baseline=args.update_ruff_baseline,
                )
            elif job == "types":
                job_types(
                    root, python_bin, env, update_baseline=args.update_mypy_baseline
                )
            elif job == "test":
                job_test(root, python_bin, env)
            elif job == "security":
                job_security(root, python_bin, env)
            elif job == "frontend":
                job_frontend(root, env)
            elif job == "imports":
                job_imports(root, python_bin, env)
            else:
                raise SystemExit(f"unknown job: {job}")
    except subprocess.CalledProcessError as exc:
        print(f"\nLocal quality gate failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode

    print("\nOK: run_ci_local.py finished successfully (all selected jobs passed).")
    if len(jobs) == len(ALL_JOBS):
        print(
            f"See {CI_DOC} for the recommended cross-platform matrix before large merges."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
