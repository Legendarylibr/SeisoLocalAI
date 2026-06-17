#!/usr/bin/env python3
"""Run the codellama-compress quality gate locally (see docs/CI_LOCAL.md).

Jobs (former .github/workflows/*.yml):

  1. test      — install, Ruff, Black, smoke import, pytest
  2. imports   — eval-extra and code-eval import smokes
  3. security  — detect-secrets + pip-audit

Usage:
  python3 scripts/run_ci_local.py              # all jobs
  python3 scripts/run_ci_local.py --list       # show jobs and recommended matrix
  python3 scripts/run_ci_local.py --job test
  python3 scripts/run_ci_local.py --fast       # test + security (skip imports)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CI_DOC = "docs/CI_LOCAL.md"
CI_PYTHON = "3.11"

CI_ENV: dict[str, str] = {
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

ALL_JOBS = ("test", "imports", "security")
FAST_JOBS = ("test", "security")

SECRET_SCAN_SHELL = r"""
set -euo pipefail
if [ -f .secrets.baseline ]; then
  git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline
else
  detect-secrets scan . > .secrets.baseline
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
        candidate = root / "venv" / "Scripts" / "python.exe"
        if not candidate.is_file():
            candidate = root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = root / "venv" / "bin" / "python"
        if not candidate.is_file():
            candidate = root / ".venv" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _ci_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(CI_ENV)
    venv_bin = str(Path(resolve_python_bin(repo_root())).resolve().parent)
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


def _pip_upgrade_install(python: str, *packages: str) -> list[str]:
    cmd = [python, "-m", "pip", "install", "--upgrade", "pip"]
    if packages:
        cmd.extend(packages)
    return cmd


def job_test(root: Path, python: str, env: dict[str, str]) -> None:
    _banner(f"Job: test (Python {CI_PYTHON}, host {platform.system()})")

    _step("Install package and dev tools", _pip_upgrade_install(python), cwd=root, env=env)
    _step("Install project", [python, "-m", "pip", "install", "."], cwd=root, env=env)
    _step(
        "Install dev requirements",
        [python, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        cwd=root,
        env=env,
    )
    _step("Ruff", [python, "-m", "ruff", "check", "."], cwd=root, env=env)
    _step("Black", [python, "-m", "black", "--check", "."], cwd=root, env=env)
    _step(
        "Smoke import",
        [
            python,
            "-c",
            "import codellama_compress; import codellama_compress.cli; "
            "print(codellama_compress.__version__)",
        ],
        cwd=root,
        env=env,
    )
    _step("Pytest", [python, "-m", "pytest", "-q"], cwd=root, env=env)


def job_imports(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: imports (eval + code-eval smoke)")

    _step("Install eval extra", _pip_upgrade_install(python), cwd=root, env=env)
    _step(
        "Install project with eval extra",
        [python, "-m", "pip", "install", ".[eval]"],
        cwd=root,
        env=env,
    )
    _step(
        "Smoke import benchmarks",
        [python, "-c", "import codellama_compress.benchmarks; print('ok')"],
        cwd=root,
        env=env,
    )
    _step("Reinstall base package", _pip_upgrade_install(python), cwd=root, env=env)
    _step("Install project", [python, "-m", "pip", "install", "."], cwd=root, env=env)
    _step(
        "Smoke import code eval",
        [
            python,
            "-c",
            "import codellama_compress.code_eval; import codellama_compress.code_exec; print('ok')",
        ],
        cwd=root,
        env=env,
    )


def job_security(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: security (secrets + pip-audit)")

    _step(
        "Install security tools",
        _pip_upgrade_install(python, "detect-secrets", "pip-audit"),
        cwd=root,
        env=env,
    )
    _shell_step("Secret scan", SECRET_SCAN_SHELL, cwd=root, env=env)
    _step("Install project for audit", [python, "-m", "pip", "install", "."], cwd=root, env=env)
    pip_audit = shutil.which("pip-audit") or "pip-audit"
    _step("Dependency vulnerability audit", [pip_audit], cwd=root, env=env)


def _print_plan(*, python_bin: str, jobs: Sequence[str]) -> None:
    host = platform.system()
    print(f"Quality gate: {CI_DOC}")
    print(f"Local host: {host}  |  interpreter: {python_bin}  |  CI Python: {CI_PYTHON}")
    print("\nRecommended matrix (run --fast before release merges):")
    print(f"  - Linux, Python {CI_PYTHON}")
    print(f"  - macOS, Python {CI_PYTHON}")
    print("\nAll local jobs:")
    for job in ALL_JOBS:
        print(f"  - {job}")
    print("\nLocal run plan:")
    for job in jobs:
        print(f"  - {job}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local quality gate (docs/CI_LOCAL.md).")
    parser.add_argument(
        "--job",
        choices=ALL_JOBS,
        action="append",
        help="Run one job (repeatable). Default: all jobs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run test + security only (skip optional eval import smokes).",
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
        _print_plan(python_bin=python_bin, jobs=jobs)
        return 0

    _print_plan(python_bin=python_bin, jobs=jobs)
    env = _ci_env()
    success = False

    try:
        for job in jobs:
            if job == "test":
                job_test(root, python_bin, env)
            elif job == "imports":
                job_imports(root, python_bin, env)
            elif job == "security":
                job_security(root, python_bin, env)
            else:
                raise SystemExit(f"unknown job: {job}")
        success = True
    except subprocess.CalledProcessError as exc:
        print(f"\nLocal quality gate failed (exit {exc.returncode}).", file=sys.stderr)
        return exc.returncode

    if not success:
        return 1

    print("\nOK: run_ci_local.py finished successfully (all selected jobs passed).")
    if len(jobs) == len(ALL_JOBS):
        print(f"See {CI_DOC} for the recommended cross-platform matrix before large merges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
