#!/usr/bin/env python3
"""Run the Seiso local quality gate (see docs/CI_LOCAL.md).

Jobs:

  lint      — Ruff check + format, Pylint (E/F)
  types     — Mypy on seiso, forge, seiso_cli
  test      — smoke imports + pytest
  security  — Bandit, detect-secrets, pip-audit, pip check
  deps      — dependency lockfile integrity
  frontend  — bun/npm typecheck + production build (forge-ui)
  imports   — optional-extra import smokes (train, mlx)

Usage:
  python3 scripts/run_ci_local.py              # all jobs
  python3 scripts/run_ci_local.py --list       # show jobs and matrix
  python3 scripts/run_ci_local.py --job test
  python3 scripts/run_ci_local.py --fast       # lint + types + test + security
  python3 scripts/run_ci_local.py --fix        # auto-fix Ruff issues before lint
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
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
_DIAGNOSTIC_LOCATION_RE = re.compile(r"^(?P<path>.+?):\d+(?::\d+)?:\s*(?P<message>.+)$")

PY_PACKAGES = ("seiso", "forge", "seiso_cli", "tests")
PY_TYPE_PACKAGES = ("seiso", "forge", "seiso_cli")
PY_SOURCE_ROOTS = ("seiso", "forge", "seiso_cli")

ALL_JOBS = ("deps", "lint", "types", "test", "security", "frontend", "imports")
FAST_JOBS = ("deps", "lint", "types", "test", "security")
CHANGED_JOBS = ("lint", "test")

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


def _git_lines(root: Path, *args: str, check: bool = True) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=check,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def changed_files(root: Path, base: str | None) -> list[Path]:
    """Return changed tracked and untracked files relative to a merge base."""
    comparison_base = base or os.environ.get("CHANGED_BASE", "origin/main")
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", comparison_base],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if merge_base.returncode == 0:
        committed = _git_lines(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{merge_base.stdout.strip()}...HEAD",
        )
    else:
        committed = _git_lines(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "HEAD~1",
            check=False,
        )
    working = _git_lines(root, "diff", "--name-only", "--diff-filter=ACMR")
    staged = _git_lines(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    untracked = _git_lines(root, "ls-files", "--others", "--exclude-standard")
    paths = {
        Path(name)
        for name in (*committed, *working, *staged, *untracked)
        if (root / name).is_file()
    }
    return sorted(paths, key=lambda path: path.as_posix())


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
    for module in ("ruff", "mypy", "pylint", "bandit", "pytest", "xdist"):
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


def _install_project_locked(root: Path, python: str, env: dict[str, str]) -> None:
    """Install hashed locks/python.lock, then the editable project without resolving deps."""
    _step(
        "Install locked dependencies",
        [python, "scripts/install_locked_deps.py", "--editable"],
        cwd=root,
        env=env,
    )


def _install_project_unlocked(
    root: Path, python: str, env: dict[str, str], *, extra: str = "forge,train,dev"
) -> None:
    _step(
        "Install project (unlocked PyPI resolve)",
        [python, "-m", "pip", "install", "-e", f".[{extra}]"],
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

    def fingerprint(line: str) -> str:
        match = _DIAGNOSTIC_LOCATION_RE.match(line.strip())
        if match is None:
            return line.strip().replace("\\", "/")
        path = match.group("path").replace("\\", "/")
        return f"{path}: {match.group('message').strip()}"

    baseline_counter = collections.Counter(
        fingerprint(line)
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    current_counter = collections.Counter(
        fingerprint(line) for line in current_lines if line.strip()
    )
    new_counter = current_counter - baseline_counter
    fixed_counter = baseline_counter - current_counter
    new_items = sorted(new_counter.elements())
    fixed_items = sorted(fixed_counter.elements())

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
    files: Sequence[Path] | None = None,
) -> None:
    _banner(f"Job: lint (Python {CI_PYTHON}, host {platform.system()})")
    ruff_targets = [
        path.as_posix()
        for path in files or ()
        if path.suffix == ".py" and path.parts and path.parts[0] in PY_PACKAGES
    ]
    if files is not None and not ruff_targets:
        print("No changed Python files to lint.")
        return
    lint_targets = ruff_targets or list(PY_PACKAGES)

    if fix:
        _step(
            "Ruff auto-fix",
            [python, "-m", "ruff", "check", *lint_targets, "--fix", "--unsafe-fixes"],
            cwd=root,
            env=env,
        )
        _step(
            "Ruff format",
            [python, "-m", "ruff", "format", *lint_targets],
            cwd=root,
            env=env,
        )

    result = subprocess.run(
        [python, "-m", "ruff", "check", *lint_targets, "--output-format=json"],
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
        "Ruff format check",
        [python, "-m", "ruff", "format", "--check", *lint_targets],
        cwd=root,
        env=env,
    )

    pylint_targets = [
        path for path in lint_targets if Path(path).parts and Path(path).parts[0] in PY_SOURCE_ROOTS
    ]
    if pylint_targets:
        _step(
            "Pylint (errors/fatals)",
            [
                python,
                "-m",
                "pylint",
                *pylint_targets,
                "--jobs=0",
                "--disable=all",
                "--enable=E,F",
                "--disable=possibly-used-before-assignment",
                "--disable=invalid-enum-extension",
                "--score=n",
            ],
            cwd=root,
            env={**env, "PYLINTHOME": str(root / ".cache" / "pylint")},
        )


def job_deps(root: Path, python: str, env: dict[str, str]) -> None:
    _banner("Job: deps (lock verification)")
    _step(
        "Verify dependency lock digests",
        [python, "scripts/verify_dep_locks.py"],
        cwd=root,
        env=env,
    )


def job_types(root: Path, python: str, env: dict[str, str], *, update_baseline: bool) -> None:
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
            f"local interpreter is {version}. Skipping mypy here; use --python-bin "
            f"with Python {CI_PYTHON} for authoritative results.",
            file=sys.stderr,
        )
        return

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
    if result.returncode not in {0, 1} or (result.returncode and not errors):
        raise subprocess.CalledProcessError(result.returncode or 1, "mypy")
    _baseline_check(
        label="mypy",
        baseline_path=baseline_path,
        current_lines=errors,
        update_baseline=update_baseline,
    )


def _pytest_worker_args(workers: str | int, dist: str) -> list[str]:
    """Return pytest-xdist args, or [] when workers is disabled."""
    value = str(workers).strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return []
    if value in {"auto", "logical"} or value.isdigit():
        return ["-n", value, "--dist", dist]
    raise SystemExit(f"Invalid --pytest-workers={workers!r}; use 0, N, auto, or logical.")


def job_test(
    root: Path,
    python: str,
    env: dict[str, str],
    *,
    files: Sequence[Path] | None = None,
    workers: str | int = "0",
    dist: str = "loadscope",
    hardware_tests: bool = False,
) -> None:
    _banner("Job: test (smoke imports + pytest)")

    if files is None:
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
        test_targets = ["tests/"]
    else:
        production_changed = any(
            path.suffix == ".py" and path.parts and path.parts[0] in PY_SOURCE_ROOTS
            for path in files
        )
        test_targets = [
            path.as_posix()
            for path in files
            if path.suffix == ".py" and path.parts and path.parts[0] == "tests"
        ]
        if production_changed:
            test_targets = ["tests/"]
        elif not test_targets:
            print("No production or test Python changes to test.")
            return

    marker = "gpu and not slow" if hardware_tests else "not slow and not gpu"
    command = [
        python,
        "-m",
        "pytest",
        *test_targets,
        "-q",
        "-m",
        marker,
        *_pytest_worker_args(workers, dist),
    ]
    _step(
        "Pytest",
        command,
        cwd=root,
        env=env,
    )


def _pip_audit_ignore_vulns(root: Path) -> list[str]:
    """Load ``[tool.pip-audit].ignore-vulns`` from pyproject.toml (single source of truth)."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = cfg.get("tool", {}).get("pip-audit", {}).get("ignore-vulns", [])
    if not isinstance(ignores, list):
        raise SystemExit("pyproject.toml [tool.pip-audit].ignore-vulns must be a list")
    return [str(item) for item in ignores]


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
    _shell_step("Secret scan (detect-secrets)", SECRET_SCAN_SHELL, cwd=root, env=secret_env)
    _step("pip check", [python, "-m", "pip", "check"], cwd=root, env=env)
    audit_cmd = [
        python,
        "-m",
        "pip_audit",
        "--cache-dir",
        str(root / ".cache" / "pip-audit"),
        "--progress-spinner=off",
    ]
    for vuln_id in _pip_audit_ignore_vulns(root):
        audit_cmd.extend(["--ignore-vuln", vuln_id])
    _step("Dependency vulnerability audit", audit_cmd, cwd=root, env=env)


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
    # `bun test` is Bun's native runner (no jsdom); the package script is vitest.
    _step("Unit tests (vitest)", [*pm_cmd, "run", "test"], cwd=ui, env=env)
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
    print(f"Local host: {host}  |  interpreter: {python_bin}  |  CI Python: {CI_PYTHON}")
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
    parser.add_argument(
        "--unlocked-install",
        action="store_true",
        help="Resolve deps from PyPI via pyproject extras instead of locks/python.lock.",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="Run lint and directly changed test modules instead of the full local gate.",
    )
    parser.add_argument(
        "--changed-base",
        default=None,
        help="Git ref used as the changed-file merge base (default: origin/main).",
    )
    parser.add_argument(
        "--pytest-workers",
        default="0",
        metavar="N",
        help="Run pytest with N xdist workers, or 'auto'/'logical' (default: serial).",
    )
    parser.add_argument(
        "--pytest-dist",
        default="loadscope",
        choices=("load", "loadscope", "loadfile", "loadgroup", "worksteal", "no"),
        help="pytest-xdist distribution mode when workers are enabled (default: loadscope).",
    )
    parser.add_argument(
        "--hardware-tests",
        action="store_true",
        help="Run non-slow tests marked gpu instead of the CPU test selection.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    python_bin = args.python_bin or resolve_python_bin(root)

    workers = str(args.pytest_workers).strip()
    if workers.isdigit() and int(workers) < 0:
        parser.error("--pytest-workers must be zero or greater")
    if workers.lower() not in {"0", "auto", "logical"} and not workers.isdigit():
        parser.error("--pytest-workers must be 0, N, auto, or logical")
    if args.changed and args.fast:
        parser.error("--changed and --fast cannot be combined")
    if args.changed and args.job and any(job not in CHANGED_JOBS for job in args.job):
        parser.error("--changed only supports --job lint and --job test")

    if args.changed:
        jobs = list(dict.fromkeys(args.job or CHANGED_JOBS))
    elif args.fast:
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
    selected_files = changed_files(root, args.changed_base) if args.changed else None
    if selected_files is not None:
        print(f"\nChanged-file mode: {len(selected_files)} files selected.")

    try:
        if not args.skip_install:
            _banner("Setup")
            _step("Upgrade pip", _pip_upgrade(python_bin), cwd=root, env=env)
            if args.unlocked_install:
                _install_project_unlocked(root, python_bin, env)
            else:
                _install_project_locked(root, python_bin, env)
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
                    files=selected_files,
                )
            elif job == "types":
                job_types(root, python_bin, env, update_baseline=args.update_mypy_baseline)
            elif job == "test":
                job_test(
                    root,
                    python_bin,
                    env,
                    files=selected_files,
                    workers=workers,
                    dist=args.pytest_dist,
                    hardware_tests=args.hardware_tests,
                )
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
        print(f"See {CI_DOC} for the recommended cross-platform matrix before large merges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
