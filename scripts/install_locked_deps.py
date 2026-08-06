#!/usr/bin/env python3
"""Install locked Python dependencies with per-package SHA-256 verification."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_LOCK = REPO_ROOT / "locks" / "python.lock"


def _pip_install(args: list[str]) -> int:
    """Prefer uv when available (honors UV_TORCH_BACKEND for torch CPU wheels)."""
    if shutil.which("uv"):
        return subprocess.run(
            ["uv", "pip", "install", *args], cwd=REPO_ROOT, check=False
        ).returncode
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", *args],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run digest checks without installing packages",
    )
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Also install the local project editable with --no-deps",
    )
    args = parser.parse_args()

    # verify_dep_locks needs tomli (Py<3.11) and packaging for pyproject coverage.
    # Prefer uv: GitHub Actions uv venvs often have no `pip` module.
    helpers_rc = _pip_install(["-q", "tomli", "packaging"])
    if helpers_rc != 0:
        return helpers_rc

    verify = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_dep_locks.py")],
        cwd=REPO_ROOT,
        check=False,
    )
    if verify.returncode != 0:
        return verify.returncode
    if args.verify_only:
        return 0

    install_rc = _pip_install(["--require-hashes", "-r", str(PYTHON_LOCK)])
    if install_rc != 0:
        return install_rc

    if args.editable:
        return _pip_install(["-e", str(REPO_ROOT), "--no-deps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
