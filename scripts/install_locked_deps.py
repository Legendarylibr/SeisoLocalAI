#!/usr/bin/env python3
"""Install locked Python dependencies with per-package SHA-256 verification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_LOCK = REPO_ROOT / "locks" / "python.lock"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run digest checks without installing packages",
    )
    args = parser.parse_args()

    verify = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_dep_locks.py")],
        cwd=REPO_ROOT,
        check=False,
    )
    if verify.returncode != 0:
        return verify.returncode
    if args.verify_only:
        return 0

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(PYTHON_LOCK),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return install.returncode


if __name__ == "__main__":
    raise SystemExit(main())
