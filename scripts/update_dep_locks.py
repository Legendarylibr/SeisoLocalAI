#!/usr/bin/env python3
"""Regenerate Python lockfile hashes and refresh locks/digests.json."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seiso.security.deps import (  # noqa: E402
    DEFAULT_DIGESTS_REL,
    build_digest_manifest,
    write_digest_manifest,
)

PYTHON_LOCK = Path("locks/python.lock")
NPM_LOCK = Path("forge-ui/package-lock.json")
PIP_COMPILE_DISPLAY_CMD = (
    "pip-compile --allow-unsafe --extra=dev --extra=forge --extra=train "
    "--generate-hashes --output-file=locks/python.lock --strip-extras pyproject.toml"
)
PIP_COMPILE_ARGS = [
    "pyproject.toml",
    "--extra",
    "forge",
    "--extra",
    "train",
    "--extra",
    "dev",
    "--generate-hashes",
    "--strip-extras",
    "--allow-unsafe",
    "--upgrade",
    "--output-file",
    str(PYTHON_LOCK),
]
UV_COMPILE_CMD = [
    "uv",
    "pip",
    "compile",
    *PIP_COMPILE_ARGS,
    "--custom-compile-command",
    PIP_COMPILE_DISPLAY_CMD,
]
PIP_COMPILE_CMD = [
    "pip-compile",
    *PIP_COMPILE_ARGS,
]


def compile_python_lock(repo_root: Path) -> int:
    if shutil.which("uv"):
        return subprocess.run(UV_COMPILE_CMD, cwd=repo_root, check=False).returncode
    if shutil.which("pip-compile"):
        return subprocess.run(PIP_COMPILE_CMD, cwd=repo_root, check=False).returncode
    print("missing uv or pip-compile; install uv or pip-tools and retry", file=sys.stderr)
    return 127


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Only refresh digests.json from existing lockfiles",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    python_lock = repo_root / PYTHON_LOCK
    npm_lock = repo_root / NPM_LOCK
    digests_path = repo_root / DEFAULT_DIGESTS_REL

    if not args.skip_python:
        result = compile_python_lock(repo_root)
        if result != 0:
            print("Python lock compilation failed", file=sys.stderr)
            return result

    if not python_lock.is_file():
        print(f"missing {python_lock}", file=sys.stderr)
        return 1
    if not npm_lock.is_file():
        print(f"missing {npm_lock}", file=sys.stderr)
        return 1

    manifest = build_digest_manifest(
        repo_root,
        {
            str(PYTHON_LOCK): python_lock,
            str(NPM_LOCK): npm_lock,
        },
    )
    write_digest_manifest(manifest, digests_path)
    print(f"updated {digests_path.relative_to(repo_root)}")
    for rel_path, digest in manifest["artifacts"].items():
        print(f"  {rel_path}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
