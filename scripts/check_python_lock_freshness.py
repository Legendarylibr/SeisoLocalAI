#!/usr/bin/env python3
"""Fail if locks/python.lock is stale relative to pyproject.toml.

Recompiles forge+train+dev with the existing lock as a constraint (no --upgrade).
Any new direct/transitive pins required by pyproject, or pin drift under constraints,
causes a non-zero exit. Run ``python scripts/update_dep_locks.py`` to refresh.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seiso.security.deps import (  # noqa: E402
    LockDigestError,
    locked_package_versions,
)

PYTHON_LOCK = Path("locks/python.lock")


def _compile_constrained(repo_root: Path, output: Path) -> None:
    if not shutil.which("uv"):
        raise LockDigestError("uv is required for lock freshness checks")
    cmd = [
        "uv",
        "pip",
        "compile",
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
        "--universal",
        "--python-version",
        "3.10",
        "--constraint",
        str(PYTHON_LOCK),
        "--output-file",
        str(output),
        "--quiet",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise LockDigestError(
            "constrained lock recompile failed (pyproject may require versions "
            f"outside locks/python.lock):\n{detail}"
        )


def verify_python_lock_freshness(repo_root: Path) -> list[str]:
    current_lock = repo_root / PYTHON_LOCK
    if not current_lock.is_file():
        raise LockDigestError(f"missing {current_lock}")

    with tempfile.TemporaryDirectory(prefix="seiso-lock-check-") as tmp:
        compiled = Path(tmp) / "python.lock"
        _compile_constrained(repo_root, compiled)
        old = locked_package_versions(current_lock)
        new = locked_package_versions(compiled)

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    changed = sorted(name for name in set(old) & set(new) if old[name] != new[name])
    if only_old or only_new or changed:
        parts: list[str] = []
        if only_new:
            parts.append(f"missing from lock: {', '.join(only_new[:20])}")
        if only_old:
            parts.append(f"extra in lock: {', '.join(only_old[:20])}")
        if changed:
            sample = ", ".join(
                f"{name} ({old[name]} -> {new[name]})" for name in changed[:20]
            )
            parts.append(f"pin drift: {sample}")
        raise LockDigestError(
            "locks/python.lock is stale vs pyproject.toml; "
            "run: python scripts/update_dep_locks.py\n" + "; ".join(parts)
        )
    return [f"{name}=={version}" for name, version in sorted(old.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    args = parser.parse_args()
    try:
        pins = verify_python_lock_freshness(args.repo_root.resolve())
    except LockDigestError as exc:
        print(f"lock freshness check failed: {exc}", file=sys.stderr)
        return 1
    print(f"ok locks/python.lock freshness ({len(pins)} pins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
