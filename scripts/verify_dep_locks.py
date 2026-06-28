#!/usr/bin/env python3
"""Verify dependency lockfiles against locks/digests.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seiso.security.deps import (  # noqa: E402
    DEFAULT_DIGESTS_REL,
    LockDigestError,
    verify_lock_digests,
    verify_python_lock_has_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--digests",
        type=Path,
        default=None,
        help=f"Digest manifest path (default: <repo>/{DEFAULT_DIGESTS_REL})",
    )
    args = parser.parse_args()

    try:
        verified = verify_lock_digests(
            repo_root=args.repo_root, digests_path=args.digests
        )
        python_lock = args.repo_root / "locks" / "python.lock"
        verify_python_lock_has_hashes(python_lock)
    except LockDigestError as exc:
        print(f"dependency lock verification failed: {exc}", file=sys.stderr)
        return 1

    for rel_path in verified:
        print(f"ok {rel_path}")
    print("ok locks/python.lock package hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
