"""Verify dependency lockfiles against a separate SHA-256 digest manifest."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any

from seiso.security import SecurityError

DEFAULT_DIGESTS_REL = Path("locks/digests.json")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

_HASH_LINE = re.compile(r"^\s+--hash=sha256:")
_PACKAGE_LINE = re.compile(r"^[a-zA-Z0-9][\w.\-]*==")


class LockDigestError(SecurityError):
    """Raised when a lockfile fails digest or hash-policy verification."""


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"File exceeds hash limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def load_digest_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LockDigestError("Digest manifest must be a JSON object")
    data: dict[str, Any] = raw
    if data.get("algorithm") != "sha256":
        raise LockDigestError(f"Unsupported digest algorithm: {data.get('algorithm')!r}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise LockDigestError("Digest manifest missing non-empty 'artifacts' object")
    return data


def verify_lock_digests(
    *,
    repo_root: Path | None = None,
    digests_path: Path | None = None,
) -> list[str]:
    """Verify tracked lockfiles match digests in the separate manifest."""
    root = (repo_root or DEFAULT_REPO_ROOT).resolve()
    manifest_path = (digests_path or (root / DEFAULT_DIGESTS_REL)).resolve()
    if not manifest_path.is_file():
        raise LockDigestError(f"Digest manifest not found: {manifest_path}")

    expected_by_path = load_digest_manifest(manifest_path)["artifacts"]
    verified: list[str] = []

    for rel_path, expected_digest in expected_by_path.items():
        if not isinstance(rel_path, str) or not isinstance(expected_digest, str):
            raise LockDigestError("Digest manifest entries must be strings")
        if len(expected_digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in expected_digest
        ):
            raise LockDigestError(f"Invalid SHA-256 digest for {rel_path!r}")

        lock_path = (root / rel_path).resolve()
        if not lock_path.is_file():
            raise LockDigestError(f"Lockfile missing: {lock_path}")

        actual_digest = sha256_file(lock_path)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise LockDigestError(
                f"Digest mismatch for {rel_path}: "
                f"expected {expected_digest}, computed {actual_digest}"
            )
        verified.append(rel_path)

    return verified


def verify_python_lock_has_hashes(lock_path: Path) -> None:
    """Require every pinned package in a pip-compile lock to declare sha256 hashes."""
    if not lock_path.is_file():
        raise LockDigestError(f"Python lockfile missing: {lock_path}")

    current_package: str | None = None
    saw_hash = False
    missing_hashes: list[str] = []

    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue

        if _PACKAGE_LINE.match(line):
            if current_package is not None and not saw_hash:
                missing_hashes.append(current_package)
            current_package = line.split()[0]
            saw_hash = _HASH_LINE.match(line) is not None
            continue

        if _HASH_LINE.match(line):
            saw_hash = True
            continue

        if current_package is not None and not line.startswith(" "):
            if not saw_hash:
                missing_hashes.append(current_package)
            current_package = None
            saw_hash = False

    if current_package is not None and not saw_hash:
        missing_hashes.append(current_package)

    if missing_hashes:
        sample = ", ".join(missing_hashes[:5])
        suffix = "..." if len(missing_hashes) > 5 else ""
        raise LockDigestError(
            f"{len(missing_hashes)} package(s) in {lock_path.name} lack sha256 hashes: "
            f"{sample}{suffix}"
        )


def build_digest_manifest(repo_root: Path, artifacts: dict[str, Path]) -> dict[str, Any]:
    manifest_artifacts: dict[str, str] = {}
    for rel_path, absolute_path in artifacts.items():
        manifest_artifacts[rel_path] = sha256_file(absolute_path)
    return {
        "version": 1,
        "algorithm": "sha256",
        "artifacts": manifest_artifacts,
    }


def write_digest_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    path.write_text(payload, encoding="utf-8")
