"""Supply-chain lockfile and digest verification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seiso.security.deps import (
    LockDigestError,
    build_digest_manifest,
    sha256_file,
    verify_lock_digests,
    verify_python_lock_has_hashes,
    write_digest_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_verify_lock_digests_passes_for_repo():
    verified = verify_lock_digests(repo_root=REPO_ROOT)
    assert "locks/python.lock" in verified
    assert "forge-ui/package-lock.json" in verified


def test_python_lock_requires_package_hashes():
    verify_python_lock_has_hashes(REPO_ROOT / "locks" / "python.lock")


def test_tampered_lockfile_fails_digest_check(tmp_path: Path):
    lock_path = tmp_path / "demo.lock"
    lock_path.write_text("pkg==1.0.0\n", encoding="utf-8")
    manifest_path = tmp_path / "digests.json"
    write_digest_manifest(
        build_digest_manifest(tmp_path, {"demo.lock": lock_path}),
        manifest_path,
    )

    lock_path.write_text("pkg==1.0.1\n", encoding="utf-8")
    with pytest.raises(LockDigestError, match="Digest mismatch"):
        verify_lock_digests(repo_root=tmp_path, digests_path=manifest_path)


def test_tampered_digest_manifest_fails(tmp_path: Path):
    lock_path = tmp_path / "demo.lock"
    lock_path.write_text("pkg==1.0.0\n", encoding="utf-8")
    manifest_path = tmp_path / "digests.json"
    write_digest_manifest(
        build_digest_manifest(tmp_path, {"demo.lock": lock_path}),
        manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["demo.lock"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LockDigestError, match="Digest mismatch"):
        verify_lock_digests(repo_root=tmp_path, digests_path=manifest_path)


def test_python_lock_without_hashes_is_rejected(tmp_path: Path):
    lock_path = tmp_path / "python.lock"
    lock_path.write_text("insecure==1.0.0\n", encoding="utf-8")
    with pytest.raises(LockDigestError, match="lack sha256 hashes"):
        verify_python_lock_has_hashes(lock_path)


def test_sha256_file_matches_known_value(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"seiso-lock-check\n")
    assert (
        sha256_file(sample)
        == "e9140ca2669aba95a6ff302815ebb84ef0137acab886757e64e0eb9266e8cfc7"
    )
