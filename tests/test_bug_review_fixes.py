"""Regression tests for the 2026-07-26 codebase bug-review fixes."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_strip_attributed_think_blocks():
    from seiso.chat.sanitize import strip_leaked_reasoning

    attributed = '<think channel="analysis">API_KEY=x</think>\nVisible'
    assert strip_leaked_reasoning(attributed) == "Visible"
    bare = "<think>secret</think>\nok"
    assert strip_leaked_reasoning(bare) == "ok"


def test_format_sample_uses_code_column():
    from seiso.training.datasets import DatasetFormat, format_sample

    assert format_sample({"code": "print(1)"}, DatasetFormat.TEXT, None) == "print(1)"


def test_bundled_result_rejects_failed_manifest(tmp_path: Path):
    from forge.orchestrators._bundled_job import (
        BundledJobContract,
        validate_bundled_result,
    )

    user_id = "user-1"
    run_dir = tmp_path / "compress" / user_id / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Manifest verification failed"):
        validate_bundled_result(
            tmp_path,
            user_id,
            {
                "run_dir": str(run_dir),
                "manifest": {"ok": False, "error": "hash mismatch"},
            },
            BundledJobContract(requires_manifest=True),
        )


@pytest.mark.asyncio
async def test_job_log_event_gen_skips_result_when_cancelled(tmp_path: Path):
    from forge.api.routes._stream import job_log_event_gen
    from forge.orchestrators.base import JobRecord, JobStatus, Orchestrator

    class _Orch(Orchestrator):
        kind = "test"

        async def execute(self, job_id: str, payload: dict) -> dict:
            return {}

    orch = _Orch(tmp_path)
    job_id = "job-cancel-result"
    rec = JobRecord(id=job_id, kind="test", user_id="u1")
    rec.status = JobStatus.CANCELLED
    rec.result = {"model_dir": "/tmp/x"}
    orch._jobs[job_id] = rec

    events = [event async for event in job_log_event_gen(orch, job_id)]
    assert not any(e.get("event") == "result" for e in events)


def test_restore_registry_keeps_modules_on_failure(monkeypatch):
    from seiso.kernels import lifecycle as life

    class Mod:
        pass

    m1, m2 = Mod(), Mod()
    m1._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m2._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m1.forward = lambda x: 1  # type: ignore[attr-defined]
    m2.forward = lambda x: 2  # type: ignore[attr-defined]

    orig_clear = life._clear_patch_markers

    def flaky(module: object) -> None:
        if module is m2:
            raise RuntimeError("restore boom")
        orig_clear(module)

    monkeypatch.setattr(life, "_clear_patch_markers", flaky)
    life._PATCH_REGISTRY.clear()
    life._PATCH_REGISTRY[42] = [m1, m2]
    with pytest.raises(RuntimeError, match="restore boom"):
        life._restore_registry_key(42)
    assert life._PATCH_REGISTRY[42] == [m2]
    life._PATCH_REGISTRY.clear()


def test_hf_token_no_host_fallback_for_user(monkeypatch, tmp_path: Path):
    from forge.db.crypto import generate_encryption_key
    from forge.services.hf_auth import resolve_hf_token

    monkeypatch.setenv("HF_TOKEN", "hf_host_secret")
    monkeypatch.delenv("SEISO_HF_ALLOW_HOST_TOKEN", raising=False)
    monkeypatch.setattr("forge.services.hf_auth._read_cli_token", lambda: "hf_cli")
    key = generate_encryption_key()
    token, source = resolve_hf_token(
        user_id="bob",
        data_dir=tmp_path,
        encryption_key=key,
    )
    assert token is None
    assert source == "none"

    monkeypatch.setenv("SEISO_HF_ALLOW_HOST_TOKEN", "1")
    token, source = resolve_hf_token(
        user_id="bob",
        data_dir=tmp_path,
        encryption_key=key,
    )
    assert token == "hf_host_secret"
    assert source == "env_hf"


def test_export_checksum_hashes_weight_files(tmp_path: Path):
    from seiso.research.provenance import directory_checksum_manifest

    (tmp_path / "model.safetensors").write_bytes(b"weights" * 1000)
    (tmp_path / "readme.txt").write_bytes(b"x" * 32)
    manifest = directory_checksum_manifest(
        tmp_path,
        max_files=None,
        max_file_bytes=8,
        always_hash_suffixes=(".safetensors",),
    )
    assert manifest["readme.txt"] == "skipped-large-file"
    assert manifest["model.safetensors"] not in {"skipped-large-file", "error"}
    assert len(manifest["model.safetensors"]) == 64


def test_knowledge_upload_refuses_symlink_destination(tmp_path: Path):
    import asyncio

    from fastapi import HTTPException

    from forge.api.routes import knowledge as kb
    from forge.config import ForgeSettings

    data = tmp_path / "data"
    uploads = data / "uploads" / "alice"
    uploads.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("secret", encoding="utf-8")
    dest = uploads / "planted.txt"
    dest.symlink_to(victim)

    class FakeUpload:
        filename = "planted.txt"

        async def read(self) -> bytes:
            return b"new-content"

    settings = ForgeSettings(data_dir=data)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            kb.upload_file(
                user_id="alice",
                settings=settings,
                file=FakeUpload(),  # type: ignore[arg-type]
            )
        )
    assert exc.value.status_code == 400
    assert victim.read_text(encoding="utf-8") == "secret"
    assert dest.is_symlink()
