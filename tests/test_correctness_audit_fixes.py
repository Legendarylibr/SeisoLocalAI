"""Regression tests for the 2026-07-26 full-tree correctness audit fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.orchestrators.inference import InferenceOrchestrator
from forge.services.models import resolve_training_model_id
from seiso.security import SecurityError


def test_resolve_training_model_id_rejects_cross_user_path(tmp_path: Path):
    bob = tmp_path / "models" / "bob" / "secret"
    bob.mkdir(parents=True)
    (bob / "config.json").write_text("{}", encoding="utf-8")
    (bob / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(SecurityError, match="models/alice"):
        resolve_training_model_id(
            str(bob.resolve()),
            data_dir=tmp_path,
            user_id="alice",
            inventory=[],
        )


def test_resolve_training_model_id_rejects_host_path_outside_data_dir(tmp_path: Path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    (outside / "config.json").write_text("{}", encoding="utf-8")
    (outside / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(SecurityError):
        resolve_training_model_id(
            str(outside.resolve()),
            data_dir=tmp_path,
            user_id="alice",
            inventory=[],
        )


@pytest.mark.asyncio
async def test_inference_execute_stale_finally_keeps_newer_epoch(tmp_path: Path):
    orch = InferenceOrchestrator(tmp_path)
    orch._provider_chat = AsyncMock(return_value="ok")  # type: ignore[method-assign]
    orch._emit_log = MagicMock()  # type: ignore[method-assign]

    epoch1 = orch.begin_generation_for_user("user-a")
    job_id = orch.create_job(user_id="user-a")

    # Simulate cancel → new reservation while first execute is about to finish.
    async def _hijack_provider(*_a, **_k):
        orch.end_generation_for_user("user-a", epoch=epoch1)
        orch.begin_generation_for_user("user-a")
        return "partial"

    orch._provider_chat = _hijack_provider  # type: ignore[method-assign]
    result = await orch.execute(
        job_id,
        {
            "user_id": "user-a",
            "messages": [{"role": "user", "content": "hi"}],
            "provider": {"provider_type": "openai", "config": {}},
        },
    )
    assert result["content"] == "partial"
    # Stale execute finally must not clear the newer reservation.
    assert orch._active_generation_user_id == "user-a"


def test_rl_quant_checkpoint_path_is_export_source_not_quality(tmp_path: Path):
    from seiso.rl_quant.config_builder import _path_overrides

    ckpt = tmp_path / "checkpoints" / "run"
    ckpt.mkdir(parents=True)
    overrides = _path_overrides(
        {"checkpoint_path": str(ckpt), "gguf_export": True},
        product=None,
    )
    assert overrides.get("llama_cpp_gguf_export_source") == str(ckpt)
    assert overrides.get("llama_cpp_gguf_export_enabled") is True
    assert "external_quality_path" not in overrides


def test_rl_quant_quality_sidecar_still_sets_external_quality(tmp_path: Path):
    from seiso.rl_quant.config_builder import _path_overrides

    side = tmp_path / "scores.json"
    side.write_text("{}", encoding="utf-8")
    overrides = _path_overrides({"quality_sidecar": str(side)}, product=None)
    assert overrides["external_quality_path"] == str(side)


def test_external_quality_applied_requires_external_sources():
    from types import SimpleNamespace

    from seiso.adaptive_quant.recommendation import _external_quality_applied

    ok = [
        SimpleNamespace(metrics=SimpleNamespace(perplexity_source="external:perplexity"))
    ]
    miss = [
        SimpleNamespace(
            metrics=SimpleNamespace(perplexity_source="simulator_missing_external_perplexity")
        )
    ]
    assert _external_quality_applied(ok) is True
    assert _external_quality_applied(miss) is False
    assert _external_quality_applied([]) is False


def test_nemo_relative_dotdot_override_is_sandboxed(tmp_path: Path, monkeypatch):
    from seiso.nemo_rl.config import NeMoRLConfig
    from seiso.nemo_rl.runner import train_nemo_rl
    from seiso.security import SecurityError

    sandbox = tmp_path / "data"
    sandbox.mkdir()
    out = sandbox / "checkpoints" / "u1" / "job"
    out.mkdir(parents=True)
    # cwd outside sandbox; relative .. must still be checked after resolve.
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    cfg = NeMoRLConfig(
        model_id="org/model",
        output_dir=out,
        sandbox_root=sandbox,
        dry_run=True,
        extra_overrides=("data.path=../escape.json",),
    )
    with pytest.raises(SecurityError, match="outside sandbox"):
        train_nemo_rl(cfg)


def test_rate_limiter_evicts_idle_ips():
    from forge.security.auth import RateLimiter

    limiter = RateLimiter(max_per_minute=100)
    for i in range(300):
        limiter._hits[f"ip-{i}"] = [0.0]  # all expired vs monotonic now
    limiter.check("fresh-ip")
    assert "fresh-ip" in limiter._hits
    assert len(limiter._hits) < 300


@pytest.mark.asyncio
async def test_training_job_config_encrypted_at_rest(tmp_path: Path):
    import base64
    import json

    from forge.db.crypto import resolve_encryption_key
    from forge.db.store import Database

    key = resolve_encryption_key(base64.b64encode(b"\x02" * 32).decode())
    db = Database(tmp_path / "forge.db", encryption_key=key, ephemeral=True)
    user = await db.create_user("hashed", "User", email="t@local.dev")
    job = await db.create_training_job(
        user["id"], {"model_id": "org/model", "dataset": "uploads/x.jsonl"}
    )

    conn = await db._ensure_conn()
    async with conn.execute(
        "SELECT config_json FROM training_jobs WHERE id = ?", (job["id"],)
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert str(row["config_json"]).startswith("enc:v1:")

    loaded = await db.get_training_job(job["id"], user["id"])
    assert json.loads(loaded["config_json"])["model_id"] == "org/model"
    await db.close()
