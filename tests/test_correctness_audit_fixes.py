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


def test_code_exec_blocks_list_times_large_int():
    from forge.tools.code_exec import _validate_code

    err = _validate_code("x = [0] * 10000000")
    assert err is not None
    assert "too large" in err.lower()


def test_compat_tools_reject_unknown_client_schemas():
    from fastapi import HTTPException

    from forge.api.schemas.compat import ChatCompletionRequest
    from forge.services.compat_chat import _assert_compat_tools_honesty

    body = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "CodebaseSearch"}}],
    )
    with pytest.raises(HTTPException) as exc:
        _assert_compat_tools_honesty(body)
    assert exc.value.status_code == 400


def test_compat_tools_allow_seiso_registry_names():
    from forge.api.schemas.compat import ChatCompletionRequest
    from forge.services.compat_chat import _assert_compat_tools_honesty

    body = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
    )
    _assert_compat_tools_honesty(body)


def test_detect_training_layout_ignores_stale_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    class _Torch:
        cuda = _Cuda()

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.training.multi_gpu import detect_training_layout

    layout = detect_training_layout()
    assert layout.world_size == 1
    assert layout.use_ddp is False


def test_detect_training_layout_keeps_multi_node(monkeypatch):
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 8

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.training.multi_gpu import detect_training_layout

    layout = detect_training_layout()
    assert layout.world_size == 16
    assert layout.local_rank == 1
    assert layout.use_ddp is True


def test_managed_vllm_tp_respects_cuda_visible_devices(monkeypatch):
    from seiso.inference import managed_vllm

    monkeypatch.setattr(
        managed_vllm, "resolve_vllm_command", lambda: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    )
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        managed_vllm.build_launch_command(
            model="org/model",
            tensor_parallel_size=8,
            cuda_visible_devices="0,1",
        )


def test_knowledge_retrieve_cache_busts_on_index_change(tmp_path: Path):
    from forge.services.knowledge_context import retrieve_knowledge_chunks
    from seiso.security import safe_join

    user = "u1"
    kb = "kb1"
    kb_dir = safe_join(tmp_path, "knowledge", user, kb)
    kb_dir.mkdir(parents=True)
    index = kb_dir / "index.jsonl"
    index.write_text(
        '{"id":"a","text":"alpha beta gamma","source":"a.txt"}\n',
        encoding="utf-8",
    )
    first = retrieve_knowledge_chunks(
        tmp_path, user_id=user, knowledge_base_id=kb, query="alpha", top_k=3
    )
    assert first
    index.write_text(
        '{"id":"b","text":"delta epsilon zeta","source":"b.txt"}\n',
        encoding="utf-8",
    )
    second = retrieve_knowledge_chunks(
        tmp_path, user_id=user, knowledge_base_id=kb, query="alpha", top_k=3
    )
    # Query no longer matches new corpus — must not return stale first hit.
    assert second == []
