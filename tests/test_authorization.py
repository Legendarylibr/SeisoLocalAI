"""Cross-user authorization tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from seiso.security import SecurityError
from tests.conftest import RETURN_TOKEN_HEADERS, make_second_user, user_path


@pytest.mark.asyncio
async def test_thread_cross_user_idor(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        token_a = reg.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        thread = await client.post(
            "/api/inference/threads", json={"title": "secret"}, headers=headers_a
        )
        tid = thread.json()["id"]

        _, token_b = await make_second_user()
        headers_b = {"Authorization": f"Bearer {token_b}"}

        res = await client.get(f"/api/inference/threads/{tid}/messages", headers=headers_b)
        assert res.status_code == 404

        res_owner = await client.get(f"/api/inference/threads/{tid}/messages", headers=headers_a)
        assert res_owner.status_code == 200


@pytest.mark.asyncio
async def test_export_job_cross_user(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )

        from forge.api.deps import get_db

        db = get_db()
        user = await db.get_sole_user()
        job = await db.create_export_job(user["id"], {"checkpoint": "x"})
        job_id = job["id"]

        _, token_b = await make_second_user("c@local.dev")
        headers_b = {"Authorization": f"Bearer {token_b}"}

        res = await client.get(f"/api/export/jobs/{job_id}/stream", headers=headers_b)
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_training_job_cross_user_cancel_and_stream(app, auth_client, monkeypatch):
    client, _token, headers_a, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    dataset = user_path(data_dir, user_a["id"], "uploads", "idor-train.jsonl")
    dataset.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]}\n'
    )
    model_dir = user_path(data_dir, user_a["id"], "models", "idor-trainable")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_text("fake-weights")

    def fake_run_training(config, on_metric=None, on_log=None, **_kwargs):
        from pathlib import Path

        out = Path(config.output_dir) / "checkpoint-idor"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text('{"base_model_name_or_path": "test/model"}')
        return out

    monkeypatch.setattr("forge.orchestrators.training.run_training", fake_run_training)
    monkeypatch.setattr("seiso.training.config.run_training", fake_run_training)
    monkeypatch.setattr(
        "forge.api.routes.training.run_dataset_analysis",
        lambda *args, **kwargs: {
            "valid": True,
            "num_samples": 1,
            "dataset_format": "chat",
        },
    )

    created = await client.post(
        "/api/training/jobs",
        headers=headers_a,
        json={
            "config": {
                "model_id": str(model_dir),
                "dataset": str(dataset),
                "method": "lora",
                "quant": "4bit",
                "epochs": 1,
                "batch_size": 1,
            },
            "multi_gpu": False,
        },
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]

    _, token_b = await make_second_user("idor-train@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    cancel = await client.post(
        f"/api/training/jobs/{job_id}/cancel",
        headers=headers_b,
    )
    assert cancel.status_code == 404

    stream = await client.get(
        f"/api/training/jobs/{job_id}/stream",
        headers=headers_b,
    )
    assert stream.status_code == 404


@pytest.mark.asyncio
async def test_cross_user_model_path_rejected(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    victim_model = user_path(data_dir, user_a["id"], "models", "secret.gguf")
    victim_model.write_text("fake")

    _, token_b = await make_second_user("path@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}
    user_b = await db.get_user_by_email("path@local.dev")
    own = user_path(data_dir, user_b["id"], "models", "own.gguf")
    own.write_text("fake")
    model = await db.add_model(user_id=user_b["id"], name="Own", path=str(own), format="gguf")

    res = await client.post(
        "/api/inference/chat",
        headers=headers_b,
        json={
            "model_id": model["id"],
            "model_path": str(victim_model),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_knowledge_bases_scoped_per_user(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    doc = user_path(data_dir, user["id"], "uploads", "doc.txt")
    doc.write_text("secret alpha beta gamma")

    ingest = await client.post(
        "/api/knowledge/ingest",
        headers=headers,
        json={"knowledge_base_id": "kb1", "source_path": str(doc)},
    )
    assert ingest.status_code == 200

    _, token_b = await make_second_user("kb@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    retrieve = await client.post(
        "/api/knowledge/retrieve",
        headers=headers_b,
        json={"knowledge_base_id": "kb1", "query": "alpha"},
    )
    assert retrieve.status_code == 200
    assert retrieve.json().get("results") == []


@pytest.mark.asyncio
async def test_cross_user_inference_cancel_rejected(app, auth_client):
    client, _token, headers_a, _data_dir = auth_client
    from forge.api.deps import get_db, get_inference_orchestrator

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    _, token_b = await make_second_user("cancel@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}
    orchestrator = get_inference_orchestrator()
    orchestrator._active_generation_user_id = user_a["id"]

    rejected = await client.post("/api/inference/cancel-generation", headers=headers_b)
    assert rejected.status_code == 403

    allowed = await client.post("/api/inference/cancel-generation", headers=headers_a)
    assert allowed.status_code == 200


@pytest.mark.asyncio
async def test_knowledge_ingest_blocks_other_user_index(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    doc = user_path(data_dir, user_a["id"], "uploads", "doc.txt")
    doc.write_text("private corpus")

    await client.post(
        "/api/knowledge/ingest",
        headers=headers,
        json={"knowledge_base_id": "kb1", "source_path": str(doc)},
    )

    victim_index = data_dir / "knowledge" / user_a["id"] / "kb1" / "index.jsonl"
    assert victim_index.exists()

    _, token_b = await make_second_user("ingest@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    steal = await client.post(
        "/api/knowledge/ingest",
        headers=headers_b,
        json={"knowledge_base_id": "stolen", "source_path": str(victim_index)},
    )
    assert steal.status_code == 400


@pytest.mark.asyncio
async def test_knowledge_rejects_outside_uploads(app, auth_client):
    client, _token, headers, data_dir = auth_client
    outside = data_dir / "raw.txt"
    outside.write_text("nope")

    res = await client.post(
        "/api/knowledge/ingest",
        headers=headers,
        json={"knowledge_base_id": "kb1", "source_path": str(outside)},
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_export_rejects_other_user_checkpoint(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    ckpt = user_path(data_dir, user_a["id"], "checkpoints", "run1")
    (ckpt / "adapter_config.json").write_text("{}")

    _, token_b = await make_second_user("export@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    res = await client.post(
        "/api/export/jobs",
        headers=headers_b,
        json={"checkpoint": str(ckpt), "formats": ["lora"]},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_training_rejects_cross_user_dataset(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    victim = user_path(data_dir, user_a["id"], "uploads", "train.jsonl")
    victim.write_text('{"messages":[{"role":"user","content":"secret"}]}\n')

    _, token_b = await make_second_user("train@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    res = await client.post(
        "/api/training/jobs",
        headers=headers_b,
        json={
            "config": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "dataset": str(victim),
                "method": "lora",
                "epochs": 1,
            },
        },
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_training_rejects_outside_uploads(app, auth_client):
    client, _token, headers, data_dir = auth_client
    outside = data_dir / "train.jsonl"
    outside.write_text('{"text":"nope"}\n')

    res = await client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "config": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "dataset": str(outside),
                "method": "lora",
                "epochs": 1,
            },
        },
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_compress_rejects_host_config_file(app, auth_client):
    client, _token, headers, _tmp = auth_client
    res = await client.post(
        "/api/compress/jobs",
        headers=headers,
        json={"preset": "smoke", "config_file": "/etc/passwd"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_cross_user_thread_messages_rejected(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    thread = await db.create_thread(user_a["id"], "Secret", None)

    _, token_b = await make_second_user("thread@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    res = await client.get(f"/api/inference/threads/{thread['id']}/messages", headers=headers_b)
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_cross_user_provider_delete_rejected(app, auth_client):
    client, _token, headers, _tmp = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    prov = await db.create_provider(
        user_a["id"], "Mine", "vllm", {"base_url": "http://127.0.0.1:8000"}
    )

    _, token_b = await make_second_user("prov@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    res = await client.delete(f"/api/providers/{prov['id']}", headers=headers_b)
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_cross_user_provider_inference_rejected(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user_a = await db.get_user_by_display_name("Admin")
    prov = await db.create_provider(
        user_a["id"], "Victim", "vllm", {"base_url": "http://127.0.0.1:8000"}
    )
    model_path = user_path(data_dir, user_a["id"], "models", "model.gguf")
    model_path.write_text("fake")
    await db.add_model(user_id=user_a["id"], name="Local", path=str(model_path), format="gguf")

    _, token_b = await make_second_user("provinf@local.dev")
    headers_b = {"Authorization": f"Bearer {token_b}"}
    own = user_path(
        data_dir,
        (await db.get_user_by_email("provinf@local.dev"))["id"],
        "models",
        "own.gguf",
    )
    own.write_text("fake")
    own_model = await db.add_model(
        user_id=(await db.get_user_by_email("provinf@local.dev"))["id"],
        name="Own",
        path=str(own),
        format="gguf",
    )

    res = await client.post(
        "/api/inference/chat",
        headers=headers_b,
        json={
            "model_id": own_model["id"],
            "provider_id": prov["id"],
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code == 404


def test_trainer_dataset_sandbox_blocks_other_user_path(tmp_path):
    from seiso.training.config import TrainConfig
    from seiso.training.datasets import load_training_dataset

    user_a_dataset = tmp_path / "uploads" / "user-a" / "private.jsonl"
    user_a_dataset.parent.mkdir(parents=True)
    user_a_dataset.write_text('{"text":"secret"}\n')
    user_b_root = tmp_path / "uploads" / "user-b"
    user_b_root.mkdir(parents=True)
    cfg = TrainConfig.model_validate(
        {
            "model_id": "test/model",
            "dataset": str(user_a_dataset),
            "sandbox_root": str(user_b_root),
        }
    )

    with pytest.raises(SecurityError):
        load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)
