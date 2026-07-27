"""Cross-user authorization tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import make_second_user, user_path


@pytest.mark.asyncio
async def test_thread_cross_user_idor(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
        )
        token_a = reg.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        thread = await client.post(
            "/api/inference/threads", json={"title": "secret"}, headers=headers_a
        )
        tid = thread.json()["id"]

        _, token_b = await make_second_user()
        headers_b = {"Authorization": f"Bearer {token_b}"}

        res = await client.get(
            f"/api/inference/threads/{tid}/messages", headers=headers_b
        )
        assert res.status_code == 404

        res_owner = await client.get(
            f"/api/inference/threads/{tid}/messages", headers=headers_a
        )
        assert res_owner.status_code == 200


@pytest.mark.asyncio
async def test_export_job_cross_user(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/auth/register",
            json={"generate": True},
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
        (out / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "test/model"}'
        )
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
