"""End-to-end API flows with mocked heavy backends."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from seiso.security import assert_within
from tests.conftest import user_path


@pytest.mark.asyncio
async def test_openai_rejects_path_outside_sandbox(app, auth_client):
    client, _token, headers, tmp_path = auth_client
    outside = tmp_path.parent / "outside-model.gguf"
    outside.write_text("fake")

    res = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": str(outside),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code in (400, 404)


@pytest.mark.asyncio
async def test_openai_chat_with_inventory_model(app, auth_client, monkeypatch):
    client, _token, headers, tmp_path = auth_client
    from forge.api.deps import get_db

    db = get_db()
    users = await db.get_user_by_display_name("Admin")
    model_file = user_path(tmp_path, users["id"], "models", "test-model", "model.gguf")
    model_file.write_text("fake")

    await db.add_model(
        user_id=users["id"],
        name="Test Model",
        path=str(model_file),
        format="gguf",
    )

    monkeypatch.setattr(
        "forge.orchestrators.inference.LocalInferenceRunner.chat",
        AsyncMock(return_value="Hello from Seiso"),
    )

    res = await client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["choices"][0]["message"]["content"] == "Hello from Seiso"


@pytest.mark.asyncio
async def test_training_job_e2e(app, auth_client, monkeypatch):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    dataset = user_path(data_dir, user["id"], "uploads", "train.jsonl")
    dataset.write_text('{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]}\n')

    def fake_run_training(config, on_metric=None):
        if on_metric:
            on_metric({"type": "training", "step": 1, "loss": 1.5, "reward": -1.5, "epoch": 0.1})
        out = Path(config.output_dir) / "checkpoint-e2e"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text('{"base_model_name_or_path": "test/model"}')
        return out

    monkeypatch.setattr("forge.orchestrators.training.run_training", fake_run_training)

    res = await client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "config": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "dataset": str(dataset),
                "method": "lora",
                "quant": "4bit",
                "epochs": 1,
                "batch_size": 1,
            },
            "multi_gpu": False,
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    match = None
    from forge.api.deps import get_training_orchestrator

    orch = get_training_orchestrator()
    for _ in range(100):
        job = orch.get_job(job_id)
        if job and job.status.value in ("completed", "failed"):
            match = {"id": job_id, "status": job.status.value}
            break
        await asyncio.sleep(0.05)

    assert match is not None
    assert match["status"] == "completed"


@pytest.mark.asyncio
async def test_export_lora_e2e(app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    ckpt = user_path(data_dir, user["id"], "checkpoints", "run1")
    (ckpt / "adapter_config.json").write_text('{"r": 16}')
    (ckpt / "adapter_model.bin").write_text("fake-weights")

    assert_within(data_dir, ckpt)

    res = await client.post(
        "/api/export/jobs",
        headers=headers,
        json={"checkpoint": str(ckpt), "formats": ["lora"]},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    job = None
    from forge.api.deps import get_export_orchestrator

    orch = get_export_orchestrator()
    for _ in range(100):
        job_rec = orch.get_job(job_id)
        if job_rec and job_rec.status.value in ("completed", "failed"):
            job = await db.get_export_job(job_id, user["id"])
            break
        await asyncio.sleep(0.05)

    assert job is not None
    assert job["status"] == "completed"
    outputs = json.loads(job.get("output_paths_json") or "{}")
    assert "lora" in outputs


@pytest.mark.asyncio
async def test_inference_chat_e2e(app, auth_client, monkeypatch):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    model_path = user_path(data_dir, user["id"], "models", "model.gguf")
    model_path.write_text("fake")
    model = await db.add_model(user_id=user["id"], name="Local", path=str(model_path), format="gguf")

    monkeypatch.setattr(
        "forge.orchestrators.inference.LocalInferenceRunner.chat",
        AsyncMock(return_value="streamed reply"),
    )

    res = await client.post(
        "/api/inference/chat",
        headers=headers,
        json={
            "model_id": model["id"],
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert res.status_code == 200
    assert "streamed reply" in str(res.json())


@pytest.mark.asyncio
async def test_dataset_sandbox_blocks_outside_path(tmp_path):
    pytest.importorskip("datasets")
    from seiso.security import SecurityError
    from seiso.training.datasets import load_training_dataset

    sandbox = tmp_path / "data"
    sandbox.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")

    with pytest.raises(SecurityError):
        load_training_dataset(outside, sandbox_root=sandbox)
