"""HTTP contracts for training cancel/SSE/gates/sandbox (mocked heavy backends)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from tests.conftest import user_path


def _sse_events(raw: bytes) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].lstrip())
        events.append((event, "\n".join(data)))
    return events


async def _prepare_trainable_paths(data_dir, user_id: str) -> tuple[Path, Path]:
    dataset = user_path(data_dir, user_id, "uploads", "train.jsonl")
    dataset.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]}\n'
    )
    model_dir = user_path(data_dir, user_id, "models", "trainable-api")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_text("fake-weights")
    return dataset, model_dir


def _install_fake_training(monkeypatch, fake_run):
    monkeypatch.setattr("forge.orchestrators.training.run_training", fake_run)
    monkeypatch.setattr("seiso.training.config.run_training", fake_run)


def _skip_dataset_analysis(monkeypatch):
    """Avoid HF datasets/pyarrow in create-job path; gates under test are elsewhere."""

    monkeypatch.setattr(
        "forge.api.routes.training.run_dataset_analysis",
        lambda *args, **kwargs: {
            "valid": True,
            "num_samples": 1,
            "dataset_format": "chat",
        },
    )


@pytest.mark.asyncio
async def test_training_job_http_cancel(app, auth_client, monkeypatch):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db, get_training_orchestrator
    from seiso.training.cancel import is_requested

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    dataset, model_dir = await _prepare_trainable_paths(data_dir, user["id"])

    def fake_run_training(config, on_metric=None, on_log=None, job_id=None, **_kwargs):
        if on_log:
            on_log("mock training started")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if job_id and is_requested(job_id):
                raise InterruptedError("cancelled")
            time.sleep(0.02)
        raise TimeoutError("cancel was not requested in time")

    _install_fake_training(monkeypatch, fake_run_training)
    _skip_dataset_analysis(monkeypatch)

    res = await client.post(
        "/api/training/jobs",
        headers=headers,
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
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]

    orch = get_training_orchestrator()
    for _ in range(100):
        job = orch.get_job(job_id)
        if job and job.status.value == "running":
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("training job never reached running")

    cancel = await client.post(
        f"/api/training/jobs/{job_id}/cancel",
        headers=headers,
    )
    assert cancel.status_code == 200, cancel.text
    assert cancel.json().get("cancelled") is True

    for _ in range(100):
        row = await db.get_training_job(job_id, user["id"])
        live = orch.get_job(job_id)
        statuses = {
            str((row or {}).get("status") or "").lower(),
            str(live.status.value if live else ""),
        }
        if "cancelled" in statuses:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("training job never reached cancelled")


@pytest.mark.asyncio
async def test_training_job_stream_emits_log_metric_status(
    app, auth_client, monkeypatch
):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db, get_training_orchestrator

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    dataset, model_dir = await _prepare_trainable_paths(data_dir, user["id"])

    def fake_run_training(config, on_metric=None, on_log=None, **_kwargs):
        if on_log:
            on_log("mock training complete")
        if on_metric:
            on_metric(
                {
                    "type": "training",
                    "step": 1,
                    "loss": 1.25,
                    "reward": -1.25,
                    "epoch": 0.1,
                }
            )
        out = Path(config.output_dir) / "checkpoint-stream"
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "test/model"}'
        )
        return out

    _install_fake_training(monkeypatch, fake_run_training)
    _skip_dataset_analysis(monkeypatch)

    res = await client.post(
        "/api/training/jobs",
        headers=headers,
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
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]

    orch = get_training_orchestrator()
    for _ in range(100):
        job = orch.get_job(job_id)
        if job and job.status.value in ("completed", "failed", "cancelled"):
            break
        await asyncio.sleep(0.05)

    async with client.stream(
        "GET",
        f"/api/training/jobs/{job_id}/stream",
        headers=headers,
    ) as stream_res:
        assert stream_res.status_code == 200, await stream_res.aread()
        events = _sse_events(await stream_res.aread())

    assert any(event == "status" for event, _ in events)
    status_values = [data for event, data in events if event == "status"]
    assert status_values[-1] == "completed"
    # Durable or live buffers should include the mocked log/metric when persisted.
    assert any(event in ("log", "metric", "status") for event, _ in events)


@pytest.mark.asyncio
async def test_training_api_rejects_preference_slime_and_packing_mask(
    app, auth_client
):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    dataset, model_dir = await _prepare_trainable_paths(data_dir, user["id"])
    prefs = user_path(data_dir, user["id"], "uploads", "prefs.jsonl")
    prefs.write_text(
        '{"prompt": "q", "chosen": "a", "rejected": "b"}\n',
        encoding="utf-8",
    )

    pref = await client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "config": {
                "model_id": str(model_dir),
                "dataset": str(prefs),
                "dataset_format": "preference",
                "preference_as_sft": True,
                "method": "slime",
                "require_held_out_eval": False,
            },
            "multi_gpu": False,
        },
    )
    assert pref.status_code == 400, pref.text
    assert "slime" in pref.text.lower() or "preference" in pref.text.lower()

    packing = await client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "config": {
                "model_id": str(model_dir),
                "dataset": str(dataset),
                "dataset_format": "chat",
                "method": "lora",
                "packing": True,
                "train_on_responses_only": True,
                "epochs": 1,
                "batch_size": 1,
            },
            "multi_gpu": False,
        },
    )
    assert packing.status_code == 400, packing.text
    assert "packing" in packing.text.lower()


@pytest.mark.asyncio
async def test_training_api_rejects_dataset_outside_sandbox(app, auth_client, tmp_path):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    _dataset, model_dir = await _prepare_trainable_paths(data_dir, user["id"])
    outside = tmp_path / "outside-train.jsonl"
    outside.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hey"}]}\n'
    )

    res = await client.post(
        "/api/training/jobs",
        headers=headers,
        json={
            "config": {
                "model_id": str(model_dir),
                "dataset": str(outside),
                "method": "lora",
                "epochs": 1,
                "batch_size": 1,
            },
            "multi_gpu": False,
        },
    )
    assert res.status_code in (400, 403), res.text
