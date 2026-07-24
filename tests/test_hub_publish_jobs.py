"""Hub publish job durability (F4-06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.db.crypto import generate_encryption_key
from forge.db.store import Database


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(
        tmp_path / "forge.db",
        encryption_key=generate_encryption_key(),
        ephemeral=True,
    )
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_hub_publish_job_persists_without_token(db: Database):
    await db.create_hub_publish_job(
        "user-1",
        {"repo_id": "org/model", "folder": "/tmp/x"},
        job_id="pub-1",
    )
    row = await db.get_hub_publish_job("pub-1", "user-1")
    assert row is not None
    assert row["status"] == "pending"
    cfg = json.loads(row["config_json"])
    assert cfg["repo_id"] == "org/model"
    assert "token" not in cfg
    assert "hf_token" not in json.dumps(cfg)

    await db.update_hub_publish_job_status(
        "pub-1",
        "completed",
        user_id="user-1",
        result={"repo_id": "org/model", "path": "/tmp/x"},
    )
    row2 = await db.get_hub_publish_job("pub-1", "user-1")
    assert row2["status"] == "completed"
    assert "org/model" in (row2.get("result_json") or "")


@pytest.mark.asyncio
async def test_hub_publish_stale_jobs_reconcile(db: Database):
    await db.create_hub_publish_job("user-1", {"repo_id": "a/b"}, job_id="pub-stale")
    n = await db.reconcile_stale_jobs(reason="Server restarted while job was active")
    assert n >= 1
    row = await db.get_hub_publish_job("pub-stale", "user-1")
    assert row["status"] == "failed"
    assert "restarted" in (row.get("error_text") or "").lower()
