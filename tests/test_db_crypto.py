"""Tests for ephemeral encrypted SQLite storage."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from forge.db.crypto import (
    decrypt_field,
    encrypt_field,
    load_encryption_key_file,
    persist_encryption_key_file,
    resolve_encryption_key,
)
from forge.db.store import Database, DatabaseCryptoError

_TEST_KEY = base64.b64encode(b"\x01" * 32).decode()


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(
        tmp_path / "forge.db",
        encryption_key=resolve_encryption_key(_TEST_KEY),
        ephemeral=True,
    )


@pytest.mark.asyncio
async def test_encrypt_decrypt_roundtrip():
    key = resolve_encryption_key(_TEST_KEY)
    plaintext = '{"api_key": "sk-secret"}'
    encrypted = encrypt_field(plaintext, key)
    assert encrypted.startswith("enc:v1:")
    assert decrypt_field(encrypted, key) == plaintext


@pytest.mark.asyncio
async def test_resolve_encryption_key_hex():
    hex_key = "01" * 32
    assert resolve_encryption_key(hex_key) == b"\x01" * 32


def test_load_encryption_key_file_raw_binary(tmp_path: Path):
    key = b"\x98\xab" + b"\x00" * 30
    key_file = tmp_path / ".db_encryption_key"
    key_file.write_bytes(key)
    assert load_encryption_key_file(key_file) == key


def test_load_encryption_key_file_base64_text(tmp_path: Path):
    key = b"\x01" * 32
    key_file = tmp_path / ".db_encryption_key"
    persist_encryption_key_file(key_file, key)
    assert load_encryption_key_file(key_file) == key


def test_forge_settings_loads_legacy_binary_db_key(tmp_path: Path, monkeypatch):
    from forge.config import ForgeSettings, get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("SEISO_DB_EPHEMERAL", raising=False)
    monkeypatch.delenv("SEISO_DB_STORAGE_MODE", raising=False)
    monkeypatch.delenv("SEISO_DB_ENCRYPTION_KEY", raising=False)

    key = b"\x98\xab" + os.urandom(30)
    (tmp_path / ".storage_mode").write_text("persistent\n", encoding="utf-8")
    (tmp_path / ".db_encryption_key").write_bytes(key)

    settings = ForgeSettings(data_dir=tmp_path, db_encryption_key="")
    assert settings.db_encryption_key_bytes == key
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_chat_messages_encrypted_at_rest(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    thread = await db.create_thread(user["id"], "Test")
    await db.add_message(thread["id"], "user", "secret prompt", {"tool": "x"})

    conn = await db._ensure_conn()
    async with conn.execute(
        "SELECT content, metadata_json FROM chat_messages WHERE thread_id = ?",
        (thread["id"],),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert not str(row["content"]).startswith("secret")
    assert str(row["content"]).startswith("enc:v1:")

    messages = await db.get_messages(thread["id"])
    assert messages[0]["content"] == "secret prompt"
    assert json.loads(messages[0]["metadata_json"]) == {"tool": "x"}


@pytest.mark.asyncio
async def test_provider_config_encrypted_at_rest(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    await db.create_provider(
        user["id"], "Local vLLM", "vllm", {"base_url": "http://127.0.0.1:8000"}
    )

    conn = await db._ensure_conn()
    async with conn.execute("SELECT config_json FROM providers") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert str(row["config_json"]).startswith("enc:v1:")

    providers = await db.list_providers(user["id"])
    assert json.loads(providers[0]["config_json"]) == {
        "base_url": "http://127.0.0.1:8000"
    }


@pytest.mark.asyncio
async def test_ephemeral_db_wiped_on_close(db: Database):
    await db.create_user("hashed", "User", email="u@local.dev")
    assert await db.user_count() == 1
    await db.close()

    fresh = Database(
        db.path,
        encryption_key=db._encryption_key,
        ephemeral=True,
    )
    assert await fresh.user_count() == 0
    await fresh.close()


@pytest.mark.asyncio
async def test_corrupt_encrypted_chat_message_fails_safely(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    thread = await db.create_thread(user["id"], "Test")
    await db.add_message(thread["id"], "user", "secret prompt")

    conn = await db._ensure_conn()
    await conn.execute(
        "UPDATE chat_messages SET content = ? WHERE thread_id = ?",
        ("enc:v1:not-valid-base64", thread["id"]),
    )
    await conn.commit()

    with pytest.raises(DatabaseCryptoError, match="could not be decrypted"):
        await db.get_messages(thread["id"])


@pytest.mark.asyncio
async def test_upsert_model_preserves_id_on_update(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    created = await db.upsert_model(
        user["id"],
        "hf:org/model",
        name="Model v1",
        path="/models/v1",
        format="hf",
        size_bytes=100,
        metadata={"rev": 1},
    )
    updated = await db.upsert_model(
        user["id"],
        "hf:org/model",
        name="Model v2",
        path="/models/v2",
        format="hf",
        size_bytes=200,
        metadata={"rev": 2},
    )
    assert updated["id"] == created["id"]
    assert updated["name"] == "Model v2"
    assert updated["path"] == "/models/v2"
    assert json.loads(updated["metadata_json"]) == {"rev": 2}
    assert len(await db.list_models(user["id"])) == 1


@pytest.mark.asyncio
async def test_job_events_append_tail_and_prune(db: Database):
    user = await db.create_user("hashed", "User", email="events@local.dev")
    await db.append_job_event(
        job_id="job-1",
        user_id=user["id"],
        kind="training",
        event_type="log",
        payload={"line": "starting"},
    )
    await db.append_job_event(
        job_id="job-1",
        user_id=user["id"],
        kind="training",
        event_type="metric",
        payload={"loss": 1.0},
    )

    rows = await db.list_job_events("job-1", user["id"])
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[0]["payload"] == {"line": "starting"}
    assert rows[1]["payload"] == {"loss": 1.0}

    metric_rows = await db.list_job_events(
        "job-1", user["id"], event_types=("metric",)
    )
    assert len(metric_rows) == 1

    deleted = await db.prune_job_events("job-1", user["id"], keep_last=1)
    assert deleted == 1
    remaining = await db.list_job_events("job-1", user["id"])
    assert [row["event_type"] for row in remaining] == ["metric"]


@pytest.mark.asyncio
async def test_job_events_concurrent_append_sequences(db: Database):
    """Shared-connection BEGIN IMMEDIATE must not nest under concurrent emitters."""
    import asyncio

    user = await db.create_user("hashed", "Concurrent", email="concurrent@local.dev")

    async def _emit(i: int) -> None:
        await db.append_job_event(
            job_id="job-concurrent",
            user_id=user["id"],
            kind="export",
            event_type="log",
            payload={"line": f"line-{i}"},
        )

    await asyncio.gather(*(_emit(i) for i in range(40)))
    rows = await db.list_job_events("job-concurrent", user["id"])
    sequences = [row["sequence"] for row in rows]
    assert sequences == list(range(1, 41))


@pytest.mark.asyncio
async def test_model_path_lookup_index_exists(db: Database):
    conn = await db._ensure_conn()
    async with conn.execute("PRAGMA index_list(local_models)") as cur:
        indexes = {row["name"] for row in await cur.fetchall()}
    assert "idx_models_user_created" in indexes
    assert "idx_models_user_path" in indexes
    assert "idx_models_user_name" in indexes


@pytest.mark.asyncio
async def test_list_models_merges_user_and_global_rows_by_created_at(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    conn = await db._ensure_conn()
    rows = [
        (
            "global-new",
            None,
            "Global new",
            "/models/global-new",
            "global:new",
            "gguf",
            1,
            "{}",
            "2026-01-03T00:00:00+00:00",
        ),
        (
            "user-mid",
            user["id"],
            "User mid",
            "/models/user-mid",
            "hf:user/mid",
            "gguf",
            1,
            "{}",
            "2026-01-02T00:00:00+00:00",
        ),
        (
            "global-old",
            None,
            "Global old",
            "/models/global-old",
            "global:old",
            "gguf",
            1,
            "{}",
            "2026-01-01T00:00:00+00:00",
        ),
    ]
    await conn.executemany(
        """INSERT INTO local_models
           (id, user_id, name, path, source, format, size_bytes, metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    await conn.commit()

    assert [row["id"] for row in await db.list_models(user["id"])] == [
        "global-new",
        "user-mid",
        "global-old",
    ]


@pytest.mark.asyncio
async def test_ordered_list_indexes_exist(db: Database):
    conn = await db._ensure_conn()
    expected = {
        "training_jobs": "idx_jobs_user_created",
        "chat_messages": "idx_messages_thread_created",
        "export_jobs": "idx_export_jobs_user_created",
        "compress_jobs": "idx_compress_jobs_user_created",
        "distill_rl_jobs": "idx_distill_rl_jobs_user_created",
        "providers": "idx_providers_user_created",
    }
    for table, index in expected.items():
        async with conn.execute(f"PRAGMA index_list({table})") as cur:
            indexes = {row["name"] for row in await cur.fetchall()}
        assert index in indexes


@pytest.mark.asyncio
async def test_get_thread_with_messages_batches_load(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    thread = await db.create_thread(user["id"], "Chat", model_id="model-a")
    await db.add_message(thread["id"], "user", "hello")

    loaded_thread, messages = await db.get_thread_with_messages(
        thread["id"], user["id"]
    )
    assert loaded_thread is not None
    assert loaded_thread["model_id"] == "model-a"
    assert messages[0]["content"] == "hello"

    missing_thread, missing_messages = await db.get_thread_with_messages(
        "missing", user["id"]
    )
    assert missing_thread is None
    assert missing_messages == []


@pytest.mark.asyncio
async def test_add_message_can_update_thread_model(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    thread = await db.create_thread(user["id"], "Chat", model_id="model-a")
    await db.add_message(thread["id"], "user", "hello", model_id="model-b")

    updated = await db.get_thread_for_user(thread["id"], user["id"])
    assert updated is not None
    assert updated["model_id"] == "model-b"

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

