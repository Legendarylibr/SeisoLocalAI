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
    assert json.loads(providers[0]["config_json"]) == {"base_url": "http://127.0.0.1:8000"}


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
async def test_get_thread_with_messages_batches_load(db: Database):
    user = await db.create_user("hashed", "User", email="u@local.dev")
    thread = await db.create_thread(user["id"], "Chat", model_id="model-a")
    await db.add_message(thread["id"], "user", "hello")

    loaded_thread, messages = await db.get_thread_with_messages(thread["id"], user["id"])
    assert loaded_thread is not None
    assert loaded_thread["model_id"] == "model-a"
    assert messages[0]["content"] == "hello"

    missing_thread, missing_messages = await db.get_thread_with_messages("missing", user["id"])
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
