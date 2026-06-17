"""Tests for ephemeral encrypted SQLite storage."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from forge.db.crypto import decrypt_field, encrypt_field, resolve_encryption_key
from forge.db.store import Database

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


@pytest.mark.asyncio
async def test_chat_messages_encrypted_at_rest(db: Database):
    user = await db.create_user("u@local.dev", "hashed", "User")
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
    user = await db.create_user("u@local.dev", "hashed", "User")
    await db.create_provider(user["id"], "OpenAI", "openai", {"api_key": "sk-test"})

    conn = await db._ensure_conn()
    async with conn.execute("SELECT config_json FROM providers") as cur:
        row = await cur.fetchone()
    assert row is not None
    assert str(row["config_json"]).startswith("enc:v1:")

    providers = await db.list_providers(user["id"])
    assert json.loads(providers[0]["config_json"]) == {"api_key": "sk-test"}


@pytest.mark.asyncio
async def test_ephemeral_db_wiped_on_close(db: Database):
    await db.create_user("u@local.dev", "hashed", "User")
    assert await db.user_count() == 1
    await db.close()

    fresh = Database(
        db.path,
        encryption_key=db._encryption_key,
        ephemeral=True,
    )
    assert await fresh.user_count() == 0
    await fresh.close()
