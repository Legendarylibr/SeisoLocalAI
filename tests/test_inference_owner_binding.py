"""Unit tests: Compat /v1 key bound to owner npub + cookie-primary sessions."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from forge.api.deps import clear_dependency_caches
from forge.config import ForgeSettings, get_settings
from forge.main import create_app
from forge.security.session_token import RETURN_TOKEN_HEADER, maybe_access_token
from tests.conftest import RETURN_TOKEN_HEADERS


def _settings(tmp_path, monkeypatch) -> ForgeSettings:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.delenv("SEISO_INFERENCE_API_KEY", raising=False)
    clear_dependency_caches()
    return ForgeSettings()


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "headers": raw, "path": "/"})


def test_maybe_access_token_requires_opt_in_header():
    token = "jwt-example"
    assert maybe_access_token(_request(), token) is None
    assert maybe_access_token(_request({"X-Seiso-Return-Token": "0"}), token) is None
    assert maybe_access_token(_request({RETURN_TOKEN_HEADER: "1"}), token) == token
    assert maybe_access_token(_request({"x-seiso-return-token": "1"}), token) == token


def test_bind_get_clear_inference_api_key_owner(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    pubkey = "ab" * 32
    assert settings.get_inference_api_key_owner() is None

    settings.bind_inference_api_key_owner(pubkey.upper())
    assert settings.get_inference_api_key_owner() == pubkey
    assert settings.inference_api_key_owner_file.is_file()
    assert oct(settings.inference_api_key_owner_file.stat().st_mode)[-3:] == "600"

    settings.clear_inference_api_key_owner()
    assert settings.get_inference_api_key_owner() is None
    assert not settings.inference_api_key_owner_file.exists()
    # Idempotent.
    settings.clear_inference_api_key_owner()


def test_bind_inference_api_key_owner_rejects_invalid_pubkey(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="64-char hex"):
        settings.bind_inference_api_key_owner("not-a-pubkey")
    with pytest.raises(ValueError, match="64-char hex"):
        settings.bind_inference_api_key_owner("zz" * 32)
    with pytest.raises(ValueError, match="64-char hex"):
        settings.bind_inference_api_key_owner("ab" * 16)


def test_get_inference_api_key_owner_ignores_corrupt_file(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    path = settings.inference_api_key_owner_file
    path.write_text("not-hex\n", encoding="utf-8")
    assert settings.get_inference_api_key_owner() is None
    path.write_text("ab" * 31, encoding="utf-8")
    assert settings.get_inference_api_key_owner() is None


def test_sync_inference_api_key_owner_rotates_only_on_change(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    owner_a = "aa" * 32
    owner_b = "bb" * 32
    key0 = settings.inference_api_key

    rotated = settings.sync_inference_api_key_owner(owner_a)
    assert rotated is True
    key1 = settings.inference_api_key
    assert key1 != key0
    assert settings.get_inference_api_key_owner() == owner_a

    rotated_again = settings.sync_inference_api_key_owner(owner_a)
    assert rotated_again is False
    assert settings.inference_api_key == key1

    rotated_b = settings.sync_inference_api_key_owner(owner_b)
    assert rotated_b is True
    assert settings.inference_api_key != key1
    assert settings.get_inference_api_key_owner() == owner_b


def test_sync_inference_api_key_owner_env_bound_still_binds(tmp_path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_INFERENCE_API_KEY", "seiso_sk_env_bound_test_key_value_xx")
    clear_dependency_caches()
    settings = ForgeSettings()
    assert settings.inference_api_key == "seiso_sk_env_bound_test_key_value_xx"

    owner = "cc" * 32
    rotated = settings.sync_inference_api_key_owner(owner)
    assert rotated is False  # cannot rotate env-bound key
    assert settings.inference_api_key == "seiso_sk_env_bound_test_key_value_xx"
    assert settings.get_inference_api_key_owner() == owner


def test_sync_env_bound_refuses_owner_rebind(tmp_path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_INFERENCE_API_KEY", "seiso_sk_env_bound_test_key_value_xx")
    clear_dependency_caches()
    settings = ForgeSettings()
    settings.sync_inference_api_key_owner("cc" * 32)
    with pytest.raises(RuntimeError, match="env-bound"):
        settings.sync_inference_api_key_owner("dd" * 32)


@pytest.mark.asyncio
async def test_login_omits_access_token_unless_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
        nsec = reg.json()["nsec"]
        assert reg.json().get("access_token")

        # Drop session so login is required.
        await client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {reg.json()['access_token']}"},
        )

        bare = await client.post("/api/auth/login", json={"nsec": nsec})
        assert bare.status_code == 200
        assert bare.json().get("access_token") in (None, "")
        assert client.cookies.get("seiso_token")

        opted = await client.post(
            "/api/auth/login",
            json={"nsec": nsec},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert opted.status_code == 200
        assert opted.json().get("access_token")


@pytest.mark.asyncio
async def test_legacy_unbound_inference_key_binds_on_first_compat_use(
    tmp_path, monkeypatch
):
    """Pre-binding installs: first valid /v1 call binds key to sole owner npub."""
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
        pubkey = reg.json()["user"]["nostr_pubkey"]
        settings = get_settings()
        key = settings.inference_api_key
        # Simulate legacy data dir with key but no owner file.
        settings.clear_inference_api_key_owner()
        assert settings.get_inference_api_key_owner() is None

        res = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        assert res.status_code == 200
        assert settings.get_inference_api_key_owner() == pubkey


@pytest.mark.asyncio
async def test_import_nsec_rebinds_compat_owner(tmp_path, monkeypatch):
    from seiso.research.nostr.keys import generate_keypair

    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        settings = get_settings()
        old_key = settings.inference_api_key
        old_owner = settings.get_inference_api_key_owner()

        pair = generate_keypair()
        imported = await client.put(
            "/api/settings/nostr/key",
            headers=headers,
            json={"secret": pair.nsec},
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["npub"] == pair.npub

        settings = get_settings()
        assert settings.get_inference_api_key_owner() == pair.public_hex
        assert settings.get_inference_api_key_owner() != old_owner
        assert settings.inference_api_key != old_key

        ok = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {settings.inference_api_key}"},
        )
        assert ok.status_code == 200
        stale = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {old_key}"},
        )
        assert stale.status_code == 401
