"""Forge Nostr npub/nsec authentication."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from forge.config import get_settings
from forge.main import create_app
from seiso.research.nostr.keys import generate_keypair
from tests.conftest import RETURN_TOKEN_HEADERS


@pytest.mark.asyncio
async def test_register_omits_access_token_without_opt_in():
    """Browser path: HttpOnly cookie only — no JWT in the JSON body."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201, reg.text
        assert reg.json().get("access_token") in (None, "")
        assert client.cookies.get("seiso_token")
        me = await client.get("/api/auth/me")
        assert me.status_code == 200
        status = await client.get("/api/auth/status")
        assert status.json()["owner_npub"] == me.json()["npub"]


@pytest.mark.asyncio
async def test_register_keygen_default_and_login():
    """Empty / generate-default register is the product path."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["auth_method"] == "nostr"
        assert status.json()["needs_onboarding"] is True
        assert status.json().get("owner_npub") is None

        reg = await client.post("/api/auth/register", json={}, headers=RETURN_TOKEN_HEADERS)
        assert reg.status_code == 201, reg.text
        data = reg.json()
        assert data["nsec"] and str(data["nsec"]).startswith("nsec1")
        assert data["user"]["npub"].startswith("npub1")
        assert data.get("access_token")
        nsec = data["nsec"]
        npub = data["user"]["npub"]

        status2 = await client.get("/api/auth/status")
        assert status2.json()["owner_npub"] == npub
        settings = get_settings()
        assert settings.get_inference_api_key_owner() == data["user"]["nostr_pubkey"]

        await client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {data['access_token']}"},
        )

        other = generate_keypair()
        bad = await client.post("/api/auth/login", json={"nsec": other.nsec})
        assert bad.status_code == 401

        ok = await client.post("/api/auth/login", json={"nsec": nsec})
        assert ok.status_code == 200
        body = ok.json()
        assert body["user"]["npub"] == npub
        # Login never re-echoes the private key.
        assert body.get("nsec") in (None, "")


@pytest.mark.asyncio
async def test_register_import_nsec_and_login():
    pair = generate_keypair()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["auth_method"] == "nostr"
        assert status.json()["needs_onboarding"] is True

        reg = await client.post(
            "/api/auth/register",
            json={"nsec": pair.nsec},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201, reg.text
        data = reg.json()
        assert data["nsec"] is None  # imported keys are not echoed
        assert data["user"]["npub"] == pair.npub
        assert data["user"]["nostr_pubkey"] == pair.public_hex

        await client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {data['access_token']}"},
        )

        other = generate_keypair()
        bad = await client.post("/api/auth/login", json={"nsec": other.nsec})
        assert bad.status_code == 401

        ok = await client.post("/api/auth/login", json={"nsec": pair.nsec})
        assert ok.status_code == 200
        assert ok.json()["user"]["npub"] == pair.npub


@pytest.mark.asyncio
async def test_password_register_rejected():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert res.status_code == 422


@pytest.mark.asyncio
async def test_register_generate_and_nsec_mutually_exclusive():
    pair = generate_keypair()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/auth/register",
            json={"generate": True, "nsec": pair.nsec},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert res.status_code == 422


@pytest.mark.asyncio
async def test_user_public_view_never_includes_nsec():
    from forge.services.nostr_auth import user_public_view

    pair = generate_keypair()
    view = user_public_view(
        {
            "id": "u1",
            "email": None,
            "display_name": "Admin",
            "nostr_pubkey": pair.public_hex,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert view["npub"] == pair.npub
    assert "nsec" not in view
    assert view["nostr_pubkey"] == pair.public_hex


@pytest.mark.asyncio
async def test_register_keygen_returns_nsec_once_for_backup():
    """Product contract: generate returns nsec once; /me never echoes it."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        assert reg.status_code == 201, reg.text
        data = reg.json()
        nsec = data["nsec"]
        assert nsec.startswith("nsec1")
        assert data["user"]["npub"].startswith("npub1")
        token = data["access_token"]

        me = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me.status_code == 200
        me_body = me.json()
        assert me_body["npub"] == data["user"]["npub"]
        assert "nsec" not in me_body
        assert nsec not in me.text


@pytest.mark.asyncio
async def test_second_register_forbidden():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        assert first.status_code == 201
        second = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        assert second.status_code == 403


@pytest.mark.asyncio
async def test_login_with_invalid_nsec_shape():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        bad = await client.post("/api/auth/login", json={"nsec": "not-a-valid-key"})
        assert bad.status_code == 401
