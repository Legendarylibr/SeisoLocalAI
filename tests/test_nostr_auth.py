"""Forge Nostr npub/nsec authentication."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app
from seiso.research.nostr.keys import generate_keypair


@pytest.mark.asyncio
async def test_register_keygen_default_and_login():
    """Empty / generate-default register is the product path."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["auth_method"] == "nostr"
        assert status.json()["needs_onboarding"] is True

        reg = await client.post("/api/auth/register", json={})
        assert reg.status_code == 201, reg.text
        data = reg.json()
        assert data["nsec"] and str(data["nsec"]).startswith("nsec1")
        assert data["user"]["npub"].startswith("npub1")
        nsec = data["nsec"]
        npub = data["user"]["npub"]

        await client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {data['access_token']}"},
        )

        other = generate_keypair()
        bad = await client.post("/api/auth/login", json={"nsec": other.nsec})
        assert bad.status_code == 401

        ok = await client.post("/api/auth/login", json={"nsec": nsec})
        assert ok.status_code == 200
        assert ok.json()["user"]["npub"] == npub


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
async def test_second_register_forbidden():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/auth/register", json={"generate": True})
        assert first.status_code == 201
        second = await client.post("/api/auth/register", json={"generate": True})
        assert second.status_code == 403


@pytest.mark.asyncio
async def test_login_with_invalid_nsec_shape():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"generate": True})
        bad = await client.post("/api/auth/login", json={"nsec": "not-a-valid-key"})
        assert bad.status_code == 401
