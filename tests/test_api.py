import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches
from forge.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_health(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_onboarding_flow(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["needs_onboarding"] is True

        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]

        me = await client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert me.status_code == 200
        assert me.json()["display_name"] == "Admin"

        reg2 = await client.post(
            "/api/auth/register",
            json={"password": "securepass2"},
        )
        assert reg2.status_code == 403


@pytest.mark.asyncio
async def test_onboarding_requires_storage_choice_when_unconfigured(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.delenv("SEISO_DB_EPHEMERAL", raising=False)
    monkeypatch.delenv("SEISO_DB_STORAGE_MODE", raising=False)
    monkeypatch.delenv("SEISO_DB_ENCRYPTION_KEY", raising=False)
    clear_dependency_caches()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.status_code == 200
        assert status.json()["storage_mode_configured"] is False

        missing = await client.post(
            "/api/auth/register", json={"password": "securepass1"}
        )
        assert missing.status_code == 400

        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1", "storage_mode": "persistent"},
        )
        assert reg.status_code == 201
        assert (tmp_path / ".storage_mode").read_text(
            encoding="utf-8"
        ).strip() == "persistent"
        assert (tmp_path / "forge.db").exists()


@pytest.mark.asyncio
async def test_reset_session_returns_instance_to_onboarding(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        thread = await client.post(
            "/api/inference/threads",
            json={"title": "old session"},
            headers=headers,
        )
        assert thread.status_code == 200

        csrf = client.cookies.get("seiso_csrf")
        assert csrf, "register should issue CSRF cookie"
        csrf_headers = {"X-CSRF-Token": csrf}

        bad = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "wrong"},
            headers=csrf_headers,
        )
        assert bad.status_code == 400

        blocked = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "RESET"},
        )
        assert blocked.status_code == 403

        reset = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "RESET"},
            headers=csrf_headers,
        )
        assert reset.status_code == 200
        assert reset.json()["needs_onboarding"] is True
        assert reset.json()["rows_deleted"] >= 2

        status = await client.get("/api/auth/status")
        assert status.status_code == 200
        assert status.json()["needs_onboarding"] is True

        old_me = await client.get("/api/auth/me", headers=headers)
        assert old_me.status_code in {401, 404}

        # Compat /v1 must also reject ghost JWTs after wipe.
        old_v1 = await client.get("/v1/models", headers=headers)
        assert old_v1.status_code == 401

        reg2 = await client.post(
            "/api/auth/register",
            json={"password": "securepass2"},
        )
        assert reg2.status_code == 201


@pytest.mark.asyncio
async def test_ephemeral_onboarding_preserves_existing_forge_db(monkeypatch, tmp_path):
    legacy = tmp_path / "forge.db"
    legacy.write_text("do-not-delete", encoding="utf-8")
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.delenv("SEISO_DB_EPHEMERAL", raising=False)
    monkeypatch.delenv("SEISO_DB_STORAGE_MODE", raising=False)
    monkeypatch.delenv("SEISO_DB_ENCRYPTION_KEY", raising=False)
    clear_dependency_caches()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1", "storage_mode": "ephemeral"},
        )
        assert reg.status_code == 201
    assert legacy.read_text(encoding="utf-8") == "do-not-delete"


@pytest.mark.asyncio
async def test_protected_route_requires_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/models")
        assert res.status_code == 401
