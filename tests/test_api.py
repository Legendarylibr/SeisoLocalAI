import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches
from forge.main import create_app
from tests.conftest import RETURN_TOKEN_HEADERS


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
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
        body = reg.json()
        token = body["access_token"]
        assert body.get("nsec", "").startswith("nsec1")
        assert body["user"].get("npub", "").startswith("npub1")

        me = await client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert me.status_code == 200
        assert me.json()["display_name"] == "Admin"
        assert me.json().get("npub", "").startswith("npub1")

        # Login with the generated nsec (CSRF required for cookie-session POSTs).
        csrf = client.cookies.get("seiso_csrf")
        assert csrf
        logged_out = await client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {token}", "X-CSRF-Token": csrf},
        )
        assert logged_out.status_code == 200
        login = await client.post("/api/auth/login", json={"nsec": body["nsec"]})
        assert login.status_code == 200
        assert login.json()["user"]["npub"] == body["user"]["npub"]

        reg2 = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
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
            "/api/auth/register", json={"generate": True}
        , headers=RETURN_TOKEN_HEADERS)
        assert missing.status_code == 400

        reg = await client.post(
            "/api/auth/register",
            json={"generate": True, "storage_mode": "persistent"},
            headers=RETURN_TOKEN_HEADERS,
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
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
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

        from forge.config import get_settings

        old_inference_key = get_settings().inference_api_key
        assert old_inference_key

        reset = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "RESET"},
            headers=csrf_headers,
        )
        assert reset.status_code == 200
        assert reset.json()["needs_onboarding"] is True
        assert reset.json()["rows_deleted"] >= 2
        assert reset.json().get("inference_key_rotated") is True
        assert reset.json().get("owner_cleared") is True
        assert reset.json().get("owner_npub") is None

        status = await client.get("/api/auth/status")
        assert status.status_code == 200
        assert status.json()["needs_onboarding"] is True
        assert status.json().get("owner_npub") is None

        old_me = await client.get("/api/auth/me", headers=headers)
        assert old_me.status_code in {401, 404}

        # Compat /v1 must also reject ghost JWTs after wipe.
        old_v1 = await client.get("/v1/models", headers=headers)
        assert old_v1.status_code == 401

        # Prior Compat inference key must not survive ownership wipe.
        stale_key = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {old_inference_key}"},
        )
        assert stale_key.status_code == 401
        assert get_settings().inference_api_key != old_inference_key
        assert get_settings().get_inference_api_key_owner() is None

        reg2 = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg2.status_code == 201
        assert get_settings().get_inference_api_key_owner() == reg2.json()["user"][
            "nostr_pubkey"
        ]


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
            json={"generate": True, "storage_mode": "ephemeral"},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
    assert legacy.read_text(encoding="utf-8") == "do-not-delete"


@pytest.mark.asyncio
async def test_protected_route_requires_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/models")
        assert res.status_code == 401
