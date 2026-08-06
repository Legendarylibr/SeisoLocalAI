"""CSRF protection, cookie auth, and rate limiting tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches
from forge.main import create_app
from tests.conftest import RETURN_TOKEN_HEADERS


@pytest.fixture
def app():
    return create_app()


def _csrf_headers(client) -> dict[str, str]:
    """Extract CSRF token from client cookies for double-submit header."""
    token = client.cookies.get("seiso_csrf")
    assert token, "CSRF cookie should be set on login/register"
    return {"X-CSRF-Token": token}


@pytest.mark.asyncio
async def test_csrf_blocks_cookie_mutation_without_header(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        # Cookie session without CSRF header should be rejected
        res = await client.post("/api/inference/threads", json={"title": "test"})
        assert res.status_code == 403
        assert "CSRF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_csrf_allows_cookie_mutation_with_header(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        res = await client.post(
            "/api/inference/threads",
            json={"title": "test"},
            headers=_csrf_headers(client),
        )
        assert res.status_code == 200


@pytest.mark.asyncio
async def test_bearer_auth_bypasses_csrf(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        token = reg.json()["access_token"]

        res = await client.post(
            "/api/inference/threads",
            json={"title": "bearer"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200


@pytest.mark.asyncio
async def test_empty_bearer_does_not_bypass_csrf(app):
    """Empty Authorization: Bearer must not skip CSRF while cookie auth works."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        res = await client.post(
            "/api/inference/threads",
            json={"title": "empty-bearer"},
            headers={"Authorization": "Bearer "},
        )
        assert res.status_code == 403
        assert "CSRF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_junk_bearer_does_not_bypass_csrf(app):
    """S1-010: non-JWT Bearer text must not skip CSRF."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        res = await client.post(
            "/api/inference/threads",
            json={"title": "junk-bearer"},
            headers={"Authorization": "Bearer not-a-valid-jwt"},
        )
        assert res.status_code == 403
        assert "CSRF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_inference_api_key_bypasses_csrf_on_v1(app):
    """Configured inference API key is a real Bearer credential (not junk)."""
    from forge.config import get_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        # Register binds/rotates the Compat key to the owner npub — read after.
        key = get_settings().inference_api_key
        assert key
        res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        # May fail on missing model, but must not fail CSRF.
        assert res.status_code in {400, 404, 409, 500}
        assert "CSRF" not in str(res.json().get("detail", ""))


@pytest.mark.asyncio
async def test_csrf_blocks_v1_without_header(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        res = await client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 403
        assert "CSRF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_csrf_rejects_mutation_outside_api_prefixes(app):
    """Default-deny: mutating requests outside /api and /v1 fail CSRF."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201

        res = await client.post("/other")
        assert res.status_code == 403
        assert "CSRF" in res.json()["detail"]


def test_validate_csrf_denies_mutation_outside_api_unit():
    from unittest.mock import MagicMock

    from forge.security.csrf import validate_csrf

    request = MagicMock()
    request.method = "POST"
    request.url.path = "/other"
    request.headers = {}
    request.cookies = {}
    assert not validate_csrf(request)


def test_validate_csrf_allows_safe_method_outside_api_unit():
    from unittest.mock import MagicMock

    from forge.security.csrf import validate_csrf

    request = MagicMock()
    request.method = "GET"
    request.url.path = "/other"
    request.headers = {}
    request.cookies = {}
    assert validate_csrf(request)


@pytest.mark.asyncio
async def test_cookie_session_auth(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201
        assert client.cookies.get("seiso_token")
        assert client.cookies.get("seiso_csrf")

        me = await client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["display_name"] == "Admin"


@pytest.mark.asyncio
async def test_login_rate_limit(monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_REMOTE", "true")
    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    monkeypatch.setenv("SEISO_RATE_LIMIT", "1000")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )

        saw_unauthorized = False
        for _ in range(12):
            res = await client.post(
                "/api/auth/login",
                json={"nsec": "nsec1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"},
            )
            # Invalid nsec → 401; after the login limiter threshold → 429.
            assert res.status_code in {400, 401, 429}
            if res.status_code in {400, 401}:
                saw_unauthorized = True
            if res.status_code == 429:
                break
        else:
            raise AssertionError("expected login rate limit (429)")
        assert saw_unauthorized


@pytest.mark.asyncio
async def test_global_rate_limit(monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_REMOTE", "true")
    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    monkeypatch.setenv("SEISO_RATE_LIMIT", "3")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            res = await client.get("/api/models")
            assert res.status_code == 401

        res = await client.get("/api/models")
        assert res.status_code == 429


@pytest.mark.asyncio
async def test_auth_status_no_user_count_leak(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.status_code == 200
        data = status.json()
        assert "needs_onboarding" in data
        assert "user_count" not in data


@pytest.mark.asyncio
async def test_settings_includes_security_posture(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        token = reg.json()["access_token"]

        res = await client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        sec = res.json()["security"]
        assert sec["bind_localhost"] is True
        assert sec["db_encrypted"] is True
        assert sec["allow_tools"] is False
        assert sec["rate_limit_enabled"] is True


@pytest.mark.asyncio
async def test_localhost_uses_relaxed_rate_limit(monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_REMOTE", "false")
    monkeypatch.setenv("SEISO_RATE_LIMIT", "3")
    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(10):
            res = await client.get("/api/models")
            assert res.status_code == 401
            assert res.status_code != 429


def test_jti_revocation_store_enforces_cap(monkeypatch):
    from forge.security import token_revocation as tr

    monkeypatch.setattr(tr, "_MAX_ENTRIES", 100)
    tr.clear_revocations_for_tests()
    now = 1_700_000_000.0
    for idx in range(110):
        tr.revoke_jti(f"jti-{idx}", now + 3600 + idx)
    assert len(tr._revoked) <= 100
    tr.clear_revocations_for_tests()

def test_csrf_empty_bearer_helper():
    from starlette.requests import Request

    from forge.security.csrf import validate_csrf

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/threads",
        "raw_path": b"/api/inference/threads",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer ")],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    assert validate_csrf(Request(scope)) is False

