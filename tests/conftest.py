"""Pytest fixtures — isolated settings and dependency caches."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches, get_db
from forge.main import create_app
from forge.security.auth import create_access_token, hash_password


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_DB_ENCRYPTION_KEY", "01" * 32)
    clear_dependency_caches()
    yield
    clear_dependency_caches()


@pytest.fixture
def app():
    return create_app()


def user_path(data_dir: Path, user_id: str, category: str, *parts: str) -> Path:
    """Create a user-scoped path under data_dir (directory or file parent)."""
    target = data_dir.joinpath(category, user_id, *parts)
    if parts and "." in parts[-1]:
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def enable_tools(monkeypatch):
    """Enable tool/MCP features and reload settings."""
    monkeypatch.setenv("SEISO_ALLOW_TOOLS", "true")
    clear_dependency_caches()
    yield
    clear_dependency_caches()


@pytest.fixture
def enable_autodefense(monkeypatch):
    """Enable AutoDefense and reload settings."""
    from forge.api.deps import clear_dependency_caches

    monkeypatch.setenv("SEISO_AUTODEFENSE_ENABLED", "true")
    clear_dependency_caches()
    yield
    clear_dependency_caches()


@pytest.fixture
async def autodefense_auth_client(app, tmp_path, monkeypatch):
    """Auth client with AutoDefense enabled before DB init (avoids ephemeral DB wipe)."""
    monkeypatch.setenv("SEISO_AUTODEFENSE_ENABLED", "true")
    clear_dependency_caches()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"email": "admin@local.dev", "password": "securepass1", "display_name": "Admin"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        yield client, token, headers, tmp_path


@pytest.fixture
async def auth_client(app, tmp_path):
    """Register default user and return (client, token, headers)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"email": "admin@local.dev", "password": "securepass1", "display_name": "Admin"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        yield client, token, headers, tmp_path


async def make_second_user(email: str = "b@local.dev", password: str = "securepass2") -> tuple[str, str]:
    """Insert a second user directly (onboarding blocks second registration)."""
    from forge.config import get_settings

    db = get_db()
    user = await db.create_user(email, hash_password(password), "User B")
    token = create_access_token(user["id"], get_settings())
    return user["id"], token
