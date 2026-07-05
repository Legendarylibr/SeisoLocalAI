"""Pytest fixtures — isolated settings and dependency caches."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches, close_dependency_caches, get_db
from forge.main import create_app
from forge.security.auth import create_access_token, hash_password
from forge.security.token_revocation import clear_revocations_for_tests


def pytest_configure(config):
    """Bootstrap CUDA toolkit paths before torch cpp_extension reads CUDA_HOME."""
    try:
        from seiso.kernels.cuda_env import configure_cuda_build_env

        configure_cuda_build_env()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
async def _reset_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_DB_ENCRYPTION_KEY", "01" * 32)
    monkeypatch.setenv("SEISO_DB_EPHEMERAL", "false")
    monkeypatch.setenv("SEISO_SKIP_MLX_PROBE", "true")
    clear_revocations_for_tests()
    clear_dependency_caches()
    yield
    with contextlib.suppress(Exception):
        await close_dependency_caches()
        # Let aiosqlite worker threads publish their close result before pytest closes the loop.
        await asyncio.sleep(0.2)


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
def enable_tools(monkeypatch, tmp_path):
    """Enable agent tool features and reload settings."""
    workspace = (tmp_path.parent / f"ws_{tmp_path.name}").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SEISO_ALLOW_TOOLS", "true")
    monkeypatch.setenv("SEISO_CODE_WORKSPACE", str(workspace))
    clear_dependency_caches()
    yield
    clear_dependency_caches()


@pytest.fixture
async def auth_client(app, tmp_path):
    """Register default user and return (client, token, headers)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"password": "securepass1"},
        )
        assert reg.status_code == 201
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        yield client, token, headers, tmp_path


async def make_second_user(
    email: str = "b@local.dev", password: str = "securepass2"
) -> tuple[str, str]:
    """Insert a second user directly (onboarding blocks second registration)."""
    from forge.config import get_settings

    db = get_db()
    user = await db.create_user(hash_password(password), "User B", email=email)
    token = create_access_token(user["id"], get_settings())
    return user["id"], token
