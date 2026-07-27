"""Pytest fixtures — isolated settings and dependency caches."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from forge.api.deps import clear_dependency_caches, close_dependency_caches, get_db
from forge.main import create_app
from forge.security.auth import create_access_token
from forge.security.token_revocation import clear_revocations_for_tests
from forge.services.nostr_auth import NOSTR_PASSWORD_SENTINEL
from seiso.inference.runner import reset_inference_runtime
from seiso.research.nostr.keys import generate_keypair

pytest_plugins = ("gguf_fixtures",)


def pytest_configure(config):
    """Bootstrap CUDA toolkit paths before torch cpp_extension reads CUDA_HOME."""
    try:
        from seiso.kernels.cuda_env import configure_cuda_build_env

        configure_cuda_build_env()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
async def _reset_caches(request, monkeypatch, tmp_path):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_DB_ENCRYPTION_KEY", "01" * 32)
    monkeypatch.setenv("SEISO_DB_EPHEMERAL", "false")
    monkeypatch.setenv("SEISO_SKIP_MLX_PROBE", "true")
    # Product slime defaults require held-out eval. Opt unit tests into tiny RL
    # except suites that assert real-path / Distill floor gates.
    nodeid = getattr(request.node, "nodeid", "") or ""
    if "test_rl_real_paths" not in nodeid and "test_distill_rl" not in nodeid:
        monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    clear_revocations_for_tests()
    clear_dependency_caches()
    reset_inference_runtime(wait=False)
    with contextlib.suppress(Exception):
        from forge.api.routes.auth import _login_limiter

        _login_limiter.reset()
    yield
    with contextlib.suppress(Exception):
        had_database = await close_dependency_caches()
        if had_database:
            # Let aiosqlite publish its close result before the loop closes.
            await asyncio.sleep(0.2)
        reset_inference_runtime(wait=False)


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
    """Enable agent tool features and reload settings."""
    monkeypatch.setenv("SEISO_ALLOW_TOOLS", "true")
    clear_dependency_caches()
    yield
    clear_dependency_caches()


# Opt into body JWT for tests that still use Authorization: Bearer.
RETURN_TOKEN_HEADERS = {"X-Seiso-Return-Token": "1"}


@pytest.fixture
async def auth_client(app, tmp_path):
    """Register default user and return (client, token, headers)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register",
            json={"generate": True},
            headers=RETURN_TOKEN_HEADERS,
        )
        assert reg.status_code == 201, reg.text
        token = reg.json()["access_token"]
        assert token, "register with X-Seiso-Return-Token must return access_token"
        headers = {"Authorization": f"Bearer {token}"}
        yield client, token, headers, tmp_path


async def make_second_user(
    email: str = "b@local.dev", password: str = "securepass2"
) -> tuple[str, str]:
    """Insert a second user directly (onboarding blocks second registration)."""
    from forge.config import get_settings

    db = get_db()
    pair = generate_keypair()
    # password arg kept for call-site compat; auth is Nostr pubkey based.
    _ = password
    user = await db.create_user(
        NOSTR_PASSWORD_SENTINEL,
        "User B",
        email=email,
        nostr_pubkey=pair.public_hex,
    )
    token = create_access_token(user["id"], get_settings())
    return user["id"], token
