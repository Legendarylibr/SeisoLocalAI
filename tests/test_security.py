"""Security path validation tests."""

from pathlib import Path

import pytest

from forge.security.token_revocation import (
    clear_revocations_for_tests,
    is_jti_revoked,
    revoke_jti,
)
from forge.services.models import resolve_training_model_id
from seiso.security import (
    USER_SCOPED_DATA_ROOTS,
    SecurityError,
    assert_user_scoped_path,
    assert_within,
    safe_join,
    sanitize_filename,
)
from tests.conftest import RETURN_TOKEN_HEADERS


def test_safe_join_blocks_traversal(tmp_path: Path):
    base = tmp_path / "sandbox"
    base.mkdir()
    with pytest.raises(SecurityError):
        safe_join(base, "..", "etc", "passwd")


def test_safe_join_valid(tmp_path: Path):
    base = tmp_path / "sandbox"
    base.mkdir()
    result = safe_join(base, "models", "llama.gguf")
    assert result.exists() is False
    assert str(result.resolve()).startswith(str(base.resolve()))


def test_assert_within_outside_path(tmp_path: Path):
    base = tmp_path / "data"
    base.mkdir()
    inner = base / "file.txt"
    inner.write_text("ok")
    assert assert_within(base, inner) == inner.resolve()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(SecurityError):
        assert_within(base, outside)


def test_assert_within_prefix_bypass_blocked(tmp_path: Path):
    """Paths like sandbox-evil must not pass when base is sandbox."""
    base = tmp_path / "sandbox"
    base.mkdir()
    evil = tmp_path / "sandbox-evil"
    evil.mkdir()
    (evil / "secret.txt").write_text("nope")
    with pytest.raises(SecurityError):
        assert_within(base, evil / "secret.txt")


def test_safe_join_prefix_bypass_blocked(tmp_path: Path):
    base = tmp_path / "data"
    base.mkdir()
    with pytest.raises(SecurityError):
        safe_join(base, "..", "sandbox-evil", "file.txt")


def test_safe_join_embedded_traversal_blocked(tmp_path: Path):
    """A single segment must not smuggle ../ past safe_join."""
    base = tmp_path / "sandbox"
    (base / "uploads" / "attacker").mkdir(parents=True)
    (base / "knowledge" / "victim").mkdir(parents=True)
    with pytest.raises(SecurityError):
        safe_join(base, "uploads", "attacker/../../knowledge/victim")
    with pytest.raises(SecurityError):
        safe_join(base, "alice/../bob")


def test_safe_join_rejects_symlink_segment(tmp_path: Path):
    """Planted tenant symlink must not redirect safe_join write sinks."""
    base = tmp_path / "sandbox"
    bob = base / "knowledge" / "bob" / "kb1"
    bob.mkdir(parents=True)
    (bob / "index.jsonl").write_text("bob\n", encoding="utf-8")
    alice_link = base / "knowledge" / "alice"
    alice_link.parent.mkdir(parents=True, exist_ok=True)
    alice_link.symlink_to(base / "knowledge" / "bob")
    with pytest.raises(SecurityError, match="Symlink rejected"):
        safe_join(base, "knowledge", "alice", "kb1")


def test_sanitize_filename():
    assert "evil" in sanitize_filename("../../evil")
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("My Model v1.safetensors") == "My Model v1.safetensors"


def test_user_scoped_data_roots_cover_tenant_categories():
    """Canonical set used by CLI + Forge; keep categories explicit."""
    expected = {
        "uploads",
        "knowledge",
        "artifacts",
        "sandbox",
        "models",
        "checkpoints",
        "exports",
        "compress",
        "distill_rl",
        "recipes",
    }
    assert expected == USER_SCOPED_DATA_ROOTS
    assert "hf_cache" not in USER_SCOPED_DATA_ROOTS


def test_assert_user_scoped_path_allows_owner_tree(tmp_path: Path):
    user_id = "user-1"
    target = tmp_path / "uploads" / user_id / "data.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert assert_user_scoped_path(tmp_path, user_id, target) == target.resolve()


def test_assert_user_scoped_path_rejects_cross_user(tmp_path: Path):
    target = tmp_path / "models" / "other" / "m.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    with pytest.raises(SecurityError, match="Path must be under"):
        assert_user_scoped_path(tmp_path, "user-1", target)


def test_assert_user_scoped_path_rejects_hf_cache_direct(tmp_path: Path):
    """Shared cache is not a user-scoped root (Forge inventory links handle access)."""
    cache = tmp_path / "hf_cache" / "blob.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x")
    with pytest.raises(SecurityError, match="Access denied to path root"):
        assert_user_scoped_path(tmp_path, "user-1", cache)


def test_revocation_persist_is_atomic(tmp_path: Path, monkeypatch):
    from forge.security import token_revocation as tr

    store = tmp_path / ".revoked_jtis.json"
    monkeypatch.setattr(tr, "_store_path", store)
    tr.clear_revocations_for_tests()
    tr.revoke_jti("abc", 4_000_000_000.0)
    assert store.is_file()
    assert not store.with_name(store.name + ".tmp").exists()
    raw = store.read_text(encoding="utf-8")
    assert "abc" in raw


def test_resolve_training_model_id_rejects_cross_user_path(tmp_path: Path):
    bob = tmp_path / "models" / "bob" / "secret"
    bob.mkdir(parents=True)
    (bob / "config.json").write_text("{}", encoding="utf-8")
    (bob / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(SecurityError, match="models/alice"):
        resolve_training_model_id(
            str(bob.resolve()),
            data_dir=tmp_path,
            user_id="alice",
            inventory=[],
        )


def test_resolve_training_model_id_rejects_host_path_outside_data_dir(tmp_path: Path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    (outside / "config.json").write_text("{}", encoding="utf-8")
    (outside / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(SecurityError):
        resolve_training_model_id(
            str(outside.resolve()),
            data_dir=tmp_path,
            user_id="alice",
            inventory=[],
        )


def test_rate_limiter_evicts_idle_ips():
    from forge.security.auth import RateLimiter

    limiter = RateLimiter(max_per_minute=100)
    for i in range(300):
        limiter._hits[f"ip-{i}"] = [0.0]  # all expired vs monotonic now
    limiter.check("fresh-ip")
    assert "fresh-ip" in limiter._hits
    assert len(limiter._hits) < 300


def test_jwt_revocation_overflow_evicts_oldest(monkeypatch):
    import time

    monkeypatch.setattr("forge.security.token_revocation._MAX_ENTRIES", 3)
    now = time.time()
    for idx in range(5):
        revoke_jti(f"jti-{idx}", now + 3600 + idx)

    assert not is_jti_revoked("jti-0")
    assert not is_jti_revoked("jti-1")
    assert all(is_jti_revoked(f"jti-{idx}") for idx in range(2, 5))
    clear_revocations_for_tests()


@pytest.mark.asyncio
async def test_jwt_revocation_retained_until_expiry(monkeypatch):
    from forge.config import get_settings
    from forge.security import auth as auth_mod
    from forge.security.token_revocation import (
        clear_revocations_for_tests,
        is_jti_revoked,
    )

    clear_revocations_for_tests()
    settings = get_settings()
    tokens = [auth_mod.create_access_token(f"user-{i}", settings) for i in range(5)]
    for token in tokens:
        auth_mod.revoke_access_token(token, settings)

    for token in tokens:
        with pytest.raises(auth_mod.InvalidTokenError):
            auth_mod.decode_token(token, settings)

    # Prune should not resurrect revoked tokens before JWT exp.
    from jose import jwt

    payload = jwt.decode(tokens[0], settings.secret_key, algorithms=[auth_mod.ALGORITHM])
    assert is_jti_revoked(str(payload["jti"]))


@pytest.mark.asyncio
async def test_jwt_revoked_after_logout(app, auth_client):
    client, token, headers, _tmp = auth_client
    logout = await client.post("/api/auth/logout", headers=headers)
    assert logout.status_code == 200

    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 401


@pytest.mark.asyncio
async def test_registration_rejects_second_user(app):
    from httpx import ASGITransport, AsyncClient

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


def test_client_ip_ignores_forwarded_without_trusted_proxy(monkeypatch):
    from unittest.mock import MagicMock

    from forge.config import ForgeSettings
    from forge.security.client_ip import client_ip

    settings = ForgeSettings(trust_proxy=False)
    monkeypatch.setattr("forge.security.client_ip.get_settings", lambda: settings)

    request = MagicMock()
    request.client.host = "203.0.113.10"
    request.headers = {"x-forwarded-for": "198.51.100.99"}

    assert client_ip(request) == "203.0.113.10"


def test_remote_access_requires_ack(monkeypatch, tmp_path):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_ALLOW_REMOTE", "true")
    monkeypatch.delenv("SEISO_REMOTE_ACK", raising=False)
    from forge.api.deps import clear_dependency_caches

    clear_dependency_caches()
    with pytest.raises(RuntimeError, match="SEISO_REMOTE_ACK"):
        from forge.config import ForgeSettings

        ForgeSettings()


def test_remote_code_exec_always_refused(monkeypatch):
    """Remote + code-exec is refused entirely (AST sandbox is not OS isolation)."""
    from types import SimpleNamespace

    from forge.security.startup import validate_security_settings

    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    # Legacy ack must not re-enable remote code-exec.
    monkeypatch.setenv("SEISO_REMOTE_CODE_EXEC_ACK", "1")
    monkeypatch.setenv("SEISO_REMOTE_DANGEROUS_ACK", "1")

    settings = SimpleNamespace(
        allow_remote=True,
        allow_code_exec=True,
        allow_tools=False,
        allow_compat_tools=False,
        trust_proxy=False,
        trusted_proxy_ips="",
        debug=False,
    )
    with pytest.raises(RuntimeError, match="cannot be combined with code execution"):
        validate_security_settings(settings)


def test_debug_plus_remote_refused(monkeypatch):
    """S1-016: debug CSP unsafe-inline must not combine with remote bind."""
    from types import SimpleNamespace

    from forge.security.startup import validate_security_settings

    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    settings = SimpleNamespace(
        allow_remote=True,
        allow_code_exec=False,
        allow_tools=False,
        allow_compat_tools=False,
        trust_proxy=False,
        trusted_proxy_ips="",
        debug=True,
    )
    with pytest.raises(RuntimeError, match="SEISO_DEBUG"):
        validate_security_settings(settings)


def test_remote_tools_still_use_dangerous_ack(monkeypatch):
    from types import SimpleNamespace

    from forge.security.startup import validate_security_settings

    monkeypatch.setenv("SEISO_REMOTE_ACK", "1")
    monkeypatch.delenv("SEISO_REMOTE_DANGEROUS_ACK", raising=False)
    monkeypatch.delenv("SEISO_REMOTE_CODE_EXEC_ACK", raising=False)

    settings = SimpleNamespace(
        allow_remote=True,
        allow_code_exec=False,
        allow_tools=True,
        allow_compat_tools=False,
        trust_proxy=False,
        trusted_proxy_ips="",
        debug=False,
    )
    with pytest.raises(RuntimeError, match="SEISO_REMOTE_DANGEROUS_ACK"):
        validate_security_settings(settings)

    monkeypatch.setenv("SEISO_REMOTE_DANGEROUS_ACK", "1")
    validate_security_settings(settings)


def test_trust_proxy_requires_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_SECRET_KEY", "test-secret-key-for-jwt-signing-32b")
    monkeypatch.setenv("SEISO_TRUST_PROXY", "true")
    monkeypatch.delenv("SEISO_TRUSTED_PROXY_IPS", raising=False)
    from forge.api.deps import clear_dependency_caches

    clear_dependency_caches()
    with pytest.raises(RuntimeError, match="SEISO_TRUSTED_PROXY_IPS"):
        from forge.config import ForgeSettings

        ForgeSettings()


@pytest.mark.asyncio
async def test_inference_api_key_scoped_to_compat(app, auth_client, tmp_path):
    from forge.config import get_settings

    settings = get_settings()
    assert settings.inference_api_key
    assert settings.get_inference_api_key_owner()
    client, _token, _headers, _tmp = auth_client
    res = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.inference_api_key}"},
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert res.status_code in {400, 500}

    admin = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {settings.inference_api_key}"},
    )
    assert admin.status_code == 401


@pytest.mark.asyncio
async def test_inference_api_key_rejects_owner_mismatch(app, auth_client):
    """Compat key must match the sole owner's npub binding."""
    from forge.config import get_settings

    client, _token, _headers, _tmp = auth_client
    settings = get_settings()
    key = settings.inference_api_key
    # Point owner file at a different pubkey without rotating the key material.
    settings.bind_inference_api_key_owner("ab" * 32)
    stale = await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    assert stale.status_code == 401
    assert "owner npub" in stale.json()["detail"].lower()


@pytest.mark.asyncio
async def test_npub_keygen_rotates_compat_owner_binding(app, auth_client):
    """Rotating the account npub rebinds and rotates the Compat /v1 key."""
    from forge.config import get_settings

    client, _token, headers, _tmp = auth_client
    settings = get_settings()
    old_key = settings.inference_api_key
    old_owner = settings.get_inference_api_key_owner()
    assert old_key and old_owner

    regen = await client.post("/api/settings/nostr/keygen", headers=headers)
    assert regen.status_code == 200, regen.text
    settings = get_settings()
    assert settings.inference_api_key != old_key
    assert settings.get_inference_api_key_owner() != old_owner
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


@pytest.mark.asyncio
async def test_inference_api_key_cannot_use_compat_tools(monkeypatch, app, auth_client):
    """Inference API key stays chat-only even when Compat tools are server-enabled."""
    from forge.api.deps import clear_dependency_caches
    from forge.config import get_settings
    from forge.security.compat_auth import CompatIdentity

    assert CompatIdentity("u1", "inference_key").tools_allowed is False
    assert CompatIdentity("u1", "session").tools_allowed is True

    monkeypatch.setenv("SEISO_ALLOW_COMPAT_TOOLS", "true")
    clear_dependency_caches()
    try:
        settings = get_settings()
        assert settings.allow_compat_tools is True
        client, token, headers, _tmp = auth_client
        key_res = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.inference_api_key}"},
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "web_search"}}],
                "stream": False,
            },
        )
        assert key_res.status_code == 403
        assert "chat-only" in key_res.json()["detail"].lower()

        # Session JWT may proceed past the auth-method gate (may fail later on model).
        jwt_res = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "web_search"}}],
                "stream": False,
            },
        )
        assert "chat-only" not in jwt_res.text.lower()
        assert token  # keep fixture unpack used
    finally:
        monkeypatch.delenv("SEISO_ALLOW_COMPAT_TOOLS", raising=False)
        clear_dependency_caches()


@pytest.mark.asyncio
async def test_router_status_requires_auth(app, auth_client):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        anon = await client.get("/api/inference/router/status")
        assert anon.status_code == 401

    client, _token, headers, _tmp = auth_client
    authed = await client.get("/api/inference/router/status", headers=headers)
    assert authed.status_code == 200
    assert authed.json().get("enabled") is False
