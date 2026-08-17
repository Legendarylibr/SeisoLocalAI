"""End-to-end: TUI and Forge web auth share one owner, keys, and reset."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from rich.console import Console

from forge.api.deps import clear_dependency_caches
from forge.main import create_app
from seiso.research.nostr.keys import generate_keypair, load_npub
from seiso.tui.app import run_tui
from seiso.tui.auth import TuiAuth, resolve_secret, session_path, write_encrypted_backup
from seiso.tui.keys import Key
from tests.conftest import RETURN_TOKEN_HEADERS


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=True, width=120, height=40, color_system=None)


def _chars(text: str) -> list[Key]:
    return [Key("char", ch) for ch in text] + [Key("enter")]


@pytest.fixture
def no_hub(monkeypatch):
    monkeypatch.setattr("seiso.tui.app.search_hub", lambda *_a, **_k: ([], [], None))


@pytest.mark.asyncio
async def test_forge_register_then_tui_login_and_session(tmp_path: Path):
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        assert reg.status_code == 201, reg.text
        nsec = reg.json()["nsec"]
        npub = reg.json()["user"]["npub"]
        user_id = reg.json()["user"]["id"]
        owner = (await client.get("/api/auth/status")).json()["owner_npub"]
        assert owner == npub

    tui = TuiAuth(tmp_path)
    snap = tui.status()
    assert snap.needs_onboarding is False
    assert snap.owner_npub == npub
    assert snap.session_valid is False

    user = tui.login(nsec)
    assert user.id == user_id
    assert user.npub == npub
    assert tui.restore_session() is not None
    assert load_npub(identity=user_id, data_dir=tmp_path) == npub

    # TUI JWT is the same HS256 session Forge issues — Bearer works on /api/auth/me.
    token = session_path(tmp_path).read_text(encoding="utf-8").strip()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["npub"] == npub


@pytest.mark.asyncio
async def test_tui_register_then_forge_login(tmp_path: Path):
    tui = TuiAuth(tmp_path)
    user, nsec = tui.register(generate=True, storage_mode="persistent")
    assert nsec and user.npub.startswith("npub1")

    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["needs_onboarding"] is False
        assert status.json()["owner_npub"] == user.npub
        assert status.json()["auth_method"] == "nostr"

        other = generate_keypair()
        bad = await client.post("/api/auth/login", json={"nsec": other.nsec})
        assert bad.status_code == 401

        ok = await client.post("/api/auth/login", json={"nsec": nsec}, headers=RETURN_TOKEN_HEADERS)
        assert ok.status_code == 200, ok.text
        assert ok.json()["user"]["npub"] == user.npub
        assert ok.json()["user"]["id"] == user.id
        assert ok.json().get("nsec") in (None, "")


@pytest.mark.asyncio
async def test_tui_ncryptsec_backup_logs_into_forge(tmp_path: Path):
    tui = TuiAuth(tmp_path)
    user, nsec = tui.register(generate=True, storage_mode="persistent")
    assert nsec
    dest = write_encrypted_backup(
        nsec, user.npub, "backup-pass", tmp_path / "seiso-ncryptsec-backup.txt"
    )
    assert nsec not in dest.read_text(encoding="utf-8")
    recovered = resolve_secret(str(dest), "backup-pass")
    assert recovered == nsec

    tui.logout()
    clear_dependency_caches()
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        ok = await client.post(
            "/api/auth/login", json={"nsec": recovered}, headers=RETURN_TOKEN_HEADERS
        )
        assert ok.status_code == 200
        assert ok.json()["user"]["npub"] == user.npub


@pytest.mark.asyncio
async def test_key_rotate_visible_on_both_surfaces(tmp_path: Path):
    tui = TuiAuth(tmp_path)
    user, _ = tui.register(generate=True, storage_mode="persistent")
    new_nsec, new_npub = tui.rotate_key(user.id)
    token = session_path(tmp_path).read_text(encoding="utf-8").strip()

    clear_dependency_caches()
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["npub"] == new_npub
        status = await client.get("/api/auth/status")
        assert status.json()["owner_npub"] == new_npub

    tui.logout()
    again = TuiAuth(tmp_path)
    assert again.login(new_nsec).npub == new_npub


@pytest.mark.asyncio
async def test_forge_keygen_then_tui_sees_new_npub(tmp_path: Path):
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        regen = await client.post("/api/settings/nostr/keygen", headers=headers)
        assert regen.status_code == 200
        new_npub = regen.json()["npub"]
        new_nsec = regen.json()["nsec"]

    tui = TuiAuth(tmp_path)
    snap = tui.status()
    assert snap.owner_npub == new_npub
    user = tui.login(new_nsec)
    assert user.npub == new_npub


@pytest.mark.asyncio
async def test_reset_on_either_surface_returns_both_to_onboarding(tmp_path: Path):
    tui = TuiAuth(tmp_path)
    user, nsec = tui.register(generate=True, storage_mode="persistent")
    assert nsec
    tui.reset_session("RESET")
    assert tui.status().needs_onboarding is True
    assert tui.restore_session() is None
    assert load_npub(identity=user.id, data_dir=tmp_path) is None

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        status = await client.get("/api/auth/status")
        assert status.json()["needs_onboarding"] is True
        assert status.json()["owner_npub"] is None
        bad = await client.post("/api/auth/login", json={"nsec": nsec})
        assert bad.status_code == 401

        # Re-onboard on Forge, then wipe from the API and check TUI.
        reg = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        assert reg.status_code == 201
        csrf = client.cookies.get("seiso_csrf")
        reset = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "RESET"},
            headers={"X-CSRF-Token": csrf} if csrf else {},
        )
        assert reset.status_code == 200

    tui2 = TuiAuth(tmp_path)
    assert tui2.status().needs_onboarding is True


@pytest.mark.asyncio
async def test_tui_logout_purges_chat_like_forge(tmp_path: Path):
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        reg = await client.post(
            "/api/auth/register", json={"generate": True}, headers=RETURN_TOKEN_HEADERS
        )
        nsec = reg.json()["nsec"]
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        thread = await client.post(
            "/api/inference/threads", json={"title": "tui-logout"}, headers=headers
        )
        assert thread.status_code == 200
        listed = await client.get("/api/inference/threads", headers=headers)
        assert listed.status_code == 200
        assert listed.json()

    tui = TuiAuth(tmp_path)
    tui.login(nsec)
    tui.logout()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        again = await client.post(
            "/api/auth/login", json={"nsec": nsec}, headers=RETURN_TOKEN_HEADERS
        )
        assert again.status_code == 200
        new_token = again.json()["access_token"]
        listed = await client.get(
            "/api/inference/threads", headers={"Authorization": f"Bearer {new_token}"}
        )
        assert listed.status_code == 200
        assert listed.json() == []


def test_scripted_tui_create_confirm_then_restore(tmp_path: Path, no_hub) -> None:
    """Same first-run path as AuthPage: generate → write-down → continue."""
    run_tui(
        data_dir=tmp_path,
        console=_console(),
        repo_root=tmp_path,
        keys=[
            Key("down"),
            Key("down"),
            Key("enter"),  # Create account
            Key("enter"),  # I saved my recovery key
            *_chars("/quit"),
        ],
    )
    store = TuiAuth(tmp_path)
    snap = store.status()
    assert snap.needs_onboarding is False
    assert snap.session_valid is True
    assert snap.user is not None
    assert snap.user.npub.startswith("npub1")
    assert session_path(tmp_path).is_file()

    # Second launch restores the session (Welcome back is skipped).
    run_tui(
        data_dir=tmp_path,
        console=_console(),
        repo_root=tmp_path,
        keys=[*_chars("/quit")],
    )
    assert TuiAuth(tmp_path).restore_session() is not None


def test_scripted_tui_login_after_logout(tmp_path: Path, no_hub) -> None:
    store = TuiAuth(tmp_path)
    user, nsec = store.register(generate=True, storage_mode="persistent")
    assert nsec
    store.logout()
    assert store.restore_session() is None

    run_tui(
        data_dir=tmp_path,
        console=_console(),
        repo_root=tmp_path,
        keys=[
            *_chars(nsec),
            Key("enter"),
            *_chars("/quit"),
        ],
    )
    restored = TuiAuth(tmp_path).restore_session()
    assert restored is not None
    assert restored.npub == user.npub
