"""Forge Nostr prefs, settings API, and auto-attest orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge.main import create_app
from forge.services.nostr_settings import (
    NostrPrefs,
    _candidate_manifests,
    clear_user_nostr_key,
    forge_maybe_attest,
    generate_user_nostr_key,
    import_user_nostr_key,
    load_nostr_prefs,
    nostr_status,
    save_nostr_prefs,
    wipe_nostr_identity_material,
)
from seiso.research.nostr.keys import generate_keypair, load_keypair, load_npub
from seiso.security import SecurityError


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve non-literal relay hosts to a public IP (offline-safe)."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(None, None, None, None, ("8.8.8.8", 0))]

    monkeypatch.setattr(
        "seiso.research.nostr.policy.socket.getaddrinfo", fake_getaddrinfo
    )


def test_prefs_roundtrip_and_relay_validation(tmp_path: Path, public_dns):
    prefs = save_nostr_prefs(
        tmp_path,
        "user-1",
        NostrPrefs(
            auto_attest=True,
            relays=["wss://relay.example.com/", "ws://127.0.0.1:9"],
            allow_loopback=True,
        ),
    )
    assert prefs.relays == ["wss://relay.example.com", "ws://127.0.0.1:9"]
    loaded = load_nostr_prefs(tmp_path, "user-1")
    assert loaded.auto_attest is True
    assert loaded.allow_loopback is True
    assert loaded.relays == prefs.relays

    with pytest.raises(SecurityError):
        save_nostr_prefs(
            tmp_path,
            "user-1",
            NostrPrefs(relays=["wss://192.168.1.1"], allow_loopback=False),
        )


def test_keygen_import_clear_and_status(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    assert nostr_status(tmp_path, "u1")["key_saved"] is False
    gen = generate_user_nostr_key(tmp_path, "u1")
    assert gen["npub"].startswith("npub1")
    assert gen["nsec"].startswith("nsec1")
    assert nostr_status(tmp_path, "u1")["key_saved"] is True
    assert load_npub(identity="u1", data_dir=tmp_path) == gen["npub"]

    pair = generate_keypair()
    imported = import_user_nostr_key(tmp_path, "u1", pair.nsec)
    assert imported["npub"] == pair.npub
    clear_user_nostr_key(tmp_path, "u1")
    assert nostr_status(tmp_path, "u1")["key_saved"] is False


def test_wipe_nostr_identity_material(tmp_path: Path):
    gen = generate_user_nostr_key(tmp_path, "u1")
    save_nostr_prefs(tmp_path, "u1", NostrPrefs(auto_attest=True))
    assert load_npub(identity="u1", data_dir=tmp_path) == gen["npub"]
    report = wipe_nostr_identity_material(tmp_path)
    assert report["removed_files"] >= 1
    assert load_npub(identity="u1", data_dir=tmp_path) is None
    assert load_keypair(identity="u1", data_dir=tmp_path) is None


def test_candidate_manifests_discovery(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "foo_replay_manifest.json").write_text("{}", encoding="utf-8")
    explicit = tmp_path / "custom.json"
    explicit.write_text("{}", encoding="utf-8")
    found = _candidate_manifests(
        {
            "run_dir": str(run),
            "manifest": {"manifest_path": str(explicit)},
        },
        output_dir=str(run),
    )
    paths = {p.name for p in found}
    assert "manifest.json" in paths
    assert "foo_replay_manifest.json" in paths
    assert "custom.json" in paths


def test_forge_maybe_attest_gates_and_success(tmp_path: Path, monkeypatch, public_dns):
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    assert forge_maybe_attest(data_dir=tmp_path, user_id="u1") is None

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    save_nostr_prefs(tmp_path, "u1", NostrPrefs(auto_attest=False))
    assert forge_maybe_attest(data_dir=tmp_path, user_id="u1") is None

    save_nostr_prefs(
        tmp_path,
        "u1",
        NostrPrefs(auto_attest=True, relays=[], allow_loopback=False),
    )
    assert forge_maybe_attest(data_dir=tmp_path, user_id="u1") == {
        "ok": False,
        "error": "no relays configured",
    }

    save_nostr_prefs(
        tmp_path,
        "u1",
        NostrPrefs(
            auto_attest=True,
            relays=["wss://relay.example.com"],
            allow_loopback=False,
        ),
    )
    assert forge_maybe_attest(data_dir=tmp_path, user_id="u1") == {
        "ok": False,
        "error": "no nostr key",
    }

    gen = generate_user_nostr_key(tmp_path, "u1")
    outside = tmp_path / "out" / "manifest.json"
    outside.parent.mkdir(parents=True)
    payload = {
        "pipeline": "compress",
        "run_id": "r1",
        "config_fingerprint": "aa" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    outside.write_text(json.dumps(payload), encoding="utf-8")
    with patch("seiso.research.nostr.attest.attest_manifest") as mocked:
        blocked = forge_maybe_attest(
            data_dir=tmp_path,
            user_id="u1",
            result={"output_dir": str(outside.parent)},
            expected_pubkey=gen["pubkey_hex"],
        )
    assert blocked is not None
    assert blocked["ok"] is False
    assert "sandbox" in blocked["reports"][0]["error"]
    mocked.assert_not_called()

    man = tmp_path / "compress" / "u1" / "run" / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps(payload), encoding="utf-8")

    mismatch = forge_maybe_attest(
        data_dir=tmp_path,
        user_id="u1",
        result={"output_dir": str(man.parent)},
        expected_pubkey="ab" * 32,
    )
    assert mismatch == {
        "ok": False,
        "error": "signing key does not match account npub",
        "auth_pubkey": "ab" * 32,
        "attest_pubkey": gen["pubkey_hex"],
    }

    def fake_attest(path, **kwargs):
        return {
            "ok": True,
            "event_id": "ab" * 32,
            "receipt": {"attestation_sha256": "cd" * 32},
            "manifest_path": str(path),
        }

    with patch(
        "seiso.research.nostr.attest.attest_manifest", side_effect=fake_attest
    ):
        report = forge_maybe_attest(
            data_dir=tmp_path,
            user_id="u1",
            result={"output_dir": str(man.parent)},
            expected_pubkey=gen["pubkey_hex"],
        )
    assert report is not None
    assert report["ok"] is True
    assert len(report["reports"]) == 1


@pytest.mark.asyncio
async def test_settings_nostr_api_roundtrip(tmp_path: Path, monkeypatch, public_dns):
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201, reg.text
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        auth_npub = reg.json()["user"]["npub"]

        status = await client.get("/api/settings/nostr", headers=headers)
        assert status.status_code == 200
        body = status.json()
        assert body["server_allow_nostr"] is True
        assert body["key_saved"] is True
        assert body["npub"]
        assert body["identity_match"] is True
        assert body["npub"] == auth_npub
        assert "nsec" not in json.dumps(body)

        bad = await client.put(
            "/api/settings/nostr",
            headers=headers,
            json={
                "auto_attest": True,
                "relays": ["wss://10.0.0.1"],
                "allow_loopback": False,
            },
        )
        assert bad.status_code == 400

        ok = await client.put(
            "/api/settings/nostr",
            headers=headers,
            json={
                "auto_attest": True,
                "relays": ["wss://relay.example.com"],
                "allow_loopback": False,
            },
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["auto_attest"] is True
        assert ok.json()["relays"] == ["wss://relay.example.com"]

        pair = generate_keypair()
        imported = await client.put(
            "/api/settings/nostr/key",
            headers=headers,
            json={"secret": pair.nsec},
        )
        assert imported.status_code == 200
        assert imported.json()["npub"] == pair.npub
        assert "nsec" not in imported.json()

        me = await client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["npub"] == pair.npub

        cleared = await client.delete("/api/settings/nostr/key", headers=headers)
        assert cleared.status_code == 200
        assert cleared.json()["status"] == "cleared"

        regen = await client.post("/api/settings/nostr/keygen", headers=headers)
        assert regen.status_code == 200
        assert regen.json()["npub"].startswith("npub1")
        assert regen.json()["nsec"].startswith("nsec1")
        me2 = await client.get("/api/auth/me", headers=headers)
        assert me2.json()["npub"] == regen.json()["npub"]


@pytest.mark.asyncio
async def test_login_refreshes_signing_key(tmp_path: Path):
    pair = generate_keypair()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"nsec": pair.nsec})
        assert reg.status_code == 201
        user_id = reg.json()["user"]["id"]
        clear_user_nostr_key(tmp_path, user_id)
        assert load_npub(identity=user_id, data_dir=tmp_path) is None
        login = await client.post("/api/auth/login", json={"nsec": pair.nsec})
        assert login.status_code == 200
        assert load_npub(identity=user_id, data_dir=tmp_path) == pair.npub


@pytest.mark.asyncio
async def test_reset_session_wipes_nostr_keys(tmp_path: Path):
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201
        user_id = reg.json()["user"]["id"]
        assert load_npub(identity=user_id, data_dir=tmp_path) is not None

        csrf = client.cookies.get("seiso_csrf")
        assert csrf
        reset = await client.post(
            "/api/auth/reset-session",
            json={"confirmation": "RESET"},
            headers={"X-CSRF-Token": csrf},
        )
        assert reset.status_code == 200
        assert reset.json()["nostr_identity_wiped"] is True
        assert load_npub(identity=user_id, data_dir=tmp_path) is None
        assert not (tmp_path / ".nostr_key_encryption_key").exists()


@pytest.mark.asyncio
async def test_ephemeral_mode_skips_nostr_key_persist(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DB_EPHEMERAL", "true")
    from forge.api.deps import clear_dependency_caches

    clear_dependency_caches()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201, reg.text
        user_id = reg.json()["user"]["id"]
        assert load_npub(identity=user_id, data_dir=tmp_path) is None
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        status = await client.get("/api/settings/nostr", headers=headers)
        assert status.status_code == 200
        assert status.json()["key_saved"] is False
        assert status.json()["identity_match"] is False
        assert status.json()["key_persisted"] is False

        regen = await client.post("/api/settings/nostr/keygen", headers=headers)
        assert regen.status_code == 200
        assert regen.json()["nsec"].startswith("nsec1")
        assert load_npub(identity=user_id, data_dir=tmp_path) is None
        me = await client.get("/api/auth/me", headers=headers)
        assert me.json()["npub"] == regen.json()["npub"]


@pytest.mark.asyncio
async def test_clear_nostr_key_leaves_auth_pubkey_and_login_works(tmp_path: Path):
    pair = generate_keypair()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"nsec": pair.nsec})
        assert reg.status_code == 201
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        user_id = reg.json()["user"]["id"]

        cleared = await client.delete("/api/settings/nostr/key", headers=headers)
        assert cleared.status_code == 200
        assert load_npub(identity=user_id, data_dir=tmp_path) is None

        status = await client.get("/api/settings/nostr", headers=headers)
        assert status.json()["key_saved"] is False
        assert status.json()["identity_match"] is False

        me = await client.get("/api/auth/me", headers=headers)
        assert me.json()["npub"] == pair.npub

        login = await client.post("/api/auth/login", json={"nsec": pair.nsec})
        assert login.status_code == 200
        assert load_npub(identity=user_id, data_dir=tmp_path) == pair.npub


@pytest.mark.asyncio
async def test_settings_keygen_updates_auth_and_me_never_echoes_nsec(tmp_path: Path):
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        old_npub = reg.json()["user"]["npub"]

        regen = await client.post("/api/settings/nostr/keygen", headers=headers)
        assert regen.status_code == 200
        new_nsec = regen.json()["nsec"]
        new_npub = regen.json()["npub"]
        assert new_npub != old_npub

        me = await client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        assert me.json()["npub"] == new_npub
        assert "nsec" not in me.json()
        assert new_nsec not in me.text

        status = await client.get("/api/settings/nostr", headers=headers)
        assert status.json()["identity_match"] is True
        assert status.json()["npub"] == new_npub


def test_wipe_nostr_identity_removes_prefs_and_encryption_key(
    tmp_path: Path, public_dns
):
    from seiso.research.nostr.keys import encryption_key_path

    gen = generate_user_nostr_key(tmp_path, "u1")
    save_nostr_prefs(
        tmp_path,
        "u1",
        NostrPrefs(auto_attest=True, relays=["wss://relay.example.com"]),
    )
    enc = encryption_key_path(tmp_path)
    assert enc.is_file()
    assert load_npub(identity="u1", data_dir=tmp_path) == gen["npub"]
    prefs_path = tmp_path / "nostr_keys" / "u1.prefs.json"
    assert prefs_path.is_file()

    report = wipe_nostr_identity_material(tmp_path)
    assert report["encryption_key_rotated"] is True
    assert not enc.exists()
    assert not prefs_path.exists()
    assert load_keypair(identity="u1", data_dir=tmp_path) is None


@pytest.mark.asyncio
async def test_update_user_nostr_pubkey_rejects_non_hex(tmp_path: Path):
    from forge.api.deps import get_db

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        reg = await client.post("/api/auth/register", json={"generate": True})
        assert reg.status_code == 201
        user_id = reg.json()["user"]["id"]
        db = get_db()
        with pytest.raises(ValueError, match="64-char hex"):
            await db.update_user_nostr_pubkey(user_id, "z" * 64)
        with pytest.raises(ValueError, match="64-char hex"):
            await db.update_user_nostr_pubkey(user_id, "abcd")
