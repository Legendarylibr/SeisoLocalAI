"""TUI Nostr auth: same owner, keys, and session persistence as Forge."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.research.nostr.keys import generate_keypair, load_keypair
from seiso.research.nostr.nip49 import decrypt_ncryptsec
from seiso.tui.auth import (
    AuthError,
    TuiAuth,
    extract_ncryptsec,
    format_key_backup_txt,
    looks_like_ncryptsec,
    resolve_secret,
    session_path,
    write_encrypted_backup,
)


def test_resolve_nsec_and_ncryptsec_file(tmp_path: Path) -> None:
    pair = generate_keypair()
    dest = write_encrypted_backup(pair.nsec, pair.npub, "correct-pass", tmp_path / "backup.txt")
    text = dest.read_text(encoding="utf-8")
    assert "This file does NOT contain your raw nsec" in text
    assert pair.nsec not in text
    assert looks_like_ncryptsec(text)
    restored = resolve_secret(str(dest), "correct-pass")
    assert restored == pair.nsec
    assert resolve_secret(pair.nsec) == pair.nsec
    with pytest.raises(AuthError):
        resolve_secret(text)
    with pytest.raises(AuthError):
        resolve_secret(text, "wrong-pass")


def test_format_backup_matches_forge_labels() -> None:
    blob = format_key_backup_txt("ncryptsec1abc", "npub1public")
    assert "ncryptsec=ncryptsec1abc" in blob
    assert "npub=npub1public" in blob
    assert extract_ncryptsec(blob) == "ncryptsec1abc"


def test_register_generate_persists_and_restores(tmp_path: Path) -> None:
    store = TuiAuth(tmp_path)
    assert store.status().needs_onboarding is True
    user, nsec = store.register(generate=True, storage_mode="persistent")
    assert nsec and nsec.startswith("nsec1")
    assert user.npub.startswith("npub1")
    token_file = session_path(tmp_path)
    assert token_file.is_file()
    assert oct(token_file.stat().st_mode)[-3:] == "600"

    pair = load_keypair(identity=user.id, data_dir=tmp_path)
    assert pair is not None
    assert pair.npub == user.npub

    again = TuiAuth(tmp_path)
    restored = again.restore_session()
    assert restored is not None
    assert restored.id == user.id
    assert restored.npub == user.npub
    assert again.status().needs_onboarding is False
    assert again.status().owner_npub == user.npub


def test_login_requires_matching_nsec(tmp_path: Path) -> None:
    store = TuiAuth(tmp_path)
    user, nsec = store.register(generate=True, storage_mode="persistent")
    assert nsec
    store.logout()
    assert store.restore_session() is None

    other = generate_keypair()
    with pytest.raises(AuthError, match="Invalid"):
        store.login(other.nsec)

    back = store.login(nsec)
    assert back.npub == user.npub
    assert store.restore_session() is not None


def test_register_import_does_not_echo_nsec(tmp_path: Path) -> None:
    pair = generate_keypair()
    store = TuiAuth(tmp_path)
    user, shown = store.register(generate=False, nsec=pair.nsec, storage_mode="persistent")
    assert shown is None
    assert user.npub == pair.npub


def test_reset_session_returns_to_onboarding(tmp_path: Path) -> None:
    store = TuiAuth(tmp_path)
    user, _nsec = store.register(generate=True, storage_mode="persistent")
    with pytest.raises(AuthError, match="RESET"):
        store.reset_session("nope")
    store.reset_session("RESET")
    assert store.status().needs_onboarding is True
    assert store.restore_session() is None
    assert load_keypair(identity=user.id, data_dir=tmp_path) is None


def test_rotate_and_import_key(tmp_path: Path) -> None:
    store = TuiAuth(tmp_path)
    user, old_nsec = store.register(generate=True, storage_mode="persistent")
    assert old_nsec
    new_nsec, new_npub = store.rotate_key(user.id)
    assert new_nsec != old_nsec
    assert new_npub != user.npub
    imported = generate_keypair()
    got = store.import_key(user.id, imported.nsec)
    assert got == imported.npub
    snap = store.status()
    assert snap.user is not None
    assert snap.user.npub == imported.npub


def test_nostr_prefs_roundtrip(tmp_path: Path) -> None:
    store = TuiAuth(tmp_path)
    user, _ = store.register(generate=True, storage_mode="persistent")
    store.save_prefs(
        user.id,
        auto_attest=True,
        relays=["wss://relay.damus.io"],
        allow_loopback=False,
    )
    status = store.nostr_status(user.id, user.nostr_pubkey)
    assert status["auto_attest"] is True
    assert status["relays"] == ["wss://relay.damus.io"]
    assert status["identity_match"] is True
    assert status["key_saved"] is True


def test_encrypted_backup_decrypts_to_same_secret(tmp_path: Path) -> None:
    pair = generate_keypair()
    dest = write_encrypted_backup(pair.nsec, pair.npub, "unit-test-pass", tmp_path / "k.txt")
    secret = decrypt_ncryptsec(extract_ncryptsec(dest.read_text()), "unit-test-pass")
    assert secret.hex() == pair.secret_hex


def test_auth_choices_cover_web_ui_actions() -> None:
    from seiso.tui.browse import auth_choices, page_choices

    welcome = {c.action for c in auth_choices("welcome")}
    assert welcome >= {"create", "restore", "storage_persistent", "storage_ephemeral"}
    login = {c.action for c in auth_choices("login")}
    assert login >= {"login", "reset"}
    reveal = {c.action for c in auth_choices("reveal")}
    assert reveal >= {"confirm_backup", "encrypt_backup"}
    integ = {c.action for c in page_choices("integrations")}
    assert integ >= {"attest_toggle", "keygen", "import_key"}
