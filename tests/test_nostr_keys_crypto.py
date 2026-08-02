"""Nostr key storage, AES-GCM crypto, and bech32 edge cases."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from seiso.research.nostr.bech32 import bech32_decode, bech32_encode
from seiso.research.nostr.crypto import (
    PREFIX,
    decrypt_field,
    encrypt_field,
    generate_encryption_key,
    load_or_create_encryption_key,
    resolve_encryption_key,
)
from seiso.research.nostr.keys import (
    clear_keypair,
    generate_keypair,
    key_store_path,
    keypair_from_secret,
    load_keypair,
    load_npub,
    save_keypair,
)
from seiso.security import SecurityError


def test_resolve_encryption_key_hex_and_base64():
    raw = generate_encryption_key()
    assert resolve_encryption_key(raw.hex()) == raw
    assert resolve_encryption_key(base64.b64encode(raw).decode("ascii")) == raw
    with pytest.raises(ValueError, match="required"):
        resolve_encryption_key(None)
    with pytest.raises(ValueError, match="32 bytes|base64|hex"):
        resolve_encryption_key("not-a-key")
    with pytest.raises(ValueError, match="32 bytes"):
        resolve_encryption_key(base64.b64encode(b"short").decode("ascii"))


def test_encrypt_decrypt_roundtrip_and_tamper():
    key = generate_encryption_key()
    token = encrypt_field("secret-hex-value", key)
    assert token.startswith(PREFIX)
    assert decrypt_field(token, key) == "secret-hex-value"
    # Plaintext passthrough (forge-compatible).
    assert decrypt_field("plaintext", key) == "plaintext"
    # Wrong key / tampered ciphertext must fail.
    other = generate_encryption_key()
    with pytest.raises(InvalidTag):
        decrypt_field(token, other)
    tampered = token[:-4] + ("0" if token[-4] != "0" else "1") + token[-3:]
    with pytest.raises(InvalidTag):
        decrypt_field(tampered, key)


def test_load_or_create_encryption_key_persists(tmp_path: Path):
    path = tmp_path / "enc.key"
    first = load_or_create_encryption_key(path)
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    second = load_or_create_encryption_key(path)
    assert first == second
    # Raw 32-byte file is accepted.
    raw_path = tmp_path / "raw.key"
    raw = generate_encryption_key()
    raw_path.write_bytes(raw)
    assert load_or_create_encryption_key(raw_path) == raw


def test_keypair_save_load_clear_and_wrong_key(tmp_path: Path):
    pair = generate_keypair()
    path = save_keypair(pair, identity="cli", data_dir=tmp_path)
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_npub(identity="cli", data_dir=tmp_path) == pair.npub
    loaded = load_keypair(identity="cli", data_dir=tmp_path)
    assert loaded is not None
    assert loaded.public_hex == pair.public_hex
    assert loaded.secret_hex == pair.secret_hex

    # Wrong encryption key → None (no plaintext leak via exception).
    bad = load_keypair(
        identity="cli",
        data_dir=tmp_path,
        encryption_key=generate_encryption_key(),
    )
    assert bad is None

    clear_keypair(identity="cli", data_dir=tmp_path)
    assert load_keypair(identity="cli", data_dir=tmp_path) is None
    assert load_npub(identity="cli", data_dir=tmp_path) is None


def test_load_keypair_rejects_plaintext_secret_on_disk(tmp_path: Path):
    pair = generate_keypair()
    path = key_store_path(tmp_path, "cli")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pair.secret_hex, encoding="utf-8")
    path.chmod(0o600)
    assert load_keypair(identity="cli", data_dir=tmp_path) is None


def test_generate_keypair_uses_os_urandom(monkeypatch):
    calls: list[int] = []
    # Valid secp256k1 scalar (1) → deterministic pubkey for the mock draw.
    secret = (1).to_bytes(32, "big")

    def fake_urandom(n: int) -> bytes:
        calls.append(n)
        assert n == 32
        return secret

    monkeypatch.setattr("seiso.research.nostr.keys.os.urandom", fake_urandom)
    pair = generate_keypair()
    assert calls == [32]
    assert pair.secret_hex == secret.hex()
    assert len(pair.public_hex) == 64


def test_keypair_from_hex_and_invalid_secrets():
    pair = generate_keypair()
    restored = keypair_from_secret(pair.secret_hex)
    assert restored.public_hex == pair.public_hex
    with pytest.raises(ValueError, match="nsec|64-char"):
        keypair_from_secret("too-short")
    with pytest.raises(ValueError, match="invalid nsec|bech32|checksum"):
        keypair_from_secret("nsec1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")


def test_identity_path_traversal_rejected(tmp_path: Path):
    with pytest.raises(SecurityError):
        key_store_path(tmp_path, "../escape")
    with pytest.raises(SecurityError):
        save_keypair(generate_keypair(), identity="alice/../bob", data_dir=tmp_path)


def test_bech32_rejects_mixed_case_and_bad_checksum():
    pair = generate_keypair()
    mixed = pair.npub[:8].upper() + pair.npub[8:]
    if mixed != pair.npub.lower() and mixed != pair.npub.upper():
        with pytest.raises(ValueError, match="mixed-case"):
            bech32_decode(mixed)
    # Flip a character in the checksum region.
    chars = list(pair.npub)
    chars[-1] = "q" if chars[-1] != "q" else "p"
    with pytest.raises(ValueError, match="checksum|character|bech32"):
        bech32_decode("".join(chars))
    # Encode/decode empty and known sizes.
    encoded = bech32_encode("npub", bytes(32))
    hrp, data = bech32_decode(encoded)
    assert hrp == "npub"
    assert data == bytes(32)
