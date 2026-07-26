"""Nostr key generation and encrypted-at-rest storage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from seiso.research.nostr.bech32 import bech32_decode, bech32_encode
from seiso.research.nostr.crypto import (
    decrypt_field,
    encrypt_field,
    load_or_create_encryption_key,
)
from seiso.research.nostr.schnorr import pubkey_xonly_from_secret
from seiso.security import resolve_data_dir, safe_join


@dataclass(frozen=True)
class NostrKeyPair:
    secret_hex: str
    public_hex: str

    @property
    def nsec(self) -> str:
        return bech32_encode("nsec", bytes.fromhex(self.secret_hex))

    @property
    def npub(self) -> str:
        return bech32_encode("npub", bytes.fromhex(self.public_hex))


def generate_keypair() -> NostrKeyPair:
    while True:
        secret = os.urandom(32)
        try:
            public = pubkey_xonly_from_secret(secret)
        except ValueError:
            continue
        return NostrKeyPair(secret_hex=secret.hex(), public_hex=public.hex())


def keypair_from_secret(secret: str) -> NostrKeyPair:
    """Accept nsec bech32 or 64-char hex secret."""
    raw = (secret or "").strip()
    if raw.startswith("nsec1"):
        hrp, data = bech32_decode(raw)
        if hrp != "nsec" or len(data) != 32:
            raise ValueError("invalid nsec")
        secret_bytes = data
    else:
        if len(raw) != 64:
            raise ValueError("secret must be nsec or 64-char hex")
        secret_bytes = bytes.fromhex(raw)
    public = pubkey_xonly_from_secret(secret_bytes)
    return NostrKeyPair(secret_hex=secret_bytes.hex(), public_hex=public.hex())


def npub_from_hex(public_hex: str) -> str:
    return bech32_encode("npub", bytes.fromhex(public_hex))


def encryption_key_path(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    return root / ".nostr_key_encryption_key"


def key_store_path(data_dir: Path | None, identity: str) -> Path:
    root = resolve_data_dir(data_dir)
    return safe_join(root, "nostr_keys", identity)


def save_keypair(
    pair: NostrKeyPair,
    *,
    identity: str = "cli",
    data_dir: Path | None = None,
    encryption_key: bytes | None = None,
) -> Path:
    root = resolve_data_dir(data_dir)
    key = encryption_key or load_or_create_encryption_key(encryption_key_path(root))
    path = key_store_path(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encrypt_field(pair.secret_hex, key)
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    meta = path.with_suffix(".npub")
    meta.write_text(pair.npub + "\n", encoding="utf-8")
    meta.chmod(0o600)
    return path


def load_keypair(
    *,
    identity: str = "cli",
    data_dir: Path | None = None,
    encryption_key: bytes | None = None,
) -> NostrKeyPair | None:
    root = resolve_data_dir(data_dir)
    path = key_store_path(root, identity)
    if not path.is_file():
        return None
    key = encryption_key or load_or_create_encryption_key(encryption_key_path(root))
    try:
        secret_hex = decrypt_field(path.read_text(encoding="utf-8").strip(), key)
    except Exception:
        return None
    return keypair_from_secret(secret_hex)


def load_npub(
    *,
    identity: str = "cli",
    data_dir: Path | None = None,
) -> str | None:
    root = resolve_data_dir(data_dir)
    meta = key_store_path(root, identity).with_suffix(".npub")
    if meta.is_file():
        value = meta.read_text(encoding="utf-8").strip()
        if value:
            return value
    pair = load_keypair(identity=identity, data_dir=root)
    return pair.npub if pair else None


def clear_keypair(*, identity: str = "cli", data_dir: Path | None = None) -> None:
    root = resolve_data_dir(data_dir)
    path = key_store_path(root, identity)
    meta = path.with_suffix(".npub")
    if path.exists():
        path.unlink()
    if meta.exists():
        meta.unlink()
