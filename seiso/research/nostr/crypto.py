"""AES-256-GCM helpers for Nostr key material (compatible with forge enc:v1)."""

from __future__ import annotations

import base64
import binascii
import os
import re
from pathlib import Path
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PREFIX: Final = "enc:v1:"
_IV_LEN: Final = 12
_KEY_LEN: Final = 32
_HEX_KEY_RE = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)


def resolve_encryption_key(raw: str | None) -> bytes:
    if not raw:
        raise ValueError("Nostr encryption key is required")
    if _HEX_KEY_RE.fullmatch(raw.strip()):
        key = bytes.fromhex(raw.strip())
    else:
        try:
            key = base64.b64decode(raw.strip(), validate=True)
        except binascii.Error as exc:
            raise ValueError("Nostr encryption key must be base64 or 64-char hex") from exc
    if len(key) != _KEY_LEN:
        raise ValueError("Nostr encryption key must be exactly 32 bytes")
    return key


def generate_encryption_key() -> bytes:
    return os.urandom(_KEY_LEN)


def load_or_create_encryption_key(path: Path) -> bytes:
    """Load a 32-byte AES key from disk, or create and persist one."""
    if path.is_file():
        raw = path.read_bytes()
        if len(raw) == _KEY_LEN:
            return raw
        return resolve_encryption_key(raw.decode("utf-8").strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_encryption_key()
    path.write_text(base64.b64encode(key).decode("ascii"), encoding="utf-8")
    path.chmod(0o600)
    return key


def encrypt_field(plaintext: str, key: bytes) -> str:
    iv = os.urandom(_IV_LEN)
    ciphertext = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), None)
    return PREFIX + base64.b64encode(iv + ciphertext).decode("ascii")


def decrypt_field(value: str, key: bytes) -> str:
    if not value.startswith(PREFIX):
        return value
    blob = base64.b64decode(value[len(PREFIX) :])
    iv, ciphertext = blob[:_IV_LEN], blob[_IV_LEN:]
    return AESGCM(key).decrypt(iv, ciphertext, None).decode("utf-8")
