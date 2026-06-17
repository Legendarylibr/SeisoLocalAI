"""Application-layer AES-256-GCM field encryption for SQLite payloads."""

from __future__ import annotations

import base64
import binascii
import os
import re
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PREFIX: Final = "enc:v1:"
_IV_LEN: Final = 12
_KEY_LEN: Final = 32
_HEX_KEY_RE = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)


def resolve_encryption_key(raw: str | None) -> bytes:
    """Decode a 32-byte AES key from base64 or hex, or raise ValueError."""
    if not raw:
        raise ValueError(
            "SEISO_DB_ENCRYPTION_KEY is required when database encryption is enabled. "
            "Generate one with: python -c \"import base64, os; print(base64.b64encode(os.urandom(32)).decode())\""
        )

    if _HEX_KEY_RE.fullmatch(raw.strip()):
        key = bytes.fromhex(raw.strip())
    else:
        try:
            key = base64.b64decode(raw.strip(), validate=True)
        except binascii.Error as exc:
            raise ValueError(
                "SEISO_DB_ENCRYPTION_KEY must decode to exactly 32 bytes "
                "(base64 from os.urandom(32) or 64-char hex)."
            ) from exc

    if len(key) != _KEY_LEN:
        raise ValueError(
            "SEISO_DB_ENCRYPTION_KEY must decode to exactly 32 bytes "
            "(base64 from os.urandom(32) or 64-char hex)."
        )
    return key


def generate_encryption_key() -> bytes:
    return os.urandom(_KEY_LEN)


def encrypt_field(plaintext: str, key: bytes) -> str:
    iv = os.urandom(_IV_LEN)
    ciphertext = AESGCM(key).encrypt(iv, plaintext.encode("utf-8"), None)
    blob = iv + ciphertext
    return PREFIX + base64.b64encode(blob).decode("ascii")


def decrypt_field(value: str, key: bytes) -> str:
    if not value.startswith(PREFIX):
        return value
    blob = base64.b64decode(value[len(PREFIX) :])
    iv, ciphertext = blob[:_IV_LEN], blob[_IV_LEN:]
    return AESGCM(key).decrypt(iv, ciphertext, None).decode("utf-8")
