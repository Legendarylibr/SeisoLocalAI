"""Application-layer AES-256-GCM field encryption for SQLite payloads.

Uses the shared ``enc:v1:`` helpers in ``seiso.research.nostr.crypto`` so Forge
DB columns and Nostr key-at-rest encryption cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from seiso.research.nostr import crypto as _aes

PREFIX: Final = _aes.PREFIX

generate_encryption_key = _aes.generate_encryption_key
encrypt_field = _aes.encrypt_field
decrypt_field = _aes.decrypt_field


def resolve_encryption_key(raw: str | None) -> bytes:
    """Decode a 32-byte AES key from base64 or hex, or raise ValueError."""
    try:
        return _aes.resolve_encryption_key(raw)
    except ValueError as exc:
        msg = str(exc)
        if "required" in msg:
            raise ValueError(
                "SEISO_DB_ENCRYPTION_KEY is required when database encryption is enabled. "
                'Generate one with: python -c "import base64, os; '
                'print(base64.b64encode(os.urandom(32)).decode())"'
            ) from exc
        raise ValueError(
            "SEISO_DB_ENCRYPTION_KEY must decode to exactly 32 bytes "
            "(base64 from os.urandom(32) or 64-char hex)."
        ) from exc


def load_encryption_key_file(path: Path) -> bytes:
    """Load a 32-byte AES key from disk (raw binary or base64/hex text)."""
    raw = path.read_bytes()
    if len(raw) == 32:
        return raw
    return resolve_encryption_key(raw.decode("utf-8").strip())


def persist_encryption_key_file(path: Path, key: bytes) -> None:
    """Write a 32-byte AES key as base64 text for portable, UTF-8-safe storage."""
    # Shared helper writes base64 + chmod 0600.
    path.parent.mkdir(parents=True, exist_ok=True)
    # load_or_create creates; for explicit persist reuse the same format.
    import base64

    if len(key) != 32:
        raise ValueError("encryption key must be exactly 32 bytes")
    path.write_text(base64.b64encode(key).decode("ascii"), encoding="utf-8")
    path.chmod(0o600)
