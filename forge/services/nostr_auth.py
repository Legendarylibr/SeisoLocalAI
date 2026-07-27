"""Nostr key helpers for Forge authentication (nsec proves ownership of npub)."""

from __future__ import annotations

from dataclasses import dataclass

from seiso.research.nostr.bech32 import bech32_encode
from seiso.research.nostr.keys import (
    NostrKeyPair,
    generate_keypair,
    keypair_from_secret,
    save_keypair,
)

# Sentinel stored in password_hash so legacy NOT NULL column stays valid.
# Not a secret — marks Nostr-auth rows; real credentials are nsec proofs.
NOSTR_PASSWORD_SENTINEL = "!nostr:v1"  # nosec B105


@dataclass(frozen=True)
class NostrAuthIdentity:
    pair: NostrKeyPair

    @property
    def pubkey_hex(self) -> str:
        return self.pair.public_hex

    @property
    def npub(self) -> str:
        return self.pair.npub

    @property
    def nsec(self) -> str:
        return self.pair.nsec


def resolve_identity(
    *,
    nsec: str | None = None,
    generate: bool = False,
) -> NostrAuthIdentity:
    if generate:
        if nsec:
            raise ValueError("Pass either generate=true or nsec, not both")
        return NostrAuthIdentity(generate_keypair())
    if not nsec or not str(nsec).strip():
        raise ValueError("nsec is required (or set generate=true on first setup)")
    return NostrAuthIdentity(keypair_from_secret(str(nsec).strip()))


def npub_from_pubkey_hex(pubkey_hex: str) -> str:
    return bech32_encode("npub", bytes.fromhex(pubkey_hex))


def persist_user_signing_key(
    *,
    data_dir,
    user_id: str,
    pair: NostrKeyPair,
) -> None:
    """Store encrypted nsec for provenance attestation under the user id."""
    save_keypair(pair, identity=user_id, data_dir=data_dir)


def user_public_view(user: dict) -> dict:
    """Safe user dict for API responses (never includes nsec)."""
    pubkey = str(user.get("nostr_pubkey") or "").strip()
    view = {
        "id": user["id"],
        "email": user.get("email"),
        "display_name": user.get("display_name"),
        "npub": npub_from_pubkey_hex(pubkey) if len(pubkey) == 64 else None,
        "nostr_pubkey": pubkey or None,
    }
    if user.get("created_at") is not None:
        view["created_at"] = user["created_at"]
    return view
