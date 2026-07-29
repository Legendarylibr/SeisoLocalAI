"""Resolve the Buzz agent Nostr keypair for local Seiso crypto.

Mesh plan/announce signing uses BIP-340 Schnorr over the agent ``BUZZ_PRIVATE_KEY``.
Seiso still does not speak NIP-98 to the Buzz relay — that remains buzz-cli —
but signed mesh artifacts are real Nostr events (NIP-01) verifiable offline.
"""

from __future__ import annotations

import os

from seiso.research.nostr.keys import NostrKeyPair, keypair_from_secret


def buzz_private_key_raw() -> str:
    return (os.environ.get("BUZZ_PRIVATE_KEY") or "").strip()


def get_buzz_keypair() -> NostrKeyPair | None:
    """Return the agent keypair when ``BUZZ_PRIVATE_KEY`` is a valid nsec/hex."""
    raw = buzz_private_key_raw()
    if not raw:
        return None
    try:
        return keypair_from_secret(raw)
    except Exception:
        return None


def require_buzz_nsec(*, feature: str = "Mesh") -> NostrKeyPair:
    """Require a real Buzz agent nsec for Nostr-signed mesh operations."""
    raw = buzz_private_key_raw()
    if not raw:
        tag = (os.environ.get("BUZZ_AUTH_TAG") or "").strip()
        if tag:
            raise RuntimeError(
                f"{feature} Nostr signing requires BUZZ_PRIVATE_KEY (nsec). "
                "BUZZ_AUTH_TAG alone cannot produce BIP-340 signatures in Seiso. "
                "Configure the agent nsec for signed mesh plans."
            )
        raise RuntimeError(
            f"{feature} requires BUZZ_PRIVATE_KEY (valid nsec) for Nostr/"
            "BIP-340 signed mesh plans. Set SEISO_ALLOW_MESH=1 and the agent nsec. "
            "Forge UI cannot start mesh."
        )
    try:
        return keypair_from_secret(raw)
    except Exception as exc:
        raise RuntimeError(
            f"{feature} requires a valid Buzz agent nsec in BUZZ_PRIVATE_KEY "
            "(bech32 nsec1… or 64-char hex)."
        ) from exc
