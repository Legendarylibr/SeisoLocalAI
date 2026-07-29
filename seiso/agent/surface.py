"""Training access surfaces: frontend (Forge UI) vs generic agent.

Mesh / multi-node coordination is Buzz-agent-only. Local single-node and
local multi-GPU DDP remain available on both surfaces.

Buzz→Seiso trust note: Seiso does **not** speak NIP-98 to the Buzz relay.
``BUZZ_PRIVATE_KEY`` is validated as a real Nostr secret (nsec/hex) so a
trivial ``export BUZZ_PRIVATE_KEY=1`` cannot unlock mesh. Actual Buzz relay
auth remains with buzz-cli. Peer mesh binding is ``SEISO_MESH_TOKEN``
(HMAC fingerprint), not cryptographic Buzz identity verification.
"""

from __future__ import annotations

import os
from enum import Enum


class TrainingSurface(str, Enum):
    """Where a training request originated."""

    FRONTEND = "frontend"
    AGENT = "agent"


_TRIVIAL_AUTH_TAGS = frozenset({"1", "true", "yes", "on", "agent", "buzz"})
# Managed Desktop injects a non-trivial session tag; reject spoofable short values.
_MIN_AUTH_TAG_LEN = 16


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _valid_buzz_private_key(raw: str) -> bool:
    """True when ``raw`` parses as a Nostr secret (nsec bech32 or 64-char hex)."""
    try:
        from seiso.research.nostr.keys import keypair_from_secret

        keypair_from_secret(raw)
    except Exception:
        return False
    return True


def _valid_buzz_auth_tag(raw: str) -> bool:
    """True for a non-trivial managed-session tag (not a spoofable ``1``/``true``)."""
    tag = (raw or "").strip()
    if len(tag) < _MIN_AUTH_TAG_LEN:
        return False
    if tag.lower() in _TRIVIAL_AUTH_TAGS:
        return False
    return True


def buzz_agent_present() -> bool:
    """True when a verified Buzz agent identity marker is configured.

    - ``BUZZ_PRIVATE_KEY`` must be a valid nsec/hex secret (never logged).
    - ``BUZZ_AUTH_TAG`` must be a non-trivial managed Desktop session tag.

    This is **not** a NIP-98 round-trip to the Buzz relay — Seiso never calls
    the relay. It only prevents trivial env spoofing of the mesh/multi-node gate.
    """
    key = (os.environ.get("BUZZ_PRIVATE_KEY") or "").strip()
    if key:
        return _valid_buzz_private_key(key)
    tag = (os.environ.get("BUZZ_AUTH_TAG") or "").strip()
    if tag:
        return _valid_buzz_auth_tag(tag)
    return False


def agent_context_present() -> bool:
    """True for any agent harness (Buzz or generic)."""
    if _truthy(os.environ.get("SEISO_AGENT")):
        return True
    surface = (os.environ.get("SEISO_TRAINING_SURFACE") or "").strip().lower()
    if surface == TrainingSurface.AGENT.value:
        return True
    return buzz_agent_present()


def resolve_training_surface(*, explicit: str | None = None) -> TrainingSurface:
    """Resolve the active training surface.

    Precedence: explicit argument → ``SEISO_TRAINING_SURFACE`` → agent env →
    frontend (safe default for Forge API / UI).
    """
    raw = (explicit or os.environ.get("SEISO_TRAINING_SURFACE") or "").strip().lower()
    if raw == TrainingSurface.AGENT.value:
        return TrainingSurface.AGENT
    if raw == TrainingSurface.FRONTEND.value:
        return TrainingSurface.FRONTEND
    if agent_context_present():
        return TrainingSurface.AGENT
    return TrainingSurface.FRONTEND


def require_buzz_agent(*, feature: str = "Mesh") -> None:
    """Refuse features that must only run under a Buzz agent identity marker."""
    key = (os.environ.get("BUZZ_PRIVATE_KEY") or "").strip()
    if key:
        if _valid_buzz_private_key(key):
            return
        raise RuntimeError(
            f"{feature} requires a valid Buzz agent nsec in BUZZ_PRIVATE_KEY "
            "(bech32 nsec1… or 64-char hex). The value present is not a valid "
            "Nostr secret — refusing (presence-only spoofing is not enough). "
            "Forge UI / frontend training cannot start mesh or multi-node plans."
        )
    tag = (os.environ.get("BUZZ_AUTH_TAG") or "").strip()
    if tag and _valid_buzz_auth_tag(tag):
        return
    if tag:
        raise RuntimeError(
            f"{feature} requires a non-trivial BUZZ_AUTH_TAG from a managed "
            f"Buzz Desktop agent session (min {_MIN_AUTH_TAG_LEN} chars). "
            "A trivial tag like '1'/'true' is refused."
        )
    raise RuntimeError(
        f"{feature} is Buzz-agent-only. "
        "Configure a valid BUZZ_PRIVATE_KEY (nsec) or a managed BUZZ_AUTH_TAG, "
        "and opt in with SEISO_ALLOW_MESH=1. "
        "Seiso does not NIP-98-auth to the Buzz relay; peer binding uses "
        "SEISO_MESH_TOKEN. Forge UI cannot start mesh or multi-node plans."
    )
