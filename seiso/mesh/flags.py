"""Mesh opt-in flags — off by default; Nostr-signed Buzz-agent mesh when enabled."""

from __future__ import annotations

import os

from seiso.agent.nostr_identity import require_buzz_nsec


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def mesh_allowed() -> bool:
    return _truthy(os.environ.get("SEISO_ALLOW_MESH"))


def require_mesh_allowed() -> None:
    if not mesh_allowed():
        raise RuntimeError(
            "Mesh is disabled. Set SEISO_ALLOW_MESH=1 to opt in (experimental). "
            "Self-hosted single-node training does not need this flag. "
            "Mesh is Buzz-agent-only — Forge UI cannot start mesh jobs."
        )
    # Mesh artifacts are NIP-01 / BIP-340 signed — requires a real agent nsec.
    require_buzz_nsec(feature="Mesh")


def mesh_token() -> str:
    return (os.environ.get("SEISO_MESH_TOKEN") or "").strip()
