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


def mesh_allow_loopback() -> bool:
    """Opt-in loopback master for single-host mesh smoke (never the default)."""
    return _truthy(os.environ.get("SEISO_MESH_ALLOW_LOOPBACK"))


def mesh_allow_any_planner() -> bool:
    """Opt-out of planner allowlist (single-operator smoke only — never default).

    Buzz channel membership is **not** a Seiso ACL. Without
    ``SEISO_MESH_TRUSTED_NPUBS`` / ``SEISO_MESH_TRUSTED_PUBKEYS``, plans are
    refused unless this flag is set.
    """
    return _truthy(os.environ.get("SEISO_MESH_ALLOW_ANY_PLANNER"))


def mesh_confirm_launch() -> bool:
    """True when operator explicitly confirmed a mesh train launch."""
    return _truthy(os.environ.get("SEISO_MESH_CONFIRM_LAUNCH"))


def mesh_debug_local() -> bool:
    """Allow extra local-only debug fields (e.g. hostname) on disk records."""
    return _truthy(os.environ.get("SEISO_MESH_DEBUG_LOCAL"))


def require_mesh_planner_allowlist() -> None:
    """Fail closed unless planner allowlist is set (or ANY_PLANNER smoke opt-out)."""
    from seiso.mesh.nostr_bind import _trusted_pubkey_hexes

    if _trusted_pubkey_hexes():
        return
    if mesh_allow_any_planner():
        return
    raise RuntimeError(
        "SEISO_MESH_TRUSTED_NPUBS (or SEISO_MESH_TRUSTED_PUBKEYS) is required. "
        "Buzz room membership is not a Seiso ACL — list planner npubs explicitly. "
        "For single-operator smoke only, set SEISO_MESH_ALLOW_ANY_PLANNER=1."
    )
