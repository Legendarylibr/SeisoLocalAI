"""Per-user Nostr provenance settings and Forge-side auto-attest."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.security.audit import audit_event
from forge.security.url_policy import validate_nostr_relay_url
from seiso.research.nostr.keys import (
    clear_keypair,
    encryption_key_path,
    generate_keypair,
    keypair_from_secret,
    load_keypair,
    load_npub,
    save_keypair,
)
from seiso.research.nostr.policy import nostr_allowed
from seiso.security import SecurityError, assert_user_scoped_path, safe_join

logger = logging.getLogger(__name__)


@dataclass
class NostrPrefs:
    auto_attest: bool = False
    relays: list[str] | None = None
    allow_loopback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_attest": self.auto_attest,
            "relays": list(self.relays or []),
            "allow_loopback": self.allow_loopback,
        }


def _prefs_path(data_dir: Path, user_id: str) -> Path:
    return safe_join(data_dir, "nostr_keys", f"{user_id}.prefs.json")


def load_nostr_prefs(data_dir: Path, user_id: str) -> NostrPrefs:
    path = _prefs_path(data_dir, user_id)
    if not path.is_file():
        return NostrPrefs()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return NostrPrefs()
    if not isinstance(raw, dict):
        return NostrPrefs()
    relays = raw.get("relays") or []
    if not isinstance(relays, list):
        relays = []
    return NostrPrefs(
        auto_attest=bool(raw.get("auto_attest")),
        relays=[str(r) for r in relays if str(r).strip()],
        allow_loopback=bool(raw.get("allow_loopback")),
    )


def save_nostr_prefs(data_dir: Path, user_id: str, prefs: NostrPrefs) -> NostrPrefs:
    validated: list[str] = []
    for relay in prefs.relays or []:
        validated.append(
            validate_nostr_relay_url(
                str(relay), allow_loopback=prefs.allow_loopback
            )
        )
    prefs = NostrPrefs(
        auto_attest=bool(prefs.auto_attest),
        relays=validated,
        allow_loopback=bool(prefs.allow_loopback),
    )
    path = _prefs_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs.to_dict(), indent=2), encoding="utf-8")
    path.chmod(0o600)
    return prefs


def nostr_status(
    data_dir: Path,
    user_id: str,
    *,
    auth_pubkey: str | None = None,
    persist_keys: bool = True,
) -> dict[str, Any]:
    prefs = load_nostr_prefs(data_dir, user_id)
    npub = load_npub(identity=user_id, data_dir=data_dir)
    pair = load_keypair(identity=user_id, data_dir=data_dir) if npub else None
    attest_pubkey = pair.public_hex if pair else None
    auth = (auth_pubkey or "").strip().lower() or None
    # True only when both sides exist and match. Missing attest material is a
    # mismatch when an account npub is configured (e.g. after clear / ephemeral).
    if auth and attest_pubkey:
        identity_match = auth == attest_pubkey
    elif auth and not attest_pubkey:
        identity_match = False
    else:
        identity_match = True
    return {
        "server_allow_nostr": nostr_allowed(),
        "key_saved": npub is not None,
        "key_persisted": bool(persist_keys and npub is not None),
        "npub": npub,
        "auth_pubkey": auth,
        "attest_pubkey": attest_pubkey,
        "identity_match": identity_match,
        "auto_attest": prefs.auto_attest,
        "relays": list(prefs.relays or []),
        "allow_loopback": prefs.allow_loopback,
    }


def generate_user_nostr_key(
    data_dir: Path,
    user_id: str,
    *,
    persist: bool = True,
) -> dict[str, str]:
    """Generate a new signing key. Returns nsec once for offline backup on rotate."""
    pair = generate_keypair()
    if persist:
        save_keypair(pair, identity=user_id, data_dir=data_dir)
    audit_event("nostr_keygen", user_id=user_id, npub=pair.npub, persisted=persist)
    return {
        "status": "saved" if persist else "ephemeral",
        "npub": pair.npub,
        "nsec": pair.nsec,
        "pubkey_hex": pair.public_hex,
    }


def import_user_nostr_key(
    data_dir: Path,
    user_id: str,
    secret: str,
    *,
    persist: bool = True,
) -> dict[str, str]:
    pair = keypair_from_secret(secret)
    if persist:
        save_keypair(pair, identity=user_id, data_dir=data_dir)
    audit_event("nostr_key_import", user_id=user_id, npub=pair.npub, persisted=persist)
    return {
        "status": "saved" if persist else "ephemeral",
        "npub": pair.npub,
        "pubkey_hex": pair.public_hex,
    }


def clear_user_nostr_key(data_dir: Path, user_id: str) -> dict[str, str]:
    clear_keypair(identity=user_id, data_dir=data_dir)
    prefs = _prefs_path(data_dir, user_id)
    if prefs.is_file():
        prefs.unlink()
    audit_event("nostr_key_clear", user_id=user_id)
    return {"status": "cleared"}


def wipe_nostr_identity_material(data_dir: Path) -> dict[str, Any]:
    """Remove all Nostr keys/prefs and rotate the field-encryption key (reset-session)."""
    root = Path(data_dir)
    keys_dir = root / "nostr_keys"
    removed_files = 0
    if keys_dir.is_dir():
        for child in keys_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
                removed_files += 1
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed_files += 1
    enc = encryption_key_path(root)
    rotated = False
    if enc.is_file():
        enc.unlink(missing_ok=True)
        rotated = True
    audit_event(
        "nostr_identity_wipe",
        removed_files=removed_files,
        encryption_key_rotated=rotated,
    )
    return {
        "removed_files": removed_files,
        "encryption_key_rotated": rotated,
    }


def _candidate_manifests(result: dict[str, Any] | None, output_dir: str | None) -> list[Path]:
    candidates: list[Path] = []
    roots: list[Path] = []
    if isinstance(result, dict):
        for key in ("run_dir", "output_root", "output_dir", "manifest_path"):
            value = result.get(key)
            if value:
                roots.append(Path(str(value)))
        nested = result.get("nostr") or result.get("manifest")
        if isinstance(nested, dict) and nested.get("manifest_path"):
            candidates.append(Path(str(nested["manifest_path"])))
    if output_dir:
        roots.append(Path(output_dir))
    for root in roots:
        if root.is_file() and root.suffix == ".json":
            candidates.append(root)
            continue
        for name in (
            "manifest.json",
            "seiso_manifest.json",
            "seiso_export_metadata.json",
        ):
            path = root / name
            if path.is_file():
                candidates.append(path)
        if root.is_dir():
            candidates.extend(sorted(root.glob("*replay_manifest.json")))
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _scoped_manifest_path(data_dir: Path, user_id: str, path: Path) -> Path | None:
    """Return path if it is under data_dir/<scoped>/<user_id>/; else None."""
    try:
        return assert_user_scoped_path(data_dir, user_id, path)
    except SecurityError as exc:
        logger.warning(
            "Skipping Nostr attest for out-of-sandbox manifest %s (%s): %s",
            path,
            user_id,
            exc,
        )
        return None


def forge_maybe_attest(
    *,
    data_dir: Path,
    user_id: str,
    result: dict[str, Any] | None = None,
    output_dir: str | None = None,
    expected_pubkey: str | None = None,
) -> dict[str, Any] | None:
    """Attest job manifests when user prefs + server gate allow it."""
    if not nostr_allowed():
        return None
    prefs = load_nostr_prefs(data_dir, user_id)
    if not prefs.auto_attest:
        return None
    if not prefs.relays:
        logger.warning("Nostr auto-attest enabled but no relays configured for %s", user_id)
        return {"ok": False, "error": "no relays configured"}
    pair = load_keypair(identity=user_id, data_dir=data_dir)
    if pair is None:
        logger.warning("Nostr auto-attest enabled but no key for %s", user_id)
        return {"ok": False, "error": "no nostr key"}
    expected = (expected_pubkey or "").strip().lower()
    if expected and pair.public_hex != expected:
        logger.warning(
            "Nostr auto-attest blocked: signing key does not match auth npub for %s",
            user_id,
        )
        return {
            "ok": False,
            "error": "signing key does not match account npub",
            "auth_pubkey": expected,
            "attest_pubkey": pair.public_hex,
        }

    from seiso.research.nostr.attest import attest_manifest

    reports: list[dict[str, Any]] = []
    for manifest_path in _candidate_manifests(result, output_dir):
        if not manifest_path.is_file():
            continue
        scoped = _scoped_manifest_path(data_dir, user_id, manifest_path)
        if scoped is None:
            reports.append(
                {
                    "ok": False,
                    "error": "manifest path outside user sandbox",
                    "manifest_path": str(manifest_path),
                }
            )
            continue
        try:
            report = attest_manifest(
                scoped,
                relays=list(prefs.relays or []),
                pair=pair,
                identity=user_id,
                data_dir=data_dir,
                allow_loopback=prefs.allow_loopback,
                require_allow=True,
            )
            audit_event(
                "nostr_attest",
                user_id=user_id,
                event_id=report.get("event_id"),
                attestation_sha256=(report.get("receipt") or {}).get(
                    "attestation_sha256"
                ),
                manifest_path=str(scoped),
            )
            reports.append(report)
        except (SecurityError, Exception) as exc:
            logger.warning(
                "Nostr attest failed for %s (%s): %s", user_id, scoped, exc
            )
            reports.append(
                {"ok": False, "error": str(exc), "manifest_path": str(scoped)}
            )
    if not reports:
        return None
    return {"ok": all(r.get("ok") for r in reports), "reports": reports}
