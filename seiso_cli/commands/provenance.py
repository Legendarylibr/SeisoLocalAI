"""CLI for Nostr provenance attestation and verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from seiso_cli.console import console

provenance_app = typer.Typer(
    name="provenance",
    help="Nostr provenance attestation (opt-in; digests only).",
    no_args_is_help=True,
)


def _data_dir() -> Path:
    try:
        from forge.config import get_settings

        return get_settings().data_dir
    except Exception:
        from seiso.security import resolve_data_dir

        return resolve_data_dir()


@provenance_app.command("keygen")
def provenance_keygen(
    identity: str = typer.Option("cli", help="Key identity slot under nostr_keys/"),
    import_nsec: str | None = typer.Option(
        None, "--import", help="Import existing nsec or hex secret instead of generating"
    ),
) -> None:
    """Create or import an encrypted Nostr key; print npub."""
    from seiso.research.nostr.keys import (
        generate_keypair,
        keypair_from_secret,
        save_keypair,
    )

    data_dir = _data_dir()
    (data_dir / "nostr_keys").mkdir(parents=True, exist_ok=True)
    pair = keypair_from_secret(import_nsec) if import_nsec else generate_keypair()
    path = save_keypair(pair, identity=identity, data_dir=data_dir)
    console.print(f"Saved encrypted key → {path}")
    console.print(f"npub: {pair.npub}")
    console.print(
        "Enable outbound attest with SEISO_ALLOW_NOSTR=1 and SEISO_NOSTR_RELAYS=wss://..."
    )


@provenance_app.command("attest")
def provenance_attest(
    manifest: Annotated[str, typer.Argument(help="Path to local manifest JSON")],
    relay: Annotated[
        list[str] | None,
        typer.Option("--relay", "-r", help="Relay URL (repeatable); else SEISO_NOSTR_RELAYS"),
    ] = None,
    identity: Annotated[str, typer.Option(help="Key identity slot")] = "cli",
    allow_loopback: Annotated[
        bool, typer.Option(help="Allow ws://127.0.0.1 relays (dev/mock only)")
    ] = False,
) -> None:
    """Seal a manifest to allowlisted Nostr relays and write a receipt."""
    os.environ.setdefault("SEISO_ALLOW_NOSTR", "1")
    from seiso.research.nostr.attest import attest_manifest

    report = attest_manifest(
        Path(manifest),
        relays=list(relay) if relay else None,
        identity=identity,
        data_dir=_data_dir(),
        allow_loopback=allow_loopback,
        require_allow=True,
    )
    console.print_json(data=report)


@provenance_app.command("verify")
def provenance_verify(
    manifest: Annotated[str, typer.Argument(help="Path to local manifest JSON")],
    relay: Annotated[
        list[str] | None,
        typer.Option(
            "--relay", "-r", help="Relay URL (repeatable); else receipt/env relays"
        ),
    ] = None,
    allow_loopback: Annotated[
        bool, typer.Option(help="Allow ws://127.0.0.1 relays (dev/mock only)")
    ] = False,
    local_only: Annotated[
        bool,
        typer.Option(
            "--local-only", help="Skip relay fetch; check local receipt digests only"
        ),
    ] = False,
) -> None:
    """Recompute digests and verify the Nostr event commitment."""
    if not local_only:
        os.environ.setdefault("SEISO_ALLOW_NOSTR", "1")
    from seiso.research.nostr.attest import verify_attestation

    report = verify_attestation(
        Path(manifest),
        relays=list(relay) if relay else None,
        allow_loopback=allow_loopback,
        require_network=not local_only,
    )
    console.print_json(data=report)
    if not report.get("ok"):
        raise typer.Exit(code=1)


@provenance_app.command("show")
def provenance_show(
    manifest: str = typer.Argument(..., help="Path to local manifest JSON"),
) -> None:
    """Print attestation payload and nostr receipt summary."""
    from seiso.research.provenance import build_attestation_v1

    path = Path(manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter("manifest must be a JSON object")
    attestation = build_attestation_v1(data, manifest_path=path)
    console.print_json(
        data={
            "manifest_path": str(path),
            "attestation": attestation,
            "receipt": data.get("nostr"),
        }
    )


def _load_json_object(path: Path, *, label: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{label} must be a JSON object")
    return data


def _resolve_merkle_sidecar(manifest_path: Path, manifest: dict) -> Path:
    rel = str(manifest.get("dataset_merkle_sidecar") or "dataset_merkle.json")
    candidate = (manifest_path.parent / rel).resolve()
    if not candidate.is_file():
        raise typer.BadParameter(f"dataset merkle sidecar not found: {candidate}")
    return candidate


@provenance_app.command("dataset-prove")
def provenance_dataset_prove(
    manifest: Annotated[str, typer.Argument(help="Path to seiso_manifest.json")],
    row: Annotated[
        str,
        typer.Option(
            "--row",
            "-r",
            help="JSON file with one training row (same fields used in training)",
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write proof JSON to this path"),
    ] = None,
) -> None:
    """Build a membership proof for a row against the local dataset merkle sidecar."""
    from seiso.research.dataset_merkle import (
        build_membership_proof,
        load_dataset_merkle_sidecar,
        row_content_fingerprint,
    )

    manifest_path = Path(manifest)
    man = _load_json_object(manifest_path, label="manifest")
    if not man.get("dataset_merkle_root"):
        raise typer.BadParameter(
            "manifest has no dataset_merkle_root; retrain with dataset_merkle=true"
        )
    sidecar = load_dataset_merkle_sidecar(_resolve_merkle_sidecar(manifest_path, man))
    row_obj = _load_json_object(Path(row), label="row")
    fingerprint = row_content_fingerprint(row_obj)
    try:
        proof = build_membership_proof(sidecar, fingerprint)
    except KeyError as exc:
        console.print(f"[red]Row not in committed train corpus[/red]: {exc}")
        raise typer.Exit(code=1) from exc
    # Prefer manifest root if present (should match sidecar).
    proof["dataset_merkle_root"] = str(
        man.get("dataset_merkle_root") or proof["dataset_merkle_root"]
    )
    proof["manifest_path"] = str(manifest_path)
    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
        console.print(f"Wrote proof → {out_path}")
    console.print_json(data=proof)


@provenance_app.command("dataset-verify-proof")
def provenance_dataset_verify_proof(
    proof: Annotated[str, typer.Argument(help="Path to membership proof JSON")],
    event_id: Annotated[
        str | None,
        typer.Option(
            "--event-id",
            help="Nostr event id to fetch for dataset_merkle_root (optional)",
        ),
    ] = None,
    manifest: Annotated[
        str | None,
        typer.Option(
            "--manifest",
            help="Local manifest with nostr receipt (alternative to --event-id)",
        ),
    ] = None,
    relay: Annotated[
        list[str] | None,
        typer.Option("--relay", "-r", help="Relay URL (repeatable)"),
    ] = None,
    allow_loopback: Annotated[
        bool, typer.Option(help="Allow ws://127.0.0.1 relays")
    ] = False,
    local_only: Annotated[
        bool,
        typer.Option(
            "--local-only",
            help="Only check merkle path vs root embedded in the proof",
        ),
    ] = False,
) -> None:
    """Verify a membership proof; optionally match root to a Nostr attestation."""
    from seiso.research.dataset_merkle import verify_membership_proof

    proof_path = Path(proof)
    proof_obj = _load_json_object(proof_path, label="proof")
    root = str(proof_obj.get("dataset_merkle_root") or "").strip()
    path_ok = verify_membership_proof(proof_obj, root=root or None)
    report: dict = {
        "ok": path_ok,
        "merkle_path_ok": path_ok,
        "proof_path": str(proof_path),
        "dataset_merkle_root": root or None,
        "event_root_match": None,
        "event_verified": None,
    }
    if not path_ok:
        report["error"] = "merkle path does not recompute to dataset_merkle_root"
        console.print_json(data=report)
        raise typer.Exit(code=1)

    if local_only:
        console.print_json(data=report)
        return

    remote_root: str | None = None
    if manifest:
        os.environ.setdefault("SEISO_ALLOW_NOSTR", "1")
        from seiso.research.nostr.attest import verify_attestation

        att_report = verify_attestation(
            Path(manifest),
            relays=list(relay) if relay else None,
            allow_loopback=allow_loopback,
            require_network=True,
        )
        report["attestation_verify"] = {
            "ok": att_report.get("ok"),
            "error": att_report.get("error"),
            "event_id": (att_report.get("receipt") or {}).get("event_id"),
        }
        remote_att = att_report.get("attestation") or {}
        remote_root = str(remote_att.get("dataset_merkle_root") or "").strip() or None
        report["event_verified"] = bool(att_report.get("event_verified"))
    elif event_id:
        os.environ.setdefault("SEISO_ALLOW_NOSTR", "1")
        from seiso.research.nostr.events import verify_event
        from seiso.research.nostr.policy import (
            normalize_relay_list,
            relay_allowlist_from_env,
        )
        from seiso.research.nostr.relays import fetch_event_by_id

        relay_urls = list(relay) if relay else relay_allowlist_from_env()
        if not relay_urls:
            report["ok"] = False
            report["error"] = "no relays configured for --event-id fetch"
            console.print_json(data=report)
            raise typer.Exit(code=1)
        from urllib.parse import urlparse

        host_allowlist = [
            (urlparse(r).hostname or "").lower() for r in relay_urls if r.strip()
        ]
        normalized = normalize_relay_list(
            relay_urls, allowlist=host_allowlist, allow_loopback=allow_loopback
        )
        event = fetch_event_by_id(
            event_id,
            normalized,
            allowlist=host_allowlist,
            allow_loopback=allow_loopback,
        )
        if event is None:
            report["ok"] = False
            report["error"] = "event not found on configured relays"
            console.print_json(data=report)
            raise typer.Exit(code=1)
        sig_ok = verify_event(event)
        report["event_verified"] = sig_ok
        try:
            remote_att = json.loads(str(event.get("content") or ""))
        except json.JSONDecodeError:
            remote_att = None
        if isinstance(remote_att, dict):
            remote_root = (
                str(remote_att.get("dataset_merkle_root") or "").strip() or None
            )
        if not sig_ok:
            report["ok"] = False
            report["error"] = "invalid event signature"
            console.print_json(data=report)
            raise typer.Exit(code=1)
    else:
        # No external root source — local path check is enough.
        console.print_json(data=report)
        return

    match = (
        remote_root is not None and root.lower() == remote_root.lower()
    )
    report["event_root_match"] = match
    report["remote_dataset_merkle_root"] = remote_root
    report["ok"] = bool(path_ok and match and report.get("event_verified") is not False)
    if not match:
        report["error"] = "proof root does not match Nostr attestation dataset_merkle_root"
    console.print_json(data=report)
    if not report["ok"]:
        raise typer.Exit(code=1)
