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
