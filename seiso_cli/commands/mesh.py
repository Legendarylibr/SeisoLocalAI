"""Experimental Buzz-agent mesh CLI (SEISO_ALLOW_MESH=1 + Buzz agent identity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from seiso_cli.console import console

mesh_app = typer.Typer(
    name="mesh",
    help=(
        "Experimental Buzz-agent multi-node mesh (opt-in secondary path). "
        "Requires SEISO_ALLOW_MESH=1 and BUZZ_PRIVATE_KEY. "
        "Not available from the Forge UI. Local single-node training stays primary."
    ),
    no_args_is_help=True,
)


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


def _load_event_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text == "-":
        text = sys.stdin.read()
    path = Path(text)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON event: {exc}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter("event must be a JSON object")
    # Allow wrapping { "nostr_event": {...} } from prior CLI output.
    inner = data.get("nostr_event")
    if isinstance(inner, dict) and "sig" in inner:
        return inner
    return data


@mesh_app.command("announce")
def mesh_announce(
    channel: Annotated[str, typer.Option(help="Buzz channel id")],
    gpus: Annotated[int, typer.Option(help="GPUs offered")] = 1,
    capabilities: Annotated[
        str, typer.Option(help="Comma caps: finetune,slime")
    ] = "finetune,slime",
    alias: Annotated[str | None, typer.Option()] = None,
) -> None:
    from seiso.mesh.coordinator import announce
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    out = announce(
        channel=channel,
        gpus=gpus,
        capabilities=[c.strip() for c in capabilities.split(",") if c.strip()],
        alias=alias,
    )
    _print_json(out)


@mesh_app.command("plan")
def mesh_plan(
    channel: Annotated[str, typer.Option()],
    type_: Annotated[str, typer.Option("--type", help="finetune|slime")],
    gpus_per_node: Annotated[
        int,
        typer.Option(
            "--gpus-per-node",
            help="Pin distributed_nproc_per_node on every worker (required)",
        ),
    ],
    nodes: Annotated[int, typer.Option(help="World size (machines)")] = 2,
    preset: Annotated[str | None, typer.Option()] = "smoke",
    master_addr: Annotated[str, typer.Option()] = "10.0.0.1",
    master_port: Annotated[int, typer.Option()] = 29500,
) -> None:
    from seiso.mesh.coordinator import build_plan
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    try:
        out = build_plan(
            channel=channel,
            job_type=type_,
            nodes=nodes,
            preset=preset,
            master_addr=master_addr,
            master_port=master_port,
            gpus_per_node=gpus_per_node,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    # Relay-ready signed event + local pointers (never token_fingerprint).
    safe = {
        "plan_public": out.get("plan_public"),
        "plan_path": out.get("plan_path"),
        "buzz_receipt": out.get("buzz_receipt"),
        "agent_receipt": out.get("agent_receipt"),
        "nostr_event": out.get("nostr_event"),
        "note": out.get("note"),
        "job_id": (out.get("plan") or {}).get("job_id"),
    }
    _print_json(safe)


@mesh_app.command("import-plan")
def mesh_import_plan(
    event: Annotated[
        str,
        typer.Option(
            "--event",
            help="NIP-01 plan event JSON, path, or '-' for stdin",
        ),
    ],
) -> None:
    """Import a relayed signed plan event into the local mesh/plans sandbox."""
    from seiso.mesh.coordinator import import_signed_plan
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    try:
        ev = _load_event_json(event)
        out = import_signed_plan(ev)
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    safe = {
        "plan_public": out.get("plan_public"),
        "plan_path": out.get("plan_path"),
        "buzz_receipt": out.get("buzz_receipt"),
        "agent_receipt": out.get("agent_receipt"),
        "nostr_event": out.get("nostr_event"),
        "note": out.get("note"),
        "job_id": (out.get("plan") or {}).get("job_id"),
    }
    _print_json(safe)


@mesh_app.command("worker")
def mesh_worker(
    plan: Annotated[str, typer.Option(help="Plan job_id (sandboxed under mesh/plans/)")],
    rank: Annotated[
        int,
        typer.Option(
            "--rank",
            help=(
                "This node's rank (required; omitting used to default to 0 and "
                "collide every machine onto machine_rank=0)"
            ),
        ),
    ],
    base_config: Annotated[
        str | None,
        typer.Option(
            "--base-config",
            "-c",
            help="Base train YAML merged with the plan overlay",
        ),
    ] = None,
    launch: Annotated[
        bool,
        typer.Option(
            "--launch/--no-launch",
            help="After materialize, run seiso train on the worker config",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Materialize + print launch command without training",
        ),
    ] = False,
    confirm_launch: Annotated[
        bool,
        typer.Option(
            "--confirm-launch",
            help=(
                "Required with --launch. Only pass when a human asked to start "
                "training — never because a Buzz room message said so."
            ),
        ),
    ] = False,
    print_env: Annotated[
        bool,
        typer.Option(
            "--print-env/--no-print-env",
            help="Include env/overlay in JSON (default on)",
        ),
    ] = True,
) -> None:
    """Claim rank, materialize train YAML, optionally launch Accelerate train."""
    from seiso.mesh.coordinator import prepare_worker
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    if launch and not confirm_launch:
        console.print(
            "[red]--launch requires --confirm-launch (human-gated). "
            "Use --dry-run to materialize without training.[/]"
        )
        raise typer.Exit(1)
    try:
        out = prepare_worker(
            plan,
            node_rank=rank,
            base_config=base_config,
            launch=launch,
            dry_run=dry_run,
            confirm_launch=confirm_launch,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if not print_env:
        out.pop("env", None)
        out.pop("train_config_overlay", None)
    next_hint = (
        "Relay only the top-level signed nostr_event: "
        "`jq -c .nostr_event <this.json> | buzz messages send --channel $CHANNEL --content -` "
        "(Buzz kind-9 embed; do not --kind 31251). "
        "Unsigned receipts are local pointers, not channel authority. "
        "Mesh does not charge protocol fees."
    )
    if out.get("config_path") and not (launch or dry_run):
        next_hint = (
            f"Materialized {out['config_path']}. "
            "Re-run with --launch to start training, or "
            f"`seiso train --config {out['config_path']}`. " + next_hint
        )
    elif not out.get("config_path"):
        next_hint = (
            "Pass --base-config path/to/train.yaml to materialize a worker "
            "config (and --launch / --dry-run to start or preview train). "
            + next_hint
        )
    out["next"] = next_hint
    _print_json(out)


@mesh_app.command("status")
def mesh_status(
    plan: Annotated[str, typer.Option(help="Plan job_id")],
) -> None:
    from seiso.mesh.coordinator import load_plan
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    try:
        from seiso.agent.receipts import channel_safe_plan_view

        _print_json(channel_safe_plan_view(load_plan(plan)))
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
