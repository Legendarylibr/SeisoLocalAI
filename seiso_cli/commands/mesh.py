"""Experimental Buzz-agent mesh CLI (SEISO_ALLOW_MESH=1 + Buzz agent identity)."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from seiso_cli.console import console

mesh_app = typer.Typer(
    name="mesh",
    help=(
        "Experimental Buzz-agent multi-node mesh (opt-in). "
        "Requires SEISO_ALLOW_MESH=1 and BUZZ_PRIVATE_KEY. "
        "Not available from the Forge UI. Not functional for real multi-node yet."
    ),
    no_args_is_help=True,
)


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


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
    _print_json(out)


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
    print_env: Annotated[
        bool, typer.Option("--print-env", help="Print env/config only")
    ] = True,
) -> None:
    """Prepare worker env for Accelerate multi-node (does not auto-launch GPU job)."""
    from seiso.mesh.coordinator import (
        buzz_heartbeat,
        load_plan,
        worker_env,
        worker_train_config_overlay,
    )
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    try:
        p = load_plan(plan)
        env = worker_env(p, node_rank=rank)
        overlay = worker_train_config_overlay(p, node_rank=rank)
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    out = {
        "env": env,
        "train_config_overlay": overlay,
        "buzz_receipt": buzz_heartbeat(p, node_rank=rank, status="joining"),
        "agent_receipt": buzz_heartbeat(p, node_rank=rank, status="joining"),
        "surface": "agent",
        "next": (
            "Apply train_config_overlay to your train YAML / Accelerate launch "
            "(Seiso honors the overlay, not env-only NNODES). "
            "Post buzz_receipt to the channel. Mesh does not charge protocol fees. "
            "Not functional for real multi-node yet."
        ),
    }
    if print_env:
        _print_json(out)
    else:
        _print_json(out)


@mesh_app.command("status")
def mesh_status(
    plan: Annotated[str, typer.Option(help="Plan job_id")],
) -> None:
    from seiso.mesh.coordinator import load_plan
    from seiso.mesh.flags import require_mesh_allowed

    require_mesh_allowed()
    try:
        _print_json(load_plan(plan))
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
