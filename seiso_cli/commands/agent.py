"""CLI helpers for Buzz-facing signed agent status (relay only with signing)."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from seiso_cli.console import console

agent_app = typer.Typer(
    name="agent",
    help=(
        "Buzz-compatible agent helpers. "
        "Relay only signed NIP-01 / BIP-340 events via buzz-cli."
    ),
    no_args_is_help=True,
)


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


@agent_app.command("status")
def agent_status(
    role: Annotated[str, typer.Option(help="Milestone role, e.g. train|export|doctor")],
    status: Annotated[str, typer.Option(help="Status string, e.g. started|done|failed")],
    channel: Annotated[
        str | None, typer.Option(help="Buzz channel id (tagged into the event)")
    ] = None,
    job_id: Annotated[str | None, typer.Option(help="Optional job / run id")] = None,
    message: Annotated[str | None, typer.Option(help="Short human summary")] = None,
) -> None:
    """Emit a signed agent status event + local receipt pointer.

    Post ``nostr_event`` to Buzz via buzz-cli. Do not treat ``buzz_receipt`` as
    channel authority.
    """
    from seiso.agent.signed_relay import signed_agent_interaction

    fields: dict[str, Any] = {}
    if job_id:
        fields["job_id"] = job_id
    if message:
        fields["message"] = message
    try:
        out = signed_agent_interaction(
            role=role,
            status=status,
            channel=channel,
            require_nsec=True,
            **fields,
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _print_json(out)
