"""CLI: model-aware route picker (local inventory + optional external router)."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from seiso_cli.console import console


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


def _candidates_from_json(raw: str | None) -> list[Any]:
    from seiso.routing.types import Candidate

    if not raw or not raw.strip():
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise typer.BadParameter("--inventory-json must be a JSON array")
    out: list[Candidate] = []
    for item in payload:
        if not isinstance(item, dict):
            raise typer.BadParameter("inventory entries must be objects")
        out.append(
            Candidate(
                model_id=str(item["model_id"]),
                backend=str(item.get("backend") or "llamacpp"),
                role=str(item.get("role") or "chat"),
                context_tokens=int(item.get("context_tokens") or 8192),
                vram_mb=int(item.get("vram_mb") or 4096),
                downloaded=bool(item.get("downloaded", True)),
                quant=item.get("quant"),
                params_b=float(item["params_b"]) if item.get("params_b") is not None else None,
            )
        )
    return out


def route(
    task: Annotated[str, typer.Option(help="Task kind: chat, code, finetune, …")] = "chat",
    context: Annotated[int, typer.Option(help="Required context tokens")] = 8192,
    vram_mb: Annotated[int, typer.Option("--vram-mb", help="Available VRAM (MiB)")] = 8192,
    inventory_json: Annotated[
        str | None,
        typer.Option("--inventory-json", help="JSON array of Candidate objects"),
    ] = None,
    route_class: Annotated[str, typer.Option("--route-class")] = "allow_paid",
    external: Annotated[bool, typer.Option("--external/--no-external")] = False,
    router_url: Annotated[str | None, typer.Option("--router-url")] = None,
) -> None:
    """Print a RouteDecision JSON (no GPU, no network)."""
    from seiso.routing.select import select_route_from_args
    from seiso.routing.types import NoRouteError

    try:
        decision = select_route_from_args(
            task=task,
            required_context=context,
            available_vram_mb=vram_mb,
            inventory=_candidates_from_json(inventory_json),
            external_router_enabled=external,
            external_router_url=router_url,
            route_class=route_class,
        )
    except (NoRouteError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    _print_json(decision.as_dict())
