"""CLI helpers for Buzz-facing signed agent status (relay only with signing)."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from seiso_cli.console import console

agent_app = typer.Typer(
    name="agent",
    help=("Buzz-compatible agent helpers. Relay only signed NIP-01 / BIP-340 events via buzz-cli."),
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

    Embed ``nostr_event`` JSON in a Buzz kind-9 ``buzz messages send`` (Buzz
    rejects ``--kind 31254``). Do not treat ``buzz_receipt`` as channel authority.
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


@agent_app.command("decide")
def agent_decide(
    job: Annotated[
        str, typer.Option(help="Job / task kind, e.g. finetune|chat|slime")
    ] = "finetune",
    local_healthy: Annotated[
        bool, typer.Option("--local-healthy/--no-local-healthy", help="Local Forge/CLI healthy")
    ] = True,
    mesh_peers: Annotated[bool, typer.Option("--mesh-peers/--no-mesh-peers")] = False,
    pay_url: Annotated[str | None, typer.Option("--pay-url")] = None,
    route_class: Annotated[str, typer.Option("--route-class")] = "allow_paid",
    surface: Annotated[str | None, typer.Option(help="frontend|agent")] = None,
) -> None:
    """Print a ComputeDecision JSON (local → mesh → pay → ask_human)."""
    from seiso.agent.kernel import decide_compute

    decision = decide_compute(
        local_healthy=local_healthy,
        mesh_peers_ok=mesh_peers,
        pay_url=pay_url,
        route_class=route_class,
        job_kind=job,
        surface=surface,
    )
    _print_json(decision.as_dict())


@agent_app.command("plan")
def agent_plan(
    task: Annotated[str, typer.Option(help="Single-step task kind")] = "chat",
    goal: Annotated[str, typer.Option(help="Plan goal")] = "dry-run",
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = True,
    local_healthy: Annotated[bool, typer.Option("--local-healthy/--no-local-healthy")] = True,
    route_class: Annotated[str, typer.Option("--route-class")] = "allow_paid",
    confirm: Annotated[bool, typer.Option("--confirm/--no-confirm")] = False,
    context: Annotated[int, typer.Option(help="Required context tokens")] = 2048,
    vram_mb: Annotated[int, typer.Option("--vram-mb")] = 8192,
) -> None:
    """Build a one-step harness plan. Default is --dry-run (no jobs)."""
    from seiso.agent.harness import HarnessContext, run_harness
    from seiso.agent.tasks import Plan, Step, parse_task_kind
    from seiso.routing.types import Candidate

    kind = parse_task_kind(task)
    plan = Plan(
        id="cli-plan",
        goal=goal,
        route_class=route_class,
        steps=(Step(id="step-1", kind=kind, action="run", required_context=context),),
    )
    inventory = (
        Candidate(
            model_id="local-default",
            backend="llamacpp",
            role="chat" if kind.value not in {"code", "embed", "draft", "target"} else kind.value,
            context_tokens=max(context, 2048),
            vram_mb=4096,
            downloaded=True,
            params_b=7.0,
        ),
    )
    result = run_harness(
        plan,
        HarnessContext(
            local_healthy=local_healthy,
            inventory=inventory,
            available_vram_mb=vram_mb,
            confirm=confirm,
            dry_run=dry_run,
        ),
    )
    _print_json({"plan": plan.as_dict(), "result": result.as_dict()})


@agent_app.command("harnesses")
def agent_harnesses() -> None:
    """Detect Pi / OMP / Hermes / Cline / OpenClaw on PATH."""
    from seiso.agent.adapters.detect import detect_all

    rows = detect_all(include_version=True)
    _print_json({"harnesses": [row.as_dict() for row in rows]})


@agent_app.command("endpoint")
def agent_endpoint(
    source: Annotated[str, typer.Option(help="auto|ollama|router|forge")] = "auto",
    route_class: Annotated[str, typer.Option("--route-class")] = "allow_paid",
) -> None:
    """Print the resolved localhost OpenAI-compat URL (no secrets)."""
    from seiso.agent.adapters.endpoint import resolve_endpoint

    endpoint = resolve_endpoint(source=source, route_class=route_class, probe=False)
    _print_json(endpoint.public_dict())


@agent_app.command("swarm")
def agent_swarm(
    goal: Annotated[str, typer.Option(help="Swarm goal")] = "dry-run",
    harness: Annotated[str, typer.Option(help="pi|omp|hermes|cline|openclaw")] = "hermes",
    preset: Annotated[str, typer.Option(help="single|pair|plan_act_verify")] = "single",
    dry_run: Annotated[bool, typer.Option("--dry-run/--run")] = True,
    subagents: Annotated[bool, typer.Option("--subagents/--no-subagents")] = False,
    route_class: Annotated[str, typer.Option("--route-class")] = "allow_paid",
    source: Annotated[str, typer.Option(help="auto|ollama|router|forge")] = "auto",
    confirm: Annotated[bool, typer.Option("--confirm/--no-confirm")] = False,
) -> None:
    """Build and optionally run a swarm (default --dry-run, no child processes)."""
    from seiso.agent.adapters.types import parse_harness_id
    from seiso.agent.harness import HarnessContext
    from seiso.agent.swarm.presets import build_plan
    from seiso.agent.swarm.run import run_swarm
    from seiso.agent.swarm.types import AgentSettings, parse_preset
    from seiso.routing.types import Candidate

    settings = AgentSettings(
        harness=parse_harness_id(harness),
        model_source=source,
        seiso_subagents=subagents,
        preset=parse_preset(preset),
        route_class=route_class,
    )
    # activate_subagents() already ran in AgentSettings.__post_init__ when on:
    # pair + completion/correctness, no extra LLM.
    plan = build_plan(goal, settings, plan_id="cli-swarm")
    inventory = (
        Candidate(
            model_id="local-default",
            backend="llamacpp",
            role="code",
            context_tokens=8192,
            vram_mb=4096,
            downloaded=True,
            params_b=7.0,
        ),
    )
    result = run_swarm(
        goal,
        settings,
        HarnessContext(
            local_healthy=True,
            inventory=inventory,
            confirm=confirm,
            dry_run=dry_run,
        ),
        plan_id=plan.id,
    )
    _print_json({"plan": plan.as_dict(), "result": result.as_dict()})
