"""Compute decision kernel: local → mesh → pay → ask_human.

Self-hosted is always free. Localhost marketplace URLs are refused.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from seiso.agent.policy import RouteClass, parse_route_class
from seiso.agent.surface import TrainingSurface, buzz_agent_present, resolve_training_surface
from seiso.agent.tasks import JOB_KINDS, job_kind_for_task, parse_task_kind
from seiso.compat import StrEnum
from seiso.mesh.flags import mesh_allowed
from seiso.pay.flags import pay_allowed
from seiso.routing.external import is_loopback_url


class ComputeTarget(StrEnum):
    LOCAL = "local"
    MESH = "mesh"
    PAY = "pay"
    ASK_HUMAN = "ask_human"


@dataclass(frozen=True, slots=True)
class ComputeDecision:
    target: ComputeTarget
    fee_sats: int
    reason: str
    route_class: RouteClass
    job_kind: str | None = None
    quote: dict[str, Any] | None = None
    consulted_pay: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.value
        data["route_class"] = self.route_class.value
        return data


def pay_url_from_env() -> str | None:
    raw = (os.environ.get("SEISO_PAY_URL") or "").strip()
    return raw or None


def _local_decision(route_class: RouteClass, job_kind: str | None, reason: str) -> ComputeDecision:
    return ComputeDecision(
        target=ComputeTarget.LOCAL,
        fee_sats=0,
        reason=reason,
        route_class=route_class,
        job_kind=job_kind,
        quote=None,
        consulted_pay=False,
    )


def _ask(
    route_class: RouteClass,
    job_kind: str | None,
    reason: str,
    *,
    consulted_pay: bool = False,
    quote: dict[str, Any] | None = None,
) -> ComputeDecision:
    return ComputeDecision(
        target=ComputeTarget.ASK_HUMAN,
        fee_sats=0,
        reason=reason,
        route_class=route_class,
        job_kind=job_kind,
        quote=quote,
        consulted_pay=consulted_pay,
    )


def decide_compute(
    *,
    local_healthy: bool,
    mesh_peers_ok: bool = False,
    pay_url: str | None = None,
    route_class: str | RouteClass | None = None,
    job_kind: str | None = None,
    surface: str | TrainingSurface | None = None,
    allow_mesh: bool | None = None,
    allow_pay: bool | None = None,
    buzz_agent: bool | None = None,
    quote: dict[str, Any] | None = None,
) -> ComputeDecision:
    """Where should this job run?

    Order: healthy local (free) → mesh peers (free) → paid marketplace → human.
    ``never_leave`` never leaves the machine. ``local_then_mesh`` never pays.
    A loopback ``pay_url`` is refused so localhost is never billed.
    """
    policy = parse_route_class(route_class)
    kind: str | None = None
    if job_kind:
        parsed = parse_task_kind(job_kind)
        kind = job_kind_for_task(parsed)
        if kind not in JOB_KINDS and job_kind.strip().lower() not in JOB_KINDS:
            kind = parsed.value

    if local_healthy:
        return _local_decision(policy, kind, "local_healthy")

    if policy is RouteClass.NEVER_LEAVE:
        return _ask(policy, kind, "never_leave:local_unhealthy")

    # Kernel default is the agent surface. Forge UI must pass frontend explicitly.
    if surface is None:
        resolved_surface = TrainingSurface.AGENT
    else:
        resolved_surface = resolve_training_surface(
            explicit=surface.value if isinstance(surface, TrainingSurface) else surface
        )
    mesh_flag = mesh_allowed() if allow_mesh is None else bool(allow_mesh)
    buzz = buzz_agent_present() if buzz_agent is None else bool(buzz_agent)
    frontend = resolved_surface is TrainingSurface.FRONTEND

    mesh_ok = (
        mesh_flag
        and buzz
        and mesh_peers_ok
        and not frontend
        and policy in {RouteClass.LOCAL_THEN_MESH, RouteClass.ALLOW_PAID}
    )
    if mesh_ok:
        return ComputeDecision(
            target=ComputeTarget.MESH,
            fee_sats=0,
            reason="mesh_peers",
            route_class=policy,
            job_kind=kind,
            quote=None,
            consulted_pay=False,
        )

    if policy is RouteClass.LOCAL_THEN_MESH:
        why = "local_then_mesh:"
        if frontend:
            why += "frontend_surface"
        elif not mesh_flag:
            why += "mesh_flag_off"
        elif not buzz:
            why += "buzz_agent_missing"
        elif not mesh_peers_ok:
            why += "peers_insufficient"
        else:
            why += "unavailable"
        return _ask(policy, kind, why)

    pay_flag = pay_allowed() if allow_pay is None else bool(allow_pay)
    url = (pay_url if pay_url is not None else pay_url_from_env()) or ""
    url = url.strip()

    if policy is not RouteClass.ALLOW_PAID:
        return _ask(policy, kind, "pay_not_in_route_class")

    if not pay_flag:
        return _ask(policy, kind, "pay_flag_off")

    if not url:
        return _ask(policy, kind, "pay_url_unset")

    if is_loopback_url(url):
        return _ask(policy, kind, "refuse_localhost_pay", consulted_pay=True)

    fee = 0
    if quote:
        try:
            fee = int(quote.get("total_sats") or quote.get("compute_sats") or 0)
        except (TypeError, ValueError):
            fee = 0
    return ComputeDecision(
        target=ComputeTarget.PAY,
        fee_sats=max(0, fee),
        reason="marketplace",
        route_class=policy,
        job_kind=kind,
        quote=dict(quote) if quote else None,
        consulted_pay=True,
    )
