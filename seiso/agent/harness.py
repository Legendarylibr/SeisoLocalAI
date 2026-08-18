"""Deterministic harness: observe → decide → route → act → verify → receipt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from seiso.agent.kernel import ComputeDecision, ComputeTarget, decide_compute
from seiso.agent.policy import RouteClass, parse_route_class
from seiso.agent.receipts import agent_receipt, buzz_compatible_receipt
from seiso.agent.tasks import Plan, Step, StepResult
from seiso.routing.select import select_route
from seiso.routing.types import Candidate, NoRouteError, RouteDecision, RouteRequest

# Mirror forge/tools/agent_loop._MAX_TOOL_CALLS_PER_ROUND.
MAX_HARNESS_STEPS = 8

StepExecutor = Callable[[Step, ComputeDecision, RouteDecision | None], Mapping[str, Any]]
StepVerifier = Callable[[Step, Mapping[str, Any]], bool]


class HarnessError(RuntimeError):
    """Harness stopped without completing the plan."""


@dataclass(frozen=True, slots=True)
class HarnessResult:
    status: str
    plan_id: str
    results: tuple[StepResult, ...]
    receipts: tuple[dict[str, Any], ...]
    blocked_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan_id": self.plan_id,
            "blocked_reason": self.blocked_reason,
            "results": [r.as_dict() for r in self.results],
            "receipts": list(self.receipts),
        }


@dataclass
class HarnessContext:
    local_healthy: bool
    inventory: tuple[Candidate, ...] = ()
    available_vram_mb: int = 8192
    mesh_peers_ok: bool = False
    pay_url: str | None = None
    quote: dict[str, Any] | None = None
    surface: str | None = "agent"
    allow_mesh: bool | None = None
    allow_pay: bool | None = None
    buzz_agent: bool | None = None
    external_router_enabled: bool = False
    external_router_url: str | None = None
    confirm: bool = False
    dry_run: bool = False
    executors: Mapping[str, StepExecutor] = field(default_factory=dict)
    verify: StepVerifier | None = None


def _privileged(target: ComputeTarget) -> bool:
    return target in {ComputeTarget.MESH, ComputeTarget.PAY}


def _receipt(
    *,
    role: str,
    status: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return agent_receipt(role=role, status=status, surface="agent", **dict(extra))


def run_harness(plan: Plan, ctx: HarnessContext) -> HarnessResult:
    """Run a plan. Mesh/pay acts require ``ctx.confirm``; dry-run never acts."""
    policy = parse_route_class(plan.route_class)
    results: list[StepResult] = []
    receipts: list[dict[str, Any]] = []

    if not plan.steps:
        receipts.append(_receipt(role="harness", status="done", extra={"plan_id": plan.id}))
        return HarnessResult("done", plan.id, (), tuple(receipts), None)

    steps = plan.steps[:MAX_HARNESS_STEPS]
    capped = len(plan.steps) > MAX_HARNESS_STEPS

    for step in steps:
        decision = decide_compute(
            local_healthy=ctx.local_healthy,
            mesh_peers_ok=ctx.mesh_peers_ok,
            pay_url=ctx.pay_url,
            route_class=policy,
            job_kind=step.kind.value,
            surface=ctx.surface,
            allow_mesh=ctx.allow_mesh,
            allow_pay=ctx.allow_pay,
            buzz_agent=ctx.buzz_agent,
            quote=ctx.quote,
        )
        if decision.target is ComputeTarget.ASK_HUMAN:
            result = StepResult(
                step_id=step.id,
                status="blocked",
                compute_target=decision.target.value,
                detail=decision.reason,
            )
            results.append(result)
            receipts.append(
                _receipt(
                    role=step.kind.value,
                    status="blocked",
                    extra={
                        "plan_id": plan.id,
                        "step_id": step.id,
                        "reason": decision.reason,
                        "target": decision.target.value,
                    },
                )
            )
            return HarnessResult(
                "blocked", plan.id, tuple(results), tuple(receipts), decision.reason
            )

        if _privileged(decision.target) and not ctx.confirm:
            reason = f"confirm_required:{decision.target.value}"
            result = StepResult(
                step_id=step.id,
                status="blocked",
                compute_target=decision.target.value,
                detail=reason,
            )
            results.append(result)
            receipts.append(
                _receipt(
                    role=step.kind.value,
                    status="blocked",
                    extra={
                        "plan_id": plan.id,
                        "step_id": step.id,
                        "reason": reason,
                    },
                )
            )
            return HarnessResult("blocked", plan.id, tuple(results), tuple(receipts), reason)

        route: RouteDecision | None = None
        try:
            route = select_route(
                RouteRequest(
                    task=step.kind,
                    required_context=step.required_context,
                    available_vram_mb=ctx.available_vram_mb,
                    inventory=ctx.inventory,
                    external_router_enabled=ctx.external_router_enabled
                    and policy is not RouteClass.NEVER_LEAVE,
                    external_router_url=ctx.external_router_url,
                    route_class=policy,
                )
            )
        except (NoRouteError, ValueError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            result = StepResult(
                step_id=step.id,
                status="failed",
                compute_target=decision.target.value,
                detail=reason,
            )
            results.append(result)
            receipts.append(
                _receipt(
                    role=step.kind.value,
                    status="failed",
                    extra={"plan_id": plan.id, "step_id": step.id, "reason": reason},
                )
            )
            return HarnessResult("failed", plan.id, tuple(results), tuple(receipts), reason)

        if ctx.dry_run:
            output: Mapping[str, Any] = {
                "dry_run": True,
                "compute": decision.as_dict(),
                "route": route.as_dict(),
            }
        else:
            executor = ctx.executors.get(step.action) or ctx.executors.get(step.kind.value)
            if executor is None:
                reason = f"no_executor:{step.action}"
                result = StepResult(
                    step_id=step.id,
                    status="failed",
                    compute_target=decision.target.value,
                    route_backend=route.backend,
                    route_model=route.model_id,
                    detail=reason,
                )
                results.append(result)
                receipts.append(
                    _receipt(
                        role=step.kind.value,
                        status="failed",
                        extra={"plan_id": plan.id, "step_id": step.id, "reason": reason},
                    )
                )
                return HarnessResult("failed", plan.id, tuple(results), tuple(receipts), reason)
            output = buzz_compatible_receipt(dict(executor(step, decision, route)))

        if ctx.verify is not None and not ctx.dry_run and not ctx.verify(step, output):
            reason = "verify_failed"
            result = StepResult(
                step_id=step.id,
                status="failed",
                compute_target=decision.target.value,
                route_backend=route.backend,
                route_model=route.model_id,
                detail=reason,
                output=output,
            )
            results.append(result)
            receipts.append(
                _receipt(
                    role=step.kind.value,
                    status="failed",
                    extra={"plan_id": plan.id, "step_id": step.id, "reason": reason},
                )
            )
            return HarnessResult("failed", plan.id, tuple(results), tuple(receipts), reason)

        result = StepResult(
            step_id=step.id,
            status="done",
            compute_target=decision.target.value,
            route_backend=route.backend,
            route_model=route.model_id,
            detail=route.reason,
            output=output,
        )
        results.append(result)
        receipts.append(
            _receipt(
                role=step.kind.value,
                status="done",
                extra={
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "target": decision.target.value,
                    "fee_sats": decision.fee_sats,
                    "backend": route.backend,
                    "model_id": route.model_id,
                },
            )
        )

    if capped:
        reason = f"step_cap:{MAX_HARNESS_STEPS}"
        receipts.append(
            _receipt(
                role="harness",
                status="blocked",
                extra={"plan_id": plan.id, "reason": reason},
            )
        )
        return HarnessResult("blocked", plan.id, tuple(results), tuple(receipts), reason)

    receipts.append(_receipt(role="harness", status="done", extra={"plan_id": plan.id}))
    return HarnessResult("done", plan.id, tuple(results), tuple(receipts), None)
