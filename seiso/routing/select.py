"""Pick a backend + model from local inventory, then an optional external router."""

from __future__ import annotations

from seiso.agent.policy import RouteClass, parse_route_class
from seiso.agent.tasks import TaskKind, parse_task_kind
from seiso.routing.external import validate_router_url
from seiso.routing.table import roles_for_task
from seiso.routing.types import Candidate, NoRouteError, RouteDecision, RouteRequest

ROUTER_MODEL_ID = "__seiso_router__"
ROUTER_BACKEND = "router"


def _params(candidate: Candidate) -> float:
    if candidate.params_b is not None:
        return float(candidate.params_b)
    return 0.0


def _fits(candidate: Candidate, request: RouteRequest) -> tuple[bool, str]:
    if not candidate.downloaded:
        return False, f"not_downloaded:{candidate.model_id}"
    if candidate.context_tokens < request.required_context:
        return False, (
            f"context_miss:{candidate.model_id}:"
            f"{candidate.context_tokens}<{request.required_context}"
        )
    if candidate.vram_mb > request.available_vram_mb:
        return False, (
            f"vram_miss:{candidate.model_id}:{candidate.vram_mb}>{request.available_vram_mb}"
        )
    return True, "ok"


def _local_candidates(request: RouteRequest) -> tuple[list[Candidate], list[str]]:
    preferred = roles_for_task(request.task)
    role_rank = {role: index for index, role in enumerate(preferred)}
    considered: list[Candidate] = []
    misses: list[str] = []
    for candidate in request.inventory:
        if candidate.role not in role_rank:
            continue
        ok, why = _fits(candidate, request)
        if not ok:
            misses.append(why)
            considered.append(candidate)
            continue
        considered.append(candidate)
    fitting = [c for c in request.inventory if c.role in role_rank and _fits(c, request)[0]]
    fitting.sort(key=lambda c: (role_rank[c.role], -_params(c), c.vram_mb, c.model_id))
    return fitting, misses


def _maybe_validate_external(request: RouteRequest) -> None:
    if not request.external_router_enabled:
        return
    url = (request.external_router_url or "").strip()
    validate_router_url(url)


def select_route(request: RouteRequest) -> RouteDecision:
    """Choose ``{backend, model, reason}`` — local inventory first.

    Raises:
        ValueError: external router enabled with a non-loopback / invalid URL.
        NoRouteError: nothing local fits and the external router is not allowed.
    """
    route_class = parse_route_class(request.route_class)
    _maybe_validate_external(request)

    fitting, misses = _local_candidates(request)
    if fitting:
        chosen = fitting[0]
        larger_missed = any(
            _params(c) > _params(chosen)
            and not _fits(c, request)[0]
            and c.role in {r for r in roles_for_task(request.task)}
            for c in request.inventory
        )
        reason = f"local:{chosen.model_id}:fits"
        if larger_missed:
            reason = f"step_down:{chosen.model_id}:larger_missed"
        return RouteDecision(
            backend=chosen.backend,
            model_id=chosen.model_id,
            reason=reason,
            route_class=route_class,
            source="local",
            stepped_down=larger_missed,
        )

    if request.external_router_enabled and route_class != RouteClass.NEVER_LEAVE:
        return RouteDecision(
            backend=ROUTER_BACKEND,
            model_id=ROUTER_MODEL_ID,
            reason="external_router:no_local_fit",
            route_class=route_class,
            source="external",
            stepped_down=False,
        )

    if not request.inventory:
        detail = "missing_inventory"
    elif not any(c.downloaded for c in request.inventory):
        detail = "not_downloaded"
    elif misses:
        detail = misses[0]
    else:
        detail = f"no_role_match:{request.task.value}"
    if route_class == RouteClass.NEVER_LEAVE and request.external_router_enabled:
        detail = f"never_leave:{detail}"
    raise NoRouteError(detail)


def select_route_from_args(
    *,
    task: str | TaskKind,
    required_context: int,
    available_vram_mb: int,
    inventory: list[Candidate] | tuple[Candidate, ...],
    external_router_enabled: bool = False,
    external_router_url: str | None = None,
    route_class: str | RouteClass = RouteClass.ALLOW_PAID,
) -> RouteDecision:
    request = RouteRequest(
        task=parse_task_kind(task),
        required_context=int(required_context),
        available_vram_mb=int(available_vram_mb),
        inventory=tuple(inventory),
        external_router_enabled=external_router_enabled,
        external_router_url=external_router_url,
        route_class=parse_route_class(route_class),
    )
    return select_route(request)
