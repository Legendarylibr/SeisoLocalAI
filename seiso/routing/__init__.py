"""Model-aware routing: local inventory first, optional localhost external router."""

from seiso.routing.external import (
    LOOPBACK_HOSTS,
    is_loopback_host,
    is_loopback_url,
    validate_router_url,
)
from seiso.routing.select import (
    ROUTER_BACKEND,
    ROUTER_MODEL_ID,
    select_route,
    select_route_from_args,
)
from seiso.routing.table import ROLES_FOR_TASK, roles_for_task
from seiso.routing.types import Candidate, NoRouteError, RouteDecision, RouteRequest

__all__ = [
    "LOOPBACK_HOSTS",
    "ROUTER_BACKEND",
    "ROUTER_MODEL_ID",
    "Candidate",
    "NoRouteError",
    "ROLES_FOR_TASK",
    "RouteDecision",
    "RouteRequest",
    "is_loopback_host",
    "is_loopback_url",
    "roles_for_task",
    "select_route",
    "select_route_from_args",
    "validate_router_url",
]
