"""Route request / decision types for model-aware routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seiso.agent.policy import RouteClass
from seiso.agent.tasks import TaskKind


@dataclass(frozen=True, slots=True)
class Candidate:
    """A local (or router) model that might handle a step."""

    model_id: str
    backend: str
    role: str
    context_tokens: int
    vram_mb: int
    downloaded: bool = True
    quant: str | None = None
    params_b: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouteRequest:
    task: TaskKind
    required_context: int
    available_vram_mb: int
    inventory: tuple[Candidate, ...]
    external_router_enabled: bool = False
    external_router_url: str | None = None
    route_class: RouteClass = RouteClass.ALLOW_PAID

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "required_context": self.required_context,
            "available_vram_mb": self.available_vram_mb,
            "inventory": [c.as_dict() for c in self.inventory],
            "external_router_enabled": self.external_router_enabled,
            "external_router_url": self.external_router_url,
            "route_class": self.route_class.value,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    backend: str
    model_id: str
    reason: str
    route_class: RouteClass
    source: str
    stepped_down: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "reason": self.reason,
            "route_class": self.route_class.value,
            "source": self.source,
            "stepped_down": self.stepped_down,
        }


class NoRouteError(ValueError):
    """Nothing in inventory (or the external router) can serve this request."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
