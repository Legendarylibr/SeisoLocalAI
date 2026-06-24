"""Fallback routing when primary specialist is unavailable."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from seiso.model_router.backend_lifecycle import BackendLifecycleManager, BackendState
from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute

logger = logging.getLogger(__name__)


@dataclass
class FallbackChain:
    catalog: SpecialistCatalog
    lifecycle: BackendLifecycleManager
    default_route_id: str = "general"

    def ordered_fallbacks(self, primary: SpecialistRoute) -> list[SpecialistRoute]:
        others = [r for r in self.catalog if r.route_id != primary.route_id]
        others.sort(key=lambda r: r.fallback_priority)
        return [primary, *others]

    async def resolve_awake_route(self, primary: SpecialistRoute) -> tuple[SpecialistRoute, str]:
        for route in self.ordered_fallbacks(primary):
            record = await self.lifecycle.wake_for_route(route)
            if record.state in (BackendState.AWAKE, BackendState.UNKNOWN):
                if route.route_id != primary.route_id:
                    logger.warning(
                        "fallback route %s -> %s (primary %s state=%s)",
                        primary.route_id,
                        route.route_id,
                        primary.route_id,
                        record.state.value,
                    )
                return route, (
                    "primary"
                    if route.route_id == primary.route_id
                    else f"fallback_from_{primary.route_id}"
                )
        # Last resort — return primary anyway
        return primary, "unreachable_primary"
