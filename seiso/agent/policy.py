"""OS-level route classes — how far work is allowed to leave this machine."""

from __future__ import annotations

from seiso.compat import StrEnum


class RouteClass(StrEnum):
    """Policy for compute and model routing.

    ``never_leave``
        This machine only. No mesh, no marketplace, no external router.
    ``local_then_mesh``
        Local first, then trusted Buzz mesh peers. Never paid remote.
    ``allow_paid``
        Local → mesh → opt-in marketplace. Default for agent surfaces.
    """

    NEVER_LEAVE = "never_leave"
    LOCAL_THEN_MESH = "local_then_mesh"
    ALLOW_PAID = "allow_paid"


def parse_route_class(raw: str | RouteClass | None) -> RouteClass:
    """Parse a route class; empty / None defaults to ``allow_paid``."""
    if raw is None:
        return RouteClass.ALLOW_PAID
    if isinstance(raw, RouteClass):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return RouteClass.ALLOW_PAID
    try:
        return RouteClass(text)
    except ValueError as exc:
        allowed = ", ".join(c.value for c in RouteClass)
        raise ValueError(f"unknown route_class {raw!r}; expected one of: {allowed}") from exc
