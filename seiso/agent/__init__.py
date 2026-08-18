"""Generic agent cover for Seiso — Buzz-compatible, not Buzz-only.

Agent-driven workflows (CLI, chat harnesses, Buzz rooms) share one surface.
Buzz is an optional control-plane / receipt channel, not a required runtime.

Heavy kernel / harness / routing symbols are loaded lazily so
``seiso.inference`` can import this package without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seiso.agent.nostr_identity import get_buzz_keypair, require_buzz_nsec
from seiso.agent.receipts import agent_receipt, buzz_compatible_receipt, channel_safe_plan_view
from seiso.agent.signed_relay import (
    relay_policy_note,
    relay_signed_event,
    signed_agent_interaction,
)
from seiso.agent.surface import (
    TrainingSurface,
    buzz_agent_present,
    require_buzz_agent,
    resolve_training_surface,
)

if TYPE_CHECKING:
    from seiso.agent.harness import (
        MAX_HARNESS_STEPS,
        HarnessContext,
        HarnessResult,
        run_harness,
    )
    from seiso.agent.kernel import ComputeDecision, ComputeTarget, decide_compute
    from seiso.agent.policy import RouteClass, parse_route_class
    from seiso.agent.tasks import Plan, Step, StepResult, TaskKind

__all__ = [
    "ComputeDecision",
    "ComputeTarget",
    "HarnessContext",
    "HarnessResult",
    "MAX_HARNESS_STEPS",
    "Plan",
    "RouteClass",
    "Step",
    "StepResult",
    "TaskKind",
    "TrainingSurface",
    "agent_receipt",
    "buzz_agent_present",
    "buzz_compatible_receipt",
    "channel_safe_plan_view",
    "decide_compute",
    "get_buzz_keypair",
    "parse_route_class",
    "relay_policy_note",
    "relay_signed_event",
    "require_buzz_agent",
    "require_buzz_nsec",
    "resolve_training_surface",
    "run_harness",
    "signed_agent_interaction",
]


def __getattr__(name: str) -> Any:
    if name in {"RouteClass", "parse_route_class"}:
        from seiso.agent import policy as _policy

        return getattr(_policy, name)
    if name in {"TaskKind", "Plan", "Step", "StepResult"}:
        from seiso.agent import tasks as _tasks

        return getattr(_tasks, name)
    if name in {"ComputeDecision", "ComputeTarget", "decide_compute"}:
        from seiso.agent import kernel as _kernel

        return getattr(_kernel, name)
    if name in {"MAX_HARNESS_STEPS", "HarnessContext", "HarnessResult", "run_harness"}:
        from seiso.agent import harness as _harness

        return getattr(_harness, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
