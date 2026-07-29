"""Generic agent cover for Seiso — Buzz-compatible, not Buzz-only.

Agent-driven workflows (CLI, chat harnesses, Buzz rooms) share one surface.
Buzz is an optional control-plane / receipt channel, not a required runtime.
"""

from seiso.agent.receipts import agent_receipt, buzz_compatible_receipt
from seiso.agent.surface import (
    TrainingSurface,
    buzz_agent_present,
    require_buzz_agent,
    resolve_training_surface,
)

__all__ = [
    "TrainingSurface",
    "agent_receipt",
    "buzz_agent_present",
    "buzz_compatible_receipt",
    "require_buzz_agent",
    "resolve_training_surface",
]
