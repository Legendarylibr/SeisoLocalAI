"""Generic agent cover for Seiso — Buzz-compatible, not Buzz-only.

Agent-driven workflows (CLI, chat harnesses, Buzz rooms) share one surface.
Buzz is an optional control-plane / receipt channel, not a required runtime.
"""

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

__all__ = [
    "TrainingSurface",
    "agent_receipt",
    "buzz_agent_present",
    "buzz_compatible_receipt",
    "channel_safe_plan_view",
    "get_buzz_keypair",
    "relay_policy_note",
    "relay_signed_event",
    "require_buzz_agent",
    "require_buzz_nsec",
    "resolve_training_surface",
    "signed_agent_interaction",
]
