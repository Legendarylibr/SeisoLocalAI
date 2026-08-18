"""Seiso-native swarm: one worker harness + optional configurable subagents."""

from __future__ import annotations

from seiso.agent.swarm.presets import build_plan, enabled_roles
from seiso.agent.swarm.run import run_swarm
from seiso.agent.swarm.types import (
    MODEL_SOURCES,
    SUBAGENT_ROLES,
    SWARM_PRESETS,
    AgentSettings,
    SubagentSpec,
    SwarmResult,
    Verdict,
    parse_model_source,
    parse_preset,
)
from seiso.agent.swarm.verify import check_completion, check_correctness, parse_judge_json

__all__ = [
    "MODEL_SOURCES",
    "SUBAGENT_ROLES",
    "SWARM_PRESETS",
    "AgentSettings",
    "SubagentSpec",
    "SwarmResult",
    "Verdict",
    "build_plan",
    "check_completion",
    "check_correctness",
    "enabled_roles",
    "parse_judge_json",
    "parse_model_source",
    "parse_preset",
    "run_swarm",
]
