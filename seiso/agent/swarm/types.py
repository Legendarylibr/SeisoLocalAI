"""Swarm + Seiso-subagent config types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from seiso.agent.adapters.detect import default_harness_id
from seiso.agent.adapters.types import parse_harness_id
from seiso.agent.policy import parse_route_class

SUBAGENT_ROLES: tuple[str, ...] = ("planner", "completion", "correctness", "synthesizer")
SWARM_PRESETS: tuple[str, ...] = ("single", "pair", "plan_act_verify")
MODEL_SOURCES: tuple[str, ...] = ("auto", "ollama", "router", "forge")

SubagentRole = Literal["planner", "completion", "correctness", "synthesizer"]


def parse_preset(raw: str | None) -> str:
    text = str(raw or "single").strip().lower().replace("-", "_")
    if text not in SWARM_PRESETS:
        allowed = ", ".join(SWARM_PRESETS)
        raise ValueError(f"unknown swarm preset {raw!r}; expected one of: {allowed}")
    return text


def parse_model_source(raw: str | None) -> str:
    text = str(raw or "auto").strip().lower()
    aliases = {"local": "auto", "smart": "router", "smart_router": "router"}
    text = aliases.get(text, text)
    if text not in MODEL_SOURCES:
        allowed = ", ".join(MODEL_SOURCES)
        raise ValueError(f"unknown model source {raw!r}; expected one of: {allowed}")
    return text


@dataclass
class SubagentSpec:
    role: str
    enabled: bool = False
    model_id: str = "auto"
    system_prompt: str = ""
    max_tokens: int = 256
    allow_llm: bool = False

    def __post_init__(self) -> None:
        if self.role not in SUBAGENT_ROLES:
            allowed = ", ".join(SUBAGENT_ROLES)
            raise ValueError(f"unknown subagent role {self.role!r}; expected one of: {allowed}")
        self.max_tokens = max(16, min(int(self.max_tokens or 256), 1024))
        self.model_id = (self.model_id or "auto").strip() or "auto"
        self.system_prompt = str(self.system_prompt or "")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_subagents() -> dict[str, SubagentSpec]:
    return {role: SubagentSpec(role=role) for role in SUBAGENT_ROLES}


@dataclass
class AgentSettings:
    harness: str = "hermes"
    model_source: str = "auto"
    seiso_subagents: bool = False
    preset: str = "single"
    route_class: str = "allow_paid"
    subagents: dict[str, SubagentSpec] = field(default_factory=default_subagents)

    def __post_init__(self) -> None:
        try:
            self.harness = parse_harness_id(self.harness)
        except ValueError:
            self.harness = default_harness_id()
        self.model_source = parse_model_source(self.model_source)
        self.preset = parse_preset(self.preset)
        self.route_class = parse_route_class(self.route_class).value
        merged = default_subagents()
        for key, spec in (self.subagents or {}).items():
            if isinstance(spec, SubagentSpec) and spec.role in merged:
                merged[spec.role] = spec
            elif isinstance(spec, Mapping) and key in merged:
                merged[key] = SubagentSpec(role=key, **{k: v for k, v in spec.items() if k != "role"})
        self.subagents = merged
        if self.seiso_subagents:
            self.activate_subagents()

    def activate_subagents(self) -> None:
        """Turn swarms on. Default useful set: pair + completion/correctness, no LLM."""
        self.seiso_subagents = True
        if self.preset == "single":
            self.preset = "pair"
        if not any(spec.enabled for spec in self.subagents.values()):
            for role in ("completion", "correctness"):
                spec = self.subagents[role]
                spec.enabled = True
                spec.allow_llm = False

    def deactivate_subagents(self) -> None:
        """Master off — worker only. Per-role flags kept for the next turn-on."""
        self.seiso_subagents = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "model_source": self.model_source,
            "seiso_subagents": self.seiso_subagents,
            "preset": self.preset,
            "route_class": self.route_class,
            "subagents": {k: v.as_dict() for k, v in self.subagents.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> AgentSettings:
        data = dict(raw or {})
        subs = data.pop("subagents", None)
        settings = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if isinstance(subs, Mapping):
            settings.subagents = default_subagents()
            for key, spec in subs.items():
                if key not in settings.subagents:
                    continue
                if isinstance(spec, Mapping):
                    settings.subagents[key] = SubagentSpec(
                        role=key, **{k: v for k, v in spec.items() if k != "role"}
                    )
        return settings


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    used_llm: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "used_llm": self.used_llm,
        }


@dataclass(frozen=True, slots=True)
class SwarmResult:
    status: str
    plan_id: str
    harness: str
    blocked_reason: str | None = None
    results: tuple[Any, ...] = ()
    receipts: tuple[dict[str, Any], ...] = ()
    verdicts: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan_id": self.plan_id,
            "harness": self.harness,
            "blocked_reason": self.blocked_reason,
            "results": [r.as_dict() if hasattr(r, "as_dict") else r for r in self.results],
            "receipts": list(self.receipts),
            "verdicts": list(self.verdicts),
        }



