"""Persist TUI agent-harness settings under SEISO_DATA_DIR/agent/."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.agent.adapters.detect import default_harness_id, detect_all
from seiso.agent.adapters.endpoint import resolve_endpoint
from seiso.agent.adapters.profiles import isolated_dir, write_profile
from seiso.agent.adapters.types import HARNESS_IDS, HARNESS_LABELS
from seiso.agent.policy import RouteClass
from seiso.agent.swarm.presets import cycle_value
from seiso.agent.swarm.types import (
    MODEL_SOURCES,
    SUBAGENT_ROLES,
    SWARM_PRESETS,
    AgentSettings,
    SubagentSpec,
)

SETTINGS_NAME = "settings.json"


def settings_path(data_dir: Path) -> Path:
    return Path(data_dir) / "agent" / SETTINGS_NAME


def load_settings(data_dir: Path) -> AgentSettings:
    path = settings_path(data_dir)
    if not path.is_file():
        settings = AgentSettings(harness=default_harness_id())
        return settings
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AgentSettings(harness=default_harness_id())
    if not isinstance(raw, dict):
        return AgentSettings(harness=default_harness_id())
    return AgentSettings.from_dict(raw)


def save_settings(data_dir: Path, settings: AgentSettings) -> Path:
    path = settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def summary_line(settings: AgentSettings) -> str:
    flag = "on" if settings.seiso_subagents else "off"
    return f"{HARNESS_LABELS.get(settings.harness, settings.harness)} · subagents {flag} · {settings.model_source}"


def cycle_harness(settings: AgentSettings) -> AgentSettings:
    settings.harness = cycle_value(settings.harness, HARNESS_IDS)
    return settings


def cycle_source(settings: AgentSettings) -> AgentSettings:
    settings.model_source = cycle_value(settings.model_source, MODEL_SOURCES)
    return settings


def cycle_preset(settings: AgentSettings) -> AgentSettings:
    settings.preset = cycle_value(settings.preset, SWARM_PRESETS)
    return settings


def cycle_route(settings: AgentSettings) -> AgentSettings:
    order = tuple(item.value for item in RouteClass)
    settings.route_class = cycle_value(settings.route_class, order)
    return settings


def toggle_subagents(settings: AgentSettings) -> AgentSettings:
    if settings.seiso_subagents:
        settings.deactivate_subagents()
    else:
        settings.activate_subagents()
    return settings


def toggle_role(settings: AgentSettings, role: str) -> AgentSettings:
    spec = settings.subagents.get(role)
    if spec is None:
        return settings
    spec.enabled = not spec.enabled
    return settings


def toggle_role_llm(settings: AgentSettings, role: str) -> AgentSettings:
    spec = settings.subagents.get(role)
    if spec is None:
        return settings
    spec.allow_llm = not spec.allow_llm
    return settings


def set_role_prompt(settings: AgentSettings, role: str, text: str) -> AgentSettings:
    spec = settings.subagents.get(role)
    if spec is None:
        return settings
    spec.system_prompt = text
    return settings


def cycle_role_model(settings: AgentSettings, role: str, models: list[str]) -> AgentSettings:
    spec = settings.subagents.get(role)
    if spec is None:
        return settings
    options = ("auto", *tuple(models))
    spec.model_id = cycle_value(spec.model_id, options)
    return settings


def prepare_endpoint(data_dir: Path, settings: AgentSettings, *, probe: bool = False):
    endpoint = resolve_endpoint(
        source=settings.model_source,
        data_dir=data_dir,
        route_class=settings.route_class,
        probe=probe,
    )
    dest = isolated_dir(data_dir, settings.harness)
    if endpoint.url:
        write_profile(dest, settings.harness, endpoint)
    return endpoint


__all__ = [
    "HARNESS_IDS",
    "HARNESS_LABELS",
    "SUBAGENT_ROLES",
    "AgentSettings",
    "SubagentSpec",
    "cycle_harness",
    "cycle_preset",
    "cycle_role_model",
    "cycle_route",
    "cycle_source",
    "detect_all",
    "load_settings",
    "prepare_endpoint",
    "save_settings",
    "set_role_prompt",
    "summary_line",
    "toggle_role",
    "toggle_role_llm",
    "toggle_subagents",
]
