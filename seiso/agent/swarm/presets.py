"""Turn AgentSettings + a goal into a run_harness Plan."""

from __future__ import annotations

from seiso.agent.swarm.types import SUBAGENT_ROLES, AgentSettings
from seiso.agent.tasks import Plan, Step, TaskKind

_PRESET_ROLES: dict[str, tuple[str, ...]] = {
    "single": (),
    "pair": ("completion", "correctness"),
    "plan_act_verify": ("planner", "completion", "correctness", "synthesizer"),
}


def enabled_roles(settings: AgentSettings) -> tuple[str, ...]:
    if not settings.seiso_subagents:
        return ()
    wanted = _PRESET_ROLES.get(settings.preset, ())
    out: list[str] = []
    for role in wanted:
        spec = settings.subagents.get(role)
        if spec is not None and spec.enabled:
            out.append(role)
    return tuple(out)


def build_plan(
    goal: str,
    settings: AgentSettings,
    *,
    plan_id: str = "swarm",
    active_roles: tuple[str, ...] | None = None,
) -> Plan:
    """Turn AgentSettings + a goal into a run_harness Plan.

    ``active_roles`` overrides the preset-derived role set — the escalating
    swarm uses it to add subagents one rung at a time.
    """
    text = (goal or "").strip() or "swarm"
    steps: list[Step] = []
    roles = active_roles if active_roles is not None else enabled_roles(settings)
    if "planner" in roles:
        spec = settings.subagents["planner"]
        steps.append(
            Step(
                id="planner",
                kind=TaskKind.CODE,
                action="planner",
                payload={
                    "goal": text,
                    "role": "planner",
                    "model_id": spec.model_id,
                    "system_prompt": spec.system_prompt,
                    "max_tokens": spec.max_tokens,
                    "allow_llm": spec.allow_llm,
                },
            )
        )
    steps.append(
        Step(
            id="worker",
            kind=TaskKind.CODE,
            action="worker",
            payload={"goal": text, "role": "worker", "harness": settings.harness},
        )
    )
    for role in ("completion", "correctness", "synthesizer"):
        if role not in roles:
            continue
        spec = settings.subagents[role]
        steps.append(
            Step(
                id=role,
                kind=TaskKind.CODE,
                action=role,
                payload={
                    "goal": text,
                    "role": role,
                    "model_id": spec.model_id,
                    "system_prompt": spec.system_prompt,
                    "max_tokens": spec.max_tokens,
                    "allow_llm": spec.allow_llm,
                },
            )
        )
    return Plan(
        id=plan_id,
        goal=text,
        steps=tuple(steps),
        route_class=settings.route_class,
    )


def cycle_value(current: str, options: tuple[str, ...]) -> str:
    if current not in options:
        return options[0]
    return options[(options.index(current) + 1) % len(options)]


# Keep SUBAGENT_ROLES imported for callers that want the full set.
_ = SUBAGENT_ROLES
