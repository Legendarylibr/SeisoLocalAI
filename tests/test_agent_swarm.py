"""Swarm presets, subagent toggles, verifiers, OOM skip."""

from __future__ import annotations

from seiso.agent.harness import HarnessContext
from seiso.agent.swarm.presets import build_plan, enabled_roles
from seiso.agent.swarm.run import run_swarm
from seiso.agent.swarm.types import AgentSettings, SubagentSpec
from seiso.agent.swarm.verify import check_completion, check_correctness, parse_judge_json
from seiso.routing.types import Candidate


def _ctx(**kwargs) -> HarnessContext:
    inventory = (
        Candidate(
            model_id="local-7b",
            backend="llamacpp",
            role="code",
            context_tokens=8192,
            vram_mb=4000,
            downloaded=True,
            params_b=7.0,
        ),
    )
    base = dict(local_healthy=True, inventory=inventory, dry_run=True)
    base.update(kwargs)
    return HarnessContext(**base)


def test_defaults_leave_swarms_off() -> None:
    settings = AgentSettings()
    assert settings.seiso_subagents is False
    assert settings.preset == "single"
    assert enabled_roles(settings) == ()
    assert all(not spec.enabled for spec in settings.subagents.values())


def test_turning_on_enables_pair_verifiers_without_llm() -> None:
    settings = AgentSettings()
    settings.activate_subagents()
    assert settings.seiso_subagents is True
    assert settings.preset == "pair"
    assert enabled_roles(settings) == ("completion", "correctness")
    assert settings.subagents["completion"].allow_llm is False
    assert settings.subagents["planner"].enabled is False
    settings.deactivate_subagents()
    assert settings.seiso_subagents is False
    assert enabled_roles(settings) == ()


def test_subagents_off_is_worker_only() -> None:
    settings = AgentSettings(seiso_subagents=False, preset="plan_act_verify")
    settings.subagents["planner"] = SubagentSpec(role="planner", enabled=True)
    plan = build_plan("goal", settings)
    assert [step.id for step in plan.steps] == ["worker"]
    assert enabled_roles(settings) == ()


def test_pair_includes_enabled_verifiers_only() -> None:
    settings = AgentSettings(seiso_subagents=True, preset="pair")
    settings.subagents["completion"] = SubagentSpec(role="completion", enabled=True)
    settings.subagents["correctness"] = SubagentSpec(role="correctness", enabled=False)
    plan = build_plan("goal", settings)
    assert [step.id for step in plan.steps] == ["worker", "completion"]


def test_custom_prompt_on_plan_payload() -> None:
    settings = AgentSettings(seiso_subagents=True, preset="pair")
    settings.subagents["completion"] = SubagentSpec(
        role="completion",
        enabled=True,
        system_prompt="JSON only. Be strict.",
        allow_llm=True,
    )
    plan = build_plan("goal", settings)
    step = next(s for s in plan.steps if s.id == "completion")
    assert step.payload["system_prompt"] == "JSON only. Be strict."
    assert step.payload["allow_llm"] is True


def test_dry_run_swarm_no_worker() -> None:
    def boom(*_a, **_k):
        raise AssertionError("worker must not run")

    settings = AgentSettings()
    result = run_swarm("smoke", settings, _ctx(), worker=boom)
    assert result.status == "done"
    assert result.results[0].output["dry_run"] is True


def test_completion_fail_fast() -> None:
    from seiso.agent.adapters.types import LaunchResult, LaunchSpec

    def fail_worker(_spec: LaunchSpec) -> LaunchResult:
        return LaunchResult("hermes", 1, detail="failed")

    settings = AgentSettings(seiso_subagents=True, preset="pair")
    settings.subagents["completion"] = SubagentSpec(role="completion", enabled=True)
    result = run_swarm("x", settings, _ctx(dry_run=False), worker=fail_worker)
    assert result.status == "failed"
    assert any(v.get("ok") is False for v in result.verdicts)


def test_tests_beat_llm_yes_man() -> None:
    verdict = check_correctness({"tests_ran": True, "tests_ok": False, "exit_code": 0})
    assert verdict.ok is False
    assert verdict.reason == "tests_failed"


def test_judge_parse_and_empty() -> None:
    ok = parse_judge_json('{"ok": true, "reasons": ["files exist"]}')
    assert ok.ok is True
    bad = parse_judge_json("not json")
    assert bad.ok is False


def test_oom_preflight_skips_llm() -> None:
    from seiso.agent.adapters.types import LaunchResult, LaunchSpec

    def ok_worker(_spec: LaunchSpec) -> LaunchResult:
        return LaunchResult("hermes", 0, detail="ok")

    called = {"n": 0}

    def judge(*_a, **_k) -> str:
        called["n"] += 1
        return '{"ok": true, "reasons": ["nope"]}'

    settings = AgentSettings(seiso_subagents=True, preset="pair")
    settings.subagents["completion"] = SubagentSpec(
        role="completion", enabled=True, allow_llm=True
    )
    result = run_swarm(
        "x",
        settings,
        _ctx(dry_run=False),
        worker=ok_worker,
        judge=judge,
        preflight=lambda _m: False,
    )
    assert called["n"] == 0
    assert result.verdicts
    assert result.verdicts[0].get("used_llm") is False


def test_compile_failure_fails_correctness(tmp_path) -> None:
    from seiso.agent.swarm.verify import check_correctness

    broken = tmp_path / "broken.py"
    broken.write_text("def nope(\n", encoding="utf-8")
    verdict = check_correctness(
        {"exit_code": 0, "stdout": "edited broken.py"},
        workdir=tmp_path,
    )
    assert verdict.ok is False
    assert verdict.reason == "compile_failed"


def test_planner_draft_is_prepended_to_worker_goal() -> None:
    from seiso.agent.adapters.types import LaunchResult, LaunchSpec

    seen: list[str] = []

    def capture(spec: LaunchSpec) -> LaunchResult:
        seen.append(spec.goal)
        return LaunchResult("hermes", 0, detail="ok")

    def judge(system: str, user: str, _ev) -> str:
        _ = system, user
        return '{"steps": ["write tests"]}'

    settings = AgentSettings(seiso_subagents=True, preset="plan_act_verify")
    settings.subagents["planner"] = SubagentSpec(
        role="planner", enabled=True, allow_llm=True
    )
    settings.subagents["completion"] = SubagentSpec(role="completion", enabled=False)
    settings.subagents["correctness"] = SubagentSpec(role="correctness", enabled=False)
    settings.subagents["synthesizer"] = SubagentSpec(role="synthesizer", enabled=False)
    run_swarm("add tests", settings, _ctx(dry_run=False), worker=capture, judge=judge)
    assert seen
    assert "write tests" in seen[0]
    assert "add tests" in seen[0]


def test_check_completion_missing_file(tmp_path) -> None:
    verdict = check_completion(
        {"exit_code": 0, "expect_files": ["nope.txt"]},
        workdir=tmp_path,
    )
    assert verdict.ok is False
    assert verdict.reason == "missing_artifacts"
