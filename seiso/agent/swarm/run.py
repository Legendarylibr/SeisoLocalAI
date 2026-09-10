"""Run a swarm Plan through the existing run_harness kernel.

The swarm escalates sensibly: start with just the worker (no subagents),
verify completion, and only add subagent roles one at a time when the
verification demands more scrutiny. Each escalation re-runs the worker
with continuation feedback from the failed verdicts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from seiso.agent.adapters.types import LaunchResult, LaunchSpec
from seiso.agent.harness import HarnessContext, HarnessResult, run_harness
from seiso.agent.kernel import ComputeDecision
from seiso.agent.swarm.presets import build_plan, enabled_roles
from seiso.agent.swarm.types import AgentSettings, SwarmResult
from seiso.agent.swarm.verify import check_completion, check_correctness, maybe_judge
from seiso.agent.tasks import Plan, Step
from seiso.routing.types import RouteDecision

WorkerFn = Callable[[LaunchSpec], LaunchResult]
JudgeFn = Callable[[str, str, Mapping[str, Any]], str]
PreflightFn = Callable[[str], bool]

# Verifier roles added one at a time as verification demands more scrutiny.
# Rung 0 = worker only (deterministic completion check, no LLM).
_ESCALATION_ORDER: tuple[str, ...] = ("completion", "correctness")


def _payload(step: Step) -> dict[str, Any]:
    return dict(step.payload or {})


def default_preflight(model_id: str) -> bool:
    """Skip LLM roles when headroom is too small. Never overcommit."""
    _ = model_id
    try:
        from seiso.hardware.profile import hardware_profile
        from seiso.hardware.tiers import vram_headroom_mb
        from seiso.memory.estimates import estimate_chat_vram_gb
        from seiso.memory.protection.oom import allow_memory_overcommit
    except Exception:
        return True
    if allow_memory_overcommit():
        return True
    try:
        profile = hardware_profile()
        headroom = vram_headroom_mb(profile)
    except Exception:
        return True
    need_mb = int(estimate_chat_vram_gb("7B") * 1024)
    if str(model_id).lower() not in {"", "auto", "default"}:
        need_mb = max(need_mb, 512)
    if headroom <= 0:
        return False
    return headroom >= need_mb


def _launch_spec(
    step: Step,
    settings: AgentSettings,
    *,
    endpoint_url: str,
    model_id: str,
    api_key: str,
    workdir: Path,
    isolated: Path,
) -> LaunchSpec:
    payload = _payload(step)
    return LaunchSpec(
        goal=str(payload.get("goal") or step.id),
        workdir=str(workdir),
        isolated_config_dir=str(isolated),
        endpoint_url=endpoint_url,
        model_id=model_id,
        api_key=api_key,
    )


def _escalation_ladder(settings: AgentSettings) -> tuple[str, ...]:
    """Verifier roles enabled for this swarm, in escalation order."""
    enabled = set(enabled_roles(settings))
    return tuple(role for role in _ESCALATION_ORDER if role in enabled)


def _with_continuation(plan: Plan, continuation: str) -> Plan:
    """Append verifier feedback to the worker's goal for its next run."""
    if not continuation:
        return plan
    steps: list[Step] = []
    for step in plan.steps:
        if step.id == "worker":
            payload = dict(step.payload)
            goal = str(payload.get("goal") or "")
            payload["goal"] = (
                f"{goal}\n\nVerifier feedback — continue and address:\n{continuation}"
            )
            step = Step(
                id=step.id,
                kind=step.kind,
                action=step.action,
                payload=payload,
                required_context=step.required_context,
                estimated_vram_mb=step.estimated_vram_mb,
            )
        steps.append(step)
    return Plan(id=plan.id, goal=plan.goal, steps=tuple(steps), route_class=plan.route_class)


def _build_feedback(verdicts: list[dict[str, Any]]) -> str:
    """Concise continuation prompt from the failed verdicts of one rung."""
    parts: list[str] = []
    for verdict in verdicts:
        if verdict.get("ok"):
            continue
        reason = str(verdict.get("reason") or "incomplete")
        evidence = (
            verdict.get("evidence") if isinstance(verdict.get("evidence"), Mapping) else {}
        )
        detail = ""
        missing = evidence.get("missing")
        if missing:
            detail = f" missing: {', '.join(str(item) for item in missing)}"
        elif evidence.get("tests_failed"):
            detail = " tests failed"
        elif evidence.get("compile_errors"):
            detail = (
                f" compile errors: {'; '.join(str(e) for e in evidence['compile_errors'][:2])}"
            )
        parts.append(f"- {reason}{detail}")
    return "\n".join(parts)


def run_swarm(
    goal: str,
    settings: AgentSettings,
    ctx: HarnessContext,
    *,
    worker: WorkerFn | None = None,
    judge: JudgeFn | None = None,
    preflight: PreflightFn | None = None,
    workdir: Path | None = None,
    isolated_dir: Path | None = None,
    endpoint_url: str = "",
    model_id: str = "default",
    api_key: str = "",
    plan_id: str | None = None,
) -> SwarmResult:
    pid = plan_id or f"swarm-{uuid4().hex[:8]}"
    root = workdir or Path.cwd()
    isolated = isolated_dir or (root / ".seiso-agent" / settings.harness)
    check = preflight or default_preflight

    # Escalation ladder: verifier roles added one at a time as verification
    # demands more scrutiny. Rung 0 = worker only (deterministic completion
    # check, no LLM). Each escalation re-runs the worker with continuation
    # feedback from the failed verdicts.
    ladder = _escalation_ladder(settings)
    max_rungs = len(ladder)
    planner_enabled = "planner" in enabled_roles(settings)

    last_worker: dict[str, Any] = {}
    planner_draft = ""
    continuation = ""
    verdicts: list[dict[str, Any]] = []
    results: list[Any] = []
    receipts: list[dict[str, Any]] = []
    final_status = "done"
    blocked_reason: str | None = None

    def exec_worker(
        step: Step, _decision: ComputeDecision, route: RouteDecision | None
    ) -> Mapping[str, Any]:
        nonlocal last_worker
        if ctx.dry_run:
            last_worker = {"dry_run": True, "harness": settings.harness}
            return last_worker
        if worker is None:
            last_worker = {"exit_code": 1, "status": "failed", "detail": "no_worker"}
            return last_worker
        chosen = (route.model_id if route is not None else None) or model_id
        spec = _launch_spec(
            step,
            settings,
            endpoint_url=endpoint_url,
            model_id=chosen,
            api_key=api_key,
            workdir=root,
            isolated=isolated,
        )
        if planner_draft:
            spec = LaunchSpec(
                goal=f"Follow this plan, then do the goal.\n\nPlan:\n{planner_draft}\n\nGoal:\n{spec.goal}",
                workdir=spec.workdir,
                isolated_config_dir=spec.isolated_config_dir,
                endpoint_url=spec.endpoint_url,
                model_id=spec.model_id,
                api_key=spec.api_key,
                timeout_sec=spec.timeout_sec,
                extra_env=spec.extra_env,
            )
        result = worker(spec)
        last_worker = result.as_dict()
        # Worker ran; completion/correctness decide pass/fail from exit_code.
        last_worker["status"] = "done"
        last_worker["goal"] = spec.goal
        return last_worker

    def exec_planner(
        step: Step, _decision: ComputeDecision, _route: RouteDecision | None
    ) -> Mapping[str, Any]:
        nonlocal planner_draft
        payload = _payload(step)
        if ctx.dry_run:
            return {"dry_run": True, "role": "planner"}
        if planner_draft:
            # Planner runs once; later escalation rungs reuse the cached draft.
            return {
                "ok": True,
                "role": "planner",
                "plan": plan.as_dict(),
                "used_llm": False,
                "cached": True,
            }
        if not payload.get("allow_llm") or judge is None:
            return {"ok": True, "role": "planner", "plan": plan.as_dict(), "used_llm": False}
        if not check(str(payload.get("model_id") or "auto")):
            return {"ok": True, "role": "planner", "blocked": "oom_guard", "plan": plan.as_dict()}
        raw = judge(
            str(payload.get("system_prompt") or "Return a short JSON plan."),
            str(payload.get("goal") or ""),
            {},
        )
        planner_draft = raw[:2000]
        return {"ok": True, "role": "planner", "draft": planner_draft, "used_llm": True}

    def exec_completion(
        step: Step, _decision: ComputeDecision, _route: RouteDecision | None
    ) -> Mapping[str, Any]:
        payload = _payload(step)
        verdict = check_completion(last_worker, workdir=root)
        verdict = maybe_judge(
            verdict,
            allow_llm=bool(payload.get("allow_llm")),
            system_prompt=str(payload.get("system_prompt") or ""),
            evidence=verdict.evidence,
            judge=judge,
            preflight_ok=check(str(payload.get("model_id") or "auto")),
        )
        verdicts.append(verdict.as_dict())
        out = verdict.as_dict()
        if not verdict.ok:
            out["status"] = "failed"
        return out

    def exec_correctness(
        step: Step, _decision: ComputeDecision, _route: RouteDecision | None
    ) -> Mapping[str, Any]:
        payload = _payload(step)
        verdict = check_correctness(last_worker, workdir=root)
        verdict = maybe_judge(
            verdict,
            allow_llm=bool(payload.get("allow_llm")),
            system_prompt=str(payload.get("system_prompt") or ""),
            evidence=verdict.evidence,
            judge=judge,
            preflight_ok=check(str(payload.get("model_id") or "auto")),
        )
        verdicts.append(verdict.as_dict())
        out = verdict.as_dict()
        if not verdict.ok:
            out["status"] = "failed"
        return out

    def exec_synth(
        step: Step, _decision: ComputeDecision, _route: RouteDecision | None
    ) -> Mapping[str, Any]:
        payload = _payload(step)
        if ctx.dry_run:
            return {"dry_run": True, "role": "synthesizer"}
        summary = f"harness={settings.harness} worker={last_worker.get('detail') or last_worker.get('status')}"
        if (
            payload.get("allow_llm")
            and judge is not None
            and check(str(payload.get("model_id") or "auto"))
        ):
            extra = judge(
                str(payload.get("system_prompt") or "Summarize the swarm result in two sentences."),
                summary,
                last_worker,
            )
            return {"ok": True, "summary": extra[:2000], "used_llm": True}
        return {"ok": True, "summary": summary, "used_llm": False}

    executors = dict(ctx.executors)
    executors.setdefault("worker", exec_worker)
    executors.setdefault("planner", exec_planner)
    executors.setdefault("completion", exec_completion)
    executors.setdefault("correctness", exec_correctness)
    executors.setdefault("synthesizer", exec_synth)

    def _step_verify(step: Step, output: Mapping[str, Any]) -> bool:
        if step.action in {"completion", "correctness"}:
            return bool(output.get("ok", True))
        return str(output.get("status") or "") != "failed"

    verify = ctx.verify or _step_verify
    harness_ctx = HarnessContext(
        local_healthy=ctx.local_healthy,
        inventory=ctx.inventory,
        available_vram_mb=ctx.available_vram_mb,
        mesh_peers_ok=ctx.mesh_peers_ok,
        pay_url=ctx.pay_url,
        quote=ctx.quote,
        surface=ctx.surface,
        allow_mesh=ctx.allow_mesh,
        allow_pay=ctx.allow_pay,
        buzz_agent=ctx.buzz_agent,
        external_router_enabled=ctx.external_router_enabled,
        external_router_url=ctx.external_router_url,
        confirm=ctx.confirm,
        dry_run=ctx.dry_run,
        executors=executors,
        verify=verify,
    )

    for rung in range(max_rungs + 1):
        rung_start = len(verdicts)
        active_roles = (("planner",) if planner_enabled else ()) + ladder[:rung]
        plan = build_plan(goal, settings, plan_id=pid, active_roles=active_roles)
        if continuation:
            plan = _with_continuation(plan, continuation)

        result: HarnessResult = run_harness(plan, harness_ctx)
        results.extend(result.results)
        receipts.extend(result.receipts)

        if result.status == "blocked":
            final_status = "blocked"
            blocked_reason = result.blocked_reason
            break

        exit_code = last_worker.get("exit_code")
        if exit_code not in (None, 0):
            # Hard worker failure — no escalation, report failed.
            verdict = check_completion(last_worker, workdir=root)
            verdicts.append(verdict.as_dict())
            final_status = "failed"
            blocked_reason = f"worker_exit:{exit_code}"
            break

        if rung == 0:
            # Rung 0 has no verifier steps in the plan: run the deterministic
            # completion check explicitly (no LLM).
            verdict = check_completion(last_worker, workdir=root)
            verdicts.append(verdict.as_dict())
            if verdict.ok:
                final_status = "done"
                break
        elif result.status == "done":
            final_status = "done"
            break

        # Incomplete: escalate with continuation feedback, or report partial.
        if rung < max_rungs:
            continuation = _build_feedback(verdicts[rung_start:])
        else:
            final_status = "partial"
            break

    # Synthesizer (if enabled) wraps the final result — not an escalation gate.
    if "synthesizer" in enabled_roles(settings) and final_status in {"done", "partial"}:
        synth_plan = build_plan(goal, settings, plan_id=pid, active_roles=("synthesizer",))
        synth_result = run_harness(synth_plan, harness_ctx)
        results.extend(synth_result.results)
        receipts.extend(synth_result.receipts)

    return SwarmResult(
        status=final_status,
        plan_id=pid,
        harness=settings.harness,
        blocked_reason=blocked_reason,
        results=tuple(results),
        receipts=tuple(receipts),
        verdicts=tuple(verdicts),
    )
