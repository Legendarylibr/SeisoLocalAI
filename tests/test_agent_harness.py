"""Unit tests for run_harness (plan → decide → route → act → verify → receipt)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seiso.agent.harness import MAX_HARNESS_STEPS, HarnessContext, run_harness
from seiso.agent.kernel import ComputeDecision, ComputeTarget
from seiso.agent.policy import RouteClass
from seiso.agent.receipts import agent_receipt
from seiso.agent.tasks import Plan, Step, TaskKind
from seiso.routing.types import Candidate, RouteDecision


def _cand(**kwargs: Any) -> Candidate:
    base = dict(
        model_id="local-7b",
        backend="llamacpp",
        role="chat",
        context_tokens=8192,
        vram_mb=4000,
        downloaded=True,
        params_b=7.0,
    )
    base.update(kwargs)
    return Candidate(**base)


def _step(step_id: str = "s1", kind: TaskKind = TaskKind.CHAT, action: str = "run") -> Step:
    return Step(id=step_id, kind=kind, action=action, required_context=2048)


def _plan(*steps: Step, route_class: str = "allow_paid") -> Plan:
    return Plan(id="p1", goal="test", steps=steps, route_class=route_class)


def _ok_executor(
    step: Step, decision: ComputeDecision, route: RouteDecision | None
) -> Mapping[str, Any]:
    return {"ok": True, "step": step.id, "target": decision.target.value}


def test_happy_path_local() -> None:
    calls: list[str] = []

    def exec_(
        step: Step, decision: ComputeDecision, route: RouteDecision | None
    ) -> Mapping[str, Any]:
        calls.append(step.id)
        assert decision.target is ComputeTarget.LOCAL
        assert route is not None
        return {"ok": True}

    result = run_harness(
        _plan(_step("a"), _step("b")),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            executors={"run": exec_},
        ),
    )
    assert result.status == "done"
    assert calls == ["a", "b"]
    assert all(r.status == "done" for r in result.results)
    assert result.receipts[-1]["status"] == "done"


def test_dry_run_does_not_call_executor() -> None:
    def boom(*_a: Any, **_k: Any) -> Mapping[str, Any]:
        raise AssertionError("executor must not run in dry_run")

    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            dry_run=True,
            executors={"run": boom},
        ),
    )
    assert result.status == "done"
    assert result.results[0].output["dry_run"] is True


def test_ask_human_stops_without_retry() -> None:
    runs = {"n": 0}

    def exec_(*_a: Any, **_k: Any) -> Mapping[str, Any]:
        runs["n"] += 1
        return {}

    result = run_harness(
        _plan(_step("a"), _step("b")),
        HarnessContext(
            local_healthy=False,
            allow_pay=False,
            allow_mesh=False,
            inventory=(_cand(),),
            executors={"run": exec_},
        ),
    )
    assert result.status == "blocked"
    assert result.blocked_reason
    assert runs["n"] == 0
    assert result.results[0].status == "blocked"
    assert len(result.results) == 1


def test_verify_fail_stops() -> None:
    result = run_harness(
        _plan(_step("a"), _step("b")),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            executors={"run": _ok_executor},
            verify=lambda _step, _out: False,
        ),
    )
    assert result.status == "failed"
    assert result.blocked_reason == "verify_failed"
    assert len(result.results) == 1
    assert result.results[0].status == "failed"


def test_verify_pass_continues() -> None:
    result = run_harness(
        _plan(_step("a"), _step("b")),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            executors={"run": _ok_executor},
            verify=lambda _step, _out: True,
        ),
    )
    assert result.status == "done"
    assert len(result.results) == 2


def test_step_cap() -> None:
    steps = tuple(_step(f"s{i}") for i in range(MAX_HARNESS_STEPS + 3))
    result = run_harness(
        _plan(*steps),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            dry_run=True,
        ),
    )
    assert result.status == "blocked"
    assert result.blocked_reason == f"step_cap:{MAX_HARNESS_STEPS}"
    assert len(result.results) == MAX_HARNESS_STEPS


def test_mesh_requires_confirm() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=False,
            mesh_peers_ok=True,
            allow_mesh=True,
            buzz_agent=True,
            allow_pay=False,
            inventory=(_cand(),),
            confirm=False,
            dry_run=True,
        ),
    )
    assert result.status == "blocked"
    assert result.blocked_reason == "confirm_required:mesh"


def test_pay_requires_confirm() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=False,
            allow_mesh=False,
            allow_pay=True,
            pay_url="https://pay.example.com",
            inventory=(_cand(),),
            confirm=False,
            dry_run=True,
        ),
    )
    assert result.status == "blocked"
    assert result.blocked_reason == "confirm_required:pay"


def test_mesh_with_confirm_runs() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=False,
            mesh_peers_ok=True,
            allow_mesh=True,
            buzz_agent=True,
            inventory=(_cand(),),
            confirm=True,
            dry_run=True,
        ),
    )
    assert result.status == "done"
    assert result.results[0].compute_target == "mesh"


def test_no_executor_fails() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            dry_run=False,
            executors={},
        ),
    )
    assert result.status == "failed"
    assert result.blocked_reason == "no_executor:run"


def test_receipt_redacts_injected_secrets() -> None:
    def exec_(
        step: Step, decision: ComputeDecision, route: RouteDecision | None
    ) -> Mapping[str, Any]:
        return {
            "ok": True,
            "nsec": "nsec1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
            "message": "token seiso_pay_token=seiso_pay_abc",
        }

    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=True,
            inventory=(_cand(),),
            executors={"run": exec_},
        ),
    )
    blob = repr(result.as_dict())
    assert "nsec1" not in blob
    assert "seiso_pay_abc" not in blob
    for receipt in result.receipts:
        assert "nsec" not in receipt
        assert "seiso_pay_token" not in receipt
        assert receipt.get("hostname") is None


def test_receipt_uses_same_scrubber_as_agent_receipt() -> None:
    raw = agent_receipt(
        role="train",
        status="done",
        nsec="nsec1shouldneverappearxxxxxxxxxxxxxxxx",
        message="seiso_mesh_token=supersecret",
    )
    result = run_harness(
        _plan(_step()),
        HarnessContext(local_healthy=True, inventory=(_cand(),), dry_run=True),
    )
    assert "nsec1shouldneverappear" not in repr(raw)
    assert "[redacted-secret]" in raw["message"]
    assert all("nsec" not in r for r in result.receipts)


def test_empty_plan_done() -> None:
    result = run_harness(
        Plan(id="empty", goal="noop", steps=()),
        HarnessContext(local_healthy=True),
    )
    assert result.status == "done"
    assert result.results == ()


def test_route_failure_is_failed() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(
            local_healthy=True,
            inventory=(),
            dry_run=True,
        ),
    )
    assert result.status == "failed"
    assert result.blocked_reason
    assert "missing_inventory" in result.blocked_reason


def test_never_leave_does_not_use_external() -> None:
    result = run_harness(
        _plan(_step(), route_class=RouteClass.NEVER_LEAVE.value),
        HarnessContext(
            local_healthy=True,
            inventory=(),
            external_router_enabled=True,
            external_router_url="http://127.0.0.1:8780",
            dry_run=True,
        ),
    )
    assert result.status == "failed"


def test_harness_result_as_dict() -> None:
    result = run_harness(
        _plan(_step()),
        HarnessContext(local_healthy=True, inventory=(_cand(),), dry_run=True),
    )
    data = result.as_dict()
    assert data["status"] == "done"
    assert data["plan_id"] == "p1"
    assert isinstance(data["receipts"], list)
