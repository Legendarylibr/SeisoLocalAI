"""Unit tests for decide_compute (local → mesh → pay → ask_human)."""

from __future__ import annotations

from typing import Any

import pytest

from seiso.agent.kernel import ComputeTarget, decide_compute
from seiso.agent.policy import RouteClass, parse_route_class
from seiso.agent.surface import TrainingSurface
from seiso.pay.pricing import quote_job


def _decide(**kwargs: Any):
    defaults: dict[str, Any] = {
        "local_healthy": False,
        "mesh_peers_ok": False,
        "allow_mesh": False,
        "allow_pay": False,
        "buzz_agent": False,
        "surface": TrainingSurface.AGENT,
        "route_class": RouteClass.ALLOW_PAID,
    }
    defaults.update(kwargs)
    return decide_compute(**defaults)


def test_local_healthy_is_free_and_ignores_pay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    monkeypatch.setenv("SEISO_PAY_URL", "https://pay.example.com")
    d = _decide(local_healthy=True, allow_pay=True, pay_url="https://pay.example.com")
    assert d.target is ComputeTarget.LOCAL
    assert d.fee_sats == 0
    assert d.consulted_pay is False
    assert d.quote is None
    assert d.reason == "local_healthy"


def test_local_healthy_ignores_mesh_flags() -> None:
    d = _decide(
        local_healthy=True,
        mesh_peers_ok=True,
        allow_mesh=True,
        buzz_agent=True,
    )
    assert d.target is ComputeTarget.LOCAL
    assert d.fee_sats == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://[::1]:8787",
        "https://127.0.0.1",
        "127.0.0.1:8787",
        "localhost",
        "http://127.0.0.9:9",
    ],
)
def test_loopback_pay_url_refused(url: str) -> None:
    d = _decide(allow_pay=True, pay_url=url)
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.reason == "refuse_localhost_pay"
    assert d.consulted_pay is True
    assert d.fee_sats == 0


def test_pay_when_local_down_and_mesh_unavailable() -> None:
    quote = quote_job("finetune", preset="smoke")
    d = _decide(
        allow_pay=True,
        pay_url="https://pay.example.com",
        quote=quote,
        job_kind="finetune",
    )
    assert d.target is ComputeTarget.PAY
    assert d.consulted_pay is True
    assert d.fee_sats == quote["total_sats"]
    assert d.quote is not None
    assert d.quote["job_type"] == "finetune"
    assert d.job_kind == "finetune"


def test_mesh_preferred_over_pay() -> None:
    d = _decide(
        mesh_peers_ok=True,
        allow_mesh=True,
        buzz_agent=True,
        allow_pay=True,
        pay_url="https://pay.example.com",
        quote=quote_job("slime", preset="smoke"),
    )
    assert d.target is ComputeTarget.MESH
    assert d.fee_sats == 0
    assert d.consulted_pay is False


def test_mesh_requires_flag() -> None:
    d = _decide(mesh_peers_ok=True, allow_mesh=False, buzz_agent=True)
    assert d.target is ComputeTarget.ASK_HUMAN
    assert "mesh" not in d.reason or d.reason in {
        "pay_flag_off",
        "local_then_mesh:mesh_flag_off",
    }


def test_mesh_requires_buzz_agent() -> None:
    d = _decide(mesh_peers_ok=True, allow_mesh=True, buzz_agent=False)
    assert d.target is not ComputeTarget.MESH


def test_mesh_requires_peers() -> None:
    d = _decide(mesh_peers_ok=False, allow_mesh=True, buzz_agent=True)
    assert d.target is not ComputeTarget.MESH


def test_frontend_surface_refuses_mesh() -> None:
    d = _decide(
        mesh_peers_ok=True,
        allow_mesh=True,
        buzz_agent=True,
        surface=TrainingSurface.FRONTEND,
    )
    assert d.target is not ComputeTarget.MESH


def test_frontend_surface_can_still_use_local() -> None:
    d = _decide(local_healthy=True, surface=TrainingSurface.FRONTEND)
    assert d.target is ComputeTarget.LOCAL


def test_never_leave_skips_mesh_and_pay() -> None:
    d = _decide(
        route_class=RouteClass.NEVER_LEAVE,
        mesh_peers_ok=True,
        allow_mesh=True,
        buzz_agent=True,
        allow_pay=True,
        pay_url="https://pay.example.com",
    )
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.reason == "never_leave:local_unhealthy"
    assert d.consulted_pay is False


def test_never_leave_still_uses_local() -> None:
    d = _decide(local_healthy=True, route_class=RouteClass.NEVER_LEAVE)
    assert d.target is ComputeTarget.LOCAL


def test_local_then_mesh_never_pays() -> None:
    d = _decide(
        route_class=RouteClass.LOCAL_THEN_MESH,
        allow_pay=True,
        pay_url="https://pay.example.com",
        quote=quote_job("finetune"),
    )
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.consulted_pay is False
    assert d.reason.startswith("local_then_mesh:")


def test_local_then_mesh_uses_mesh() -> None:
    d = _decide(
        route_class=RouteClass.LOCAL_THEN_MESH,
        mesh_peers_ok=True,
        allow_mesh=True,
        buzz_agent=True,
        allow_pay=True,
        pay_url="https://pay.example.com",
    )
    assert d.target is ComputeTarget.MESH
    assert d.fee_sats == 0


@pytest.mark.parametrize(
    "reason_part,kwargs",
    [
        (
            "frontend_surface",
            {
                "surface": TrainingSurface.FRONTEND,
                "allow_mesh": True,
                "buzz_agent": True,
                "mesh_peers_ok": True,
            },
        ),
        ("mesh_flag_off", {"allow_mesh": False, "buzz_agent": True, "mesh_peers_ok": True}),
        ("buzz_agent_missing", {"allow_mesh": True, "buzz_agent": False, "mesh_peers_ok": True}),
        ("peers_insufficient", {"allow_mesh": True, "buzz_agent": True, "mesh_peers_ok": False}),
    ],
)
def test_local_then_mesh_reason_codes(reason_part: str, kwargs: dict[str, Any]) -> None:
    d = _decide(route_class=RouteClass.LOCAL_THEN_MESH, **kwargs)
    assert d.target is ComputeTarget.ASK_HUMAN
    assert reason_part in d.reason


def test_pay_flag_off_asks_human() -> None:
    d = _decide(allow_pay=False, pay_url="https://pay.example.com")
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.reason == "pay_flag_off"


def test_pay_url_unset_asks_human() -> None:
    d = _decide(allow_pay=True, pay_url="")
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.reason == "pay_url_unset"


def test_pay_url_none_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_URL", "https://pay.example.com")
    d = decide_compute(
        local_healthy=False,
        allow_pay=True,
        allow_mesh=False,
        buzz_agent=False,
        surface=TrainingSurface.AGENT,
        pay_url=None,
    )
    assert d.target is ComputeTarget.PAY


def test_pay_url_none_and_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_URL", raising=False)
    d = decide_compute(
        local_healthy=False,
        allow_pay=True,
        allow_mesh=False,
        buzz_agent=False,
        surface=TrainingSurface.AGENT,
        pay_url=None,
    )
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.reason == "pay_url_unset"


def test_flags_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_PAY", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.delenv("SEISO_PAY_URL", raising=False)
    d = decide_compute(local_healthy=False, mesh_peers_ok=True)
    assert d.target is ComputeTarget.ASK_HUMAN
    assert d.target is not ComputeTarget.PAY
    assert d.target is not ComputeTarget.MESH


def test_job_kind_maps_chat_to_inference() -> None:
    d = _decide(local_healthy=True, job_kind="chat")
    assert d.job_kind == "inference"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("finetune", "finetune"),
        ("slime", "slime"),
        ("distill-rl", "distill_rl"),
        ("nemo_rl", "nemo_rl"),
        ("code", "inference"),
        ("doctor", "doctor"),
    ],
)
def test_job_kind_normalization(raw: str, expected: str) -> None:
    d = _decide(local_healthy=True, job_kind=raw)
    assert d.job_kind == expected


def test_unknown_job_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown task kind"):
        _decide(local_healthy=True, job_kind="mining")


def test_unknown_route_class_raises() -> None:
    with pytest.raises(ValueError, match="unknown route_class"):
        parse_route_class("cloud_only")


def test_parse_route_class_empty_defaults() -> None:
    assert parse_route_class(None) is RouteClass.ALLOW_PAID
    assert parse_route_class("") is RouteClass.ALLOW_PAID
    assert parse_route_class(RouteClass.NEVER_LEAVE) is RouteClass.NEVER_LEAVE


def test_quote_fee_split_agrees_with_marketplace() -> None:
    quote = quote_job("finetune", preset="smoke")
    d = _decide(
        allow_pay=True,
        pay_url="https://operator.example",
        quote=quote,
        job_kind="finetune",
    )
    assert d.fee_sats == quote["total_sats"]
    assert quote["protocol_fee_sats"] == (quote["compute_sats"] * 500 + 9999) // 10_000


def test_bad_quote_total_does_not_raise() -> None:
    d = _decide(
        allow_pay=True,
        pay_url="https://pay.example.com",
        quote={"total_sats": "nope"},
    )
    assert d.target is ComputeTarget.PAY
    assert d.fee_sats == 0


def test_decision_as_dict_jsonable() -> None:
    d = _decide(local_healthy=True, job_kind="export")
    data = d.as_dict()
    assert data["target"] == "local"
    assert data["route_class"] == "allow_paid"
    assert data["fee_sats"] == 0


def test_mesh_env_flag_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    d = decide_compute(
        local_healthy=False,
        mesh_peers_ok=True,
        buzz_agent=True,
        allow_mesh=None,
        allow_pay=False,
        surface=TrainingSurface.AGENT,
    )
    assert d.target is ComputeTarget.MESH


def test_pay_env_flag_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    d = decide_compute(
        local_healthy=False,
        allow_pay=None,
        allow_mesh=False,
        buzz_agent=False,
        pay_url="https://pay.example.com",
        surface=TrainingSurface.AGENT,
    )
    assert d.target is ComputeTarget.PAY
