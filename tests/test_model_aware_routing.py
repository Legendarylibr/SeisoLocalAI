"""Unit tests for model-aware routing (local first, localhost external router)."""

from __future__ import annotations

import pytest

from seiso.agent.policy import RouteClass
from seiso.agent.tasks import TaskKind
from seiso.routing.external import is_loopback_host, is_loopback_url, validate_router_url
from seiso.routing.select import (
    ROUTER_BACKEND,
    ROUTER_MODEL_ID,
    select_route,
    select_route_from_args,
)
from seiso.routing.table import roles_for_task
from seiso.routing.types import Candidate, NoRouteError, RouteRequest


def _c(
    model_id: str,
    *,
    role: str = "chat",
    ctx: int = 8192,
    vram: int = 4000,
    downloaded: bool = True,
    backend: str = "llamacpp",
    params_b: float | None = None,
    quant: str | None = None,
) -> Candidate:
    return Candidate(
        model_id=model_id,
        backend=backend,
        role=role,
        context_tokens=ctx,
        vram_mb=vram,
        downloaded=downloaded,
        params_b=params_b,
        quant=quant,
    )


def _req(
    inventory: list[Candidate],
    *,
    task: TaskKind = TaskKind.CHAT,
    ctx: int = 4096,
    vram: int = 8192,
    external: bool = False,
    url: str | None = "http://127.0.0.1:8780",
    route_class: RouteClass = RouteClass.ALLOW_PAID,
) -> RouteRequest:
    return RouteRequest(
        task=task,
        required_context=ctx,
        available_vram_mb=vram,
        inventory=tuple(inventory),
        external_router_enabled=external,
        external_router_url=url,
        route_class=route_class,
    )


def test_picks_downloaded_local_that_fits() -> None:
    d = select_route(_req([_c("qwen-7b", params_b=7)]))
    assert d.source == "local"
    assert d.model_id == "qwen-7b"
    assert d.backend == "llamacpp"
    assert d.stepped_down is False
    assert "fits" in d.reason


def test_steps_down_when_70b_does_not_fit() -> None:
    inv = [
        _c("llama-70b", params_b=70, vram=48000),
        _c("llama-7b-q4", params_b=7, vram=5000, quant="q4"),
    ]
    d = select_route(_req(inv, vram=8000))
    assert d.model_id == "llama-7b-q4"
    assert d.stepped_down is True
    assert d.reason.startswith("step_down:")


def test_prefers_larger_model_when_both_fit() -> None:
    inv = [
        _c("small", params_b=3, vram=3000),
        _c("mid", params_b=14, vram=7000),
    ]
    d = select_route(_req(inv, vram=16000))
    assert d.model_id == "mid"
    assert d.stepped_down is False


def test_missing_inventory() -> None:
    with pytest.raises(NoRouteError, match="missing_inventory"):
        select_route(_req([]))


def test_not_downloaded_skipped() -> None:
    with pytest.raises(NoRouteError, match="not_downloaded"):
        select_route(_req([_c("missing", downloaded=False)]))


def test_context_miss() -> None:
    with pytest.raises(NoRouteError, match="context_miss"):
        select_route(_req([_c("short", ctx=2048)], ctx=8192))


def test_vram_miss() -> None:
    with pytest.raises(NoRouteError, match="vram_miss"):
        select_route(_req([_c("fat", vram=20000, params_b=70)], vram=4000))


def test_code_prefers_code_role() -> None:
    inv = [
        _c("chat-only", role="chat", params_b=32),
        _c("coder", role="code", params_b=7),
    ]
    d = select_route(_req(inv, task=TaskKind.CODE))
    assert d.model_id == "coder"


def test_code_falls_back_to_chat() -> None:
    d = select_route(_req([_c("general", role="chat")], task=TaskKind.CODE))
    assert d.model_id == "general"


def test_embed_requires_embed_role() -> None:
    with pytest.raises(NoRouteError, match="no_role_match"):
        select_route(_req([_c("chat")], task=TaskKind.EMBED))


def test_draft_and_target_only_from_inventory() -> None:
    inv = [
        _c("draft-1b", role="draft", params_b=1),
        _c("target-8b", role="target", params_b=8),
        _c("chat-7b", role="chat", params_b=7),
    ]
    draft = select_route(_req(inv, task=TaskKind.DRAFT))
    target = select_route(_req(inv, task=TaskKind.TARGET))
    assert draft.model_id == "draft-1b"
    assert target.model_id == "target-8b"


def test_never_leave_excludes_external_even_if_enabled() -> None:
    with pytest.raises(NoRouteError, match="never_leave"):
        select_route(
            _req(
                [],
                external=True,
                url="http://127.0.0.1:8780",
                route_class=RouteClass.NEVER_LEAVE,
            )
        )


def test_external_used_when_no_local_fit() -> None:
    d = select_route(_req([], external=True, url="http://127.0.0.1:8780"))
    assert d.source == "external"
    assert d.backend == ROUTER_BACKEND
    assert d.model_id == ROUTER_MODEL_ID
    assert "no_local_fit" in d.reason


def test_local_wins_over_external() -> None:
    d = select_route(_req([_c("local-7b", params_b=7)], external=True))
    assert d.source == "local"
    assert d.model_id == "local-7b"


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/router",
        "https://8.8.8.8/v1",
        "http://169.254.169.254/",
        "http://router.internal",
        "ftp://127.0.0.1:8780",
        "not-a-url",
        "",
    ],
)
def test_bad_router_url_raises_when_enabled(url: str) -> None:
    with pytest.raises(ValueError, match="model_router_url"):
        select_route(_req([_c("local")], external=True, url=url))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8780",
        "http://localhost:8780",
        "http://[::1]:8780",
        "https://127.0.0.1",
    ],
)
def test_loopback_router_urls_accepted(url: str) -> None:
    d = select_route(_req([], external=True, url=url))
    assert d.source == "external"


def test_ipv6_localhost_validate() -> None:
    assert validate_router_url("http://[::1]:8780") == "http://[::1]:8780"
    assert is_loopback_host("::1")
    assert is_loopback_url("http://[::1]/")


@pytest.mark.parametrize(
    "url,ok",
    [
        ("127.0.0.1:8787", True),
        ("localhost", True),
        ("localhost:8787", True),
        ("[::1]:8780", True),
        ("http://127.0.0.1:8787", True),
        ("http://127.0.0.9:9", True),
        ("example.com:8787", False),
        ("192.168.1.1:80", False),
    ],
)
def test_is_loopback_url_scheme_optional(url: str, ok: bool) -> None:
    assert is_loopback_url(url) is ok


@pytest.mark.parametrize(
    "host,ok",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("127.0.0.99", True),
        ("192.168.1.1", False),
        ("example.com", False),
        ("", False),
        (None, False),
    ],
)
def test_is_loopback_host(host: str | None, ok: bool) -> None:
    assert is_loopback_host(host) is ok


def test_validate_router_url_strips_trailing_slash() -> None:
    assert validate_router_url("http://127.0.0.1:8780/") == "http://127.0.0.1:8780"


def test_roles_for_known_tasks() -> None:
    assert roles_for_task(TaskKind.CHAT) == ("chat",)
    assert "code" in roles_for_task(TaskKind.CODE)
    assert roles_for_task(TaskKind.DRAFT) == ("draft",)


def test_select_route_from_args_wrapper() -> None:
    d = select_route_from_args(
        task="chat",
        required_context=2048,
        available_vram_mb=8000,
        inventory=[_c("a", params_b=3)],
    )
    assert d.model_id == "a"


def test_request_and_decision_as_dict() -> None:
    req = _req([_c("m")])
    data = req.as_dict()
    assert data["task"] == "chat"
    assert data["inventory"][0]["model_id"] == "m"
    d = select_route(req)
    dumped = d.as_dict()
    assert dumped["source"] == "local"
    assert dumped["route_class"] == "allow_paid"


def test_local_then_mesh_may_use_external() -> None:
    d = select_route(
        _req([], external=True, url="http://127.0.0.1:1", route_class=RouteClass.LOCAL_THEN_MESH)
    )
    assert d.source == "external"


def test_quant_preserved_on_step_down() -> None:
    inv = [
        _c("big-fp16", params_b=70, vram=40000, quant="fp16"),
        _c("small-q4", params_b=8, vram=4500, quant="q4_k_m"),
    ]
    d = select_route(_req(inv, vram=6000))
    assert d.model_id == "small-q4"
    assert d.stepped_down is True
