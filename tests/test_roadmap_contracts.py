"""Lock ROADMAP.md claims to the executable kernel APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.agent.harness import run_harness
from seiso.agent.kernel import decide_compute
from seiso.mesh.flags import mesh_allowed
from seiso.pay.catalog import Listing
from seiso.pay.flags import pay_allowed
from seiso.routing.select import select_route

REPO = Path(__file__).resolve().parents[1]


def _roadmap() -> str:
    return (REPO / "ROADMAP.md").read_text(encoding="utf-8")


def test_roadmap_exists() -> None:
    assert (REPO / "ROADMAP.md").is_file()


@pytest.mark.parametrize(
    "phrase",
    [
        "local agentic operating system",
        "harness",
        "model-aware routing",
        "marketplace",
        "no Seiso token",
        "decide_compute",
        "select_route",
        "run_harness",
        "Listing",
        "Bitcoin",
        "self-hosted",
    ],
)
def test_roadmap_names_contracts(phrase: str) -> None:
    text = _roadmap()
    assert phrase.lower() in text.lower()


def test_no_circular_import_inference_and_routing() -> None:
    import seiso.inference.backends  # noqa: F401
    from seiso.agent.kernel import decide_compute
    from seiso.routing.select import select_route

    assert callable(decide_compute)
    assert callable(select_route)


def test_router_model_id_matches_forge() -> None:
    from forge.services.model_router_client import ROUTER_MODEL_ID as forge_id
    from seiso.routing.select import ROUTER_MODEL_ID as core_id

    assert forge_id == core_id == "__seiso_router__"


def test_kernel_apis_importable() -> None:
    assert callable(decide_compute)
    assert callable(select_route)
    assert callable(run_harness)
    listing = Listing(
        kind="inference",
        label="t",
        operator_id="op",
        compute_sats=1,
    )
    assert listing.kind == "inference"


def test_pay_and_mesh_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_PAY", raising=False)
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    assert pay_allowed() is False
    assert mesh_allowed() is False


def test_readme_links_roadmap() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "ROADMAP.md" in readme


def test_docs_hub_links_roadmap() -> None:
    docs = (REPO / "docs/README.md").read_text(encoding="utf-8")
    assert "../ROADMAP.md" in docs


def test_skill_points_at_decide_compute() -> None:
    skill = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    assert "decide_compute" in skill


def test_cli_docs_cover_route_and_agent_decide() -> None:
    cli = (REPO / "docs/cli.md").read_text(encoding="utf-8")
    assert "seiso route" in cli
    assert "seiso agent decide" in cli
    assert "seiso agent plan" in cli


def test_cli_registers_route_and_agent() -> None:
    from seiso_cli.main import app

    names = {cmd.name or getattr(cmd.callback, "__name__", "") for cmd in app.registered_commands}
    groups = {g.name for g in app.registered_groups}
    assert "route" in names
    assert "agent" in groups


def test_no_seiso_token_in_pay_catalog_module() -> None:
    src = (REPO / "seiso/pay/catalog.py").read_text(encoding="utf-8")
    assert "There is no Seiso token" in src
    assert "SEISO_COIN" not in src
