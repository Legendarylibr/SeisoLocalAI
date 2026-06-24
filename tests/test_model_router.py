"""Tests for model router classifier, policy, and API."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute
from seiso.model_router.classifier import classify_messages, detect_domain
from seiso.model_router.config import RouterSettings
from seiso.model_router.policy import SpecialistRouteBandit, pick_route_with_hints


def test_detect_domain_code():
    domain, conf, _ = detect_domain("def merge(a, b): return {**a, **b}")
    assert domain == "code"
    assert conf > 0.4


def test_detect_domain_math():
    domain, _, _ = detect_domain("solve the equation 2x + 5 = 15 step by step")
    assert domain in ("math", "reasoning", "qa")


def test_specialist_catalog_roundtrip(tmp_path: Path):
    catalog = SpecialistCatalog(
        routes=[
            SpecialistRoute(
                route_id="general",
                llamaswap_model="seiso-general",
                backend_url="http://localhost:8001",
                backend_type="vllm",
                vram_hot=True,
            )
        ]
    )
    path = tmp_path / "specialists.json"
    catalog.save_json(path)
    loaded = SpecialistCatalog.from_json(path)
    assert len(loaded) == 1
    assert loaded.by_id("general").llamaswap_model == "seiso-general"


def test_specialist_catalog_vllm_url_alias(tmp_path: Path):
    path = tmp_path / "specialists.json"
    path.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "route_id": "general",
                        "llamaswap_model": "seiso-general",
                        "vllm_url": "http://localhost:8080",
                        "backend_type": "llamacpp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = SpecialistCatalog.from_json(path)
    route = loaded.by_id("general")
    assert route.backend_url == "http://localhost:8080"
    assert route.is_llamacpp


def test_bandit_select_and_update():
    catalog = SpecialistCatalog(
        routes=[
            SpecialistRoute("general", "g", "http://a:8000", domain_hints=("general",)),
            SpecialistRoute("code", "c", "http://b:8000", domain_hints=("code",)),
        ]
    )
    bandit = SpecialistRouteBandit(catalog=catalog, seed=1)
    messages = [{"role": "user", "content": "import os; print(os.getcwd())"}]
    classification, context = classify_messages(
        messages, hardware="gpu", known_domains=catalog.known_domains()
    )
    selection = pick_route_with_hints(bandit, context, classification.domain)
    bandit.update(selection.route.route_id, context, 0.8)
    assert bandit._total_pulls == 1


def test_router_health_endpoint(tmp_path: Path):
    specialists = tmp_path / "specialists.json"
    specialists.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "route_id": "general",
                        "llamaswap_model": "seiso-general",
                        "vllm_url": "http://127.0.0.1:59999",
                        "vram_hot": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    settings = RouterSettings(
        specialists_path=specialists,
        policy_state_path=policy_path,
        llamaswap_url="",
        mode="local",
        enable_rl_policy=False,
    )
    from seiso.model_router.main import build_app

    client = TestClient(build_app(settings))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "seiso-general"
