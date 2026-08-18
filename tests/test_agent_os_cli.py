"""CLI smoke for seiso route / seiso agent decide / seiso agent plan."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from seiso_cli.main import app

runner = CliRunner()


def test_route_cli_picks_local() -> None:
    inventory = json.dumps(
        [
            {
                "model_id": "qwen-7b",
                "backend": "llamacpp",
                "role": "chat",
                "context_tokens": 8192,
                "vram_mb": 5000,
                "downloaded": True,
                "params_b": 7,
            }
        ]
    )
    result = runner.invoke(
        app,
        ["route", "--task", "chat", "--inventory-json", inventory],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_id"] == "qwen-7b"
    assert payload["source"] == "local"


def test_route_cli_rejects_remote_router() -> None:
    result = runner.invoke(
        app,
        [
            "route",
            "--task",
            "chat",
            "--external",
            "--router-url",
            "https://evil.example/v1",
            "--inventory-json",
            "[]",
        ],
    )
    assert result.exit_code == 1
    assert "localhost" in result.output.lower() or "model_router_url" in result.output


def test_agent_decide_local() -> None:
    result = runner.invoke(app, ["agent", "decide", "--job", "finetune", "--local-healthy"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == "local"
    assert payload["fee_sats"] == 0


def test_agent_decide_refuses_localhost_pay(monkeypatch) -> None:
    monkeypatch.setenv("SEISO_ALLOW_PAY", "1")
    result = runner.invoke(
        app,
        [
            "agent",
            "decide",
            "--job",
            "chat",
            "--no-local-healthy",
            "--pay-url",
            "http://127.0.0.1:8787",
            "--route-class",
            "allow_paid",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == "ask_human"
    assert payload["reason"] == "refuse_localhost_pay"


def test_agent_plan_dry_run() -> None:
    result = runner.invoke(app, ["agent", "plan", "--dry-run", "--task", "chat"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["plan"]["steps"][0]["kind"] == "chat"
    assert payload["result"]["status"] == "done"
    assert payload["result"]["results"][0]["output"]["dry_run"] is True
