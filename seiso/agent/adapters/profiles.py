"""Isolated provider snippets so Seiso never rewrites ~/.pi / ~/.hermes / etc."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.agent.adapters.endpoint import ResolvedEndpoint


def isolated_dir(data_dir: Path, harness_id: str) -> Path:
    dest = Path(data_dir) / "agent" / "harnesses" / harness_id
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def write_openai_provider(
    dest: Path,
    *,
    harness_id: str,
    endpoint: ResolvedEndpoint,
) -> Path:
    """Write a generic OpenAI-compat provider JSON (Pi / OMP / Cline / OpenClaw)."""
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "providers": {
            "seiso": {
                "baseUrl": endpoint.url,
                "api": "openai-completions",
                "apiKey": endpoint.api_key or "not-needed",
                "models": [
                    {
                        "id": endpoint.model_id or "default",
                        "name": endpoint.model_id or "default",
                    }
                ],
            }
        }
    }
    path = dest / "models.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _ = harness_id
    return path


def write_hermes_overlay(dest: Path, *, endpoint: ResolvedEndpoint) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    lines = [
        "model:",
        f"  default: {endpoint.model_id or 'default'}",
        "  provider: custom",
        f"  base_url: {endpoint.url}",
        "custom_providers:",
        "  - name: Seiso",
        f"    base_url: {endpoint.url}",
        f"    model: {endpoint.model_id or 'default'}",
        "",
    ]
    path = dest / "config.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_profile(dest: Path, harness_id: str, endpoint: ResolvedEndpoint) -> Path:
    if harness_id == "hermes":
        return write_hermes_overlay(dest, endpoint=endpoint)
    return write_openai_provider(dest, harness_id=harness_id, endpoint=endpoint)
