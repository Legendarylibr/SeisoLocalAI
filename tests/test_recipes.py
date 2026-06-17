"""Tests for visual recipe orchestration."""

from __future__ import annotations

import json

import pytest

from forge.orchestrators.recipes import RecipeOrchestrator


@pytest.mark.asyncio
async def test_recipe_execute_writes_user_scoped_output_and_snapshot(tmp_path):
    user_id = "u1"
    source = tmp_path / "uploads" / user_id / "rows.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"text": "hello"}\n', encoding="utf-8")

    recipe = {
        "name": "smoke",
        "nodes": [
            {"id": "in", "type": "import", "config": {"path": str(source), "format": "jsonl"}},
            {"id": "out", "type": "output", "config": {"source": "in"}},
        ],
        "edges": [{"source": "in", "target": "out"}],
    }
    orchestrator = RecipeOrchestrator(tmp_path)

    result = await orchestrator.execute("job-1", {"user_id": user_id, "recipe": recipe})

    output = tmp_path / "recipes" / user_id / "job-1" / "output.jsonl"
    snapshot = tmp_path / "recipes" / user_id / "job-1" / "recipe_snapshot.json"
    assert result["output_path"] == str(output)
    assert result["recipe_snapshot"] == str(snapshot)
    assert result["row_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {"text": "hello"}
    assert json.loads(snapshot.read_text(encoding="utf-8"))["name"] == "smoke"
