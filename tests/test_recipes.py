"""Tests for visual recipe orchestration."""

from __future__ import annotations

import json

import pytest

from forge.orchestrators.recipes import RecipeOrchestrator, _recipe_template_fields


def test_recipe_template_fields_messages_format():
    row = {
        "messages": [
            {"role": "user", "content": "What is Seiso?"},
            {"role": "assistant", "content": "A local AI workspace."},
        ]
    }
    fields = _recipe_template_fields(row)
    assert "What is Seiso?" in fields["text"]
    assert "local AI workspace" in fields["text"]


def test_resolve_import_suffix_prefers_file_extension():
    path = __import__("pathlib").Path("/tmp/sample.jsonl")
    assert RecipeOrchestrator._resolve_import_suffix(path, "txt") == "jsonl"
    assert RecipeOrchestrator._resolve_import_suffix(path, "auto") == "jsonl"


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


@pytest.mark.asyncio
async def test_recipe_default_pipeline_parses_messages_jsonl(tmp_path):
    user_id = "u1"
    source = tmp_path / "uploads" / user_id / "sample.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "World"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recipe = {
        "name": "messages",
        "nodes": [
            {"id": "import", "type": "import", "config": {"path": str(source), "format": "txt"}},
            {
                "id": "transform",
                "type": "transform",
                "config": {"source": "import", "template": "Instruction: {text}\nOutput:"},
            },
            {"id": "filter", "type": "filter", "config": {"source": "transform", "min_length": 5}},
            {"id": "out", "type": "output", "config": {"source": "filter"}},
        ],
        "edges": [
            {"source": "import", "target": "transform"},
            {"source": "transform", "target": "filter"},
            {"source": "filter", "target": "out"},
        ],
    }
    orchestrator = RecipeOrchestrator(tmp_path)
    result = await orchestrator.execute("job-2", {"user_id": user_id, "recipe": recipe})

    output = tmp_path / "recipes" / user_id / "job-2" / "output.jsonl"
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert result["row_count"] == 1
    assert "Hello" in row["text"]
    assert "World" in row["text"]
