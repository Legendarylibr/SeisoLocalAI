"""Verify user-facing docs match the codebase (paths, config, API routes)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "rel_path",
    [
        "docs/training/quickstart.md",
        "docs/getting-started.md",
        "docs/forge.md",
        "docs/cli.md",
        "docs/README.md",
        "configs/example_lora.yaml",
        "data/sample.jsonl",
        "AGENTS.md",
    ],
)
def test_doc_referenced_files_exist(rel_path: str):
    assert (REPO_ROOT / rel_path).is_file(), f"missing doc asset: {rel_path}"


def test_example_lora_yaml_loads_as_train_config():
    from seiso.training.config import TrainConfig

    data = yaml.safe_load(_read("configs/example_lora.yaml"))
    cfg = TrainConfig.model_validate(data)
    assert cfg.method.value == "lora"
    assert cfg.dataset_format.value == "auto"
    assert cfg.preprocess_dataset is True


def test_quickstart_documents_real_train_config_fields():
    from seiso.training.config import TrainConfig

    quickstart = _read("docs/training/quickstart.md")
    table_fields: set[str] = set()
    in_config_table = False
    for line in quickstart.splitlines():
        if line.strip() == "| Field | Description |":
            in_config_table = True
            continue
        if in_config_table:
            if not line.startswith("|"):
                break
            if line.startswith("| `"):
                match = re.match(r"\|\s*`([a-z_]+)`\s*\|", line)
                if match:
                    table_fields.add(match.group(1))
    model_fields = set(TrainConfig.model_fields.keys())
    undocumented = table_fields - model_fields
    assert table_fields, "expected config field table in quickstart"
    assert not undocumented, f"quickstart table lists unknown TrainConfig fields: {undocumented}"


def test_sample_jsonl_is_valid_chat_dataset():
    rows = []
    for line in _read("data/sample.jsonl").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "messages" in row
        assert any(m.get("role") == "assistant" for m in row["messages"])
        rows.append(row)
    assert len(rows) >= 1


def test_training_api_routes_documented_in_forge_and_quickstart():
    training_py = _read("forge/api/routes/training.py")
    for route in (
        '"/analyze-dataset"',
        '"/validate-dataset"',
        '"/recommendations"',
        '"/jobs"',
    ):
        assert route in training_py

    forge_doc = _read("docs/forge.md")
    quickstart = _read("docs/training/quickstart.md")
    for fragment in (
        "/api/training/analyze-dataset",
        "/api/training/validate-dataset",
        "/api/training/recommendations",
        "/api/training/jobs",
    ):
        assert fragment in forge_doc or fragment in quickstart


def test_quickstart_mentions_dataset_analysis_artifacts():
    quickstart = _read("docs/training/quickstart.md")
    assert "dataset_analysis.json" in quickstart
    assert "analyze-dataset" in quickstart
    assert "GGUF" in quickstart and "safetensors" in quickstart


def test_docs_internal_markdown_links_resolve():
    """Relative links like [text](../install.md) must point at real files under docs/."""
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []
    for md_path in DOCS.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if target.startswith("http") or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (md_path.parent / target).resolve()
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                broken.append(f"{md_path.relative_to(REPO_ROOT)} -> {target} (outside repo)")
                continue
            if not resolved.exists():
                broken.append(f"{md_path.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "broken doc links:\n" + "\n".join(broken)


def test_repository_layout_snippet_matches_core_packages():
    readme = _read("docs/README.md")
    assert "training/" in readme and "dataset analysis" in readme
    quickstart = _read("docs/training/quickstart.md").lower()
    assert "dataset analysis" in quickstart
    assert (REPO_ROOT / "seiso/training/dataset_analysis.py").is_file()
    assert (REPO_ROOT / "seiso/training/practices.py").is_file()


def test_cli_docs_cover_experiment_command():
    """seiso experiment quant-regression must appear in user-facing CLI docs."""
    cli_doc = _read("docs/cli.md")
    readme = _read("README.md")
    assert "seiso experiment" in cli_doc
    assert "quant-regression" in cli_doc
    assert "configs/examples/quant_regression_study.yaml" in cli_doc
    assert "seiso experiment" in readme and "quant-regression" in readme


def test_docs_do_not_reference_nonexistent_rl_quant_extra():
    """pyproject.toml has no [rl-quant] optional extra — docs must not claim one."""
    for rel in ("docs/compression.md", "docs/install.md"):
        text = _read(rel)
        assert ".[rl-quant]" not in text, f"{rel} references nonexistent .[rl-quant] extra"
        assert "`rl-quant`" not in text or "seiso rl-quant" in text or "rl_quant/" in text, (
            f"{rel} may list rl-quant as pip extra"
        )


def test_forge_doc_covers_settings_api():
    forge_doc = _read("docs/forge.md")
    for fragment in (
        "/api/settings",
        "/api/settings/hf-token",
        "/api/settings/hf-status",
    ):
        assert fragment in forge_doc


def test_quickstart_extra_field_documents_fused_lora_qkv():
    quickstart = _read("docs/training/quickstart.md")
    assert "use_fused_lora_qkv" in quickstart
    assert "`extra`" in quickstart
