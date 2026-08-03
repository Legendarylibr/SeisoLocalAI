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
    assert (
        not undocumented
    ), f"quickstart table lists unknown TrainConfig fields: {undocumented}"


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
                broken.append(
                    f"{md_path.relative_to(REPO_ROOT)} -> {target} (outside repo)"
                )
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
    """Experiment CLI stub points researchers to Adaptive-RL-Quantization."""
    cli_doc = _read("docs/cli.md")
    readme = _read("README.md")
    assert "seiso experiment" in cli_doc
    assert "Adaptive-RL-Quantization" in cli_doc
    assert "seiso experiment" in readme
    from seiso_cli.main import app

    group_names = {g.name for g in app.registered_groups}
    assert "experiment" in group_names


def test_cli_docs_cover_provenance_nostr():
    """Nostr provenance CLI must be documented and registered."""
    cli_doc = _read("docs/cli.md")
    assert "seiso provenance" in cli_doc
    assert "dataset-prove" in cli_doc
    assert (REPO_ROOT / "docs/provenance-nostr.md").is_file()
    prov = _read("docs/provenance-nostr.md")
    assert "seiso provenance" in prov
    assert "dataset_merkle_root" in prov
    from seiso_cli.main import app

    group_names = {g.name for g in app.registered_groups}
    assert "provenance" in group_names


def test_cli_docs_cover_compress_run():
    """docs/cli.md must document the real `seiso compress run` subcommand."""
    cli_doc = _read("docs/cli.md")
    compression = _read("docs/compression.md")
    assert "seiso compress run" in cli_doc
    assert "seiso compress run --preset smoke" in cli_doc
    assert "seiso compress manifest-verify" in cli_doc
    assert "seiso compress run" in compression
    assert (REPO_ROOT / "configs/example_compress.json").is_file()
    # Wrong historical form: bare `seiso compress --config` without `run`.
    assert "seiso compress --config" not in cli_doc


def test_cli_docs_cover_slime_training():
    """Slime post-training must be documented with a real example config + CLI path."""
    cli_doc = _read("docs/cli.md")
    quickstart = _read("docs/training/quickstart.md")
    example = "configs/example_training_slime.yaml"
    assert example in cli_doc
    assert (REPO_ROOT / example).is_file()
    assert "method: slime" in cli_doc or "method: slime" in quickstart
    assert "rollout_backend" in quickstart
    # Dedicated CLI exists; docs may say `seiso slime` and/or `seiso train -c …`.
    assert "seiso slime" in cli_doc or "seiso train --config configs/example_training_slime.yaml" in cli_doc
    from seiso_cli.main import app

    registered = {cmd.name for cmd in app.registered_commands}
    assert "slime" in registered


def test_cli_docs_cover_nemo_rl_training():
    """NeMo RL must be documented with example config + CLI path."""
    cli_doc = _read("docs/cli.md")
    quickstart = _read("docs/training/quickstart.md")
    example = "configs/example_training_nemo_rl.yaml"
    assert example in cli_doc
    assert (REPO_ROOT / example).is_file()
    assert (REPO_ROOT / "configs/smoke_nemo_rl.yaml").is_file()
    assert "method: nemo_rl" in cli_doc or "method: nemo_rl" in quickstart
    assert "SEISO_NEMO_RL_ROOT" in cli_doc or "SEISO_NEMO_RL_ROOT" in quickstart
    assert "seiso nemo-rl" in cli_doc
    from seiso_cli.main import app

    registered = {cmd.name for cmd in app.registered_commands}
    assert "nemo-rl" in registered


def test_smoke_configs_exist_and_are_referenced():
    """F6-04: smoke presets must exist and be discoverable from AGENTS/docs."""
    smokes = [
        "configs/smoke_train_cpu.yaml",
        "configs/smoke_train_gpu.yaml",
        "configs/smoke_train_moe_cpu.yaml",
        "configs/smoke_slime_cpu.yaml",
        "configs/smoke_nemo_rl.yaml",
        "configs/distill_rl_smoke.json",
    ]
    agents = _read("AGENTS.md")
    docs_ci = _read("docs/CI_LOCAL.md")
    for rel in smokes:
        path = REPO_ROOT / rel
        assert path.is_file(), f"missing smoke config {rel}"
        name = path.name
        assert (
            name in agents or name in docs_ci or rel in agents or rel in docs_ci
        ), f"{rel} is unreferenced in AGENTS.md / docs/CI_LOCAL.md"


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
