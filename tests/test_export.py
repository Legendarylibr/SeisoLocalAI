"""Tests for export pipeline, Hub precheck, profiles, and API integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seiso.export.formats import ExportFormat, ExportOptions, export_checkpoint
from seiso.export.hub_precheck import (
    HubPrecheckResult,
    assert_hub_precheck_ok,
    precheck_hub_export,
    validate_repo_id,
)
from seiso.export.model_card import HubModelMetadata, render_readme, write_hub_artifacts
from seiso.export.pipeline import (
    auto_export_after_training,
    prepare_export,
    profile_catalog,
)
from seiso.export.profiles import (
    ExportProfile,
    detect_checkpoint_kind,
    formats_for_profile,
    resolve_formats,
    suggest_profile,
)


def _meta(**kwargs) -> HubModelMetadata:
    defaults = {"username": "alice", "model_name": "my-model", "author": "Alice"}
    defaults.update(kwargs)
    return HubModelMetadata(**defaults)


# --- Hub precheck ---


def test_validate_repo_id_accepts_valid():
    validate_repo_id("alice/my-model")
    validate_repo_id("org-name/model_v1.0")


@pytest.mark.parametrize(
    "repo_id",
    ["", "no-slash", "alice/", "/model", "alice/../evil", "alice/model/extra"],
)
def test_validate_repo_id_rejects_invalid(repo_id: str):
    with pytest.raises(ValueError):
        validate_repo_id(repo_id)


def test_hub_metadata_model_name_validation():
    with pytest.raises(ValueError, match="Model name"):
        HubModelMetadata(username="a", model_name="bad/name", author="A").validate()


def test_precheck_fails_without_token():
    result = precheck_hub_export(repo_id="alice/model", token="", metadata=_meta())
    assert not result.ok
    assert any("token" in e.lower() for e in result.errors)


def test_precheck_validates_metadata_before_hub():
    result = precheck_hub_export(
        repo_id="alice/model",
        token="hf_fake",
        metadata=_meta(model_name="bad/name"),
    )
    assert not result.ok
    assert not result.metadata_valid


@patch("seiso.export.hub_precheck.HfApi")
def test_precheck_repo_available(mock_api_cls):
    api = MagicMock()
    mock_api_cls.return_value = api
    api.whoami.return_value = {"name": "alice"}
    api.repo_info.side_effect = Exception("404")

    from huggingface_hub.utils import HfHubHTTPError

    response = MagicMock()
    response.status_code = 404
    api.repo_info.side_effect = HfHubHTTPError("missing", response=response)

    meta = _meta(quantizations=["q4_k_m"])
    result = precheck_hub_export(
        repo_id="alice/new-model",
        token="hf_test",
        metadata=meta,
        formats=["merged", "gguf"],
    )
    assert result.token_valid
    assert result.repo_available
    assert result.metadata_valid
    assert result.ok
    assert result.model_card_preview.startswith("---")


@patch("seiso.export.hub_precheck.HfApi")
def test_precheck_repo_taken_by_other_user(mock_api_cls):
    api = MagicMock()
    mock_api_cls.return_value = api
    api.whoami.return_value = {"name": "alice"}
    info = MagicMock()
    info.author = "bob"
    api.repo_info.return_value = info

    result = precheck_hub_export(
        repo_id="bob/existing", token="hf_test", metadata=_meta(username="alice")
    )
    assert not result.ok
    assert not result.repo_available
    assert any("owned by" in e for e in result.errors)


@patch("seiso.export.hub_precheck.HfApi")
def test_precheck_repo_owned_by_user_warns(mock_api_cls):
    api = MagicMock()
    mock_api_cls.return_value = api
    api.whoami.return_value = {"name": "alice"}
    info = MagicMock()
    info.author = "alice"
    api.repo_info.return_value = info

    result = precheck_hub_export(repo_id="alice/existing", token="hf_test", metadata=_meta())
    assert result.ok
    assert result.repo_owned_by_user
    assert any("already exists" in w for w in result.warnings)


def test_assert_hub_precheck_ok_raises():
    result = HubPrecheckResult(repo_id="a/b", ok=False, errors=["bad token"])
    with pytest.raises(ValueError, match="bad token"):
        assert_hub_precheck_ok(result)


# --- Profiles ---


def test_lora_bundle_profile_formats():
    fmts = formats_for_profile(ExportProfile.LORA_BUNDLE)
    assert ExportFormat.LORA in fmts
    assert ExportFormat.MERGED in fmts
    assert ExportFormat.GGUF in fmts


def test_full_finetune_profile():
    fmts = formats_for_profile(ExportProfile.FULL_FINETUNE)
    assert fmts == [ExportFormat.FULL]


def test_detect_checkpoint_kind_lora(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    assert detect_checkpoint_kind(ckpt) == "lora"


def test_detect_checkpoint_kind_full(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    assert detect_checkpoint_kind(ckpt) == "full"


def test_suggest_profile_from_manifest(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "seiso_manifest.json").write_text(json.dumps({"method": "full"}))
    assert suggest_profile(ckpt) == ExportProfile.FULL_BUNDLE


def test_resolve_formats_deduplicates():
    fmts = resolve_formats(formats=["lora", "lora", "merged"])
    assert fmts == [ExportFormat.LORA, ExportFormat.MERGED]


def test_profile_catalog_complete():
    catalog = profile_catalog()
    assert len(catalog) == len(ExportProfile)
    ids = {entry["id"] for entry in catalog}
    assert "lora_bundle" in ids
    assert "full_bundle" in ids


# --- Export formats ---


def test_export_lora_copies_and_writes_sidecar(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text('{"r": 16}')
    (ckpt / "adapter_model.safetensors").write_text("weights")

    out = sandbox / "exports" / "job1"
    results = export_checkpoint(
        ExportOptions(
            checkpoint=ckpt,
            output_dir=out,
            formats=[ExportFormat.LORA],
            sandbox_root=sandbox,
        )
    )
    assert results["lora"].is_dir()
    assert (results["lora"] / "adapter_config.json").is_file()
    assert (results["lora"] / "adapter_model.safetensors").is_file()
    assert (results["lora"] / "seiso_export_metadata.json").is_file()


def test_export_full_finetune(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "full-run"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"model_type": "llama"}')
    (ckpt / "model.safetensors").write_text("full-weights")
    (ckpt / "seiso_manifest.json").write_text(
        json.dumps({"method": "full", "model_id": "meta-llama/Llama-3.2-3B"})
    )

    out = sandbox / "exports" / "full-job"
    results = export_checkpoint(
        ExportOptions(
            checkpoint=ckpt,
            output_dir=out,
            formats=[ExportFormat.FULL],
            sandbox_root=sandbox,
        )
    )
    assert results["full"].is_dir()
    assert (results["full"] / "config.json").is_file()
    sidecar = json.loads((results["full"] / "seiso_export_metadata.json").read_text())
    assert sidecar["format"] == "full"
    assert sidecar["checkpoint_kind"] == "full"


@patch("seiso.export.formats._push_hub")
@patch("seiso.export.formats.merge_lora_checkpoint")
def test_export_skips_hub_precheck_when_disabled(mock_merge, mock_push, tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    mock_merge.side_effect = lambda c, d, log: d.mkdir(parents=True, exist_ok=True)

    out = sandbox / "exports" / "job1"
    with patch("seiso.export.formats.precheck_hub_export") as mock_precheck:
        export_checkpoint(
            ExportOptions(
                checkpoint=ckpt,
                output_dir=out,
                formats=[ExportFormat.MERGED],
                hub_repo="alice/model",
                hub_token="hf_test",
                hub_metadata=_meta(),
                sandbox_root=sandbox,
                skip_hub_precheck=True,
            )
        )
        mock_precheck.assert_not_called()
    mock_push.assert_called_once()


@patch("seiso.export.formats._push_hub")
@patch("seiso.export.formats.precheck_hub_export")
@patch("seiso.export.formats.merge_lora_checkpoint")
def test_export_runs_hub_precheck_first(mock_merge, mock_precheck, mock_push, tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    mock_merge.side_effect = lambda c, d, log: d.mkdir(parents=True, exist_ok=True)
    mock_precheck.return_value = HubPrecheckResult(repo_id="alice/model", ok=True, token_valid=True)

    out = sandbox / "exports" / "job1"
    export_checkpoint(
        ExportOptions(
            checkpoint=ckpt,
            output_dir=out,
            formats=[ExportFormat.MERGED],
            hub_repo="alice/model",
            hub_token="hf_test",
            hub_metadata=_meta(),
            sandbox_root=sandbox,
        )
    )
    mock_precheck.assert_called_once()
    mock_push.assert_called_once()


@patch("seiso.export.formats.HfApi")
@patch("seiso.models.hf_env.configure_hf_hub_cache")
def test_push_hub_uses_large_folder_for_big_uploads(mock_configure, mock_api_cls, tmp_path: Path):
    from seiso.export.formats import _push_hub

    folder = tmp_path / "gguf"
    folder.mkdir()
    big = folder / "model.gguf"
    big.write_bytes(b"x" * (101 * 1024 * 1024))

    api = MagicMock()
    mock_api_cls.return_value = api
    logs: list[str] = []

    _push_hub("alice/model", "hf_test", folder, logs.append, data_dir=tmp_path)

    mock_configure.assert_called_once_with(tmp_path)
    api.upload_large_folder.assert_called_once()
    api.upload_folder.assert_not_called()
    assert any("resumable" in line.lower() for line in logs)


# --- Pipeline ---


def test_prepare_export_plan(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")

    plan = prepare_export(
        checkpoint=ckpt,
        output_dir=sandbox / "exports" / "out",
        profile="lora_adapter",
    )
    assert plan.checkpoint_kind == "lora"
    assert plan.formats == [ExportFormat.LORA]
    assert plan.profile == "lora_adapter"


def test_auto_export_after_training_uses_suggested_profile(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text('{"r": 8}')
    (ckpt / "adapter_model.safetensors").write_text("x")

    out = sandbox / "exports" / "auto"
    results = auto_export_after_training(ckpt, out, {}, sandbox_root=sandbox)
    assert "lora" in results


# --- Model card ---


def test_model_card_includes_finetune_type():
    meta = _meta(finetune_type="qlora", base_model="meta-llama/Llama-3.2-3B")
    readme = render_readme(meta)
    assert "qlora" in readme
    assert "Fine-tune type" in readme


def test_write_hub_artifacts_includes_finetune_type(tmp_path: Path):
    meta = _meta(finetune_type="full", export_formats=["full", "gguf"])
    write_hub_artifacts(tmp_path, meta)
    payload = json.loads((tmp_path / "seiso_model_metadata.json").read_text())
    assert payload["finetune_type"] == "full"
    assert payload["export_formats"] == ["full", "gguf"]


# --- API ---


@pytest.mark.asyncio
async def test_export_profiles_api(app, auth_client):
    client, _token, headers, _data_dir = auth_client
    res = await client.get("/api/export/profiles", headers=headers)
    assert res.status_code == 200
    profiles = res.json()
    assert any(p["id"] == "lora_bundle" for p in profiles)


@pytest.mark.asyncio
async def test_export_precheck_api_no_token(app, auth_client):
    client, _token, headers, _data_dir = auth_client
    res = await client.post(
        "/api/export/precheck",
        headers=headers,
        json={
            "hub": {"username": "alice", "model_name": "test-model", "author": "Alice"},
            "formats": ["merged"],
        },
    )
    assert res.status_code == 400


@pytest.mark.asyncio
@patch("seiso.export.hub_precheck.HfApi")
async def test_export_precheck_api_ok(mock_api_cls, app, auth_client):
    client, _token, headers, _data_dir = auth_client

    api = MagicMock()
    mock_api_cls.return_value = api
    api.whoami.return_value = {"name": "alice"}
    from huggingface_hub.utils import HfHubHTTPError

    response = MagicMock()
    response.status_code = 404
    api.repo_info.side_effect = HfHubHTTPError("missing", response=response)

    res = await client.post(
        "/api/export/precheck",
        headers=headers,
        json={
            "hub": {
                "username": "alice",
                "model_name": "test-model",
                "author": "Alice",
                "hf_token": "hf_test_token",
            },
            "formats": ["merged"],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["repo_id"] == "alice/test-model"
    assert body["model_card_preview"].startswith("---")


@pytest.mark.asyncio
@patch("seiso.export.hub_precheck.precheck_hub_export")
async def test_export_job_precheck_before_start(mock_precheck, app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    ckpt = data_dir / "checkpoints" / user["id"] / "run1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")

    mock_precheck.return_value = HubPrecheckResult(
        repo_id="alice/model",
        ok=False,
        errors=["Repo taken"],
    )

    res = await client.post(
        "/api/export/jobs",
        headers=headers,
        json={
            "checkpoint": str(ckpt),
            "formats": ["merged"],
            "hub": {
                "username": "alice",
                "model_name": "model",
                "author": "Alice",
                "hf_token": "hf_x",
            },
        },
    )
    assert res.status_code == 400
    assert "Repo taken" in res.json()["detail"]


@pytest.mark.asyncio
async def test_export_with_profile_lora(tmp_path, app, auth_client):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    ckpt = data_dir / "checkpoints" / user["id"] / "profile-run"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text('{"r": 16}')
    (ckpt / "adapter_model.bin").write_text("fake")

    res = await client.post(
        "/api/export/jobs",
        headers=headers,
        json={"checkpoint": str(ckpt), "profile": "lora_adapter"},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    from forge.api.deps import get_export_orchestrator

    orch = get_export_orchestrator()
    for _ in range(100):
        job_rec = orch.get_job(job_id)
        if job_rec and job_rec.status.value in ("completed", "failed"):
            break
        await __import__("asyncio").sleep(0.05)

    job = await db.get_export_job(job_id, user["id"])
    assert job["status"] == "completed"
    outputs = json.loads(job.get("output_paths_json") or "{}")
    assert "lora" in outputs
