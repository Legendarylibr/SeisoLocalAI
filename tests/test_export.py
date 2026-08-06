"""Tests for export pipeline, Hub precheck, profiles, and API integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seiso.export.formats import ExportFormat, ExportOptions, _select_hub_folder, export_checkpoint
from seiso.export.hub_precheck import (
    HubPrecheckResult,
    assert_hub_precheck_ok,
    precheck_hub_export,
    validate_repo_id,
)
from seiso.export.model_card import (
    HubModelMetadata,
    metadata_from_manifest,
    render_readme,
    write_hub_artifacts,
)
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


def test_slime_manifest_suggests_lora_profile(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "seiso_manifest.json").write_text(json.dumps({"method": "slime", "adapter": "lora"}))

    assert detect_checkpoint_kind(ckpt) == "lora"
    assert suggest_profile(ckpt) == ExportProfile.LORA_ADAPTER


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


def test_export_full_refuses_lora_only_checkpoint(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "lora-run"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text('{"r": 16}')
    (ckpt / "adapter_model.safetensors").write_text("weights")

    out = sandbox / "exports" / "bad-full"
    with pytest.raises(ValueError, match="LoRA-only checkpoint"):
        export_checkpoint(
            ExportOptions(
                checkpoint=ckpt,
                output_dir=out,
                formats=[ExportFormat.FULL],
                sandbox_root=sandbox,
            )
        )


def test_export_full_refuses_config_json_plus_adapter_weights(tmp_path: Path):
    """EXP-02-R: config.json must not mask LoRA-only adapter weights as full."""
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "lora-with-config"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"model_type": "llama"}')
    (ckpt / "adapter_model.safetensors").write_text("weights")
    assert detect_checkpoint_kind(ckpt) == "lora"

    out = sandbox / "exports" / "bad-full-masked"
    with pytest.raises(ValueError, match="LoRA-only checkpoint"):
        export_checkpoint(
            ExportOptions(
                checkpoint=ckpt,
                output_dir=out,
                formats=[ExportFormat.FULL],
                sandbox_root=sandbox,
            )
        )


def test_export_base_refuses_lora_only_checkpoint(tmp_path: Path):
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "lora-run"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text('{"r": 8}')
    (ckpt / "adapter_model.safetensors").write_text("weights")

    out = sandbox / "exports" / "bad-base"
    with pytest.raises(ValueError, match="LoRA-only checkpoint"):
        export_checkpoint(
            ExportOptions(
                checkpoint=ckpt,
                output_dir=out,
                formats=[ExportFormat.BASE],
                sandbox_root=sandbox,
            )
        )


@patch("seiso.export.formats._push_hub")
@patch("seiso.export.formats.merge_lora_checkpoint")
def test_export_skip_hub_precheck_still_rechecks_before_push(mock_merge, mock_push, tmp_path: Path):
    """skip_hub_precheck skips the pre-export gate only; push always re-prechecks."""
    sandbox = tmp_path / "data"
    ckpt = sandbox / "checkpoints" / "run1"
    ckpt.mkdir(parents=True)
    mock_merge.side_effect = lambda c, d, log: d.mkdir(parents=True, exist_ok=True)

    out = sandbox / "exports" / "job1"
    with patch("seiso.export.formats.precheck_hub_export") as mock_precheck:
        mock_precheck.return_value = HubPrecheckResult(
            repo_id="alice/model", ok=True, token_valid=True
        )
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
        # Only the push-time re-precheck (no early pre-export call).
        mock_precheck.assert_called_once()
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
    # Pre-export gate + push-time re-precheck.
    assert mock_precheck.call_count == 2
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


def test_metadata_from_slime_manifest_marks_post_training(tmp_path: Path):
    manifest = tmp_path / "seiso_manifest.json"
    manifest.write_text(
        json.dumps({"method": "slime", "model_id": "test/base"}),
        encoding="utf-8",
    )

    meta = metadata_from_manifest(_meta(), manifest)

    assert meta.finetune_type == "slime"
    assert meta.base_model == "test/base"


# --- API ---


@pytest.mark.asyncio
async def test_export_profiles_api(app, auth_client):
    client, _token, headers, _data_dir = auth_client
    res = await client.get("/api/export/profiles", headers=headers)
    assert res.status_code == 200
    profiles = res.json()
    assert any(p["id"] == "lora_bundle" for p in profiles)


@pytest.mark.asyncio
async def test_export_precheck_api_no_token(app, auth_client, monkeypatch):
    client, _token, headers, _data_dir = auth_client
    monkeypatch.setattr("forge.api.routes.export.resolve_hub_publish_token", lambda *_a, **_k: None)
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

    job = None
    for _ in range(100):
        job = await db.get_export_job(job_id, user["id"])
        if job and job["status"] in ("completed", "failed"):
            break
        await __import__("asyncio").sleep(0.05)

    assert job is not None
    assert job["status"] == "completed"
    outputs = json.loads(job.get("output_paths_json") or "{}")
    assert "lora" in outputs


def test_export_checksum_hashes_weight_files(tmp_path: Path):
    from seiso.research.provenance import directory_checksum_manifest

    (tmp_path / "model.safetensors").write_bytes(b"weights" * 1000)
    (tmp_path / "readme.txt").write_bytes(b"x" * 32)
    manifest = directory_checksum_manifest(
        tmp_path,
        max_files=None,
        max_file_bytes=8,
        always_hash_suffixes=(".safetensors",),
    )
    assert manifest["readme.txt"] == "skipped-large-file"
    assert manifest["model.safetensors"] not in {"skipped-large-file", "error"}
    assert len(manifest["model.safetensors"]) == 64


def test_export_download_requires_exact_key(tmp_path: Path):
    import asyncio

    from fastapi import HTTPException

    from forge.api.routes import export as export_routes
    from forge.config import ForgeSettings
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database

    async def _run() -> None:
        db = Database(
            tmp_path / "forge.db",
            encryption_key=generate_encryption_key(),
            ephemeral=True,
        )
        try:
            user = await db.create_user("hashed", "User", email="export-key@local.dev")
            uid = user["id"]
            out_dir = tmp_path / "exports" / uid / "job1"
            out_dir.mkdir(parents=True)
            gguf = out_dir / "model.gguf"
            gguf.write_bytes(b"GGUF")
            other = out_dir / "adapter.safetensors"
            other.write_bytes(b"ST")
            await db.create_export_job(
                uid,
                {"checkpoint_path": str(tmp_path / "ckpt"), "formats": ["gguf"]},
                job_id="job1",
            )
            await db.update_export_job_status(
                "job1",
                "completed",
                user_id=uid,
                output_paths={"gguf": str(gguf), "safetensors": str(other)},
            )
            settings = ForgeSettings(data_dir=tmp_path)
            with pytest.raises(HTTPException) as empty:
                await export_routes.download_export_output(
                    job_id="job1",
                    user_id=uid,
                    db=db,
                    settings=settings,
                    key="  ",
                )
            assert empty.value.status_code == 400
            with pytest.raises(HTTPException) as fuzzy:
                await export_routes.download_export_output(
                    job_id="job1",
                    user_id=uid,
                    db=db,
                    settings=settings,
                    key="a",
                )
            assert fuzzy.value.status_code == 404
            resp = await export_routes.download_export_output(
                job_id="job1",
                user_id=uid,
                db=db,
                settings=settings,
                key="GGUF",
            )
            assert Path(resp.path) == gguf  # type: ignore[attr-defined]
        finally:
            await db.close()

    asyncio.run(_run())


def test_select_hub_folder_skips_empty_gguf_dirs(tmp_path: Path):
    empty = tmp_path / "q4_k_m"
    empty.mkdir()
    good = tmp_path / "q8_0"
    good.mkdir()
    (good / "model-q8_0.gguf").write_bytes(b"gguf")
    chosen = _select_hub_folder(tmp_path, [ExportFormat.GGUF])
    assert chosen == good


def test_select_hub_folder_prefers_lora_dir(tmp_path: Path):
    from seiso.export.formats import ExportFormat, _select_hub_folder

    out = tmp_path / "export"
    lora = out / "lora"
    lora.mkdir(parents=True)
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _select_hub_folder(out, [ExportFormat.LORA]) == lora
