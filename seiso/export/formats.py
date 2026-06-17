"""Checkpoint export — merged, LoRA, full fine-tune, GGUF, Hub push."""

from __future__ import annotations

import enum
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi

from seiso.export.hub_precheck import assert_hub_precheck_ok, precheck_hub_export
from seiso.export.model_card import HubModelMetadata, metadata_from_manifest, write_hub_artifacts
from seiso.security import assert_within

logger = logging.getLogger(__name__)


class ExportFormat(enum.StrEnum):
    MERGED = "merged"
    BASE = "base"
    FULL = "full"
    LORA = "lora"
    GGUF = "gguf"


@dataclass
class ExportOptions:
    checkpoint: Path
    output_dir: Path
    formats: list[ExportFormat] = field(default_factory=lambda: [ExportFormat.MERGED])
    gguf_quantizations: list[str] = field(default_factory=lambda: ["q4_k_m", "q8_0"])
    hub_repo: str | None = None
    hub_token: str | None = None
    hub_metadata: HubModelMetadata | None = None
    sandbox_root: Path | None = None
    skip_hub_precheck: bool = False


def export_checkpoint(
    options: ExportOptions,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Export checkpoint to requested formats. Returns map format -> path."""
    sandbox = options.sandbox_root or options.output_dir.parent
    ckpt = assert_within(sandbox, options.checkpoint)
    out_root = assert_within(sandbox, options.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    if options.hub_repo and options.hub_token and not options.skip_hub_precheck:
        meta = _enriched_metadata(options, ckpt)
        log(f"Running Hub precheck for {options.hub_repo} before export...")
        precheck = precheck_hub_export(
            repo_id=options.hub_repo,
            token=options.hub_token,
            metadata=meta,
            formats=[f.value for f in options.formats],
            on_log=log,
        )
        assert_hub_precheck_ok(precheck)
        log("Hub precheck passed — starting export")

    results: dict[str, Path] = {}
    from seiso.export.profiles import detect_checkpoint_kind

    kind = detect_checkpoint_kind(ckpt)
    log(f"Checkpoint kind: {kind}")

    for fmt in options.formats:
        if fmt == ExportFormat.LORA:
            dest = out_root / "lora"
            if ckpt.exists():
                shutil.copytree(ckpt, dest, dirs_exist_ok=True)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest
            log(f"LoRA adapter exported to {dest}")

        elif fmt == ExportFormat.MERGED:
            dest = out_root / "merged"
            dest.mkdir(parents=True, exist_ok=True)
            _merge_lora(ckpt, dest, log)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest

        elif fmt in (ExportFormat.BASE, ExportFormat.FULL):
            dest = out_root / ("full" if fmt == ExportFormat.FULL else "base")
            if ckpt.exists():
                shutil.copytree(ckpt, dest, dirs_exist_ok=True)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest
            log(f"{'Full fine-tune' if fmt == ExportFormat.FULL else 'Base'} checkpoint exported to {dest}")

        elif fmt == ExportFormat.GGUF:
            from seiso.export.gguf import export_gguf_from_checkpoint

            merged = results.get(ExportFormat.MERGED.value) or results.get(ExportFormat.FULL.value)
            gguf_paths = export_gguf_from_checkpoint(
                ckpt,
                out_root,
                options.gguf_quantizations,
                merged_dir=merged if merged and merged.exists() else None,
                on_log=log,
            )
            results.update(gguf_paths)

    if options.hub_repo and options.hub_token:
        push_folder = _select_hub_folder(out_root, options.formats)
        meta = _enriched_metadata(options, ckpt)
        meta.export_formats = [f.value for f in options.formats]
        _push_hub(
            options.hub_repo,
            options.hub_token,
            push_folder,
            log,
            metadata=meta,
            quantizations=options.gguf_quantizations,
        )
        log(f"Pushed to Hugging Face: {options.hub_repo}")

    return results


def publish_folder_to_hub(
    folder: Path,
    *,
    repo_id: str,
    token: str,
    metadata: HubModelMetadata,
    quantizations: list[str] | None = None,
    on_log: Callable[[str], None] | None = None,
    skip_precheck: bool = False,
) -> None:
    """Publish an existing export folder to Hugging Face with model card."""

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    meta = metadata
    if quantizations:
        meta.quantizations = list(quantizations)

    if not skip_precheck:
        precheck = precheck_hub_export(
            repo_id=repo_id,
            token=token,
            metadata=meta,
            formats=meta.export_formats or None,
            on_log=log,
        )
        assert_hub_precheck_ok(precheck)

    _push_hub(repo_id, token, folder, log, metadata=meta, quantizations=quantizations or meta.quantizations)


def _enriched_metadata(options: ExportOptions, ckpt: Path) -> HubModelMetadata:
    meta = options.hub_metadata
    if meta is None:
        raise ValueError("hub_metadata is required when pushing to Hugging Face")
    manifest = ckpt / "seiso_manifest.json"
    return metadata_from_manifest(meta, manifest)


def _write_export_sidecar(dest: Path, ckpt: Path, fmt: ExportFormat, kind: str) -> None:
    """Write seiso_export_metadata.json alongside exported artifacts."""
    if not dest.is_dir():
        return
    payload = {
        "format": fmt.value,
        "checkpoint_kind": kind,
        "source_checkpoint": str(ckpt),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest = ckpt / "seiso_manifest.json"
    if manifest.is_file():
        try:
            payload["training_manifest"] = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    (dest / "seiso_export_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _select_hub_folder(out_root: Path, formats: list[ExportFormat]) -> Path:
    """Prefer merged/full weights; fall back to GGUF quant dir or export root."""
    for key in (ExportFormat.MERGED, ExportFormat.FULL, ExportFormat.BASE):
        if key in formats:
            candidate = out_root / ("merged" if key == ExportFormat.MERGED else key.value)
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
    if ExportFormat.GGUF in formats:
        for child in sorted(out_root.iterdir()):
            if child.is_dir() and (child.name in {"q4_k_m", "q8_0", "f16"} or child.name.startswith("gguf-")):
                return child
    return out_root


def _merge_lora(checkpoint: Path, dest: Path, log: Callable[[str], None]) -> None:
    """Merge LoRA weights into base model when PEFT adapter present."""
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.exists():
        log("Merging LoRA adapter into base weights...")
        try:
            import json

            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cfg = json.loads(adapter_config.read_text())
            base_id = cfg.get("base_model_name_or_path", "")
            model = AutoModelForCausalLM.from_pretrained(base_id, device_map="cpu")
            model = PeftModel.from_pretrained(model, str(checkpoint))
            merged = model.merge_and_unload()
            merged.save_pretrained(str(dest))
            tok = AutoTokenizer.from_pretrained(str(checkpoint))
            tok.save_pretrained(str(dest))
            log(f"Merged model saved to {dest}")
        except Exception as exc:
            log(f"Merge failed, copying checkpoint: {exc}")
            shutil.copytree(checkpoint, dest, dirs_exist_ok=True)
    else:
        shutil.copytree(checkpoint, dest, dirs_exist_ok=True)
        log(f"Copied checkpoint to {dest}")


def _push_hub(
    repo: str,
    token: str,
    folder: Path,
    log: Callable[[str], None],
    *,
    metadata: HubModelMetadata | None = None,
    quantizations: list[str] | None = None,
) -> None:
    api = HfApi(token=token)
    if metadata:
        meta = metadata
        if quantizations:
            meta.quantizations = list(quantizations)
        write_hub_artifacts(folder, meta)
        log(f"Wrote Hugging Face model card for {meta.repo_id}")

    try:
        api.repo_info(repo, repo_type="model")
    except Exception:
        api.create_repo(repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo,
        repo_type="model",
        commit_message="Upload model from Seiso Forge",
    )
    if metadata:
        card_data = metadata.to_card_dict()
        try:
            api.update_repo_settings(
                repo_id=repo,
                repo_type="model",
                **{k: v for k, v in card_data.items() if k in {"license", "tags"}},
            )
        except Exception as exc:
            log(f"Note: could not update repo settings: {exc}")
