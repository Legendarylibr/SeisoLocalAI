"""Checkpoint export — merged, LoRA, full fine-tune, GGUF, Hub push."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from seiso.compat import StrEnum
from seiso.export.hub_precheck import assert_hub_precheck_ok, precheck_hub_export
from seiso.export.model_card import (
    HubModelMetadata,
    metadata_from_manifest,
    write_hub_artifacts,
)
from seiso.io.files import iter_matching_files, path_size_bytes
from seiso.io.jsonl import read_json_file
from seiso.security import assert_within

logger = logging.getLogger(__name__)

_LARGE_HUB_UPLOAD_BYTES = 100 * 1024 * 1024


class _HubUploadLogWriter:
    """Capture tqdm / upload_large_folder stdout and forward to on_log."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log
        self._buf = ""

    def write(self, text: str) -> None:
        if not text:
            return
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            cleaned = line.strip("\r").strip()
            if cleaned:
                self._log(cleaned)

    def flush(self) -> None:
        cleaned = self._buf.strip("\r").strip()
        if cleaned:
            self._log(cleaned)
        self._buf = ""


class ExportFormat(StrEnum):
    MERGED = "merged"
    # Deprecated alias of FULL (same copytree behavior, different dest dir).
    # Prefer FULL; BASE remains for backward-compatible configs.
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
            _export_lora_adapter(ckpt, dest, log)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest
            log(f"LoRA adapter exported to {dest}")

        elif fmt == ExportFormat.MERGED:
            dest = out_root / "merged"
            dest.mkdir(parents=True, exist_ok=True)
            merge_lora_checkpoint(ckpt, dest, log)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest

        elif fmt in (ExportFormat.BASE, ExportFormat.FULL):
            looks_like_adapter = (ckpt / "adapter_config.json").is_file() or (
                (ckpt / "adapter_model.safetensors").is_file()
                or (ckpt / "adapter_model.bin").is_file()
            )
            if kind == "lora" or (kind == "unknown" and looks_like_adapter):
                raise ValueError(
                    f"Cannot export LoRA-only checkpoint as {fmt.value!r}; "
                    "use formats 'lora' and/or 'merged' instead "
                    f"(checkpoint={ckpt}, kind={kind})"
                )
            if kind == "unknown":
                raise ValueError(
                    f"Cannot export checkpoint as {fmt.value!r}: unable to detect "
                    "full weights (missing config.json / adapter_config.json). "
                    f"Refuse to copytree unknown trees as full/base (checkpoint={ckpt})"
                )
            dest = out_root / ("full" if fmt == ExportFormat.FULL else "base")
            if ckpt.exists():
                shutil.copytree(ckpt, dest, dirs_exist_ok=True)
            _write_export_sidecar(dest, ckpt, fmt, kind)
            results[fmt.value] = dest
            log(
                f"{'Full fine-tune' if fmt == ExportFormat.FULL else 'Base'} checkpoint exported to {dest}"
            )

        elif fmt == ExportFormat.GGUF:
            from seiso.export.gguf import export_gguf_from_checkpoint

            # Prefer a real merge. FULL/BASE may be a raw LoRA copytree — never
            # feed adapter-only dirs into GGUF conversion.
            merged = results.get(ExportFormat.MERGED.value)
            full = results.get(ExportFormat.FULL.value)
            if full is not None and full.exists() and (full / "adapter_config.json").is_file():
                full = None
            merged_dir = merged if merged and merged.exists() else None
            if merged_dir is None and full is not None and full.exists():
                merged_dir = full
            gguf_paths = export_gguf_from_checkpoint(
                ckpt,
                out_root,
                options.gguf_quantizations,
                merged_dir=merged_dir,
                on_log=log,
            )
            results.update(gguf_paths)

    if options.hub_repo and options.hub_token:
        push_folder = _select_hub_folder(out_root, options.formats)
        meta = _enriched_metadata(options, ckpt)
        meta.export_formats = [f.value for f in options.formats]
        # Always re-precheck immediately before push (TOCTOU after long export work).
        log(f"Re-running Hub precheck for {options.hub_repo} before push...")
        push_precheck = precheck_hub_export(
            repo_id=options.hub_repo,
            token=options.hub_token,
            metadata=meta,
            formats=[f.value for f in options.formats],
            on_log=log,
        )
        assert_hub_precheck_ok(push_precheck)
        _push_hub(
            options.hub_repo,
            options.hub_token,
            push_folder,
            log,
            metadata=meta,
            quantizations=options.gguf_quantizations,
            data_dir=options.sandbox_root,
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
    data_dir: Path | None = None,
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

    _push_hub(
        repo_id,
        token,
        folder,
        log,
        metadata=meta,
        quantizations=quantizations or meta.quantizations,
        data_dir=data_dir,
    )


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
    from seiso.research.provenance import (
        directory_checksum_manifest,
        git_commit_optional,
    )

    # Always hash weight tensors; optional full tree when SEISO_EXPORT_FULL_CHECKSUMS.
    full_checksums = os.environ.get("SEISO_EXPORT_FULL_CHECKSUMS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    weight_suffixes = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx")
    checksums = directory_checksum_manifest(
        dest,
        max_files=None,
        max_file_bytes=None if full_checksums else 8 * 1024 * 1024,
        always_hash_suffixes=weight_suffixes,
    )
    incomplete = any(value in {"skipped-large-file", "error"} for value in checksums.values())
    payload = {
        "format": fmt.value,
        "checkpoint_kind": kind,
        "source_checkpoint": str(ckpt),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_optional(),
        "file_checksums_sha256": checksums,
        "checksum_coverage": "partial" if incomplete else "full",
    }
    manifest = ckpt / "seiso_manifest.json"
    training_manifest = read_json_file(manifest, default=None)
    if training_manifest is not None:
        payload["training_manifest"] = training_manifest
    sidecar = dest / "seiso_export_metadata.json"
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if incomplete:
        return
    try:
        from seiso.research.nostr import maybe_auto_attest

        maybe_auto_attest(sidecar)
    except Exception:
        logger.exception("Nostr auto-attest failed for %s", sidecar)


def _select_hub_folder(out_root: Path, formats: list[ExportFormat]) -> Path:
    """Prefer merged/full weights; then LoRA adapter dir; then GGUF; else root."""
    for key in (ExportFormat.MERGED, ExportFormat.FULL, ExportFormat.BASE):
        if key in formats:
            candidate = out_root / ("merged" if key == ExportFormat.MERGED else key.value)
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
    if ExportFormat.LORA in formats:
        lora_dir = out_root / "lora"
        if lora_dir.is_dir() and any(lora_dir.iterdir()):
            return lora_dir
    if ExportFormat.GGUF in formats:
        gguf_dirs: list[Path] = []
        for child in out_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            looks_gguf = (
                name.startswith("gguf-")
                or name.startswith("q")
                or name in {"f16", "f32", "bf16"}
                or "gguf" in name
            )
            if not looks_gguf and not any(child.glob("*.gguf")):
                continue
            ggufs = [p for p in child.glob("*.gguf") if p.is_file() and p.stat().st_size > 0]
            if ggufs:
                gguf_dirs.append(child)
        if gguf_dirs:
            # Prefer smaller/common quants over f16 when multiple exist.
            def _gguf_rank(path: Path) -> tuple[int, str]:
                key = path.name.lower()
                preferred = (
                    "q2_k",
                    "q3_k_m",
                    "q4_k_m",
                    "q4_0",
                    "q5_k_m",
                    "q6_k",
                    "q8_0",
                    "f16",
                    "bf16",
                    "f32",
                )
                try:
                    return (preferred.index(key), key)
                except ValueError:
                    return (len(preferred), key)

            return sorted(gguf_dirs, key=_gguf_rank)[0]
    return out_root


@dataclass(frozen=True)
class _MergeDeps:
    auto_model: Any
    auto_tokenizer: Any
    peft_model: Any


def _load_merge_deps() -> _MergeDeps:
    """Lazy-load merge dependencies so tests can patch this single entry point."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return _MergeDeps(AutoModelForCausalLM, AutoTokenizer, PeftModel)


def _load_training_manifest(checkpoint: Path) -> dict:
    manifest = read_json_file(checkpoint / "seiso_manifest.json", default={})
    return manifest if isinstance(manifest, dict) else {}


def _resolve_merge_base_model(checkpoint: Path) -> str:
    """Resolve base model path/id for LoRA merge — prefers local cached weights."""
    adapter_config = checkpoint / "adapter_config.json"
    adapter_cfg = read_json_file(adapter_config, default={})
    if not isinstance(adapter_cfg, dict):
        adapter_cfg = {}

    # Prefer the Seiso-recorded original base (local or Hub) over a possibly stale
    # PEFT base_model_name_or_path that points at a missing cache dir.
    for key in ("seiso_original_base_model", "base_model_name_or_path"):
        value = adapter_cfg.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return str(candidate.resolve())

    manifest = _load_training_manifest(checkpoint)
    for key in (
        "resolved_model_path",
        "base_model_path",
        "original_model_id",
        "model_id",
    ):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate.resolve())
        if "/" in value and not candidate.is_absolute():
            # Hub-style id (org/model) — fall through to load_from_hub.
            return value
        if not candidate.exists():
            return value

    for key in ("seiso_original_base_model", "base_model_name_or_path"):
        base_id = adapter_cfg.get(key, "")
        if isinstance(base_id, str) and base_id.strip():
            return base_id.strip()

    raise ValueError(
        f"Cannot resolve base model for merge from {checkpoint}. "
        "Ensure seiso_manifest.json or adapter_config.json includes a local base path."
    )


def validate_lora_checkpoint(checkpoint: Path) -> None:
    """Raise when a LoRA checkpoint is missing required adapter artifacts."""
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        raise ValueError(f"LoRA checkpoint missing adapter_config.json: {checkpoint}")

    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    )
    if not any((checkpoint / name).is_file() for name in weight_names):
        nested = any(
            path.name in {"adapter_model.safetensors", "adapter_model.bin"}
            for path in iter_matching_files(
                checkpoint, suffixes=frozenset({".safetensors", ".bin"})
            )
        )
        if not nested:
            raise ValueError(f"LoRA checkpoint missing adapter weights: {checkpoint}")


def _export_lora_adapter(checkpoint: Path, dest: Path, log: Callable[[str], None]) -> None:
    """Copy a validated LoRA adapter tree for standalone use."""
    validate_lora_checkpoint(checkpoint)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(checkpoint, dest)
    readme = dest / "README.md"
    if not readme.is_file():
        manifest = _load_training_manifest(checkpoint)
        base = manifest.get("original_model_id") or manifest.get("model_id") or "base model"
        quant = manifest.get("quant", "unknown")
        readme.write_text(
            f"# LoRA adapter\n\n"
            f"Fine-tuned adapter exported from Seiso.\n\n"
            f"- **Base model:** `{base}`\n"
            f"- **Quantization during training:** {quant}\n\n"
            f"Load with PEFT:\n\n"
            f"```python\n"
            f"from peft import PeftModel\n"
            f"from transformers import AutoModelForCausalLM\n\n"
            f'model = AutoModelForCausalLM.from_pretrained("{base}")\n'
            f'model = PeftModel.from_pretrained(model, "{dest}")\n'
            f"```\n",
            encoding="utf-8",
        )
    log(f"Validated LoRA adapter ({checkpoint.name})")


def merge_lora_checkpoint(checkpoint: Path, dest: Path, log: Callable[[str], None]) -> None:
    """Merge LoRA weights into base model when PEFT adapter present."""
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.exists():
        log("Merging LoRA adapter into base weights...")
        deps = _load_merge_deps()

        base_id = _resolve_merge_base_model(checkpoint)
        log(f"Loading base model: {base_id}")
        from seiso.memory.protection import ensure_load_fits, release_cached_memory

        ensure_load_fits(base_id, mode="chat")
        tok_path = checkpoint if (checkpoint / "tokenizer_config.json").is_file() else base_id
        tok = deps.auto_tokenizer.from_pretrained(str(tok_path), revision="main")  # nosec B615: revision pinned
        model = None
        merged = None
        try:
            model = deps.auto_model.from_pretrained(
                base_id, device_map="cpu", low_cpu_mem_usage=True, revision="main"
            )  # nosec B615: revision pinned
            if len(tok) != model.get_input_embeddings().weight.shape[0]:
                model.resize_token_embeddings(len(tok))
            model = deps.peft_model.from_pretrained(model, str(checkpoint))
            merged = model.merge_and_unload()
            merged.save_pretrained(str(dest))
            tok.save_pretrained(str(dest))
            log(f"Merged model saved to {dest}")
        finally:
            del model, merged
            release_cached_memory()
    elif (checkpoint / "config.json").is_file():
        shutil.copytree(checkpoint, dest, dirs_exist_ok=True)
        log(f"Copied full checkpoint to {dest}")
    else:
        raise ValueError(f"Checkpoint is neither a LoRA adapter nor a full model: {checkpoint}")


def _push_hub(
    repo: str,
    token: str,
    folder: Path,
    log: Callable[[str], None],
    *,
    metadata: HubModelMetadata | None = None,
    quantizations: list[str] | None = None,
    data_dir: Path | None = None,
) -> None:
    from seiso.models.hf_env import configure_hf_hub_cache

    configure_hf_hub_cache(data_dir)
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

    total_bytes = path_size_bytes(folder)
    use_large = total_bytes >= _LARGE_HUB_UPLOAD_BYTES
    if use_large:
        log(
            f"Uploading {total_bytes / 1e9:.2f} GB to {repo} (resumable XET large-folder upload)..."
        )
        writer = _HubUploadLogWriter(log)
        with contextlib.redirect_stdout(writer):
            api.upload_large_folder(
                repo_id=repo,
                folder_path=str(folder),
                repo_type="model",
                num_workers=8,
                print_report=True,
                print_report_every=30,
            )
        writer.flush()
    else:
        log(f"Uploading to {repo}...")
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
