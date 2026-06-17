"""Checkpoint export — merged, LoRA, GGUF, Hub push."""

from __future__ import annotations

import enum
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi

from seiso.security import assert_within, safe_join

logger = logging.getLogger(__name__)


class ExportFormat(enum.StrEnum):
    MERGED = "merged"
    BASE = "base"
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
    sandbox_root: Path | None = None


def export_checkpoint(
    options: ExportOptions,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Export checkpoint to requested formats. Returns map format -> path."""
    sandbox = options.sandbox_root or options.output_dir.parent
    ckpt = assert_within(sandbox, options.checkpoint)
    out_root = safe_join(sandbox, "exports", ckpt.name)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    for fmt in options.formats:
        if fmt == ExportFormat.LORA:
            dest = out_root / "lora"
            if ckpt.exists():
                shutil.copytree(ckpt, dest, dirs_exist_ok=True)
            results[fmt.value] = dest
            log(f"LoRA adapter exported to {dest}")

        elif fmt == ExportFormat.MERGED:
            dest = out_root / "merged"
            dest.mkdir(parents=True, exist_ok=True)
            _merge_lora(ckpt, dest, log)
            results[fmt.value] = dest

        elif fmt == ExportFormat.BASE:
            dest = out_root / "base"
            if ckpt.exists():
                shutil.copytree(ckpt, dest, dirs_exist_ok=True)
            results[fmt.value] = dest
            log(f"Base checkpoint copied to {dest}")

        elif fmt == ExportFormat.GGUF:
            from seiso.export.gguf import export_gguf_from_checkpoint

            merged = results.get(ExportFormat.MERGED.value)
            gguf_paths = export_gguf_from_checkpoint(
                ckpt,
                out_root,
                options.gguf_quantizations,
                merged_dir=merged if merged and merged.exists() else None,
                on_log=log,
            )
            results.update(gguf_paths)

    if options.hub_repo and options.hub_token:
        _push_hub(options.hub_repo, options.hub_token, out_root, log)
        log(f"Pushed to Hugging Face: {options.hub_repo}")

    return results


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


def _push_hub(repo: str, token: str, folder: Path, log: Callable[[str], None]) -> None:
    api = HfApi(token=token)
    log(f"Prechecking Hub repo: {repo}")
    try:
        api.repo_info(repo, repo_type="model")
    except Exception:
        api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model")
