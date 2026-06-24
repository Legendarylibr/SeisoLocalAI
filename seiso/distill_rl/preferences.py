"""Preference dataset construction with provenance and train/val splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.distill_rl.prompts import RolloutPrompt, load_rollout_prompts, split_train_val
from seiso.distill_rl.rollouts import generate_preference_rows
from seiso.io.jsonl import write_jsonl


@dataclass(frozen=True)
class PreferenceBundle:
    train_path: Path
    val_path: Path
    manifest_path: Path
    train_count: int
    val_count: int
    filtered_count: int


def build_preference_bundle(
    *,
    teacher_model: str,
    student_model: str,
    output_dir: Path,
    prompt_library_path: Path | None,
    max_prompts: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    train_fraction: float,
    use_chat_template: bool,
    teacher_revision: str | None = None,
    student_revision: str | None = None,
    on_log=None,
) -> PreferenceBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_rollout_prompts(prompt_library_path, limit=max_prompts)
    train_prompts, val_prompts = split_train_val(
        prompts,
        train_fraction=train_fraction,
        seed=seed,
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    _log(f"Rollout train={len(train_prompts)} val={len(val_prompts)} prompts")
    train_rows, train_filtered = _rows_for_split(
        teacher_model=teacher_model,
        student_model=student_model,
        prompts=train_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        teacher_revision=teacher_revision,
        student_revision=student_revision,
    )
    val_rows, val_filtered = _rows_for_split(
        teacher_model=teacher_model,
        student_model=student_model,
        prompts=val_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed + 1,
        use_chat_template=use_chat_template,
        teacher_revision=teacher_revision,
        student_revision=student_revision,
    )

    train_path = output_dir / "preferences_train.jsonl"
    val_path = output_dir / "preferences_val.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    manifest = {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_revision": teacher_revision,
        "student_revision": student_revision,
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "use_chat_template": use_chat_template,
        "train_fraction": train_fraction,
        "prompt_library": str(prompt_library_path) if prompt_library_path else "default",
        "prompt_library_sha256": _prompt_library_hash(prompt_library_path),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "filtered_degenerate": train_filtered + val_filtered,
        "train_sha256": _file_hash(train_path),
        "val_sha256": _file_hash(val_path),
    }
    manifest_path = output_dir / "preferences_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    legacy_path = output_dir / "preferences.jsonl"
    legacy_path.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")

    return PreferenceBundle(
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        train_count=len(train_rows),
        val_count=len(val_rows),
        filtered_count=train_filtered + val_filtered,
    )


def _rows_for_split(
    *,
    teacher_model: str,
    student_model: str,
    prompts: list[RolloutPrompt],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    use_chat_template: bool,
    teacher_revision: str | None = None,
    student_revision: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rows = generate_preference_rows(
        teacher_model=teacher_model,
        student_model=student_model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        teacher_revision=teacher_revision,
        student_revision=student_revision,
    )
    filtered = 0
    kept: list[dict[str, Any]] = []
    for row in rows:
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not chosen or not rejected or chosen == rejected:
            filtered += 1
            continue
        kept.append(row)
    return kept, filtered


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_library_hash(path: Path | None) -> str | None:
    from seiso.rl_quant.bootstrap import vendor_root

    source = path or (vendor_root() / "prompts" / "post_train_library.json")
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()
