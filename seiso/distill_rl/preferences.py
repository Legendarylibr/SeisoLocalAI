"""Preference dataset construction with provenance and train/val splits."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.distill_rl.prompts import (
    RolloutPrompt,
    load_rollout_prompts,
    split_train_val,
)
from seiso.distill_rl.rollouts import generate_preference_rows
from seiso.io.jsonl import write_jsonl
from seiso.rl_verify.synth_code import synthesize_code_bundle

logger = logging.getLogger(__name__)


def _is_grounded_reward_source(source: object) -> bool:
    """True when preference chosen/rejected came from a verifiable scorer."""
    text = str(source or "")
    return bool(
        text.startswith("verifiable")
        or text.startswith("code_")
        or text.startswith("synthetic_code")
    )


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
    trust_remote_code: bool = False,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    verifiable_outcome_rewards: bool = False,
    grpo_group_size: int = 1,
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
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        verifiable_outcome_rewards=verifiable_outcome_rewards,
        grpo_group_size=grpo_group_size,
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
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        verifiable_outcome_rewards=verifiable_outcome_rewards,
        grpo_group_size=grpo_group_size,
    )

    all_rows = train_rows + val_rows
    if all_rows:
        verifiable_n = sum(
            1
            for row in all_rows
            if _is_grounded_reward_source(row.get("reward_source"))
        )
        teacher_style_n = len(all_rows) - verifiable_n
        if teacher_style_n > verifiable_n:
            msg = (
                f"Preference set is dominated by teacher≻student pairs "
                f"({teacher_style_n}/{len(all_rows)}); these encode style, not "
                "verifiable correctness. Prefer prompts with answer/tests and "
                "verifiable_outcome_rewards=true for meaningful Distill-RL."
            )
            logger.warning(msg)
            _log(msg)

    train_path = output_dir / "preferences_train.jsonl"
    val_path = output_dir / "preferences_val.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    manifest = {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_revision": teacher_revision,
        "student_revision": student_revision,
        "trust_remote_code": trust_remote_code,
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "use_chat_template": use_chat_template,
        "require_thinking_trace": require_thinking_trace,
        "thinking_instruction": thinking_instruction,
        "verifiable_outcome_rewards": verifiable_outcome_rewards,
        "grpo_group_size": grpo_group_size,
        "train_fraction": train_fraction,
        "prompt_library": (
            str(prompt_library_path) if prompt_library_path else "default"
        ),
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
    trust_remote_code: bool = False,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    verifiable_outcome_rewards: bool = False,
    grpo_group_size: int = 1,
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
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        verifiable_outcome_rewards=verifiable_outcome_rewards,
        grpo_group_size=grpo_group_size,
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
    from seiso.rl_quant.bootstrap import bundle_root

    source = path or (bundle_root() / "prompts" / "post_train_library.json")
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def build_synthetic_code_preference_bundle(
    *,
    output_dir: Path,
    seed: int = 0,
    train_fraction: float = 0.85,
    limit: int | None = None,
    include_variants: bool = True,
    on_log=None,
) -> PreferenceBundle:
    """Build DPO preference JSONL from deterministic golden/mutant code pairs.

    No teacher/student model rollouts — every ``chosen`` is a sandbox-verified
    solution and every ``rejected`` is a mutant that fails unit tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    # Unit-test-grounded corpus only — not the small hand smoke catalog.
    # ``include_variants`` applies to the hand catalog path; ignored here.
    _ = include_variants
    corpus_count = max(32, int(limit) if limit is not None else 64)
    bundle = synthesize_code_bundle(
        seed=seed,
        include_variants=False,
        build_preferences=True,
        limit=limit,
        verify=True,
        corpus_count=corpus_count,
        include_hand_catalog=False,
    )
    rows = [pref.to_row() for pref in bundle.preferences]
    _log(f"Synthetic code preferences: {len(rows)} pairs (seed={seed})")

    # Deterministic train/val split by stable prompt_id order.
    rows_sorted = sorted(rows, key=lambda r: str(r.get("prompt_id", "")))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = max(1, int(len(rows_sorted) * train_fraction))
    if split_at >= len(rows_sorted):
        split_at = max(1, len(rows_sorted) - 1)
    train_rows = rows_sorted[:split_at]
    val_rows = rows_sorted[split_at:]

    train_path = output_dir / "preferences_train.jsonl"
    val_path = output_dir / "preferences_val.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    manifest = {
        "source": "synthetic_code_unit_tests",
        "seed": seed,
        "train_fraction": train_fraction,
        "include_variants": include_variants,
        "task_count": len(bundle.tasks),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "filtered_degenerate": 0,
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
        filtered_count=0,
    )
