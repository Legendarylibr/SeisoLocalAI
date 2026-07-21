"""Preference dataset construction with provenance and train/val splits."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.distill_rl.config import GROUNDED_LIBRARY_FLOOR
from seiso.distill_rl.prompts import (
    RolloutPrompt,
    is_verifiable_prompt,
    load_rollout_prompts,
    split_train_val,
)
from seiso.distill_rl.rollouts import generate_preference_rows
from seiso.io.jsonl import write_jsonl

logger = logging.getLogger(__name__)


def _is_grounded_reward_source(source: object) -> bool:
    """True when preference chosen/rejected came from a verifiable scorer."""
    text = str(source or "")
    return bool(
        text.startswith("verifiable")
        or text.startswith("code_")
        or text.startswith("synthetic_code")
        or "data_designer" in text
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
    min_grounded_prompts: int | None = None,
    on_log=None,
) -> PreferenceBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    if prompt_library_path is None:
        raise ValueError(
            "prompt_library_path is required for grounded_library / teacher_style "
            "preference construction"
        )
    # Load full library first so floor checks see true grounded count.
    prompts = load_rollout_prompts(prompt_library_path, limit=0)
    if verifiable_outcome_rewards:
        grounded = [p for p in prompts if is_verifiable_prompt(p)]
        floor = (
            GROUNDED_LIBRARY_FLOOR
            if min_grounded_prompts is None
            else int(min_grounded_prompts)
        )
        if len(grounded) < floor:
            raise ValueError(
                f"grounded_library has {len(grounded)} verifiable prompts "
                f"(need >= {floor} with answer/tests). Refusing tiny/open libraries. "
                "Use preference_source=dataset (curated verifiable Hub set), "
                "data_designer (opt-in), or a larger operator JSONL. "
                "CI fixtures: SEISO_ALLOW_TINY_RL=1 / preset=smoke."
            )
        prompts = grounded[:max_prompts] if max_prompts > 0 else grounded
    else:
        prompts = prompts[:max_prompts] if max_prompts > 0 else prompts
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
        "prompt_library": str(prompt_library_path),
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


def materialize_data_designer_prompt_library(
    *,
    output_dir: Path,
    count: int,
    seed: int,
    require_thinking_trace: bool,
    thinking_instruction: str,
    base_url: str | None,
    model: str | None,
    mix: str = "numeric:0.7,choice:0.3",
    difficulty: str = "easy:0.35,medium:0.45,hard:0.20",
    preset: str = "smoke",
    on_log=None,
) -> Path:
    """Shim → shared ``materialize_grounded_corpus`` (Data Designer source).

    Prefer ``grounded_data.materialize_distill_grounded_prompts`` for Distill jobs.
    """
    from seiso.distill_rl.config import allow_tiny_rl
    from seiso.rl_verify.synth_materialize import (
        SynthRequest,
        materialize_grounded_corpus,
        resolve_endpoint,
    )

    del thinking_instruction  # applied at generation time, not baked into rows
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "data_designer_prompts.jsonl"
    tiny = allow_tiny_rl(preset=preset)
    if on_log:
        on_log(
            f"Materializing Distill-RL prompts via shared Data Designer path "
            f"(count={count})"
        )
    materialize_grounded_corpus(
        out_path,
        SynthRequest(
            source="data_designer",
            count=max(1, int(count)),
            seed=int(seed),
            mix=mix,
            difficulty=difficulty,
            endpoint=resolve_endpoint(base_url),
            model=str(model or "local-model").strip(),
            require_thinking_trace=require_thinking_trace,
            artifact_dir=output_dir / "data_designer_artifacts",
            min_verifiable=1 if tiny else None,
            allow_tiny=tiny,
        ),
    )
    return out_path


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
    if path is None:
        return None
    source = Path(path).expanduser()
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()
