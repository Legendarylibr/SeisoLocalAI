"""Materialize Distill-RL grounded prompts via shared synth API (once per job)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seiso.distill_rl.config import allow_tiny_rl
from seiso.rl_verify.synth_materialize import (
    GROUNDED_FLOOR_DEFAULT,
    SynthRequest,
    materialize_grounded_corpus,
    resolve_endpoint,
)

if TYPE_CHECKING:
    from seiso.distill_rl.config import DistillRLConfig

GROUNDED_PROMPTS_NAME = "grounded_prompts.jsonl"
GROUNDED_FINGERPRINT_NAME = "grounded_prompts.fingerprint.json"


def grounded_prompts_path(config: DistillRLConfig) -> Path:
    return config.preferences_dir / GROUNDED_PROMPTS_NAME


def grounded_fingerprint_path(config: DistillRLConfig) -> Path:
    return config.preferences_dir / GROUNDED_FINGERPRINT_NAME


def _local_path_identity(path: Path | str | None) -> dict[str, Any] | None:
    """Path + mtime/size so in-place JSONL edits invalidate the cache."""
    if path is None:
        return None
    p = Path(path).expanduser()
    try:
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p)
    identity: dict[str, Any] = {"path": resolved}
    try:
        if p.is_file():
            st = p.stat()
            identity["mtime_ns"] = int(st.st_mtime_ns)
            identity["size"] = int(st.st_size)
        elif p.is_dir():
            st = p.stat()
            identity["mtime_ns"] = int(st.st_mtime_ns)
            identity["isdir"] = True
    except OSError:
        pass
    return identity


def _grounded_fingerprint(config: DistillRLConfig) -> dict[str, Any]:
    """Stable identity for grounded materialize inputs (reuse gate)."""
    source = str(config.preference_source).strip().lower()
    lib = (
        _local_path_identity(config.prompt_library_path)
        if config.prompt_library_path is not None
        else None
    )
    ref = (config.dataset_ref or "").strip() or None
    dataset_identity: dict[str, Any] | str | None
    dataset_identity = _local_path_identity(ref) if ref and Path(ref).expanduser().exists() else ref
    tiny = allow_tiny_rl(preset=config.preset)
    return {
        "preference_source": source,
        "dataset_ref": dataset_identity,
        "prompt_library": lib,
        "data_gen_count": int(config.data_gen_count),
        "seed": int(config.seed),
        "dataset_split": str(config.dataset_split),
        "dataset_revision": str(config.dataset_revision),
        "prompt_field": config.prompt_field,
        "answer_field": config.answer_field,
        "tests_field": config.tests_field,
        "preprocess_dataset": bool(config.preprocess_dataset),
        "deduplicate_dataset": bool(config.deduplicate_dataset),
        "data_gen_mix": str(config.data_gen_mix),
        "data_gen_difficulty": str(config.data_gen_difficulty),
        # Resolve env overrides so SEISO_*_BASE_URL invalidates the cache.
        "data_designer_base_url": resolve_endpoint(config.data_designer_base_url),
        "data_designer_model": config.data_designer_model,
        "require_thinking_trace": bool(config.require_thinking_trace),
        "thinking_instruction": str(config.thinking_instruction),
        # Tiny-allow must invalidate caches written under SEISO_ALLOW_TINY_RL / smoke.
        "allow_tiny": bool(tiny),
        "min_verifiable": 1 if tiny else GROUNDED_FLOOR_DEFAULT,
    }


def _fingerprint_matches(config: DistillRLConfig) -> bool:
    path = grounded_fingerprint_path(config)
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(saved == _grounded_fingerprint(config))


def _write_fingerprint(config: DistillRLConfig) -> None:
    path = grounded_fingerprint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_grounded_fingerprint(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_row_count(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _cache_meets_floor(path: Path, *, tiny: bool) -> bool:
    floor = 1 if tiny else GROUNDED_FLOOR_DEFAULT
    return _jsonl_row_count(path) >= floor


def materialize_distill_grounded_prompts(
    config: DistillRLConfig,
    *,
    on_log: Any = None,
) -> Path:
    """Write ``preferences/grounded_prompts.jsonl`` once for distill + rollout."""
    out = grounded_prompts_path(config)
    tiny = allow_tiny_rl(preset=config.preset)
    if (
        out.is_file()
        and out.stat().st_size > 0
        and _fingerprint_matches(config)
        and _cache_meets_floor(out, tiny=tiny)
    ):
        if on_log:
            on_log(f"Reusing grounded prompts: {out}")
        return out

    source = str(config.preference_source).strip().lower()

    if source == "teacher_style":
        raise ValueError(
            "teacher_style does not materialize grounded prompts; "
            "pass prompt_library for open-style DPO"
        )

    if source == "grounded_library":
        if config.prompt_library_path is None:
            raise ValueError("grounded_library requires prompt_library")
        req = SynthRequest(
            source="grounded_library",
            count=max(1, int(config.data_gen_count)),
            seed=int(config.seed),
            dataset_ref=config.prompt_library_path,
            require_thinking_trace=config.require_thinking_trace,
            thinking_instruction=config.thinking_instruction,
            min_verifiable=1 if tiny else None,
            allow_tiny=tiny,
        )
    elif source == "dataset":
        ref = config.dataset_ref or (
            str(config.prompt_library_path) if config.prompt_library_path else None
        )
        if not ref:
            raise ValueError(
                "preference_source=dataset requires dataset_ref or prompt_library "
                "(HF hub id or local path)"
            )
        req = SynthRequest(
            source="dataset",
            count=max(1, int(config.data_gen_count)),
            seed=int(config.seed),
            dataset_ref=ref,
            split=config.dataset_split,
            prompt_field=config.prompt_field,
            answer_field=config.answer_field,
            tests_field=config.tests_field,
            revision=config.dataset_revision,
            preprocess=config.preprocess_dataset,
            deduplicate=config.deduplicate_dataset,
            require_thinking_trace=config.require_thinking_trace,
            thinking_instruction=config.thinking_instruction,
            sandbox_root=config.sandbox_root,
            sandbox_user_id=config.user_id if config.sandbox_root is not None else None,
            min_verifiable=1 if tiny else None,
            allow_tiny=tiny,
            max_rows=config.data_gen_count,
        )
    elif source == "data_designer":
        endpoint = resolve_endpoint(config.data_designer_base_url)
        req = SynthRequest(
            source="data_designer",
            count=max(1, int(config.data_gen_count)),
            seed=int(config.seed),
            mix=config.data_gen_mix,
            difficulty=config.data_gen_difficulty,
            endpoint=endpoint,
            model=config.data_designer_model or config.student_model,
            require_thinking_trace=config.require_thinking_trace,
            thinking_instruction=config.thinking_instruction,
            artifact_dir=config.preferences_dir / "data_designer_artifacts",
            min_verifiable=1 if tiny else None,
            allow_tiny=tiny,
        )
    else:
        raise ValueError(f"unsupported preference_source for materialize: {source!r}")

    if on_log:
        on_log(
            f"Materializing grounded prompts source={source} count={config.data_gen_count} → {out}"
        )
    materialize_grounded_corpus(out, req)
    _write_fingerprint(config)
    return out
