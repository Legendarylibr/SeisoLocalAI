"""Shared grounded corpus materialization for slime and Distill-RL.

Product sources (RLVR-aligned; no silent localhost / toy generators):

* ``dataset`` — HF hub / local via training loaders + prep → verifiable JSONL
* ``data_designer`` — NVIDIA NeMo Data Designer (package + endpoint required)
* ``grounded_library`` — already-normalized local JSON/JSONL

``code_corpus`` is rejected as a product source (verifier unit tests only).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from seiso.rl_verify.data_gen import (
    DataGenResult,
    to_slime_prompt_row,
    write_jsonl,
)

logger = logging.getLogger(__name__)

# Product / Distill-RL / slime materialize sources.
SYNTH_SOURCES = (
    "data_designer",
    "dataset",
    "grounded_library",
)

SynthSource = Literal["data_designer", "dataset", "grounded_library"]


def normalize_materialize_source(source: str) -> str:
    """Normalize product materialize source names (legacy ``hf_dataset`` → ``dataset``)."""
    key = str(source or "").strip().lower()
    if key == "hf_dataset":
        return "dataset"
    return key


GROUNDED_FLOOR_DEFAULT = 256
GROUNDED_FLOOR_SMOKE = 32


@dataclass(frozen=True)
class SynthRequest:
    """Request to materialize a grounded prompt corpus."""

    source: SynthSource
    count: int = 256
    seed: int = 0
    mix: str = "numeric:0.5,choice:0.2,code:0.3"
    difficulty: str = "easy:0.35,medium:0.45,hard:0.20"
    endpoint: str | None = None
    model: str | None = None
    require_thinking_trace: bool = True
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    )
    artifact_dir: Path | None = None
    # dataset / grounded_library
    dataset_ref: str | Path | None = None
    split: str = "train"
    prompt_field: str | None = None
    answer_field: str | None = None
    tests_field: str | None = None
    revision: str = "main"
    preprocess: bool = True
    deduplicate: bool = True
    max_rows: int | None = None
    sandbox_root: Path | None = None
    # When set with sandbox_root, local refs must be under
    # ``sandbox_root/<scoped>/<sandbox_user_id>/`` (Forge multi-tenant).
    sandbox_user_id: str | None = None
    # Floor after filter (None = use default non-smoke floor).
    min_verifiable: int | None = None
    allow_tiny: bool = False


@dataclass
class SynthResult:
    """Materialized rows + light diagnostics."""

    rows: list[dict[str, Any]]
    source: str
    path: Path | None = None
    stream_counts: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "source": self.source,
            "stream_counts": dict(self.stream_counts),
            **self.meta,
        }

    def to_data_gen_result(self, *, seed: int = 0) -> DataGenResult:
        return DataGenResult(
            rows=list(self.rows),
            stream_counts=dict(self.stream_counts),
            difficulty_counts=dict(self.meta.get("difficulty_counts") or {}),
            seed=seed,
        )


def resolve_endpoint(*candidates: str | None) -> str | None:
    """Resolve OpenAI-compatible endpoint; never invent localhost."""
    for raw in candidates:
        text = (raw or "").strip()
        if text:
            return text
    for key in ("SEISO_DATA_DESIGNER_BASE_URL", "SEISO_VLLM_BASE_URL"):
        text = os.environ.get(key, "").strip()
        if text:
            return text
    return None


def materialize_grounded_corpus(
    out_path: Path,
    request: SynthRequest,
    *,
    write: bool = True,
) -> SynthResult:
    """Materialize a grounded corpus and optionally write JSONL to ``out_path``."""
    source = normalize_materialize_source(str(request.source))
    if source in {"code_corpus", "synthetic_code"}:
        raise ValueError(
            f"source={source!r} is not a product training path (toy programmatic "
            "generator). Use dataset, data_designer, or grounded_library. "
            "CI fixtures: preference_source=grounded_library + SEISO_ALLOW_TINY_RL=1."
        )
    if source not in SYNTH_SOURCES:
        raise ValueError(f"unknown synth source {source!r}; expected one of {SYNTH_SOURCES}")
    if source == "data_designer":
        result = _from_data_designer(request)
    elif source == "dataset":
        result = _from_dataset(request)
    else:
        result = _from_grounded_library(request)

    _enforce_floor(result, request)
    if write:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = write_jsonl(out_path, result.rows)
        result.path = out_path
        result.meta["written"] = n
        manifest = {
            **result.summary(),
            "path": str(out_path),
        }
        manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _enforce_floor(result: SynthResult, request: SynthRequest) -> None:
    floor = request.min_verifiable
    if floor is None:
        floor = 1 if request.allow_tiny else GROUNDED_FLOOR_DEFAULT
    if result.count < floor:
        raise ValueError(
            f"synth source={request.source!r} produced {result.count} verifiable "
            f"rows (need >= {floor}). Supply more data, or for CI fixtures set "
            "allow_tiny / SEISO_ALLOW_TINY_RL=1 with min_verifiable."
        )


def _thinking_suffix(request: SynthRequest) -> str:
    """Do not bake thinking instructions into materialized prompts.

    Generation-time ``format_generation_prompt`` / ``format_thinking_prompt``
    append the instruction and open ``<think>``. Embedding the default
    instruction (which contains a ``<think>`` substring) here suppresses that
    priming.
    """
    del request
    return ""


def _from_data_designer(request: SynthRequest) -> SynthResult:
    from seiso.rl_verify.data_designer_gen import (
        DataDesignerGenConfig,
        data_designer_available,
        materialize_data_designer_corpus,
    )

    if not data_designer_available():
        raise RuntimeError(
            "source=data_designer requires NVIDIA NeMo Data Designer. "
            "Install with: pip install -e '.[data-designer]'. "
            "Or use source=dataset or grounded_library."
        )
    endpoint = resolve_endpoint(request.endpoint)
    if not endpoint:
        raise RuntimeError(
            "source=data_designer requires an OpenAI-compatible endpoint. "
            "Set data_designer_base_url / vllm_base_url or "
            "SEISO_DATA_DESIGNER_BASE_URL / SEISO_VLLM_BASE_URL. "
            "No silent localhost default."
        )
    out = (
        Path(request.artifact_dir or Path.cwd() / "artifacts" / "data_designer")
        / "grounded_prompts.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = DataDesignerGenConfig(
        count=max(1, int(request.count)),
        seed=int(request.seed),
        vllm_base_url=endpoint,
        vllm_model=str(request.model or "local-model").strip(),
        mix=request.mix,
        difficulty=request.difficulty,
        require_thinking_trace=request.require_thinking_trace,
        thinking_instruction=request.thinking_instruction,
        artifact_dir=Path(request.artifact_dir)
        if request.artifact_dir is not None
        else out.parent / "dd_artifacts",
    )
    # Write to a temp path inside artifact_dir; caller may rewrite to out_path.
    dd_out = out
    dg = materialize_data_designer_corpus(dd_out, cfg, write_manifest=False)
    stream_counts = dict(dg.stream_counts)
    return SynthResult(
        rows=list(dg.rows),
        source="data_designer",
        stream_counts=stream_counts,
        meta={
            "seed": request.seed,
            "endpoint": endpoint,
            "generator": "nvidia.nemo.data_designer",
            "difficulty_counts": dict(dg.difficulty_counts),
        },
    )


def _from_grounded_library(request: SynthRequest) -> SynthResult:
    ref = request.dataset_ref
    if ref is None:
        raise ValueError("source=grounded_library requires dataset_ref (JSON/JSONL path)")
    path = Path(ref).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"grounded_library not found: {path}")
    if request.sandbox_root is not None:
        from seiso.security import assert_user_scoped_path, assert_within

        if request.sandbox_user_id:
            assert_user_scoped_path(request.sandbox_root, request.sandbox_user_id, path)
        else:
            assert_within(Path(request.sandbox_root), path)
    rows = _load_local_prompt_rows(path)
    mapped = [_ensure_slime_row(r, index=i) for i, r in enumerate(rows)]
    mapped = [r for r in mapped if _is_verifiable_row(r)]
    if request.max_rows is not None and request.max_rows > 0:
        mapped = mapped[: int(request.max_rows)]
    elif request.count > 0:
        mapped = mapped[: int(request.count)]
    return SynthResult(
        rows=mapped,
        source="grounded_library",
        stream_counts=_stream_counts(mapped),
        meta={"dataset_ref": str(path)},
    )


def _from_dataset(request: SynthRequest) -> SynthResult:
    from seiso.models.hf_env import configure_hf_hub_auth
    from seiso.training.config import DatasetFormat
    from seiso.training.datasets import load_training_dataset
    from seiso.training.preprocess import preprocess_training_dataset

    if request.dataset_ref is None or not str(request.dataset_ref).strip():
        raise ValueError("source=dataset requires dataset_ref (HF hub id or local path)")
    ref = str(request.dataset_ref).strip()
    local = Path(ref).expanduser()
    # Local files/dirs need no Hub token. Public Hub sets also work without a
    # token; configure_hf_hub_auth only mirrors a token when one is already set.
    if not local.exists():
        configure_hf_hub_auth()
    revision = str(request.revision or "main").strip() or "main"
    raw = load_training_dataset(
        ref,
        split=request.split,
        sandbox_root=request.sandbox_root,
        sandbox_user_id=request.sandbox_user_id,
        revision=revision,
    )

    if request.preprocess:
        prepared, stats, fmt = preprocess_training_dataset(
            raw,
            dataset_format=DatasetFormat.AUTO,
            deduplicate=request.deduplicate,
            preference_as_sft=False,
        )
    else:
        prepared, stats, fmt = raw, {"kept": len(raw)}, DatasetFormat.AUTO

    rows: list[dict[str, Any]] = []
    dropped = 0
    preference_only = 0
    scanned = 0
    limit = request.max_rows if request.max_rows is not None else request.count
    n_prepared = len(prepared)
    # Scan until enough keeps or EOF — never early-exit while rows is empty
    # (preference-only heads can be long; false preference-only is worse than cost).
    for idx in range(n_prepared):
        sample = prepared[idx]
        if not isinstance(sample, dict):
            sample = dict(sample)
        scanned += 1
        if _looks_preference_only(
            sample,
            prompt_field=request.prompt_field,
            answer_field=request.answer_field,
            tests_field=request.tests_field,
        ):
            preference_only += 1
            dropped += 1
            continue
        mapped = _map_sample_to_slime_row(
            sample,
            index=idx,
            prompt_field=request.prompt_field,
            answer_field=request.answer_field,
            tests_field=request.tests_field,
            fmt=fmt,
            thinking_suffix=_thinking_suffix(request),
        )
        if mapped is None:
            dropped += 1
            continue
        rows.append(mapped)
        if limit and len(rows) >= int(limit):
            break

    if not rows and preference_only > 0 and preference_only == scanned:
        raise ValueError(
            f"source=dataset ref={request.dataset_ref!r} looks like a "
            f"preference-only corpus ({preference_only} chosen/rejected rows without "
            f"answer/tests after scanning {scanned}). Outcome RL needs verifiable "
            "labels; use a math/code Hub set, or preference_source=teacher_style "
            "for Distill-RL style bootstrap only."
        )
    if scanned >= 20 and dropped > 0:
        drop_rate = dropped / float(scanned)
        if drop_rate >= 0.5:
            logger.warning(
                "dataset %s dropped %.0f%% of scanned rows (%s/%s) lacking "
                "answer/tests; check field mapping or use a verifiable corpus",
                request.dataset_ref,
                100.0 * drop_rate,
                dropped,
                scanned,
            )

    return SynthResult(
        rows=rows,
        source="dataset",
        stream_counts=_stream_counts(rows),
        meta={
            "dataset_ref": str(request.dataset_ref),
            "split": request.split,
            "revision": revision,
            "preprocess_stats": stats,
            "dropped_unverifiable": dropped,
            "preference_only_dropped": preference_only,
            "scanned": scanned,
            "resolved_format": getattr(fmt, "value", str(fmt)),
        },
    )


def _looks_preference_only(
    sample: dict[str, Any],
    *,
    prompt_field: str | None,
    answer_field: str | None,
    tests_field: str | None,
) -> bool:
    """True when row has chosen/rejected prefs but no outcome verifier signal."""
    has_pref = any(
        sample.get(key) is not None and str(sample.get(key)).strip()
        for key in ("chosen", "rejected", "chosen_response", "rejected_response")
    )
    if not has_pref:
        return False
    answer = _extract_answer_text(sample, answer_field=answer_field)
    tests = None
    if tests_field and sample.get(tests_field) is not None:
        tests = sample.get(tests_field)
    elif sample.get("tests") is not None:
        tests = sample.get("tests")
    elif sample.get("test") is not None:
        tests = sample.get("test")
    if answer and str(answer).strip():
        return False
    if _tests_nonempty(tests):
        return False
    # Prefer-style rows often still have a prompt; absence of answer/tests is enough.
    _ = prompt_field
    return True


def _map_sample_to_slime_row(
    sample: dict[str, Any],
    *,
    index: int,
    prompt_field: str | None,
    answer_field: str | None,
    tests_field: str | None,
    fmt: Any,
    thinking_suffix: str,
) -> dict[str, Any] | None:
    prompt = _extract_prompt_text(sample, prompt_field=prompt_field)
    answer = _extract_answer_text(sample, answer_field=answer_field)
    tests = None
    if tests_field and sample.get(tests_field) is not None:
        tests = sample.get(tests_field)
    elif sample.get("tests") is not None:
        tests = sample.get("tests")
    elif sample.get("test") is not None:
        tests = sample.get("test")

    if not prompt:
        return None
    has_answer = bool(answer and str(answer).strip())
    has_tests = _tests_nonempty(tests)
    if not has_answer and not has_tests:
        return None

    content = prompt.rstrip() + thinking_suffix
    rm_type = "code" if has_tests else _infer_rm_type(answer)
    meta: dict[str, Any] = {
        "benchmark": rm_type,
        "stream": rm_type,
        "task_id": str(sample.get("id") or sample.get("prompt_id") or f"hf_{index}"),
        "source_name": "dataset",
        "generator": "seiso.rl_verify.synth_materialize.dataset",
        "dataset_format": getattr(fmt, "value", str(fmt)),
    }
    if has_tests:
        meta["tests"] = tests
    if sample.get("timeout_s") is not None:
        meta["timeout_s"] = sample.get("timeout_s")
    return to_slime_prompt_row(
        content,
        str(answer).strip() if has_answer else "",
        rm_type=rm_type,
        metadata=meta,
    )


def _extract_prompt_text(sample: dict[str, Any], *, prompt_field: str | None) -> str:
    if prompt_field and sample.get(prompt_field) is not None:
        return _coerce_prompt(sample.get(prompt_field))
    for key in ("prompt", "question", "query", "instruction", "text"):
        if sample.get(key) is not None:
            text = _coerce_prompt(sample.get(key))
            if key == "instruction":
                inp = str(sample.get("input") or "").strip()
                if inp:
                    text = f"{text}\n\n{inp}" if text else inp
            if text:
                return text
    messages = sample.get("messages") or sample.get("conversations")
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            content = str(msg.get("content") or msg.get("value") or "").strip()
            if role in {"user", "human"} and content:
                parts.append(content)
        if parts:
            return parts[-1]
    return ""


def _extract_answer_text(sample: dict[str, Any], *, answer_field: str | None) -> str:
    if answer_field and sample.get(answer_field) is not None:
        return str(sample.get(answer_field)).strip()
    for key in ("answer", "label", "output", "response", "completion"):
        if sample.get(key) is not None and str(sample.get(key)).strip():
            return str(sample.get(key)).strip()
    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and str(msg.get("role") or "").lower() == "assistant":
                text = str(msg.get("content") or "").strip()
                if text:
                    return text
    return ""


def _coerce_prompt(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for msg in value:
            if isinstance(msg, dict) and msg.get("content") is not None:
                parts.append(str(msg["content"]))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts).strip()
    return str(value or "").strip()


def _infer_rm_type(answer: str | None) -> str:
    text = str(answer or "").strip()
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D", "E"}:
        return "choice"
    return "numeric"


def _tests_nonempty(tests: Any) -> bool:
    if tests is None:
        return False
    if isinstance(tests, list):
        return any(str(t).strip() for t in tests)
    return bool(str(tests).strip())


def _is_verifiable_row(row: dict[str, Any]) -> bool:
    answer = row.get("answer", row.get("label"))
    if answer is not None and str(answer).strip():
        return True
    return _tests_nonempty(row.get("tests") or (row.get("metadata") or {}).get("tests"))


def _ensure_slime_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Normalize Distill/JSONL library rows into slime-shaped dicts."""
    if "prompt" in row and isinstance(row.get("prompt"), list):
        return row
    text = _coerce_prompt(row.get("prompt") or row.get("text") or row.get("instruction"))
    answer = row.get("answer", row.get("label"))
    tests = row.get("tests", row.get("test"))
    rm_type = str(row.get("reward") or row.get("benchmark") or ("code" if tests else "numeric"))
    meta = dict(row.get("metadata") or {})
    meta.setdefault(
        "task_id",
        str(row.get("prompt_id") or row.get("id") or f"lib_{index}"),
    )
    if tests is not None:
        meta["tests"] = tests
    return to_slime_prompt_row(
        text,
        str(answer).strip() if answer is not None else "",
        rm_type=rm_type,
        metadata=meta,
    )


def _load_local_prompt_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("prompts", "examples", "data"):
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
    raise ValueError(f"unsupported grounded_library format: {path}")


def _stream_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        key = str(meta.get("rm_type") or row.get("reward") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts
