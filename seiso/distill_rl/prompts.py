"""Load rollout prompts with stable IDs for reproducible research runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RolloutPrompt:
    prompt_id: str
    text: str
    answer: str | None = None
    benchmark: str | None = None
    # Optional code-proof fields (unit tests as verifier).
    tests: list[str] | str | None = None
    prompt_code: str | None = None
    setup: str | None = None
    timeout_s: float | None = None


def load_rollout_prompts(path: Path | None, *, limit: int) -> list[RolloutPrompt]:
    """Return prompt records from an explicit JSON/JSONL library.

    Distill-RL no longer falls back to the open post-train chat library (no
    answers/tests). Pass a grounded path, or use ``preference_source=dataset`` /
    ``data_designer``.
    """
    if path is None:
        raise ValueError(
            "prompt_library path is required for grounded_library / teacher_style. "
            "For meaningful Distill-RL defaults use preference_source=dataset "
            "(curated verifiable Hub set), optional data_designer, or a JSON/JSONL "
            "with answer and/or tests."
        )
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Prompt library not found: {source}")

    if source.suffix.lower() == ".jsonl":
        prompts = _load_jsonl_prompts(source)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        prompts = _extract_prompt_records(payload)

    if not prompts:
        raise ValueError(f"No prompts found in {source}")
    return prompts[:limit] if limit > 0 else prompts


def prompt_texts(prompts: list[RolloutPrompt]) -> list[str]:
    return [prompt.text for prompt in prompts]


def split_train_val(
    prompts: list[RolloutPrompt],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[RolloutPrompt], list[RolloutPrompt]]:
    import random

    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    rng = random.Random(seed)  # nosec B311 — deterministic split, not cryptography
    shuffled = list(prompts)
    rng.shuffle(shuffled)
    if len(shuffled) < 2:
        # Tiny/single-prompt libraries: all train, empty val.
        return shuffled, []
    # Keep ≥1 val row when possible (including train_fraction=1.0 under tiny/smoke).
    split_at = max(1, int(len(shuffled) * train_fraction))
    if split_at >= len(shuffled):
        split_at = len(shuffled) - 1
    return shuffled[:split_at], shuffled[split_at:]


def _load_jsonl_prompts(path: Path) -> list[RolloutPrompt]:
    prompts: list[RolloutPrompt] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(_normalize_prompt_row(row, fallback_id=f"line_{line_no}"))
    return [prompt for prompt in prompts if prompt.text.strip()]


def _extract_prompt_records(payload: object) -> list[RolloutPrompt]:
    if isinstance(payload, list):
        return [
            _normalize_prompt_row(row, fallback_id=f"row_{index}")
            for index, row in enumerate(payload)
        ]
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
        return [
            _normalize_prompt_row(row, fallback_id=f"row_{index}")
            for index, row in enumerate(payload["prompts"])
        ]
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return [
            _normalize_prompt_row(row, fallback_id=f"row_{index}")
            for index, row in enumerate(payload["examples"])
        ]
    raise ValueError(
        "Unsupported prompt library format; expected list or {'prompts': [...]}"
    )


def _normalize_prompt_row(row: object, *, fallback_id: str) -> RolloutPrompt:
    if isinstance(row, str):
        return RolloutPrompt(prompt_id=fallback_id, text=row)
    if not isinstance(row, dict):
        raise ValueError(f"Prompt row must be a string or object, got {type(row)!r}")
    text = _extract_prompt_text(row)
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    prompt_id = str(
        row.get("prompt_id")
        or row.get("id")
        or meta.get("prompt_id")
        or meta.get("task_id")
        or fallback_id
    )
    answer = row.get("answer", row.get("label"))
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        answer = meta.get("answer") or meta.get("label")
    benchmark = (
        row.get("benchmark")
        or row.get("dataset")
        or row.get("task")
        or row.get("reward")
        or meta.get("benchmark")
        or meta.get("rm_type")
    )
    tests = row.get("tests", row.get("test", meta.get("tests")))
    timeout_raw = row.get("timeout_s", row.get("timeout", meta.get("timeout_s")))
    timeout_s: float | None
    try:
        timeout_s = float(timeout_raw) if timeout_raw is not None else None
    except (TypeError, ValueError):
        timeout_s = None
    prompt_code = row.get("prompt_code") or row.get("code_prefix") or meta.get("prompt_code")
    setup = row.get("setup", meta.get("setup"))
    return RolloutPrompt(
        prompt_id=prompt_id,
        text=text,
        answer=str(answer) if answer is not None and str(answer).strip() else None,
        benchmark=str(benchmark).lower() if benchmark is not None else None,
        tests=tests if tests is not None else None,
        prompt_code=str(prompt_code) if prompt_code is not None else None,
        setup=str(setup) if setup is not None else None,
        timeout_s=timeout_s,
    )


def _extract_prompt_text(row: dict) -> str:
    """Support Distill JSONL strings and slime/Data Designer chat prompts."""
    raw = row.get("prompt")
    if isinstance(raw, list):
        parts: list[str] = []
        for msg in raw:
            if isinstance(msg, dict) and msg.get("content") is not None:
                parts.append(str(msg["content"]))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts).strip()
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    text = row.get("text") or row.get("instruction")
    return str(text or "").strip()


def prompt_to_verifier_sample(prompt: RolloutPrompt) -> dict:
    """Map a rollout prompt into the sample dict expected by ``seiso.rl_verify``."""
    sample: dict = {}
    if prompt.answer is not None:
        sample["answer"] = prompt.answer
    if prompt.benchmark is not None:
        sample["benchmark"] = prompt.benchmark
    if prompt.tests is not None:
        sample["tests"] = prompt.tests
    if prompt.prompt_code is not None:
        sample["prompt_code"] = prompt.prompt_code
    if prompt.setup is not None:
        sample["setup"] = prompt.setup
    if prompt.timeout_s is not None:
        sample["timeout_s"] = prompt.timeout_s
    return sample


def is_verifiable_prompt(prompt: RolloutPrompt) -> bool:
    """True when the prompt has an answer and/or unit tests for scoring."""
    if prompt.answer is not None and str(prompt.answer).strip():
        return True
    if prompt.tests is None:
        return False
    if isinstance(prompt.tests, list):
        return any(str(t).strip() for t in prompt.tests)
    return bool(str(prompt.tests).strip())
