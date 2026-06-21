"""Load rollout prompts with stable IDs for reproducible research runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from seiso.rl_quant.bootstrap import vendor_root


@dataclass(frozen=True)
class RolloutPrompt:
    prompt_id: str
    text: str


def load_rollout_prompts(path: Path | None, *, limit: int) -> list[RolloutPrompt]:
    """Return prompt records from JSON, JSONL, or the vendored post-train library."""
    source = path or (vendor_root() / "prompts" / "post_train_library.json")
    if not source.is_file():
        raise FileNotFoundError(f"Prompt library not found: {source}")

    if source.suffix.lower() == ".jsonl":
        prompts = _load_jsonl_prompts(source)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        prompts = _extract_prompt_records(payload)

    if not prompts:
        raise ValueError(f"No prompts found in {source}")
    return prompts[:limit]


def prompt_texts(prompts: list[RolloutPrompt]) -> list[str]:
    return [prompt.text for prompt in prompts]


def split_train_val(
    prompts: list[RolloutPrompt],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[RolloutPrompt], list[RolloutPrompt]]:
    import random

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    rng = random.Random(seed)  # nosec B311 — deterministic split, not cryptography
    shuffled = list(prompts)
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * train_fraction))
    if split_at >= len(shuffled):
        split_at = max(1, len(shuffled) - 1)
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
    raise ValueError("Unsupported prompt library format; expected list or {'prompts': [...]}")


def _normalize_prompt_row(row: object, *, fallback_id: str) -> RolloutPrompt:
    if isinstance(row, str):
        return RolloutPrompt(prompt_id=fallback_id, text=row)
    if not isinstance(row, dict):
        raise ValueError(f"Prompt row must be a string or object, got {type(row)!r}")
    text = str(row.get("prompt") or row.get("text") or row.get("instruction") or "")
    prompt_id = str(row.get("prompt_id") or row.get("id") or fallback_id)
    return RolloutPrompt(prompt_id=prompt_id, text=text)
