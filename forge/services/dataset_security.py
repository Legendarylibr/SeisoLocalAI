"""Soft training-dataset security screens — warn only, never refuse training.

This does **not** censor or block datasets. It only emits an audit event when a
local file sample looks densely instruction-injection-like, so operators notice
poisoning attempts. Training always continues.
"""

from __future__ import annotations

from pathlib import Path

from forge.security.audit import audit_event
from forge.tools.sanitize import is_instruction_like

_SAMPLE_BYTES = 256 * 1024
_WARN_FLAGGED_RATIO = 0.15
_MIN_LINES_FOR_RATIO = 8


def warn_instruction_like_dataset(
    dataset: str | Path | None,
    *,
    user_id: str | None = None,
) -> dict[str, int | float] | None:
    """Sample a local dataset file and audit when instruction-like density is high.

    Returns stats when a local file was sampled; ``None`` when skipped (HF id,
    missing file, etc.). Does not raise — training must continue.
    """
    if not dataset:
        return None
    path = Path(str(dataset)).expanduser()
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()[:_SAMPLE_BYTES]
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    flagged = sum(1 for ln in lines if is_instruction_like(ln))
    ratio = flagged / len(lines)
    stats: dict[str, int | float] = {
        "lines_sampled": len(lines),
        "flagged_lines": flagged,
        "flagged_ratio": round(ratio, 4),
    }
    if flagged >= 2 and len(lines) >= _MIN_LINES_FOR_RATIO and ratio >= _WARN_FLAGGED_RATIO:
        audit_event(
            "dataset_instruction_like_warning",
            user_id=user_id,
            dataset_name=path.name,
            **stats,
        )
    return stats
