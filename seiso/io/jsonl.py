"""JSONL read/write helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: Path,
    *,
    ensure_ascii: bool = False,
) -> int:
    """Write dict rows to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield dict rows from a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
