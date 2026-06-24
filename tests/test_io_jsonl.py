from __future__ import annotations

from pathlib import Path

from seiso.io.jsonl import iter_jsonl, write_jsonl


def test_write_and_iter_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "a"}, {"text": "b"}]
    assert write_jsonl(rows, path) == 2
    assert list(iter_jsonl(path)) == rows