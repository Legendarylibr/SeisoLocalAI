from __future__ import annotations

from pathlib import Path

from seiso.io.files import matching_file_stats, path_size_bytes
from seiso.io.jsonl import iter_jsonl, read_json_file, write_jsonl


def test_write_and_iter_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "a"}, {"text": "b"}]
    assert write_jsonl(rows, path) == 2
    assert list(iter_jsonl(path)) == rows


def test_read_json_file_default_for_missing_or_bad_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert read_json_file(missing, default={"fallback": True}) == {"fallback": True}

    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert read_json_file(bad, default={}) == {}

    good = tmp_path / "good.json"
    good.write_text('{"ok": true}', encoding="utf-8")
    assert read_json_file(good, default={}) == {"ok": True}


def test_file_stats_helpers(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"123")
    (tmp_path / "b.bin").write_bytes(b"12")
    (tmp_path / "ignore.txt").write_bytes(b"12345")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.safetensors").write_bytes(b"1")

    assert path_size_bytes(tmp_path) == 11
    assert matching_file_stats(tmp_path, suffixes=frozenset({".safetensors", ".bin"})) == (3, 6)
    assert matching_file_stats(tmp_path, "*.safetensors") == (2, 4)
