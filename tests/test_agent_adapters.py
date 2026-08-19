"""Detect + isolated provider snippets. Never touch ~/.hermes or ~/.pi."""

from __future__ import annotations

from pathlib import Path

from seiso.agent.adapters.detect import default_harness_id, detect_all, detect_harness
from seiso.agent.adapters.endpoint import resolve_endpoint
from seiso.agent.adapters.profiles import isolated_dir, write_profile
from seiso.agent.adapters.types import HARNESS_IDS, parse_harness_id


def test_parse_harness_aliases() -> None:
    assert parse_harness_id("OpenClaw") == "openclaw"
    assert parse_harness_id("clawdbot") == "openclaw"
    assert parse_harness_id("omp") == "omp"


def test_detect_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr("seiso.agent.adapters.detect.shutil.which", lambda _n: None)
    monkeypatch.setattr("seiso.agent.adapters.detect._home_dir", lambda _i: None)
    row = detect_harness("pi")
    assert row.installed is False
    assert row.hint


def test_detect_all_covers_openclaw() -> None:
    ids = {row.id for row in detect_all()}
    assert ids == set(HARNESS_IDS)
    assert "openclaw" in ids


def test_isolated_profile_does_not_write_home(tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    endpoint = resolve_endpoint(source="ollama", probe=False)
    dest = isolated_dir(tmp_path / "data", "hermes")
    path = write_profile(dest, "hermes", endpoint)
    assert path.is_file()
    assert path.is_relative_to(tmp_path / "data")
    assert not (fake_home / ".hermes" / "config.yaml").exists()
    assert not (fake_home / ".pi").exists()
    assert not (fake_home / ".openclaw").exists()


def test_default_harness_prefers_installed(monkeypatch) -> None:
    from seiso.agent.adapters.types import DetectedHarness

    rows = (
        DetectedHarness("pi", "Pi", False),
        DetectedHarness("hermes", "Hermes", True, binary="/bin/hermes"),
    )
    monkeypatch.setattr("seiso.agent.adapters.detect.detect_all", lambda **_k: rows)
    assert default_harness_id(rows) == "hermes"
