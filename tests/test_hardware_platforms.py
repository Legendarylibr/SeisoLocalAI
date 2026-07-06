"""Tests for OS platform system probes."""

from seiso.hardware.platforms import disk_usage_root, probe_for


def test_disk_usage_root_windows(monkeypatch):
    monkeypatch.setenv("SYSTEMDRIVE", "D:")
    assert disk_usage_root("Windows") == "D:\\"


def test_disk_usage_root_linux():
    assert disk_usage_root("Linux") == "/"


def test_disk_usage_root_darwin():
    assert disk_usage_root("Darwin") == "/"


def test_probe_for_selects_platform_module():
    from seiso.hardware.platforms import darwin, linux, windows

    assert probe_for("Windows") is windows
    assert probe_for("Darwin") is darwin
    assert probe_for("Linux") is linux
