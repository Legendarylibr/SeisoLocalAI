"""Tests for OS platform system probes."""

from types import SimpleNamespace

from seiso.hardware.platforms import common, disk_usage_root, probe_for
from seiso.hardware.profile import _cpu_temp_from_sensors


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


def test_cpu_brand_from_proc_cpuinfo(tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\n"
        "vendor_id\t: AuthenticAMD\n"
        "model name\t: AMD Ryzen 7 7700X 8-Core Processor\n"
        "cpu cores\t: 8\n",
        encoding="utf-8",
    )
    brand = common.cpu_brand_from_proc_cpuinfo(str(cpuinfo))
    assert brand is not None
    assert "7700X" in brand
    assert "Ryzen" in brand


def test_cpu_brand_from_proc_cpuinfo_missing(tmp_path):
    missing = tmp_path / "nope"
    assert common.cpu_brand_from_proc_cpuinfo(str(missing)) is None


def test_cpu_brand_from_platform_rejects_arch_only():
    # Native Linux without py-cpuinfo: processor() == machine() == "x86_64"
    assert common.cpu_brand_from_platform("x86_64", "x86_64") == "x86_64"
    assert common.cpu_brand_from_platform("Intel(R) Core(TM) i9-13900K", "x86_64").startswith(
        "Intel"
    )


def test_linux_cpu_brand_uses_proc_when_cpuinfo_pkg_missing(monkeypatch, tmp_path):
    from seiso.hardware.platforms import linux

    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "model name\t: AMD Ryzen 7 7700X 8-Core Processor\n",
        encoding="utf-8",
    )
    real_proc = common.cpu_brand_from_proc_cpuinfo
    monkeypatch.setattr(common, "cpu_brand_from_cpuinfo", lambda: None)
    monkeypatch.setattr(
        common,
        "cpu_brand_from_proc_cpuinfo",
        lambda path="/proc/cpuinfo": real_proc(str(cpuinfo)),
    )
    # Without /proc, platform.processor() on native Linux is often just arch.
    monkeypatch.setattr("seiso.hardware.platforms.linux.platform.processor", lambda: "x86_64")
    monkeypatch.setattr("seiso.hardware.platforms.linux.platform.machine", lambda: "x86_64")

    brand = linux.cpu_brand()
    assert "7700X" in brand


def test_cpu_temp_from_sensors_amd_k10temp():
    """AMD Zen exposes k10temp (Tctl), not Intel coretemp — must not be null."""
    temps = {
        "r8169_0_500:00": [SimpleNamespace(label="", current=41.0)],
        "nvme": [SimpleNamespace(label="Composite", current=37.85)],
        "k10temp": [
            SimpleNamespace(label="Tctl", current=38.75),
            SimpleNamespace(label="Tccd1", current=30.75),
        ],
        "amdgpu": [SimpleNamespace(label="edge", current=35.0)],
    }
    assert _cpu_temp_from_sensors(temps) == 38.8


def test_cpu_temp_from_sensors_intel_coretemp_package():
    temps = {
        "coretemp": [
            SimpleNamespace(label="Package id 0", current=52.0),
            SimpleNamespace(label="Core 0", current=48.0),
            SimpleNamespace(label="Core 1", current=49.0),
        ],
    }
    assert _cpu_temp_from_sensors(temps) == 52.0


def test_cpu_temp_from_sensors_ignores_non_cpu_chips():
    temps = {
        "nvme": [SimpleNamespace(label="Composite", current=40.0)],
        "amdgpu": [SimpleNamespace(label="edge", current=55.0)],
    }
    assert _cpu_temp_from_sensors(temps) is None
