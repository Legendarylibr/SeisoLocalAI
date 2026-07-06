"""Tests for hardware probe providers."""

from seiso.hardware.probes.common import GpuMemoryProcess, sanitize_hardware_label
from seiso.hardware.probes.nvidia import parse_nvidia_smi_process_csv


def test_sanitize_hardware_label_strips_host():
    assert "@" not in sanitize_hardware_label("GPU @host.local")


def test_parse_nvidia_smi_process_csv():
    stdout = "60526, /app/.venv/bin/python3, 8334\n5874, firefox, 188"
    parsed = parse_nvidia_smi_process_csv(stdout)
    assert len(parsed) == 2
    assert parsed[0] == GpuMemoryProcess(
        pid=60526, process_name="/app/.venv/bin/python3", used_mb=8334
    )
    assert parsed[1].pid == 5874
