"""NVIDIA secure boundary and WSL2 platform tests."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from seiso.kernels.platform import detect_gpu
from seiso.security.nvidia_boundary import (
    detect_wsl2,
    enforce_nvidia_secure_boundary,
    nvidia_boundary_report,
)


def test_detect_wsl2_env():
    with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=False):
        assert detect_wsl2() is True


def test_nvidia_boundary_report_keys():
    report = nvidia_boundary_report()
    assert "linux_nvidia_host" in report
    assert "wsl2" in report
    assert "approved_tier" in report


def test_enforce_boundary_skips_non_linux():
    with patch("seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=False):
        report = enforce_nvidia_secure_boundary(context="test")
    assert report["linux_nvidia_host"] is False


def test_enforce_boundary_requires_ack_on_linux_nvidia():
    with patch("seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=True):
        with patch("seiso.security.nvidia_boundary.in_ci", return_value=False):
            with pytest.raises(SystemExit):
                enforce_nvidia_secure_boundary(context="test")


def test_enforce_boundary_wsl2_ack():
    env = {
        "SEISO_NVIDIA_WSL_ACK": "1",
        "WSL_DISTRO_NAME": "Ubuntu",
    }
    with patch("seiso.security.nvidia_boundary.is_linux_nvidia_host", return_value=True):
        with patch("seiso.security.nvidia_boundary.in_ci", return_value=False):
            with patch.dict(os.environ, env, clear=False):
                report = enforce_nvidia_secure_boundary(context="test")
    assert report.get("approved_tier") == "wsl2"


def test_gpu_platform_wsl2_field():
    platform = detect_gpu()
    assert hasattr(platform, "is_wsl2")
    assert isinstance(platform.is_wsl2, bool)
