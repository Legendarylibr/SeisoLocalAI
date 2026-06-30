"""Setuptools build entry point with optional pybind11 hot-path extensions."""

from __future__ import annotations

from setuptools import Extension, setup


def _compile_args() -> list[str]:
    import sysconfig

    compiler = (sysconfig.get_config_var("CC") or "").lower()
    if "msvc" in compiler or "cl.exe" in compiler:
        return ["/std:c++17"]
    return ["-std=c++17"]


def _extensions() -> list[Extension]:
    try:
        import pybind11
    except ImportError:
        return []

    return [
        Extension(
            "seiso.adaptive_quant.native._math_ext",
            ["seiso/adaptive_quant/native/math_ext.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=_compile_args(),
        )
    ]


setup(ext_modules=_extensions())
