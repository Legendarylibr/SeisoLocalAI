"""Compatibility shim for editable installs and CUDA extension builds.

Some build tools (including flash-attn's setuptools path) expect a setup.py next
to pyproject.toml. Hatchling remains the build backend; this file delegates to it.
"""

from setuptools import setup

setup()
