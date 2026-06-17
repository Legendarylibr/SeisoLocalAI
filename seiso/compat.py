"""Python-version compatibility helpers used across Seiso.

The project supports Python >=3.10, but several modules want enum.StrEnum,
which was added in 3.11. This module provides a tiny backport so callers
can simply import ``StrEnum`` from here.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Backport of ``enum.StrEnum`` for Python < 3.11.

    Functionally identical for the patterns used by Seiso (comparisons
    go through ``.value``). Using ``str, Enum`` also works on 3.10.
    """

    __str__ = str.__str__
