"""Compatibility re-export — use ``seiso.slime.config``."""

from __future__ import annotations

import seiso.slime.config as _impl
from seiso.slime.config import *  # noqa: F403

# Re-export private helpers so legacy ``from seiso.slime_single_gpu... import _foo`` works.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})
