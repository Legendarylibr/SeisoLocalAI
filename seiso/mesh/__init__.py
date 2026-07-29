"""Experimental Buzz-coordinated multi-node training mesh.

Opt-in via ``SEISO_ALLOW_MESH=1``. Reciprocal peers — no marketplace protocol fee.
"""

from __future__ import annotations

from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

__all__ = ["mesh_allowed", "require_mesh_allowed"]
