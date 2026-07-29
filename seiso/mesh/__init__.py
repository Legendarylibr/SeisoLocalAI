"""Experimental Buzz-agent multi-node training mesh.

Opt-in via ``SEISO_ALLOW_MESH=1`` **and** a Buzz agent identity
(``BUZZ_PRIVATE_KEY`` / ``BUZZ_AUTH_TAG``). Reciprocal peers — no marketplace
protocol fee. Not available from the Forge UI frontend surface.
Not functional yet — do not use for real multi-node jobs.
"""

from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

__all__ = ["mesh_allowed", "require_mesh_allowed"]
