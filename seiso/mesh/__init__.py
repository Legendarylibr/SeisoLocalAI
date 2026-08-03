"""Experimental Buzz-agent multi-node training mesh (secondary path).

Opt-in via ``SEISO_ALLOW_MESH=1`` **and** a valid Buzz agent
``BUZZ_PRIVATE_KEY`` nsec (BIP-340 signing). ``BUZZ_AUTH_TAG`` alone cannot
sign mesh plans. Reciprocal peers — no marketplace protocol fee. Not available
from the Forge UI frontend surface. Local single-node Forge/CLI training remains
the primary path; mesh is agent-only secondary coordination.
"""

from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

__all__ = ["mesh_allowed", "require_mesh_allowed"]
