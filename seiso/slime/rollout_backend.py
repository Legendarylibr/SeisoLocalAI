"""Online rollout backends for slime-style GRPO.

Facade module: implementation lives in ``rollout_resolve``, ``rollout_generate``,
``rollout_clients``, ``rollout_http``, and ``rollout_sync``. Import from here for
stability (tests and trainers patch ``seiso.slime.rollout_backend.*``).

* ``hf`` / ``data_gen`` — colocated Hugging Face ``generate`` (default; single-GPU)
* ``sglang`` — OpenAI-compatible HTTP generation against a running SGLang server
* ``vllm`` — OpenAI-compatible HTTP generation against a running vLLM server
"""

from __future__ import annotations

# Ensure private helpers used by tests remain available under this module path.
import seiso.slime.rollout_clients as _clients
import seiso.slime.rollout_generate as _generate
import seiso.slime.rollout_http as _http
import seiso.slime.rollout_resolve as _resolve
import seiso.slime.rollout_sync as _sync
from seiso.slime.rollout_clients import *  # noqa: F403
from seiso.slime.rollout_generate import *  # noqa: F403
from seiso.slime.rollout_http import *  # noqa: F403
from seiso.slime.rollout_resolve import *  # noqa: F403
from seiso.slime.rollout_sync import *  # noqa: F403

for _mod in (_resolve, _generate, _clients, _http, _sync):
    globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})

__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
    or name
    in {
        "_extract_completion_token_ids",
        "_prune_weight_versions",
        "_generate_http_chunk",
        "_normalize_backend_name",
        "_http_json_request",
    }
)
