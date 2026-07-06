#!/usr/bin/env python3
"""Split model_pool.py into seiso/inference/model_pool/ package."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "seiso/inference/model_pool.py"
PKG = ROOT / "seiso/inference/model_pool"

lines = SRC.read_text().splitlines(keepends=True)

# Shared header for llama_load
LLAMA_HEADER = '''"""llama.cpp load heuristics, kwargs, and model loading."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

from seiso.compat import StrEnum
from seiso.env import env_bool, env_int
from seiso.inference.backends import gguf_total_layers

logger = logging.getLogger(__name__)

'''

POOL_HEADER = '''"""VRAM-managed singleton model pool."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seiso.compat import StrEnum
from seiso.env import env_bool, env_int
from seiso.inference.model_pool.dflash import clear_dflash_draft_cache
from seiso.inference.model_pool.llama_load import (
    _clear_optimal_layers_cache,
    _default_llama_gpu_layers,
    _llama_cache_headroom_ok,
    _llama_cache_is_optimal,
    _refresh_headroom_stats,
    env_int,
)

logger = logging.getLogger(__name__)

'''

DFLASH_HEADER = '''"""DFlash / draft model cache for speculative decoding."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

'''

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)

# llama_load: lines 21-828
(PKG / "llama_load.py").write_text(LLAMA_HEADER + "".join(lines[20:828]))

# pool: lines 831-1446
(PKG / "pool.py").write_text(POOL_HEADER + "".join(lines[830:1446]))

# dflash: lines 1448-1561
(PKG / "dflash.py").write_text(
    DFLASH_HEADER
    + "from seiso.inference.model_pool.llama_load import _load_llama_model\n\n"
    + "".join(lines[1447:1561])
)

# Fix pool.py - remove duplicate env_int import, add llama_load imports properly
pool_text = (PKG / "pool.py").read_text()
pool_text = pool_text.replace(
    "from seiso.inference.model_pool.llama_load import (\n"
    "    _clear_optimal_layers_cache,\n"
    "    _default_llama_gpu_layers,\n"
    "    _llama_cache_headroom_ok,\n"
    "    _llama_cache_is_optimal,\n"
    "    _refresh_headroom_stats,\n"
    "    env_int,\n)\n",
    "from seiso.env import env_int\n"
    "from seiso.inference.model_pool.llama_load import (\n"
    "    _clear_optimal_layers_cache,\n"
    "    _default_llama_gpu_layers,\n"
    "    _llama_cache_headroom_ok,\n"
    "    _llama_cache_is_optimal,\n"
    "    _refresh_headroom_stats,\n"
    ")\n",
)
(PKG / "pool.py").write_text(pool_text)

init = '''"""VRAM-managed model pool — unloads previous model when switching."""

from __future__ import annotations

from seiso.inference.model_pool.dflash import (
    DflashDraftHandle,
    clear_dflash_draft_cache,
    dflash_draft_infer,
    get_dflash_draft,
)
from seiso.inference.model_pool.llama_load import (
    _available_cpu_count,
    _cuda_available,
    _default_llama_gpu_layers,
    _llama_batch_defaults,
    _llama_full_gpu_targets,
    _llama_gpu_layers_optimal,
    _llama_gpu_offload_ok,
    _llama_layer_attempts,
    _llama_load_retryable,
    _llama_offload_checked,
    _llama_offload_supported,
    _llama_partial_kqv_options,
    _llama_partial_memory_profiles,
    _llama_skip_partial_offload,
    _load_llama_model,
    _native_linux_nvidia,
    _nvidia_hardware_visible,
    fit_llama_gpu_layers,
    llama_load_kwargs,
    reset_llama_gpu_offload_cache,
)
from seiso.inference.model_pool.pool import (
    BackendKind,
    LoadedModel,
    ModelPool,
    get_model_pool,
)

__all__ = [
    "BackendKind",
    "DflashDraftHandle",
    "LoadedModel",
    "ModelPool",
    "clear_dflash_draft_cache",
    "dflash_draft_infer",
    "fit_llama_gpu_layers",
    "get_dflash_draft",
    "get_model_pool",
    "llama_load_kwargs",
    "reset_llama_gpu_offload_cache",
]
'''
(PKG / "__init__.py").write_text(init)
SRC.unlink()
print("Split complete:", PKG)
