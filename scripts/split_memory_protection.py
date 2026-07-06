#!/usr/bin/env python3
"""Split protection.py into seiso/memory/protection/ package."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "seiso/memory/protection.py"
PKG = ROOT / "seiso/memory/protection"

lines = SRC.read_text().splitlines(keepends=True)

SECTIONS = [
    ("constants.py", 1, 90),
    ("llama_batch.py", 91, 207),
    ("path_vram.py", 208, 325),
    ("oom.py", 326, 383),
    ("llama_kv.py", 384, 570),
    ("llama_runtime.py", 571, 1014),
    ("load_fit.py", 1015, 1261),
    ("chat_guards.py", 1262, 1464),
    ("llama_clamp.py", 1465, 1673),
    ("training_guards.py", 1674, 1820),
    ("rl_guards.py", 1821, 1881),
    ("device_map.py", 1882, len(lines)),
]

DOCS = {
    "constants.py": "Shared constants and types for memory protection.",
    "llama_batch.py": "llama.cpp batch tier caps and clamping.",
    "path_vram.py": "VRAM estimation from model paths.",
    "oom.py": "OOM detection and cache release.",
    "llama_kv.py": "KV cache sizing and offload headroom math.",
    "llama_runtime.py": "llama.cpp runtime profiles, prefill, and native Linux guards.",
    "load_fit.py": "Load preflight, headroom probes, and HF max_memory.",
    "chat_guards.py": "Chat payload trimming and sanitization.",
    "llama_clamp.py": "llama.cpp context and load-kwarg clamping.",
    "training_guards.py": "Training memory guardrails.",
    "rl_guards.py": "RL quant memory guardrails.",
    "device_map.py": "Training device map and dataset helpers.",
}

CROSS_IMPORTS = {
    "llama_batch.py": (
        "from seiso.memory.protection.constants import *  # noqa: F403\n",
    ),
    "path_vram.py": (
        "from seiso.memory.protection.constants import *  # noqa: F403\n",
    ),
    "oom.py": (
        "import gc\nimport os\n\nfrom seiso.memory.protection.constants import *  # noqa: F403\n",
    ),
    "llama_kv.py": (
        "import contextlib\nfrom pathlib import Path\n\n"
        "from seiso.env import env_bool\n"
        "from seiso.inference.backends import gguf_total_layers\n"
        "from seiso.memory.protection.constants import *  # noqa: F403\n"
        "from seiso.memory.protection.llama_batch import comfortable_vram_slack_ratio\n"
        "from seiso.memory.protection.path_vram import estimate_path_vram_mb\n",
    ),
    "llama_runtime.py": (
        "import contextlib\nfrom pathlib import Path\nfrom typing import Any\n\n"
        "from seiso import platform as seiso_platform\n"
        "from seiso.env import env_bool\n"
        "from seiso.hardware import hardware_profile\n"
        "from seiso.memory.protection.constants import *  # noqa: F403\n"
        "from seiso.memory.protection.llama_batch import (\n"
        "    clamp_llama_batch_pair,\n"
        "    discrete_gpu_total_mb,\n"
        "    gpu_batch_tier_caps,\n"
        "    resolve_llama_batch_limits,\n"
        "    tight_batch_caps,\n"
        ")\n"
        "from seiso.memory.protection.llama_kv import (\n"
        "    llama_batch_headroom_mb,\n"
        "    llama_effective_batch_headroom_mb,\n"
        "    llama_host_batch_headroom_mb,\n"
        "    llama_kv_cache_reserve_mb,\n"
        "    llama_model_is_tight_vram_fit,\n"
        "    llama_offload_fits_headroom,\n"
        ")\n"
        "from seiso.memory.protection.load_fit import headroom_mb\n",
    ),
    "load_fit.py": (
        "import logging\nimport platform\nfrom pathlib import Path\nfrom typing import Any\n\n"
        "from seiso import platform as seiso_platform\n"
        "from seiso.env import env_bool\n"
        "from seiso.hardware import assess_hardware_fit, hardware_profile, vram_headroom_mb\n"
        "from seiso.hardware.tiers import fit_headroom_mb\n"
        "from seiso.memory.protection.constants import *  # noqa: F403\n"
        "from seiso.memory.protection.oom import MemoryLoadBlockedError, allow_memory_overcommit\n"
        "from seiso.memory.protection.path_vram import estimate_path_vram_mb\n",
    ),
    "chat_guards.py": (
        "import contextlib\nimport json\nimport re\nfrom pathlib import Path\nfrom typing import Any\n\n"
        "from seiso.memory.protection.constants import *  # noqa: F403\n"
        "from seiso.memory.protection.load_fit import headroom_mb\n",
    ),
    "llama_clamp.py": (
        "import contextlib\nimport logging\nimport platform\nfrom pathlib import Path\n\n"
        "from seiso import platform as seiso_platform\n"
        "from seiso.env import env_bool\n"
        "from seiso.inference.backends import gguf_total_layers\n"
        "from seiso.memory.protection.constants import *  # noqa: F403\n"
        "from seiso.memory.protection.chat_guards import _estimate_prompt_tokens, _gguf_has_mmproj_sibling\n"
        "from seiso.memory.protection.llama_batch import clamp_llama_batch_pair\n"
        "from seiso.memory.protection.llama_kv import llama_model_is_tight_vram_fit\n"
        "from seiso.memory.protection.llama_runtime import (\n"
        "    llama_host_batch_headroom_mb,\n"
        "    native_linux_llama_context_cap,\n"
        "    resolve_llama_model_batches,\n"
        ")\n"
        "from seiso.memory.protection.load_fit import available_ram_mb, headroom_mb\n",
    ),
    "training_guards.py": (
        "import logging\nfrom typing import Any\n\n"
        "from seiso import platform as seiso_platform\n"
        "from seiso.env import env_bool\n"
        "from seiso.hardware import hardware_profile, training_defaults, vram_headroom_mb\n"
        "from seiso.memory.estimates import guess_params_from_name\n",
    ),
    "rl_guards.py": (
        "from typing import Any\n\n"
        "from seiso.env import env_bool\n"
        "from seiso.memory.protection.llama_clamp import clamp_llama_n_ctx\n"
        "from seiso.memory.protection.load_fit import headroom_mb\n",
    ),
    "device_map.py": (
        "import os\nfrom pathlib import Path\n\n"
        "from seiso.memory.protection.constants import _MAX_JSONL_LOAD_MB\n",
    ),
}

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)

for name, start, end in SECTIONS:
    body = "".join(lines[start - 1 : end])
    if name == "constants.py":
        (PKG / name).write_text(body)
        continue
    header = f'"""{DOCS[name]}"""\n\nfrom __future__ import annotations\n\n'
    extra = "".join(CROSS_IMPORTS.get(name, ()))
    (PKG / name).write_text(header + extra + "\n" + body)

mods = [s[0][:-3] for s in SECTIONS if s[0] != "constants.py"]
init = [
    '"""Cross-cutting OOM prevention — headroom probes, cache release, and fallbacks."""\n\n',
    "from __future__ import annotations\n\n",
    "from seiso.memory.protection.constants import *  # noqa: F403\n",
]
for mod in mods:
    init.append(f"from seiso.memory.protection.{mod} import *  # noqa: F403\n")
(PKG / "__init__.py").write_text("".join(init))

# Remove old module file
SRC.unlink()
print("Split complete:", PKG)
