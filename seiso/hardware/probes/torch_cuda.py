"""PyTorch CUDA GPU probing."""

from __future__ import annotations

import logging
from typing import Any

from seiso.hardware.probes.common import sanitize_hardware_label

logger = logging.getLogger(__name__)


def probe_torch_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return gpus
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = sanitize_hardware_label(props.name)
            total_mb = int(props.total_memory / (1024**2))
            used_mb: int | None = None
            try:
                free, total = torch.cuda.mem_get_info(i)
                used_mb = int((total - free) / (1024**2))
            except Exception:
                pass
            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "vram_total_mb": total_mb,
                    "vram_used_mb": used_mb,
                    "utilization_pct": None,
                    "temperature_c": None,
                }
            )
    except ImportError:
        pass
    except Exception:
        # A broken CUDA runtime (e.g. the cu12/cu13 driver split) makes
        # torch.cuda.* raise RuntimeError. Swallow it so enumerate_gpus() can
        # fall through to the nvidia-smi probe instead of erasing GPU detection
        # (which would silently disable the native-Linux inference guards).
        logger.debug("torch CUDA GPU probe failed; deferring to nvidia-smi", exc_info=True)
    return gpus
