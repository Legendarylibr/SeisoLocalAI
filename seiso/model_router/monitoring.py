"""GPU / VRAM / RAM monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_utilization_pct: list[float] = field(default_factory=list)
    vram_used_mb: list[float] = field(default_factory=list)
    vram_total_mb: list[float] = field(default_factory=list)
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_available": self.gpu_available,
            "gpu_count": self.gpu_count,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "errors": self.errors,
        }

    def prometheus_lines(self) -> list[str]:
        lines: list[str] = []
        for i, util in enumerate(self.gpu_utilization_pct):
            lines.append(f'seiso_gpu_utilization_percent{{gpu="{i}"}} {util}')
        for i, used in enumerate(self.vram_used_mb):
            lines.append(f'seiso_vram_used_mb{{gpu="{i}"}} {used}')
        for i, total in enumerate(self.vram_total_mb):
            lines.append(f'seiso_vram_total_mb{{gpu="{i}"}} {total}')
        lines.append(f"seiso_ram_used_mb {self.ram_used_mb}")
        lines.append(f"seiso_ram_total_mb {self.ram_total_mb}")
        return lines


def collect_metrics() -> SystemMetrics:
    metrics = SystemMetrics()

    try:
        import psutil

        mem = psutil.virtual_memory()
        metrics.ram_used_mb = mem.used / (1024 * 1024)
        metrics.ram_total_mb = mem.total / (1024 * 1024)
    except ImportError:
        metrics.errors.append("psutil not installed")
    except Exception as exc:
        metrics.errors.append(f"ram: {exc}")

    try:
        import pynvml

        pynvml.nvmlInit()
        metrics.gpu_count = pynvml.nvmlDeviceGetCount()
        metrics.gpu_available = metrics.gpu_count > 0
        for i in range(metrics.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.gpu_utilization_pct.append(float(util.gpu))
            metrics.vram_used_mb.append(mem.used / (1024 * 1024))
            metrics.vram_total_mb.append(mem.total / (1024 * 1024))
    except ImportError:
        metrics.errors.append("pynvml not installed (optional GPU metrics)")
    except Exception as exc:
        metrics.errors.append(f"gpu: {exc}")

    return metrics
