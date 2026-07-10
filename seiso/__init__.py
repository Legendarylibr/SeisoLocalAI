"""Seiso Core — fast local model training, export, and inference utilities.

Public re-exports for library consumers (``pip install seiso``). Forge, CLI,
and tests import submodules directly (``from seiso.training.config import …``);
those paths are stable and preferred for in-repo code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ModelKind",
    "load_model",
    "SeisoModel",
    "resolve_dtype",
    "TrainConfig",
    "ExportFormat",
    "export_checkpoint",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ExportFormat": ("seiso.export.formats", "ExportFormat"),
    "export_checkpoint": ("seiso.export.formats", "export_checkpoint"),
    "ModelKind": ("seiso.models.loader", "ModelKind"),
    "load_model": ("seiso.models.loader", "load_model"),
    "SeisoModel": ("seiso.models.seiso_model", "SeisoModel"),
    "resolve_dtype": ("seiso.models.seiso_model", "resolve_dtype"),
    "TrainConfig": ("seiso.training.config", "TrainConfig"),
}

if TYPE_CHECKING:
    from seiso.export.formats import ExportFormat, export_checkpoint
    from seiso.models.loader import ModelKind, load_model
    from seiso.models.seiso_model import SeisoModel, resolve_dtype
    from seiso.training.config import TrainConfig


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = target
    import importlib

    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
