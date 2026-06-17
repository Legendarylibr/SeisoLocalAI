"""Seiso Core — fast local model training, export, and inference utilities."""

__version__ = "0.1.0"

from seiso.export.formats import ExportFormat, export_checkpoint
from seiso.models.fast_model import FastModel, resolve_dtype
from seiso.models.loader import ModelKind, load_model
from seiso.training.config import TrainConfig

__all__ = [
    "__version__",
    "ModelKind",
    "load_model",
    "FastModel",
    "resolve_dtype",
    "TrainConfig",
    "ExportFormat",
    "export_checkpoint",
]
