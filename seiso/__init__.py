"""Seiso Core — fast local model training, export, and inference utilities.

Public re-exports for library consumers (``pip install seiso``). Forge, CLI,
and tests import submodules directly (``from seiso.training.config import …``);
those paths are stable and preferred for in-repo code.
"""

__version__ = "0.1.0"

from seiso.export.formats import ExportFormat, export_checkpoint
from seiso.models.loader import ModelKind, load_model
from seiso.models.seiso_model import SeisoModel, resolve_dtype
from seiso.training.config import TrainConfig

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
