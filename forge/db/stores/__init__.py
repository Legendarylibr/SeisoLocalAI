"""Domain store mixins for the SQLite persistence layer."""

from __future__ import annotations

from forge.db.stores.base import DatabaseCore, DatabaseCryptoError
from forge.db.stores.chat import ChatMixin
from forge.db.stores.compress import CompressMixin
from forge.db.stores.distill_rl import DistillRLMixin
from forge.db.stores.export import ExportMixin
from forge.db.stores.hub_publish import HubPublishMixin
from forge.db.stores.job_events import JobEventsMixin
from forge.db.stores.maintenance import MaintenanceMixin
from forge.db.stores.models import ModelsMixin
from forge.db.stores.providers import ProvidersMixin
from forge.db.stores.rl_quant import RLQuantMixin
from forge.db.stores.training import TrainingMixin
from forge.db.stores.users import UsersMixin


class Database(
    DatabaseCore,
    UsersMixin,
    ModelsMixin,
    TrainingMixin,
    ChatMixin,
    JobEventsMixin,
    ProvidersMixin,
    ExportMixin,
    HubPublishMixin,
    RLQuantMixin,
    CompressMixin,
    DistillRLMixin,
    MaintenanceMixin,
):
    """SQLite persistence facade composed from domain store mixins."""


__all__ = ["Database", "DatabaseCryptoError"]
