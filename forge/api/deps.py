"""Shared API dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from forge.config import get_settings
from forge.db.store import Database

if TYPE_CHECKING:
    from forge.orchestrators.base import Orchestrator

_ORCHESTRATOR_GETTERS: list[Callable[[], Orchestrator]] = []

_ORCHESTRATOR_SPECS: dict[str, tuple[str, str]] = {
    "training": ("forge.orchestrators.training", "TrainingOrchestrator"),
    "export": ("forge.orchestrators.export", "ExportOrchestrator"),
    "hub_publish": ("forge.orchestrators.hub_publish", "HubPublishOrchestrator"),
    "inference": ("forge.orchestrators.inference", "InferenceOrchestrator"),
    "compress": ("forge.orchestrators.compress", "CompressOrchestrator"),
    "distill_rl": ("forge.orchestrators.distill_rl", "DistillRLOrchestrator"),
    "recipes": ("forge.orchestrators.recipes", "RecipeOrchestrator"),
    "knowledge": ("forge.orchestrators.knowledge", "KnowledgeOrchestrator"),
}


def _orchestrator_dep(module_path: str, cls_name: str) -> Callable[[], Orchestrator]:
    @lru_cache
    def _get() -> Orchestrator:
        from forge.services.job_events import DurableJobEventSink

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        orchestrator = cls(get_settings().data_dir)
        orchestrator.set_event_sink(DurableJobEventSink(get_db()))
        return orchestrator

    _ORCHESTRATOR_GETTERS.append(_get)
    return _get


@lru_cache
def get_db() -> Database:
    settings = get_settings()
    return Database(
        settings.db_path,
        encryption_key=settings.db_encryption_key_bytes,
        ephemeral=bool(settings.db_ephemeral),
    )


get_training_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["training"])
get_export_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["export"])
get_hub_publish_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["hub_publish"])
get_inference_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["inference"])
get_compress_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["compress"])
get_distill_rl_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["distill_rl"])
get_recipe_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["recipes"])
get_knowledge_orchestrator = _orchestrator_dep(*_ORCHESTRATOR_SPECS["knowledge"])


def clear_dependency_caches() -> None:
    """Reset cached singletons — for tests and config reload."""
    get_settings.cache_clear()
    get_db.cache_clear()
    for getter in _ORCHESTRATOR_GETTERS:
        getter.cache_clear()


async def close_dependency_caches() -> bool:
    """Close async resources before resetting cached singletons."""
    had_database = bool(get_db.cache_info().currsize)
    if had_database:
        db = get_db()
        await db.close()
    clear_dependency_caches()
    return had_database
