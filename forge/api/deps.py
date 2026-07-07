"""Shared API dependencies."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from forge.config import get_settings
from forge.db.store import Database
from forge.orchestrators.base import Orchestrator
from forge.orchestrators.compress import CompressOrchestrator
from forge.orchestrators.distill_rl import DistillRLOrchestrator
from forge.orchestrators.export import ExportOrchestrator
from forge.orchestrators.hub_publish import HubPublishOrchestrator
from forge.orchestrators.inference import InferenceOrchestrator
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.orchestrators.recipes import RecipeOrchestrator
from forge.orchestrators.rl_quant import RLQuantOrchestrator
from forge.orchestrators.training import TrainingOrchestrator

_ORCHESTRATOR_GETTERS: list[Callable[[], Orchestrator]] = []


def _orchestrator_dep(cls: type[Orchestrator]) -> Callable[[], Orchestrator]:
    @lru_cache
    def _get() -> Orchestrator:
        from forge.services.job_events import DurableJobEventSink

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


get_training_orchestrator = _orchestrator_dep(TrainingOrchestrator)
get_export_orchestrator = _orchestrator_dep(ExportOrchestrator)
get_hub_publish_orchestrator = _orchestrator_dep(HubPublishOrchestrator)
get_inference_orchestrator = _orchestrator_dep(InferenceOrchestrator)
get_rl_quant_orchestrator = _orchestrator_dep(RLQuantOrchestrator)
get_compress_orchestrator = _orchestrator_dep(CompressOrchestrator)
get_distill_rl_orchestrator = _orchestrator_dep(DistillRLOrchestrator)
get_recipe_orchestrator = _orchestrator_dep(RecipeOrchestrator)
get_knowledge_orchestrator = _orchestrator_dep(KnowledgeOrchestrator)


def clear_dependency_caches() -> None:
    """Reset cached singletons — for tests and config reload."""
    get_settings.cache_clear()
    get_db.cache_clear()
    for getter in _ORCHESTRATOR_GETTERS:
        getter.cache_clear()


async def close_dependency_caches() -> None:
    """Close async resources before resetting cached singletons."""
    if get_db.cache_info().currsize:
        db = get_db()
        await db.close()
    clear_dependency_caches()
