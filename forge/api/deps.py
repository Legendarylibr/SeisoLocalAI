"""Shared API dependencies."""

from __future__ import annotations

from functools import lru_cache

from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.export import ExportOrchestrator
from forge.orchestrators.inference import InferenceOrchestrator
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.orchestrators.recipes import RecipeOrchestrator
from forge.orchestrators.training import TrainingOrchestrator


@lru_cache
def get_db() -> Database:
    settings = get_settings()
    return Database(settings.db_path)


@lru_cache
def get_training_orchestrator() -> TrainingOrchestrator:
    return TrainingOrchestrator(get_settings().data_dir)


@lru_cache
def get_export_orchestrator() -> ExportOrchestrator:
    return ExportOrchestrator(get_settings().data_dir)


@lru_cache
def get_inference_orchestrator() -> InferenceOrchestrator:
    return InferenceOrchestrator(get_settings().data_dir)


@lru_cache
def get_recipe_orchestrator() -> RecipeOrchestrator:
    return RecipeOrchestrator(get_settings().data_dir)


@lru_cache
def get_knowledge_orchestrator() -> KnowledgeOrchestrator:
    return KnowledgeOrchestrator(get_settings().data_dir)


def clear_dependency_caches() -> None:
    """Reset cached singletons — for tests and config reload."""
    get_settings.cache_clear()
    get_db.cache_clear()
    get_training_orchestrator.cache_clear()
    get_export_orchestrator.cache_clear()
    get_inference_orchestrator.cache_clear()
    get_recipe_orchestrator.cache_clear()
    get_knowledge_orchestrator.cache_clear()
