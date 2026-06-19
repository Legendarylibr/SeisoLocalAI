"""Shared API dependencies."""

from __future__ import annotations

from functools import lru_cache

from forge.config import get_settings
from forge.db.store import Database
from forge.orchestrators.compress import CompressOrchestrator
from forge.orchestrators.export import ExportOrchestrator
from forge.orchestrators.image_compress import ImageCompressOrchestrator
from forge.orchestrators.inference import InferenceOrchestrator
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.orchestrators.recipes import RecipeOrchestrator
from forge.orchestrators.rl_quant import RLQuantOrchestrator
from forge.orchestrators.training import TrainingOrchestrator


@lru_cache
def get_db() -> Database:
    settings = get_settings()
    return Database(
        settings.db_path,
        encryption_key=settings.db_encryption_key_bytes,
        ephemeral=bool(settings.db_ephemeral),
    )


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
def get_rl_quant_orchestrator() -> RLQuantOrchestrator:
    return RLQuantOrchestrator(get_settings().data_dir)


@lru_cache
def get_compress_orchestrator() -> CompressOrchestrator:
    return CompressOrchestrator(get_settings().data_dir)


@lru_cache
def get_image_compress_orchestrator() -> ImageCompressOrchestrator:
    return ImageCompressOrchestrator(get_settings().data_dir)


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
    get_rl_quant_orchestrator.cache_clear()
    get_compress_orchestrator.cache_clear()
    get_image_compress_orchestrator.cache_clear()
    get_inference_orchestrator.cache_clear()
    get_recipe_orchestrator.cache_clear()
    get_knowledge_orchestrator.cache_clear()


async def close_dependency_caches() -> None:
    """Close async resources before resetting cached singletons."""
    if get_db.cache_info().currsize:
        db = get_db()
        await db.close()
    clear_dependency_caches()
