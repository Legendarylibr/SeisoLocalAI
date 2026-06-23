from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from litellm.integrations.custom_logger import CustomLogger

from seiso.model_router.catalog import SpecialistCatalog
from seiso.model_router.config import RouterSettings, resolve_paths
from seiso.model_router.nemotron import select_route_via_nemotron

logger = logging.getLogger(__name__)

_catalog: SpecialistCatalog | None = None
_settings: RouterSettings | None = None
_http: httpx.AsyncClient | None = None


def _load_context() -> tuple[RouterSettings, SpecialistCatalog, httpx.AsyncClient]:
    global _catalog, _settings, _http
    if _settings is None:
        config_path = os.environ.get("SEISO_ROUTER_CONFIG_PATH")
        if config_path:
            _settings = resolve_paths(RouterSettings.load(path=Path(config_path)))
        else:
            _settings = resolve_paths(RouterSettings())
    if _catalog is None:
        _catalog = SpecialistCatalog.from_json(_settings.specialists_path)
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(_settings.request_timeout_sec))
    return _settings, _catalog, _http


class NemotronRouterCallback(CustomLogger):
    """LiteLLM async_pre_call_hook: Nemotron selects specialist model_name."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> dict[str, Any]:
        settings, catalog, client = _load_context()
        if not settings.nemotron_orchestrator_enabled():
            return data

        messages = data.get("messages") or []
        try:
            selection = await select_route_via_nemotron(
                client,
                orchestrator_url=settings.orchestrator_url,
                orchestrator_model=settings.orchestrator_model,
                catalog=catalog,
                messages=messages,
                fallback_route_id=settings.fallback_route_id,
                timeout=settings.orchestrator_timeout_sec,
                temperature=settings.orchestrator_temperature,
                max_tokens=settings.orchestrator_max_tokens,
            )
            data["model"] = selection.route.llamaswap_model
            data.setdefault("metadata", {})
            if isinstance(data["metadata"], dict):
                data["metadata"]["seiso_nemotron_alias"] = selection.orchestrator_alias
                data["metadata"]["seiso_route_id"] = selection.route.route_id
            logger.info(
                "Nemotron routed to %s (%s)",
                selection.route.llamaswap_model,
                selection.reasoning,
            )
        except Exception as exc:
            logger.warning("Nemotron pre-call hook failed: %s", exc)
        return data


nemotron_router_callback = NemotronRouterCallback()
