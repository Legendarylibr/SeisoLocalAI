"""Router service entrypoint with prod middleware."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from seiso.model_router.app import create_router_app
from seiso.model_router.config import RouterSettings, resolve_paths
from seiso.model_router.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)


def build_app(settings: RouterSettings | None = None):
    if settings is None:
        config_path = os.environ.get("SEISO_ROUTER_CONFIG_PATH")
        settings = RouterSettings.load(Path(config_path)) if config_path else RouterSettings()

    resolved = resolve_paths(settings)
    app = create_router_app(resolved)

    if resolved.mode == "prod":
        app.add_middleware(RequestLoggingMiddleware, json_logs=resolved.log_json)
        if resolved.rate_limit_rpm > 0:
            app.add_middleware(
                RateLimitMiddleware,
                rpm=resolved.rate_limit_rpm,
                burst=resolved.rate_limit_burst,
            )
        if resolved.api_keys:
            app.add_middleware(APIKeyMiddleware, api_keys=resolved.api_keys)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return app
