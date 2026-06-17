"""Forge FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from forge.api.routes import auth, export, inference, knowledge, mcp_servers, models, openai, providers, recipes, training
from forge.api.routes import settings as settings_routes
from forge.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    settings.write_runtime_config()
    yield


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="Seiso Forge",
        description="Local AI platform API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if cfg.debug else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        from forge.security.auth import RateLimiter

        if not hasattr(app.state, "rate_limiter"):
            app.state.rate_limiter = RateLimiter(cfg.rate_limit)
        client = request.client.host if request.client else "unknown"
        if request.url.path not in ("/health", "/api/health", "/auth/status", "/api/auth/status"):
            app.state.rate_limiter.check(client)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    prefix = "/api"
    app.include_router(auth.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(inference.router, prefix=prefix)
    app.include_router(training.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)
    app.include_router(recipes.router, prefix=prefix)
    app.include_router(knowledge.router, prefix=prefix)
    app.include_router(providers.router, prefix=prefix)
    app.include_router(mcp_servers.router, prefix=prefix)
    app.include_router(settings_routes.router, prefix=prefix)
    app.include_router(openai.router)  # /v1/chat/completions — no /api prefix (OpenAI compat)

    @app.get("/health")
    async def root_health():
        return {"status": "ok"}

    # Serve built frontend when present
    ui_dist = Path(__file__).resolve().parent.parent / "forge-ui" / "dist"
    if ui_dist.exists():
        app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            if full_path.startswith("api"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            index = ui_dist / "index.html"
            if index.exists():
                return FileResponse(index)
            return JSONResponse({"detail": "UI not built"}, status_code=404)

    return app
