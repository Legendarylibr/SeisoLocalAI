"""Forge FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from forge.api.deps import clear_dependency_caches, get_db
from forge.api.routes import (
    auth,
    compress,
    export,
    image_compress,
    inference,
    knowledge,
    models,
    openai,
    providers,
    recipes,
    rl_quant,
    system,
    training,
)
from forge.api.routes import settings as settings_routes
from forge.config import get_settings
from forge.db.store import DatabaseCryptoError
from seiso.models.hf_env import configure_hf_hub_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_hf_hub_cache(settings.data_dir)
    settings.ensure_dirs()
    settings.write_runtime_config()
    db = get_db()
    stale = await db.reconcile_stale_jobs()
    if stale:
        import logging

        logging.getLogger(__name__).info("Marked %d stale job(s) as failed after restart", stale)
    yield
    await db.close()
    clear_dependency_caches()


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="Seiso",
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
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )

    @app.exception_handler(DatabaseCryptoError)
    async def database_crypto_error(_request: Request, _exc: DatabaseCryptoError):
        return JSONResponse({"detail": "Encrypted local data could not be read"}, status_code=500)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        from forge.security.auth import RateLimiter
        from forge.security.client_ip import client_ip
        from forge.security.csrf import validate_csrf

        settings = get_settings()
        if settings.rate_limit_enabled:
            if not hasattr(app.state, "rate_limiter"):
                app.state.rate_limiter = RateLimiter(settings.rate_limit)
            client = client_ip(request)
            if request.url.path not in ("/health", "/api/health", "/auth/status", "/api/auth/status"):
                try:
                    app.state.rate_limiter.check(client)
                except HTTPException as exc:
                    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        if not validate_csrf(request):
            return JSONResponse({"detail": "CSRF validation failed"}, status_code=403)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "base-uri 'self'; "
            "object-src 'none'; "
            "form-action 'self'; "
            "connect-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; "
            "script-src-elem 'self'; "
            "script-src-attr 'none'; "
            "img-src 'self' data: blob:; "
            "font-src 'self'; "
            "worker-src 'self' blob:; "
            "frame-ancestors 'none'"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/assets/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    prefix = "/api"
    app.include_router(auth.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(inference.router, prefix=prefix)
    app.include_router(training.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)
    app.include_router(rl_quant.router, prefix=prefix)
    app.include_router(compress.router, prefix=prefix)
    app.include_router(image_compress.router, prefix=prefix)
    app.include_router(recipes.router, prefix=prefix)
    app.include_router(knowledge.router, prefix=prefix)
    app.include_router(providers.router, prefix=prefix)
    app.include_router(system.router, prefix=prefix)
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
                return FileResponse(index, headers={"Cache-Control": "no-cache"})
            return JSONResponse({"detail": "UI not built"}, status_code=404)

    return app
