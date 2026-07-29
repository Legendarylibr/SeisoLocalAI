"""Pay marketplace FastAPI sidecar — public surface; Forge stays on localhost."""

from __future__ import annotations

import os
from typing import Any

from seiso.pay.ark import funding_instructions
from seiso.pay.flags import (
    faucet_enabled,
    forge_base_url,
    pay_allowed,
    protocol_fee_bps,
    protocol_treasury_ark,
    require_pay_allowed,
)
from seiso.pay.inference import debit_inference, estimate_tokens_from_messages
from seiso.pay.jobs import cancel_job, job_receipt, start_job
from seiso.pay.pricing import JOB_TYPES, quote_job
from seiso.pay.store import (
    activate_session,
    create_session,
    list_jobs,
    load_job,
    public_session_view,
    resolve_session_by_token,
)


def build_app():
    """Construct the marketplace ASGI app (requires fastapi extra)."""
    require_pay_allowed()
    try:
        import httpx
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "seiso pay serve requires the forge/fastapi extra "
            "(pip install 'seiso[forge]' or use the project venv)"
        ) from exc

    app = FastAPI(
        title="Seiso Pay Marketplace",
        version="0.1.0",
        description="Opt-in sats marketplace for remote inference / finetune / RL",
    )

    def _bearer_session(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Bearer seiso_pay_* token required")
        token = authorization.split(" ", 1)[1].strip()
        try:
            return resolve_session_by_token(token)
        except KeyError as exc:
            raise HTTPException(401, "Invalid pay token") from exc

    # Pre-bind Depends so defaults are not a call expression (ruff B008).
    _bearer_dep = Depends(_bearer_session)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "pay_allowed": pay_allowed(),
            "protocol_fee_bps": protocol_fee_bps(),
            "treasury_configured": bool(protocol_treasury_ark()),
            "faucet": faucet_enabled(),
        }

    @app.get("/.well-known/seiso-pay.json")
    def well_known() -> dict[str, Any]:
        return {
            "name": "seiso-pay",
            "version": "0.1",
            "protocol_fee_bps": protocol_fee_bps(),
            "scopes": ["inference", "finetune", "rl"],
            "job_types": sorted(JOB_TYPES - {"inference"}),
            "endpoints": {
                "sessions": "/pay/v1/sessions",
                "quotes": "/pay/v1/quotes",
                "jobs": "/pay/v1/jobs",
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
            },
            "forge_proxied": forge_base_url(),
        }

    @app.post("/pay/v1/sessions")
    async def create_pay_session(body: dict[str, Any] | None = None) -> dict[str, Any]:
        from seiso.pay.store import load_session

        body = body or {}
        scopes = body.get("scopes") or ["inference", "finetune", "rl"]
        if isinstance(scopes, str):
            scopes = [s.strip() for s in scopes.split(",")]
        amount = int(body.get("sats") or body.get("amount_sats") or 0)
        created = create_session(scopes=list(scopes))
        token = created["token"]
        session_id = created["session_id"]
        funding = funding_instructions(session_id, amount or 0)
        if amount > 0 and faucet_enabled():
            activate_session(session_id, amount_sats=amount, funding_mode="faucet")
        record = load_session(session_id)
        return {
            "token": token,
            "session": public_session_view(record),
            "funding": funding,
        }

    @app.get("/pay/v1/sessions/me")
    def session_me(session: dict[str, Any] = _bearer_dep) -> dict[str, Any]:
        return public_session_view(session)

    @app.post("/pay/v1/quotes")
    def quotes(body: dict[str, Any]) -> dict[str, Any]:
        jt = str(body.get("type") or body.get("job_type") or "").strip()
        if not jt:
            raise HTTPException(400, "type required")
        if jt == "inference":
            from seiso.pay.pricing import quote_inference_tokens

            return quote_inference_tokens(
                int(body.get("prompt_tokens") or 0),
                int(body.get("completion_tokens") or 0),
                flat_call=bool(body.get("flat_call")),
            )
        try:
            return quote_job(jt, preset=body.get("preset"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/pay/v1/jobs")
    def jobs_start(
        body: dict[str, Any],
        session: dict[str, Any] = _bearer_dep,
    ) -> dict[str, Any]:
        if session.get("status") != "active":
            raise HTTPException(402, "session not funded/active")
        jt = str(body.get("type") or body.get("job_type") or "").strip()
        scopes = set(session.get("scopes") or [])
        if jt == "finetune" and "finetune" not in scopes:
            raise HTTPException(403, "scope finetune required")
        if jt in {"slime", "distill_rl", "rl_quant", "nemo_rl"} and "rl" not in scopes:
            raise HTTPException(403, "scope rl required")
        dry_run = bool(body.get("dry_run"))
        try:
            job = start_job(
                session_id=str(session["session_id"]),
                job_type=jt,
                preset=body.get("preset"),
                config=body.get("config"),
                dry_run=dry_run,
            )
        except RuntimeError as exc:
            raise HTTPException(402, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"job": job, "receipt": job_receipt(job)}

    @app.get("/pay/v1/jobs")
    def jobs_list(session: dict[str, Any] = _bearer_dep) -> dict[str, Any]:
        return {"jobs": list_jobs(session_id=str(session["session_id"]))}

    @app.get("/pay/v1/jobs/{job_id}")
    def jobs_get(job_id: str, session: dict[str, Any] = _bearer_dep) -> dict[str, Any]:
        try:
            job = load_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        if job.get("session_id") != session.get("session_id"):
            raise HTTPException(404, "job not found")
        return {"job": job, "receipt": job_receipt(job)}

    @app.post("/pay/v1/jobs/{job_id}/cancel")
    def jobs_cancel(job_id: str, session: dict[str, Any] = _bearer_dep) -> dict[str, Any]:
        try:
            job = load_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        if job.get("session_id") != session.get("session_id"):
            raise HTTPException(404, "job not found")
        return {"job": cancel_job(job_id)}

    async def _proxy_chat(request: Request, session: dict) -> Any:
        if session.get("status") != "active":
            raise HTTPException(402, "session not funded/active")
        if "inference" not in set(session.get("scopes") or []):
            raise HTTPException(403, "scope inference required")
        body = await request.json()
        messages = body.get("messages") or []
        prompt_est = estimate_tokens_from_messages(messages)
        # Preflight minimum balance using flat call quote
        from seiso.pay.pricing import quote_inference_tokens

        pre = quote_inference_tokens(prompt_est, 16, flat_call=False)
        if int(session.get("balance_sats") or 0) < int(pre["total_sats"]):
            raise HTTPException(402, "insufficient balance for inference")

        key = (os.environ.get("SEISO_INFERENCE_API_KEY") or "").strip()
        if not key:
            # Best-effort read local key file
            from seiso.security import resolve_data_dir

            key_path = resolve_data_dir() / ".inference_api_key"
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise HTTPException(503, "operator inference key unavailable on marketplace host")

        base = forge_base_url()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        stream = bool(body.get("stream"))
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                req = client.build_request(
                    "POST",
                    f"{base}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                upstream = await client.send(req, stream=True)
                if upstream.status_code >= 400:
                    text = (await upstream.aread()).decode("utf-8", errors="replace")
                    raise HTTPException(upstream.status_code, text)

                async def gen():
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                    await upstream.aclose()
                    # Flat post-stream debit
                    debit_inference(
                        str(session["session_id"]),
                        prompt_tokens=prompt_est,
                        completion_tokens=int(body.get("max_tokens") or 64),
                        flat_call=False,
                    )

                return StreamingResponse(gen(), media_type="text/event-stream")

            resp = await client.post(f"{base}/v1/chat/completions", json=body, headers=headers)
            if resp.status_code >= 400:
                raise HTTPException(resp.status_code, resp.text)
            data = resp.json()
            usage = data.get("usage") or {}
            completion = int(
                usage.get("completion_tokens")
                or usage.get("completion_token_estimate")
                or body.get("max_tokens")
                or 64
            )
            prompt_toks = int(
                usage.get("prompt_tokens") or usage.get("prompt_token_estimate") or prompt_est
            )
            try:
                meter = debit_inference(
                    str(session["session_id"]),
                    prompt_tokens=prompt_toks,
                    completion_tokens=completion,
                )
            except RuntimeError as exc:
                raise HTTPException(402, str(exc)) from exc
            data["seiso_pay"] = meter
            return JSONResponse(data)

    @app.get("/v1/models")
    async def list_models(session: dict[str, Any] = _bearer_dep) -> Any:
        if "inference" not in set(session.get("scopes") or []):
            raise HTTPException(403, "scope inference required")
        key = (os.environ.get("SEISO_INFERENCE_API_KEY") or "").strip()
        from seiso.security import resolve_data_dir

        if not key:
            key_path = resolve_data_dir() / ".inference_api_key"
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8").strip()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{forge_base_url()}/v1/models",
                headers={"Authorization": f"Bearer {key}"} if key else {},
            )
            return JSONResponse(resp.json(), status_code=resp.status_code)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, session: dict[str, Any] = _bearer_dep) -> Any:
        return await _proxy_chat(request, session)

    return app


def run_server(host: str = "127.0.0.1", port: int = 8787) -> None:
    require_pay_allowed()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn required for seiso pay serve") from exc
    # Bind defaults to localhost; operators put TLS proxy in front for public.
    uvicorn.run(build_app(), host=host, port=port, log_level="info")
