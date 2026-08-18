"""Pay marketplace FastAPI sidecar — public surface; Forge stays on localhost."""

from __future__ import annotations

import os
from typing import Any

from seiso.pay.ark import funding_instructions
from seiso.pay.flags import (
    faucet_enabled,
    forge_base_url,
    pay_allowed,
    payment_methods,
    protocol_fee_bps,
    protocol_treasury_ark,
    require_pay_allowed,
)
from seiso.pay.inference import debit_inference, estimate_tokens_from_messages
from seiso.pay.jobs import cancel_job, job_receipt, start_job
from seiso.pay.l402 import (
    REFERENCE_URL,
    complete_fund,
    l402_sim_enabled,
    mint_fund_challenge,
)
from seiso.pay.pricing import JOB_TYPES, quote_job
from seiso.pay.store import (
    activate_session,
    create_session,
    list_jobs,
    load_job,
    public_session_view,
    resolve_session_by_token,
)
from seiso.pay.x402 import (
    REFERENCE_URL as X402_REFERENCE_URL,
)
from seiso.pay.x402 import (
    complete_fund as complete_x402_fund,
)
from seiso.pay.x402 import (
    mint_fund_challenge as mint_x402_challenge,
)
from seiso.pay.x402 import (
    x402_sim_enabled,
)

try:
    from starlette.requests import Request as ASGIRequest
except ImportError:  # pragma: no cover
    ASGIRequest = object  # type: ignore[misc,assignment]


def build_app():
    """Construct the marketplace ASGI app (requires fastapi extra)."""
    require_pay_allowed()
    try:
        import httpx
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "seiso pay serve requires the forge/fastapi extra "
            "(pip install 'seiso[forge]' or use the project venv)"
        ) from exc

    app = FastAPI(
        title="Seiso Pay Marketplace",
        version="0.1.0",
        description=(
            "Experimental opt-in sats marketplace for remote inference / "
            "finetune / RL (SEISO_ALLOW_PAY=1; not functional for real funds yet)"
        ),
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

    def _optional_session(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any] | None:
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        token = authorization.split(" ", 1)[1].strip()
        if token.startswith("seiso_pay_"):
            try:
                return resolve_session_by_token(token)
            except KeyError as exc:
                raise HTTPException(401, "Invalid pay token") from exc
        return None

    # Pre-bind Depends so defaults are not a call expression (ruff B008).
    _bearer_dep = Depends(_bearer_session)
    _optional_session_dep = Depends(_optional_session)

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
            "payment_methods": payment_methods(),
            "payment_methods_note": (
                "Live Ark, live Lightning L402, and live x402 EVM are not "
                "functional yet — do not use for real funds. "
                "Faucet, SEISO_PAY_L402_SIM, and SEISO_PAY_X402_SIM credit "
                "sessions for smoke tests. "
                f"L402: {REFERENCE_URL} "
                f"x402: {X402_REFERENCE_URL}"
            ),
            "endpoints": {
                "sessions": "/pay/v1/sessions",
                "fund_l402": "/pay/v1/sessions/fund/l402",
                "fund_l402_complete": "/pay/v1/sessions/fund/l402/complete",
                "fund_x402": "/pay/v1/sessions/fund/x402",
                "fund_x402_complete": "/pay/v1/sessions/fund/x402/complete",
                "per_request": "/pay/v1/requests",
                "quotes": "/pay/v1/quotes",
                "jobs": "/pay/v1/jobs",
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
            },
            "l402_sim": l402_sim_enabled(),
            "x402_sim": x402_sim_enabled(),
            "per_request": True,
            "assets": ["sats", "usdc", "eth"],
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

    @app.post("/pay/v1/sessions/fund/l402")
    def fund_l402_challenge(
        body: dict[str, Any] | None = None,
        session: dict[str, Any] = _bearer_dep,
    ) -> JSONResponse:
        """Mint an L402 challenge for session top-up (sim until live LN wired).

        Requires the session Bearer token so sim_preimage cannot top up an
        arbitrary session_id guessed from the URL/body.
        """
        body = body or {}
        session_id = str(session["session_id"])
        body_sid = str(body.get("session_id") or "").strip()
        if body_sid and body_sid != session_id:
            raise HTTPException(403, "session_id does not match Bearer session")
        amount = int(body.get("sats") or body.get("amount_sats") or 0)
        if amount <= 0:
            raise HTTPException(400, "sats / amount_sats must be > 0")
        try:
            challenge = mint_fund_challenge(
                session_id=session_id,
                amount_sats=amount,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        headers = {"WWW-Authenticate": str(challenge["www_authenticate"])}
        return JSONResponse(challenge, status_code=402, headers=headers)

    @app.post("/pay/v1/sessions/fund/l402/complete")
    async def fund_l402_complete(
        body: dict[str, Any] | None = None,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Exchange L402 macaroon:preimage for session credit (+ Bearer path unchanged)."""
        body = body or {}
        auth = authorization
        if not auth:
            # Prefer header; allow JSON body for CLI/sim without custom header plumbing.
            mac = body.get("macaroon")
            pre = body.get("preimage") or body.get("sim_preimage")
            if mac and pre:
                auth = f"L402 {mac}:{pre}"
            elif body.get("authorization"):
                auth = str(body["authorization"])
        try:
            result = complete_fund(authorization=auth)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return result

    @app.post("/pay/v1/sessions/fund/x402")
    def fund_x402_challenge(
        body: dict[str, Any] | None = None,
        session: dict[str, Any] = _bearer_dep,
    ) -> JSONResponse:
        """Mint an x402 EVM challenge for session top-up (sim until live USDC wired)."""
        body = body or {}
        session_id = str(session["session_id"])
        body_sid = str(body.get("session_id") or "").strip()
        if body_sid and body_sid != session_id:
            raise HTTPException(403, "session_id does not match Bearer session")
        amount = int(body.get("sats") or body.get("amount_sats") or 0)
        if amount <= 0:
            raise HTTPException(400, "sats / amount_sats must be > 0")
        try:
            challenge = mint_x402_challenge(
                session_id=session_id,
                amount_sats=amount,
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        headers = {
            "WWW-Authenticate": str(challenge["www_authenticate"]),
            "PAYMENT-REQUIRED": str(challenge["payment_required_header"]),
        }
        return JSONResponse(challenge, status_code=402, headers=headers)

    @app.post("/pay/v1/sessions/fund/x402/complete")
    async def fund_x402_complete(
        body: dict[str, Any] | None = None,
        payment_signature: str | None = Header(default=None, alias="PAYMENT-SIGNATURE"),
    ) -> dict[str, Any]:
        """Exchange PAYMENT-SIGNATURE for session credit (+ Bearer path unchanged)."""
        body = body or {}
        sig = payment_signature
        payload = None
        if not sig:
            if isinstance(body.get("payment_signature"), str):
                sig = str(body["payment_signature"])
            elif isinstance(body.get("sim_payment_signature"), str):
                sig = str(body["sim_payment_signature"])
            elif isinstance(body.get("payload"), dict):
                payload = body["payload"]
        try:
            result = complete_x402_fund(payment_signature=sig, payload=payload)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return result

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
        if jt in {"slime", "distill_rl", "nemo_rl"} and "rl" not in scopes:
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

    async def _proxy_chat(
        request: ASGIRequest,
        session: dict | None,
        *,
        already_paid: bool = False,
    ) -> Any:
        if not already_paid:
            if not session or session.get("status") != "active":
                raise HTTPException(402, "session not funded/active")
            if "inference" not in set(session.get("scopes") or []):
                raise HTTPException(403, "scope inference required")
        body = await request.json()
        messages = body.get("messages") or []
        prompt_est = estimate_tokens_from_messages(messages)
        # Align preflight with post-call debit: bill max_tokens (not a low
        # underestimate) so clients cannot pass check then over-debit / fail.
        completion_est = max(1, int(body.get("max_tokens") or 64))
        from seiso.pay.pricing import quote_inference_tokens

        pre = quote_inference_tokens(prompt_est, completion_est, flat_call=False)
        if not already_paid and int((session or {}).get("balance_sats") or 0) < int(
            pre["total_sats"]
        ):
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
                    if not already_paid and session:
                        debit_inference(
                            str(session["session_id"]),
                            prompt_tokens=prompt_est,
                            completion_tokens=completion_est,
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
            if already_paid:
                data["seiso_pay"] = {"per_request": True, "prepaid_session": False}
                return JSONResponse(data)
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

    @app.post("/pay/v1/requests")
    def create_pay_request(body: dict[str, Any] | None = None) -> JSONResponse:
        from seiso.pay.per_request import mint_request_quote, per_request_enabled

        if not per_request_enabled():
            raise HTTPException(503, "SEISO_PAY_PER_REQUEST=0")
        body = body or {}
        messages = body.get("messages") or []
        if messages:
            prompt_est = estimate_tokens_from_messages(messages)
        else:
            prompt_est = int(body.get("prompt_tokens") or 0)
        completion_est = max(1, int(body.get("max_tokens") or body.get("completion_tokens") or 64))
        try:
            challenge = mint_request_quote(
                prompt_tokens=prompt_est,
                completion_tokens=completion_est,
                flat_call=bool(body.get("flat_call")),
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        headers = {
            "WWW-Authenticate": str(challenge["www_authenticate"]),
            "PAYMENT-REQUIRED": str(challenge["payment_required_header"]),
        }
        return JSONResponse(challenge, status_code=402, headers=headers)

    @app.get("/pay/v1/requests/{request_id}")
    def get_pay_request(request_id: str) -> dict[str, Any]:
        from seiso.pay.per_request import load_request

        try:
            rec = load_request(request_id)
        except KeyError as exc:
            raise HTTPException(404, "request not found") from exc
        return {
            "request_id": rec["request_id"],
            "status": rec["status"],
            "fx": rec.get("fx"),
            "paid_via": rec.get("paid_via"),
        }

    @app.post("/pay/v1/requests/{request_id}/complete")
    def complete_pay_request(request_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        from seiso.pay.per_request import (
            complete_eth_request,
            complete_l402_request,
            complete_sim,
            complete_x402_request,
            sim_receipt,
        )

        body = body or {}
        via = str(body.get("via") or "x402").strip().lower()
        receipt = body.get("receipt")
        try:
            if via == "x402":
                rec = complete_x402_request(
                    request_id,
                    payment_signature=body.get("payment_signature"),
                    receipt=receipt,
                )
            elif via == "eth":
                rec = complete_eth_request(request_id, receipt=receipt)
            elif via == "l402":
                rec = complete_l402_request(request_id, receipt=receipt)
            elif via == "ark":
                proof = receipt or sim_receipt(request_id, via="ark")
                rec = complete_sim(request_id, via="ark", receipt=str(proof))
            else:
                raise HTTPException(400, f"unknown via {via}")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(401, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "request_id": rec["request_id"],
            "status": rec["status"],
            "paid_via": rec["paid_via"],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ASGIRequest,
        session: dict[str, Any] | None = _optional_session_dep,
    ) -> Any:
        from seiso.pay.per_request import per_request_enabled, request_paid

        rid = (request.headers.get("x-seiso-request-id") or "").strip()
        if rid and request_paid(rid):
            return await _proxy_chat(request, session, already_paid=True)
        if session and session.get("status") == "active":
            return await _proxy_chat(request, session, already_paid=False)
        if per_request_enabled():
            body = await request.json()
            messages = body.get("messages") or []
            prompt_est = estimate_tokens_from_messages(messages)
            completion_est = max(1, int(body.get("max_tokens") or 64))
            from seiso.pay.per_request import mint_request_quote

            try:
                challenge = mint_request_quote(
                    prompt_tokens=prompt_est,
                    completion_tokens=completion_est,
                )
            except RuntimeError as exc:
                raise HTTPException(402, str(exc)) from exc
            return JSONResponse(
                challenge,
                status_code=402,
                headers={
                    "WWW-Authenticate": str(challenge["www_authenticate"]),
                    "PAYMENT-REQUIRED": str(challenge["payment_required_header"]),
                },
            )
        raise HTTPException(401, "Bearer seiso_pay_* token or per-request payment required")

    return app


def run_server(host: str = "127.0.0.1", port: int = 8787) -> None:
    require_pay_allowed()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn required for seiso pay serve") from exc
    # Bind defaults to localhost; operators put TLS proxy in front for public.
    uvicorn.run(build_app(), host=host, port=port, log_level="info")
