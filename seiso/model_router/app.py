"""OpenAI-compatible router gateway."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from seiso.model_router.backend_lifecycle import BackendLifecycleManager
from seiso.model_router.catalog import SpecialistCatalog
from seiso.model_router.classifier import classify_messages
from seiso.model_router.config import RouterSettings, resolve_paths
from seiso.model_router.fallback import FallbackChain
from seiso.model_router.monitoring import collect_metrics
from seiso.model_router.policy import SpecialistRouteBandit, pick_route_with_hints

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


def _reward_from_latency(latency_ms: float, success: bool) -> float:
    from seiso.model_router.reward import compute_route_reward

    return compute_route_reward(latency_ms, success)


def _compute_reward(
    latency_ms: float,
    success: bool,
    *,
    response_text: str = "",
    classification,
    route,
    fallback_mode: str,
    wake_latency_ms: float = 0.0,
) -> float:
    from seiso.model_router.reward import compute_route_reward

    return compute_route_reward(
        latency_ms,
        success,
        response_text=response_text,
        classified_domain=classification.domain,
        route_domain_hints=route.domain_hints,
        wake_latency_ms=wake_latency_ms,
        complexity_score=classification.complexity_score,
        fallback_used=fallback_mode != "primary",
    )


def create_router_app(settings: RouterSettings | None = None) -> FastAPI:
    base_settings = settings or RouterSettings()
    settings = resolve_paths(base_settings)

    catalog = SpecialistCatalog.from_json(settings.specialists_path)
    bandit = SpecialistRouteBandit(
        catalog=catalog,
        ucb_c=settings.rl_ucb_c,
        prior_weight=settings.rl_prior_weight,
        warmup_pulls=settings.rl_warmup_pulls,
        seed=settings.rl_seed,
    )
    if settings.policy_state_path.is_file():
        bandit.load(settings.policy_state_path)

    lifecycle = BackendLifecycleManager(settings=settings, catalog=catalog)
    fallback = FallbackChain(
        catalog=catalog, lifecycle=lifecycle, default_route_id=settings.fallback_route_id
    )

    litellm_gateway = None
    if settings.litellm_gateway_enabled():
        try:
            from seiso.model_router.litellm_gateway import LitellmGateway

            litellm_gateway = LitellmGateway(
                catalog,
                llamaswap_url=settings.llamaswap_url,
                routing_strategy=settings.litellm_routing_strategy,
                request_timeout_sec=settings.request_timeout_sec,
            )
        except Exception as exc:
            raise RuntimeError(
                "vLLM router stack requires LiteLLM — pip install -e '.[router]'"
            ) from exc
    state: dict[str, Any] = {
        "settings": settings,
        "catalog": catalog,
        "bandit": bandit,
        "lifecycle": lifecycle,
        "fallback": fallback,
        "litellm": litellm_gateway,
        "http": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await lifecycle.start()
        state["http"] = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_sec))
        yield
        await lifecycle.stop()
        client = state.get("http")
        if client:
            await client.aclose()

    app = FastAPI(title="Seiso Model Router", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": settings.mode,
            "inference_backend": settings.inference_backend,
            "routing_mode": settings.routing_mode,
            "vllm_sleep_mode": settings.vllm_sleep_mode,
            "nemotron_enabled": settings.nemotron_orchestrator_enabled(),
            "litellm_enabled": settings.litellm_gateway_enabled(),
            "orchestrator_model": settings.orchestrator_model
            if settings.orchestrator_url
            else None,
        }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        backends = lifecycle.status()["backends"]
        any_awake = any(b["state"] in ("awake", "unknown") for b in backends.values())
        return {"ready": any_awake or settings.mode == "local", "backends": backends}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        sys_m = collect_metrics()
        lines = sys_m.prometheus_lines()
        lines.append(f"seiso_router_policy_pulls {bandit._total_pulls}")
        body = "\n".join(lines) + "\n"
        return PlainTextResponse(body)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        models = [
            {
                "id": r.openai_model_name,
                "route_id": r.route_id,
                "owned_by": "seiso-router",
            }
            for r in catalog
        ]
        return {"object": "list", "data": models}

    @app.get("/router/status")
    async def router_status() -> dict[str, Any]:
        return {
            "lifecycle": lifecycle.status(),
            "metrics": collect_metrics().to_dict(),
            "policy_pulls": bandit._total_pulls,
            "routing_mode": settings.routing_mode,
            "inference_backend": settings.inference_backend,
            "vllm_sleep_mode": settings.vllm_sleep_mode,
            "nemotron_enabled": settings.nemotron_orchestrator_enabled(),
            "litellm_enabled": settings.litellm_gateway_enabled(),
            "orchestrator_url": settings.orchestrator_url or None,
            "orchestrator_model": settings.orchestrator_model,
        }

    async def _forward_chat(
        route_payload: dict[str, Any],
        target_url: str,
        api_key: str = "",
    ) -> tuple[dict[str, Any], float, bool]:
        client = state["http"]
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        path = "/v1/chat/completions"
        start = time.perf_counter()
        try:
            resp = await client.post(f"{target_url}{path}", json=route_payload, headers=headers)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code >= 400:
                return {"error": resp.text, "status": resp.status_code}, latency_ms, False
            return resp.json(), latency_ms, True
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {"error": str(exc)}, latency_ms, False

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest) -> Any:
        messages = [m.model_dump() for m in body.messages]
        known_domains = catalog.known_domains()

        classification, context = classify_messages(
            messages,
            hardware=settings.hardware,
            known_domains=known_domains,
        )

        explicit_route = None
        route_reason = ""
        orchestrator_meta: dict[str, Any] = {}

        if body.model and settings.allow_explicit_model:
            explicit_route = catalog.by_llamaswap_model(body.model)

        if explicit_route:
            primary = explicit_route
            route_reason = f"explicit_model={body.model}"
        elif settings.nemotron_orchestrator_enabled():
            from seiso.model_router.nemotron import select_route_via_nemotron

            try:
                selection = await select_route_via_nemotron(
                    state["http"],
                    orchestrator_url=settings.orchestrator_url,
                    orchestrator_model=settings.orchestrator_model,
                    catalog=catalog,
                    messages=messages,
                    fallback_route_id=settings.fallback_route_id,
                    timeout=settings.orchestrator_timeout_sec,
                    temperature=settings.orchestrator_temperature,
                    max_tokens=settings.orchestrator_max_tokens,
                )
                primary = selection.route
                route_reason = selection.reasoning
                orchestrator_meta = {
                    "orchestrator": "nvidia/Nemotron-Orchestrator-8B",
                    "orchestrator_alias": selection.orchestrator_alias,
                    "tool_calls": selection.raw_tool_calls,
                }
            except Exception as exc:
                logger.warning("nemotron orchestrator failed, falling back to heuristic: %s", exc)
                primary = catalog.by_id(settings.fallback_route_id)
                route_reason = f"nemotron_error={exc}"
                orchestrator_meta = {"orchestrator_error": str(exc)}
        elif settings.enable_rl_policy:
            selection = pick_route_with_hints(
                bandit,
                context,
                classification.domain,
                deterministic=settings.mode == "prod",
            )
            primary = selection.route
            route_reason = selection.reasoning
        else:
            primary = catalog.by_id(settings.fallback_route_id)
            route_reason = "rl_disabled"

        awake_route, fallback_mode = await fallback.resolve_awake_route(primary)
        lifecycle.touch(awake_route.route_id)
        wake_ms = lifecycle._records[awake_route.route_id].wake_latency_ms

        if settings.litellm_gateway_enabled():
            gateway = state["litellm"]
            if gateway is None:
                raise HTTPException(status_code=503, detail="LiteLLM gateway not initialized")
            if body.stream:
                return await _stream_litellm_response(
                    gateway,
                    awake_route,
                    messages,
                    body.max_tokens,
                    body.temperature,
                    classification,
                    context,
                    route_reason,
                    fallback_mode,
                    orchestrator_meta,
                )
            start = time.perf_counter()
            try:
                data = await gateway.chat_completion(
                    awake_route,
                    messages=messages,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                ok = True
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            route_payload = {
                "model": awake_route.llamaswap_model,
                "messages": messages,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
                "stream": body.stream,
            }
            target_base = (
                settings.llamaswap_url.rstrip("/")
                if settings.llamaswap_url
                else awake_route.backend_url
            )
            if body.stream:
                return await _stream_response(
                    target_base,
                    route_payload,
                    awake_route,
                    classification,
                    context,
                    route_reason,
                    fallback_mode,
                )
            data, latency_ms, ok = await _forward_chat(route_payload, target_base)
            if not ok:
                raise HTTPException(status_code=502, detail=data)

        response_text = ""
        if isinstance(data, dict):
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        reward = _compute_reward(
            latency_ms,
            ok,
            response_text=response_text,
            classification=classification,
            route=awake_route,
            fallback_mode=fallback_mode,
            wake_latency_ms=wake_ms,
        )
        if settings.enable_rl_policy:
            bandit.update(awake_route.route_id, context, reward)
            bandit.save(settings.policy_state_path)

        if isinstance(data, dict):
            data.setdefault("seiso_router", {})
            data["seiso_router"] = {
                "route_id": awake_route.route_id,
                "domain": classification.domain,
                "complexity": classification.complexity_score,
                "fallback_mode": fallback_mode,
                "latency_ms": latency_ms,
                "reward": reward,
                "reasoning": route_reason,
                "routing_mode": settings.routing_mode,
                "execution": "litellm" if settings.litellm_gateway_enabled() else "httpx",
                **orchestrator_meta,
            }
        return data

    async def _stream_litellm_response(
        gateway,
        route,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        classification,
        context,
        route_reason: str,
        fallback_mode: str,
        orchestrator_meta: dict[str, Any],
    ) -> StreamingResponse:
        start = time.perf_counter()
        bandit = state["bandit"]

        async def event_gen():
            collected: list[str] = []
            try:
                async for chunk in gateway.stream_chat_completion(
                    route,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    text = (
                        chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="ignore")
                    )
                    for line in text.split("\n"):
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            import json as _json

                            piece = _json.loads(payload)
                            delta = piece.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                collected.append(content)
                        except Exception:
                            pass
                    yield text
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                reward = _compute_reward(
                    latency_ms,
                    True,
                    response_text="".join(collected),
                    classification=classification,
                    route=route,
                    fallback_mode=fallback_mode,
                    wake_latency_ms=lifecycle._records[route.route_id].wake_latency_ms,
                )
                if settings.enable_rl_policy:
                    bandit.update(route.route_id, context, reward)
                    bandit.save(settings.policy_state_path)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    async def _stream_response(
        target_base: str,
        route_payload: dict[str, Any],
        route,
        classification,
        context,
        route_reason: str,
        fallback_mode: str,
    ) -> StreamingResponse:
        client = state["http"]
        path = "/v1/chat/completions"
        start = time.perf_counter()

        async def event_gen():
            collected: list[str] = []
            try:
                async with client.stream(
                    "POST",
                    f"{target_base}{path}",
                    json=route_payload,
                    timeout=settings.request_timeout_sec,
                ) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        yield f"data: {err.decode()}\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        text = chunk.decode("utf-8", errors="ignore")
                        for line in text.split("\n"):
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                import json as _json

                                piece = _json.loads(payload)
                                delta = piece.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    collected.append(content)
                            except Exception:
                                pass
                        yield chunk
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                reward = _compute_reward(
                    latency_ms,
                    True,
                    response_text="".join(collected),
                    classification=classification,
                    route=route,
                    fallback_mode=fallback_mode,
                    wake_latency_ms=lifecycle._records[route.route_id].wake_latency_ms,
                )
                if settings.enable_rl_policy:
                    bandit.update(route.route_id, context, reward)
                    bandit.save(settings.policy_state_path)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app
