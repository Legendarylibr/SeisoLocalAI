"""Inference API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    draft_model_id: str | None = None
    draft_model_path: str | None = None
    num_speculative_tokens: int | None = Field(default=None, ge=1, le=32)
    inference_backend: str = Field(
        default="auto", description="auto | llamacpp | llamaswap | mlx | torch"
    )
    messages: list[dict[str, str]] = Field(default_factory=list)
    # Desired overall reply length. Per-pass generation is still OOM-clamped;
    # auto-continue delivers longer totals in safe chunks when needed.
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    n_ctx: int | None = Field(default=None, ge=2048, le=131072)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = True
    tools: bool = False
    allow_code_exec: bool = False
    provider_id: str | None = None
    knowledge_base_id: str | None = None
    router_model: str | None = Field(
        default=None,
        description="Optional explicit specialist model id for Smart Router",
    )


class ThreadCreate(BaseModel):
    title: str = "New chat"
    model_id: str | None = None


class PreloadRequest(BaseModel):
    model_id: str
    inference_backend: str = Field(
        default="auto", description="auto | llamacpp | llamaswap | mlx | torch"
    )
    max_tokens: int = Field(default=2048, ge=1, le=131072)
    n_ctx: int | None = Field(default=None, ge=2048, le=131072)
