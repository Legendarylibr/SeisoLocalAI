"""OpenAI-compatible API request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=2048, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    tools: list[dict] | None = None
