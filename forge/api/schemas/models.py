"""Model inventory API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelScanRequest(BaseModel):
    path: str


class ModelDownloadRequest(BaseModel):
    repo_id: str
    filename: str | None = None
    revision: str = "main"
    variant: str = Field(default="auto", description="auto | safetensors | gguf")


class LocalModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    path: str
    source: str | None = None
    format: str | None = None
