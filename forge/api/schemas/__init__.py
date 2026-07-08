"""Pydantic request/response schemas for Forge API routes."""

from __future__ import annotations

from forge.api.schemas.export import (
    ExportStartRequest,
    HubPrecheckRequest,
    PublishToHubRequest,
)
from forge.api.schemas.inference import ChatRequest, PreloadRequest, ThreadCreate
from forge.api.schemas.models import LocalModelCreate, ModelDownloadRequest, ModelScanRequest
from forge.api.schemas.openai import ChatCompletionRequest, ChatMessage
from forge.api.schemas.training import (
    CloudGpuCredentialCreate,
    DatasetValidationRequest,
    TrainingJobResponse,
    TrainingStartRequest,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ChatRequest",
    "CloudGpuCredentialCreate",
    "DatasetValidationRequest",
    "ExportStartRequest",
    "HubPrecheckRequest",
    "LocalModelCreate",
    "ModelDownloadRequest",
    "ModelScanRequest",
    "PreloadRequest",
    "PublishToHubRequest",
    "ThreadCreate",
    "TrainingJobResponse",
    "TrainingStartRequest",
]
