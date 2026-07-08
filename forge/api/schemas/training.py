"""Training API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingStartRequest(BaseModel):
    config: dict
    project_id: str | None = None
    multi_gpu: bool = False
    dataset_analysis_token: str | None = None
    export_on_complete: dict | None = Field(
        default=None,
        description="Auto-export after training: formats, profile, gguf_quantizations, hub",
    )


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str


class CloudGpuCredentialCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    provider: str = Field(min_length=1, max_length=32)
    auth_kind: str = Field(default="api_key", max_length=32)
    api_key: str = Field(default="", max_length=4096)
    access_key_id: str = Field(default="", max_length=512)
    secret_access_key: str = Field(default="", max_length=4096)
    session_token: str = Field(default="", max_length=8192)
    ssh_username: str = Field(default="", max_length=128)
    ssh_private_key: str = Field(default="", max_length=16384)
    bootstrap_command: str = Field(default="", max_length=4096)
    region: str = Field(default="", max_length=128)
    project: str = Field(default="", max_length=128)


class DatasetValidationRequest(BaseModel):
    dataset: str
    dataset_format: str = "auto"
