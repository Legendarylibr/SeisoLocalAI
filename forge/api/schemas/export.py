"""Export API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.services.hub_publish import HubPublishRequest


class ExportStartRequest(BaseModel):
    checkpoint: str
    formats: list[str] = Field(default_factory=lambda: ["merged"])
    profile: str | None = Field(
        default=None,
        description="Export profile: lora_adapter, lora_bundle, full_finetune, full_bundle, inference, gguf_only, hub_ready",
    )
    gguf_quantizations: list[str] | None = Field(
        default=None,
        description="GGUF quant names; defaults to q4_k_m when omitted",
    )
    hub: HubPublishRequest | None = None
    hub_repo: str | None = Field(
        default=None, description="Deprecated — use hub.username + hub.model_name"
    )


class PublishToHubRequest(BaseModel):
    model_id: str | None = None
    output_path: str | None = None
    export_job_id: str | None = None
    hub: HubPublishRequest


class HubPrecheckRequest(BaseModel):
    hub: HubPublishRequest
    formats: list[str] = Field(default_factory=lambda: ["merged"])
    profile: str | None = None
    gguf_quantizations: list[str] = Field(default_factory=lambda: ["q4_k_m"])
