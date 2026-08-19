"""Shared Hugging Face Hub publish request models and helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge.config import ForgeSettings
from forge.services.hf_auth import resolve_hf_token_for_upload
from seiso.export.model_card import HubModelMetadata


class HubPublishRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    model_name: str = Field(min_length=1, max_length=128)
    author: str = Field(min_length=1, max_length=256)
    license: str = Field(default="apache-2.0", max_length=64)
    base_model: str | None = Field(default=None, max_length=256)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list)
    hf_token: str | None = Field(default=None, description="Per-request HF API token")
    use_cli: bool = Field(default=False, description="Prefer cached `hf auth login` token")


def hub_metadata_from_request(
    hub: HubPublishRequest, *, job_id: str | None = None, source: str | None = None
) -> HubModelMetadata:
    return HubModelMetadata(
        username=hub.username.strip(),
        model_name=hub.model_name.strip(),
        author=hub.author.strip(),
        license=hub.license.strip() or "apache-2.0",
        base_model=hub.base_model.strip() if hub.base_model else None,
        description=hub.description,
        tags=hub.tags,
        seiso_job_id=job_id,
        seiso_source=source,
    )


def resolve_hub_publish_token(
    settings: ForgeSettings,
    user_id: str,
    hub: HubPublishRequest | None,
) -> str | None:
    token, _ = resolve_hf_token_for_upload(
        request_token=hub.hf_token if hub else None,
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
        prefer_cli=hub.use_cli if hub else False,
    )
    return token
