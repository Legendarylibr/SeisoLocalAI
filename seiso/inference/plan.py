"""Typed generation plan shared by Forge prepare paths and the local runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationPlan:
    """Resolved local-chat generation parameters.

    Built once in ``prepare_local_chat_target`` / preload and merged into the
    runner payload so sanitize/fit/backend choices are not recomputed ad hoc.
    """

    model_path: str
    inference_backend: str
    max_tokens: int
    n_ctx: int | None = None
    model_format: str | None = None
    model_metadata: dict[str, Any] | None = None
    model_name: str | None = None
    size_bytes: int = 0
    fit_ok: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def to_payload_updates(self) -> dict[str, Any]:
        """Dict suitable for merging into a chat/preload payload."""
        out: dict[str, Any] = {
            "model_path": self.model_path,
            "inference_backend": self.inference_backend,
            "max_tokens": self.max_tokens,
            "fit_ok": self.fit_ok,
        }
        if self.n_ctx is not None:
            out["n_ctx"] = self.n_ctx
        if self.model_format is not None:
            out["model_format"] = self.model_format
        if self.model_metadata is not None:
            out["model_metadata"] = self.model_metadata
        if self.model_name is not None:
            out["model_name"] = self.model_name
        if self.size_bytes:
            out["size_bytes"] = self.size_bytes
        if self.extras:
            out.update(self.extras)
        return out

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def generation_plan_from_updates(updates: dict[str, Any]) -> GenerationPlan | None:
    """Build a plan from prepare_local_chat_target updates when path is known."""
    path = updates.get("model_path")
    backend = updates.get("inference_backend")
    if not path or not backend or updates.get("use_model_router"):
        return None
    return GenerationPlan(
        model_path=str(path),
        inference_backend=str(backend),
        max_tokens=int(updates.get("max_tokens") or 512),
        n_ctx=int(updates["n_ctx"]) if updates.get("n_ctx") is not None else None,
        model_format=(
            str(updates["model_format"]) if updates.get("model_format") is not None else None
        ),
        model_metadata=(
            dict(updates["model_metadata"])
            if isinstance(updates.get("model_metadata"), dict)
            else None
        ),
        model_name=(
            str(updates["model_name"]) if updates.get("model_name") is not None else None
        ),
        size_bytes=int(updates.get("size_bytes") or 0),
        fit_ok=bool(updates.get("fit_ok", True)),
        extras={
            k: v
            for k, v in updates.items()
            if k
            not in {
                "model_path",
                "inference_backend",
                "max_tokens",
                "n_ctx",
                "model_format",
                "model_metadata",
                "model_name",
                "size_bytes",
                "fit_ok",
            }
        },
    )
