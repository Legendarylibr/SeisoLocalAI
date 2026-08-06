"""Group local and Hub GGUF quants for fast chat switching."""

from __future__ import annotations

import json
import re
from typing import Any

from forge.db.store import Database
from forge.services.inference_models import get_inference_option, list_inference_options
from seiso.models.gguf_quant import extract_quant_label as _extract_quant_label
from seiso.models.trusted_gguf import base_model_from_tags


def extract_quant_label(
    *, name: str, path: str = "", metadata: dict[str, Any] | None = None
) -> str:
    label = _extract_quant_label(name=name, path=path, metadata=metadata)
    return label or "GGUF"


def _metadata(row_or_opt: dict[str, Any]) -> dict[str, Any]:
    raw = row_or_opt.get("metadata") or row_or_opt.get("metadata_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def variant_group_key(opt: dict[str, Any]) -> str:
    meta = _metadata(opt)
    for key in ("repo_id", "gguf_repo", "base_model"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    source = opt.get("source") or ""
    if isinstance(source, str) and source.startswith("hf:"):
        # Strip multi-quant suffix (hf:org/model:file.gguf → org/model).
        rest = source[3:]
        if ":" in rest and "/" in rest.split(":", 1)[0]:
            rest = rest.split(":", 1)[0]
        return rest.lower()
    name = str(opt.get("name") or "").lower()
    stripped = re.sub(r"[-_.]?(q\d+[_a-z0-9]*|iq\d+[_a-z0-9]*|f16|bf16).*$", "", name, flags=re.I)
    return stripped or name or str(opt.get("id") or "")


def _same_variant_group(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return variant_group_key(a) == variant_group_key(b)


def _option_variant_row(opt: dict[str, Any], *, current_id: str) -> dict[str, Any]:
    meta = _metadata(opt)
    quant = extract_quant_label(
        name=str(opt.get("name") or ""), path=str(opt.get("path") or ""), metadata=meta
    )
    return {
        "id": opt["id"],
        "name": opt.get("name"),
        "quant": quant,
        "size_bytes": int(opt.get("size_bytes") or 0),
        "path": opt.get("path"),
        "hardware_fit": opt.get("hardware_fit"),
        "hardware_fit_label": opt.get("hardware_fit_label"),
        "memory_load_blocked": bool(opt.get("memory_load_blocked")),
        "selected": opt["id"] == current_id,
        "source": "local",
        "repo_id": meta.get("repo_id") or meta.get("gguf_repo"),
        "gguf_file": meta.get("gguf_file"),
    }


def _hub_variant_rows(
    gguf_repo: str,
    *,
    token: str | None,
    local_by_file: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=token).list_repo_files(gguf_repo, repo_type="model")
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for filename in files:
        if not filename.lower().endswith(".gguf"):
            continue
        if "mmproj" in filename.lower():
            continue
        quant = extract_quant_label(name=filename, path=filename)
        local = local_by_file.get(filename)
        row: dict[str, Any] = {
            "quant": quant,
            "gguf_file": filename,
            "gguf_repo": gguf_repo,
            "source": "hub",
            "downloaded": local is not None,
            "local_id": local.get("id") if local else None,
            "selected": bool(local and local.get("selected")),
        }
        if local:
            row["size_bytes"] = local.get("size_bytes", 0)
            row["hardware_fit"] = local.get("hardware_fit")
            row["hardware_fit_label"] = local.get("hardware_fit_label")
            row["memory_load_blocked"] = local.get("memory_load_blocked")
        rows.append(row)

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        downloaded_rank = 0 if item.get("downloaded") else 1
        return (downloaded_rank, item.get("quant") or "")

    rows.sort(key=sort_key)
    return rows


async def get_model_variants(
    db: Database,
    user_id: str,
    model_id: str,
    *,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Local quant siblings plus Hub GGUF files for the same mirror/base model."""
    current = await get_inference_option(db, user_id, model_id)
    if not current:
        return {
            "model_id": model_id,
            "local_variants": [],
            "hub_variants": [],
            "variant_group": None,
        }

    all_options = await list_inference_options(db, user_id)
    local_variants = [
        _option_variant_row(opt, current_id=model_id)
        for opt in all_options
        if _same_variant_group(opt, current)
    ]
    local_variants.sort(key=lambda row: (row.get("quant") or "", row.get("name") or ""))

    meta = _metadata(current)
    gguf_repo = str(meta.get("gguf_repo") or meta.get("repo_id") or "")
    if not gguf_repo.endswith("-GGUF") and not gguf_repo.lower().endswith("gguf"):
        base = meta.get("base_model") or base_model_from_tags(tuple(meta.get("tags") or ()))
        if isinstance(base, str) and base:
            from seiso.models.trusted_gguf import gguf_mirror_candidates

            mirrors = gguf_mirror_candidates(base)
            gguf_repo = mirrors[0] if mirrors else gguf_repo

    local_by_file: dict[str, dict[str, Any]] = {}
    for row in local_variants:
        gguf_file = row.get("gguf_file")
        if isinstance(gguf_file, str):
            local_by_file[gguf_file] = row

    hub_variants = (
        _hub_variant_rows(gguf_repo, token=hf_token, local_by_file=local_by_file)
        if gguf_repo
        else []
    )

    draft_candidates = [
        {
            "id": opt["id"],
            "name": opt.get("name"),
            "size_bytes": int(opt.get("size_bytes") or 0),
            "format": opt.get("format"),
            "backends": opt.get("backends") or [],
            "hardware_fit": opt.get("hardware_fit"),
            "hardware_fit_label": opt.get("hardware_fit_label"),
        }
        for opt in all_options
        if opt["id"] != model_id
        and "torch" in (opt.get("backends") or [])
        and (opt.get("format") or "").lower() in {"safetensors", "bin", ""}
    ]
    draft_candidates.sort(key=lambda row: int(row.get("size_bytes") or 0))

    return {
        "model_id": model_id,
        "variant_group": variant_group_key(current),
        "gguf_repo": gguf_repo or None,
        "catalog_repo": meta.get("repo_id"),
        "base_model": meta.get("base_model"),
        "current_quant": extract_quant_label(
            name=str(current.get("name") or ""),
            path=str(current.get("path") or ""),
            metadata=meta,
        ),
        "local_variants": local_variants,
        "hub_variants": hub_variants,
        "draft_candidates": draft_candidates,
        "supports_speculative": "torch" in (current.get("backends") or []),
        "supports_llamacpp": "llamacpp" in (current.get("backends") or []),
    }
