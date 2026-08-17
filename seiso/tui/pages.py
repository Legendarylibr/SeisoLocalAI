"""Forge-matching nav + studio stubs for the lite offline UI."""

from __future__ import annotations

from typing import Any

# Mirrors forge-ui/src/components/Layout.tsx NAV_GROUPS (same labels / order).
NAV_GROUPS: list[dict[str, Any]] = [
    {
        "label": "Overview",
        "items": [
            {
                "id": "dashboard",
                "to": "/",
                "label": "Dashboard",
                "desc": "Hardware & workflows",
            },
        ],
    },
    {
        "label": "Models",
        "items": [
            {"id": "hub", "to": "/hub", "label": "Hub", "desc": "Local GGUF inventory"},
            {"id": "chat", "to": "/chat", "label": "Chat", "desc": "Local inference"},
            {
                "id": "knowledge",
                "to": "/knowledge",
                "label": "Knowledge",
                "desc": "Local RAG corpus",
            },
        ],
    },
    {
        "label": "Studio",
        "items": [
            {"id": "train", "to": "/train", "label": "Train", "desc": "LoRA fine-tuning"},
            {
                "id": "compress",
                "to": "/compress",
                "label": "Compress",
                "desc": "Distill & prune LLMs",
            },
            {
                "id": "distill-rl",
                "to": "/distill-rl",
                "label": "Distill-RL",
                "desc": "Distill + DPO alignment",
            },
            {"id": "export", "to": "/export", "label": "Export", "desc": "Publish to Hub"},
            {
                "id": "recipes",
                "to": "/recipes",
                "label": "Recipes",
                "desc": "Visual pipelines",
            },
        ],
    },
    {
        "label": "Platform",
        "items": [
            {
                "id": "integrations",
                "to": "/integrations",
                "label": "Integrations",
                "desc": "External LLM providers",
            },
        ],
    },
]

STUDIO_PAGES: dict[str, dict[str, str]] = {
    "knowledge": {
        "group": "Models",
        "title": "Knowledge",
        "subtitle": "Local RAG corpus ingest and retrieval.",
        "command": "Use full Forge (/knowledge) when you need the RAG studio.",
        "note": "The lite UI keeps RAG out of process so idle RAM stays small.",
    },
    "train": {
        "group": "Studio",
        "title": "Train",
        "subtitle": "LoRA / QLoRA fine-tune on this machine.",
        "command": "seiso train --config configs/example_lora.yaml",
        "note": "Training Studio graphs stay in full Forge. The CLI is the low-RAM path.",
    },
    "compress": {
        "group": "Studio",
        "title": "Compress",
        "subtitle": "Distill → prune → recover → quant.",
        "command": "seiso compress run --preset smoke",
        "note": "Same pipeline as Forge Compress, without the studio page.",
    },
    "distill-rl": {
        "group": "Studio",
        "title": "Distill-RL",
        "subtitle": "Teacher distill + DPO alignment.",
        "command": "seiso distill-rl run --preset smoke",
        "note": "Same CLI Forge Distill-RL jobs use.",
    },
    "export": {
        "group": "Studio",
        "title": "Export",
        "subtitle": "Merge LoRA, GGUF quant, Hub publish.",
        "command": "seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --formats merged,gguf",
        "note": "Exports land under ~/.seiso/exports/ by default.",
    },
    "recipes": {
        "group": "Studio",
        "title": "Recipes",
        "subtitle": "Visual data / recipe pipelines.",
        "command": "Use full Forge (/recipes) for the graph editor.",
        "note": "Recipe Studio needs the React canvas — not loaded here.",
    },
    "integrations": {
        "group": "Platform",
        "title": "Integrations",
        "subtitle": "Nostr provenance — same owner as Forge.",
        "command": "/relays wss://relay.example.com",
        "note": "Auto-attest and key rotate live here. External LLM providers stay in full Forge.",
    },
}

DASHBOARD_GOALS: list[dict[str, str]] = [
    {
        "id": "chat",
        "label": "Chat",
        "path": "/chat",
        "desc": "Run models locally with encrypted session memory",
    },
    {
        "id": "train",
        "label": "Train/Finetune",
        "path": "/train",
        "desc": "Fine-tune with LoRA on your hardware",
    },
    {
        "id": "compress",
        "label": "Compress",
        "path": "/compress",
        "desc": "Quantize and shrink models",
    },
    {
        "id": "inference",
        "label": "Local LLM Inference",
        "path": "/chat",
        "desc": "Chat with local GGUF or MLX engines",
    },
]
