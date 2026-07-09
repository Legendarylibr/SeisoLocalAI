"""Data recipe orchestrator — visual workflow execution."""

from __future__ import annotations

import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.models.chat_format import extract_messages, format_messages_for_prompt
from seiso.security import safe_join
from seiso.training.config import DatasetFormat
from seiso.training.datasets import detect_format
from seiso.training.preprocess import normalize_sample


class _SafeFormatMap(dict):
    """format_map target that ignores unknown keys."""

    def __missing__(self, key: str) -> str:
        return ""


def _recipe_template_fields(row: Any) -> dict[str, str]:
    """Map a dataset row to template placeholders for the transform node."""
    if not isinstance(row, dict):
        return {"text": str(row)}

    fmt = detect_format(row)
    normalized = normalize_sample(row, fmt)
    if not normalized:
        return {"text": ""}

    fields = {
        k: str(v) for k, v in normalized.items() if isinstance(v, (str, int, float))
    }

    if "text" in normalized:
        fields["text"] = str(normalized["text"])
    elif "messages" in normalized:
        messages = extract_messages(normalized, DatasetFormat.CHAT)
        fields["text"] = format_messages_for_prompt(
            messages, tokenizer=None, add_generation_prompt=False
        )
    elif "instruction" in normalized:
        instruction = str(normalized.get("instruction") or "")
        inp = str(normalized.get("input") or "")
        output = str(normalized.get("output") or "")
        user = f"{instruction}\n{inp}".strip() if inp else instruction
        fields.setdefault("instruction", instruction)
        fields.setdefault("input", inp)
        fields.setdefault("output", output)
        fields["text"] = f"USER: {user}\nASSISTANT: {output}".strip()
    elif "conversations" in normalized:
        messages = extract_messages(normalized, DatasetFormat.SHAREGPT)
        fields["text"] = format_messages_for_prompt(
            messages, tokenizer=None, add_generation_prompt=False
        )
    else:
        fields["text"] = str(next(iter(normalized.values()), ""))

    return fields


class RecipeOrchestrator(Orchestrator):
    kind = "recipe"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        recipe = payload["recipe"]
        nodes = recipe.get("nodes", [])
        edges = recipe.get("edges", [])
        self._emit_log(
            job_id,
            f"Executing recipe: {recipe.get('name', job_id)} ({len(nodes)} nodes)",
        )

        user_id = payload.get("user_id")
        if not user_id:
            raise PermissionError("user_id required for recipe output")
        recipe_snapshot_path = safe_join(
            self.sandbox_root, "recipes", user_id, job_id, "recipe_snapshot.json"
        )
        recipe_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_snapshot_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")

        # Topological execution of node graph
        outputs: dict[str, Any] = {}
        for node in self._order_nodes(nodes, edges):
            ntype = node.get("type")
            nid = node["id"]
            self._emit_log(job_id, f"Running node {nid} ({ntype})")
            result = await self._run_node(node, outputs, payload)
            outputs[nid] = result

        out_path = safe_join(
            self.sandbox_root, "recipes", user_id, job_id, "output.jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in outputs.get("_final", []):
                f.write(json.dumps(row) + "\n")

        self._emit_log(job_id, f"Recipe output: {out_path}")
        return {
            "output_path": str(out_path),
            "row_count": len(outputs.get("_final", [])),
            "recipe_snapshot": str(recipe_snapshot_path),
        }

    async def _run_node(
        self,
        node: dict,
        outputs: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        ntype = node.get("type")
        config = node.get("config", {})
        loop = asyncio.get_running_loop()

        if ntype == "import":
            user_id = payload.get("user_id")
            if not user_id:
                raise PermissionError("user_id required for recipe import")
            path = assert_user_path(self.sandbox_root, user_id, config["path"])
            from forge.services.executors import IO_EXECUTOR

            return await loop.run_in_executor(
                IO_EXECUTOR, self._import_file, path, config.get("format")
            )

        if ntype == "transform":
            source_id = config.get("source")
            rows = outputs.get(source_id, [])
            template = config.get("template", "{text}")
            out = []
            for r in rows:
                if isinstance(r, dict):
                    text = template.format_map(
                        _SafeFormatMap(_recipe_template_fields(r))
                    )
                else:
                    text = str(r)
                out.append({"text": text})
            return out

        if ntype == "filter":
            source_id = config.get("source")
            rows = outputs.get(source_id, [])
            min_len = config.get("min_length", 0)
            return [r for r in rows if len(str(r.get("text", r))) >= min_len]

        if ntype == "sample":
            source_id = config.get("source")
            rows = outputs.get(source_id, [])
            n = min(config.get("count", 100), len(rows))
            seed = int(config.get("seed", payload.get("seed", 42)))
            rng = random.Random(
                seed
            )  # nosec B311 - deterministic sampling from user-provided seed
            return rng.sample(rows, n) if n < len(rows) else list(rows)

        if ntype == "output":
            source_id = config.get("source")
            outputs["_final"] = outputs.get(source_id, [])
            return outputs["_final"]

        return {}

    @staticmethod
    def _resolve_import_suffix(path: Path, fmt: str | None) -> str:
        normalized = (fmt or "").strip().lower().lstrip(".")
        if normalized in ("", "auto", "txt", "text"):
            return path.suffix.lower().lstrip(".") or "txt"
        return normalized

    @staticmethod
    def _import_file(path: Path, fmt: str | None) -> list[dict]:
        suffix = RecipeOrchestrator._resolve_import_suffix(path, fmt)
        if suffix in (".csv", "csv"):
            with path.open() as f:
                return list(csv.DictReader(f))
        if suffix in (".json", "json", ".jsonl", "jsonl"):
            rows = []
            with path.open() as f:
                if suffix in (".jsonl", "jsonl"):
                    for line in f:
                        rows.append(json.loads(line))
                else:
                    data = json.load(f)
                    rows = data if isinstance(data, list) else [data]
            return rows
        return [{"text": path.read_text()}]

    @staticmethod
    def _order_nodes(nodes: list[dict], edges: list[dict]) -> list[dict]:
        """Simple topological sort by edge dependencies."""
        deps: dict[str, set[str]] = {n["id"]: set() for n in nodes}
        for e in edges:
            target = e.get("target")
            source = e.get("source")
            if target not in deps:
                raise ValueError(f"Recipe edge target not found: {target!r}")
            if source not in deps:
                raise ValueError(f"Recipe edge source not found: {source!r}")
            deps[target].add(source)
        ordered: list[dict] = []
        done: set[str] = set()
        while len(ordered) < len(nodes):
            progressed = False
            for n in nodes:
                if n["id"] in done:
                    continue
                if deps[n["id"]].issubset(done):
                    ordered.append(n)
                    done.add(n["id"])
                    progressed = True
                    break
            if not progressed:
                raise ValueError("Recipe graph contains a cycle or disconnected nodes")
        return ordered
