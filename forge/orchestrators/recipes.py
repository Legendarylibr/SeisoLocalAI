"""Data recipe orchestrator — visual workflow execution."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.security import safe_join


class _SafeFormatMap(dict):
    """format_map target that ignores unknown keys."""

    def __missing__(self, key: str) -> str:
        return ""


class RecipeOrchestrator(Orchestrator):
    kind = "recipe"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        recipe = payload["recipe"]
        nodes = recipe.get("nodes", [])
        edges = recipe.get("edges", [])
        self._emit_log(job_id, f"Executing recipe: {recipe.get('name', job_id)} ({len(nodes)} nodes)")

        # Topological execution of node graph
        outputs: dict[str, Any] = {}
        for node in self._order_nodes(nodes, edges):
            ntype = node.get("type")
            nid = node["id"]
            self._emit_log(job_id, f"Running node {nid} ({ntype})")
            result = await self._run_node(node, outputs, payload)
            outputs[nid] = result

        out_path = safe_join(self.sandbox_root, "recipes", job_id, "output.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in outputs.get("_final", []):
                f.write(json.dumps(row) + "\n")

        self._emit_log(job_id, f"Recipe output: {out_path}")
        return {"output_path": str(out_path), "row_count": len(outputs.get("_final", []))}

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
            return await loop.run_in_executor(None, self._import_file, path, config.get("format"))

        if ntype == "transform":
            source_id = config.get("source")
            rows = outputs.get(source_id, [])
            template = config.get("template", "{text}")
            _ALLOWED_TEMPLATE_KEYS = frozenset({"text", "content", "source", "id", "chunk_index"})
            out = []
            for r in rows:
                if isinstance(r, dict):
                    safe = {k: str(v) for k, v in r.items() if k in _ALLOWED_TEMPLATE_KEYS}
                    text = template.format_map(_SafeFormatMap(safe))
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
            return rows[:n]

        if ntype == "output":
            source_id = config.get("source")
            outputs["_final"] = outputs.get(source_id, [])
            return outputs["_final"]

        return {}

    @staticmethod
    def _import_file(path: Path, fmt: str | None) -> list[dict]:
        suffix = (fmt or path.suffix).lower()
        if suffix in (".csv", "csv"):
            with path.open() as f:
                return list(csv.DictReader(f))
        if suffix in (".json", ".jsonl"):
            rows = []
            with path.open() as f:
                if suffix == ".jsonl":
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
            deps[e["target"]].add(e["source"])
        ordered: list[dict] = []
        done: set[str] = set()
        node_map = {n["id"]: n for n in nodes}
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
