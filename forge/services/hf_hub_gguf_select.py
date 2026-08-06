"""GGUF filename selection, shard grouping, and inventory naming."""

from __future__ import annotations

import re
from pathlib import Path

from seiso.models.catalog import CatalogEntry
from seiso.models.gguf_quant import rank_gguf_filenames
from seiso.security import sanitize_filename

_GGUF_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.I,
)


def _pick_gguf_file(
    files: list[str],
    *,
    preferred_quant: str = "Q4_K_M",
    repo_id: str = "",
) -> str | None:
    files = _pick_gguf_files(files, preferred_quant=preferred_quant, repo_id=repo_id)
    return files[0] if files else None


def _pick_mmproj_files(files: list[str]) -> list[str]:
    return sorted(
        f
        for f in files
        if f.lower().endswith(".gguf") and ("mmproj" in f.lower() or f.lower().startswith("mmproj"))
    )


def _gguf_artifact_likely_vision(
    catalog_repo_id: str,
    *,
    gguf_filename: str | None = None,
    entry: CatalogEntry | None = None,
) -> bool:
    from seiso.inference.llama_vision import repo_likely_needs_mmproj

    task = getattr(entry, "task", None)
    task_value = task.value if hasattr(task, "value") else task
    return repo_likely_needs_mmproj(
        catalog_repo_id,
        gguf_filename=gguf_filename,
        tags=getattr(entry, "tags", None),
        task=task_value,
    )


def _pick_mmproj_file(
    files: list[str],
    *,
    preferred_quant: str = "Q8_0",
) -> str | None:
    mmprojs = _pick_mmproj_files(files)
    if not mmprojs:
        return None
    preferred_quant = preferred_quant.upper()
    for hint in (preferred_quant, "Q8_0", "Q6_K", "F16", "F32", "BF16"):
        matched = [f for f in mmprojs if hint in f.upper()]
        if matched:
            return sorted(matched)[0]
    return mmprojs[0]


def _complete_shard_group_for(files: list[str], filename: str) -> list[str]:
    """Return the full shard group for an explicit GGUF, or raise if incomplete."""
    name = Path(filename).name
    match = _GGUF_SHARD_RE.match(name)
    if not match:
        return [filename]
    prefix = match.group("prefix")
    total = match.group("total")
    key_prefix = str(Path(filename).parent / prefix)
    group: list[str] = []
    indices: set[int] = set()
    for item in files:
        item_name = Path(item).name
        item_match = _GGUF_SHARD_RE.match(item_name)
        if not item_match:
            continue
        item_key = (
            str(Path(item).parent / item_match.group("prefix")),
            item_match.group("total"),
        )
        if item_key != (key_prefix, total):
            continue
        group.append(item)
        try:
            indices.add(int(item_match.group("index")))
        except ValueError:
            continue
    try:
        expected = int(total)
    except ValueError as exc:
        raise ValueError(f"Invalid GGUF shard total in {filename}") from exc
    if len(group) != expected or indices != set(range(1, expected + 1)):
        raise ValueError(
            f"Incomplete GGUF shard group for {filename} (found {len(group)}/{expected} shards)"
        )
    return sorted(group)


def _gguf_weight_files(files: list[str]) -> list[str]:
    return [
        f
        for f in files
        if f.lower().endswith(".gguf")
        and "mmproj" not in f.lower()
        and not f.lower().startswith("mmproj")
    ]


def _complete_shard_groups(pool: list[str]) -> tuple[list[list[str]], bool]:
    """Return complete shard groups from *pool* and whether any incomplete group exists."""
    shard_groups: dict[tuple[str, str], list[str]] = {}
    for filename in pool:
        name = Path(filename).name
        match = _GGUF_SHARD_RE.match(name)
        if not match:
            continue
        key = (str(Path(filename).parent / match.group("prefix")), match.group("total"))
        shard_groups.setdefault(key, []).append(filename)
    complete_groups: list[list[str]] = []
    incomplete_shards = False
    for (_prefix, total), group in shard_groups.items():
        try:
            expected = int(total)
        except ValueError:
            continue
        indices: set[int] = set()
        for filename in group:
            match = _GGUF_SHARD_RE.match(Path(filename).name)
            if not match:
                continue
            try:
                indices.add(int(match.group("index")))
            except ValueError:
                continue
        if len(group) == expected and indices == set(range(1, expected + 1)):
            complete_groups.append(sorted(group))
        elif group:
            incomplete_shards = True
    return complete_groups, incomplete_shards


def list_complete_gguf_file_groups(files: list[str]) -> list[list[str]]:
    """Every complete GGUF artifact present in *files* (one list per quant/shard set).

    Does not prefer a quant — reflects whatever is on disk / listed on Hub.
    Incomplete shard groups are skipped; complete siblings remain.
    """
    ggufs = _gguf_weight_files(files)
    if not ggufs:
        return []

    complete_groups, _incomplete = _complete_shard_groups(ggufs)
    groups: list[list[str]] = list(complete_groups)
    for filename in ggufs:
        if _GGUF_SHARD_RE.match(Path(filename).name):
            continue
        groups.append([filename])

    # Stable order by first filename for deterministic inventory registration.
    groups.sort(key=lambda group: (group[0].lower(), len(group)))
    return groups


def _pick_gguf_files(
    files: list[str],
    *,
    preferred_quant: str = "Q4_K_M",
    repo_id: str = "",
) -> list[str]:
    ggufs = _gguf_weight_files(files)
    if not ggufs:
        return []
    preferred_quant = preferred_quant.upper()

    def quant_matches(candidates: list[str]) -> list[str]:
        normalized_preferred = preferred_quant.replace("-", "_")
        exact = [f for f in candidates if normalized_preferred in f.upper().replace("-", "_")]
        if exact:
            return exact
        return rank_gguf_filenames(candidates, preferred=preferred_quant)

    pool = quant_matches(ggufs)

    moe_match = re.search(r"a(\d+(?:\.\d+)?)b", repo_id, re.I)
    if moe_match:
        active = moe_match.group(0).lower()
        active_hits = [f for f in pool if active in f.lower().replace("_", "-")]
        if active_hits:
            pool = active_hits

    complete_groups, incomplete_shards = _complete_shard_groups(pool)
    if incomplete_shards and not complete_groups:
        return []
    if complete_groups:
        return sorted(complete_groups, key=lambda group: (len(group), len(group[0]), group[0]))[0]

    non_sharded = [filename for filename in pool if not _GGUF_SHARD_RE.match(Path(filename).name)]
    if non_sharded:
        return [sorted(non_sharded, key=len)[0]]
    return []


def _inventory_name(repo_id: str, filename: str) -> Path:
    """Stable symlink path under user models inventory."""
    safe_repo = sanitize_filename(repo_id.replace("/", "--"))
    return Path(safe_repo) / sanitize_filename(Path(filename).name)


def _inventory_name_for_files(repo_id: str, filenames: list[str]) -> Path:
    if len(filenames) == 1:
        return _inventory_name(repo_id, filenames[0])
    first = Path(filenames[0])
    match = _GGUF_SHARD_RE.match(first.name)
    name = match.group("prefix") if match else first.stem
    if first.parent != Path("."):
        name = f"{first.parent.name}-{name}"
    safe_repo = sanitize_filename(repo_id.replace("/", "--"))
    return Path(safe_repo) / sanitize_filename(name)
