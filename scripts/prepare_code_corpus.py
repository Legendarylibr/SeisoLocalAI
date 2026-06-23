#!/usr/bin/env python3
"""Stream code pretraining datasets from Hugging Face → normalized JSONL for Seiso training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seiso.models.hub_quant import infer_active_params_b  # noqa: E402
from seiso.training.code_corpus import (  # noqa: E402
    is_metadata_only_row,
    normalize_code_row,
    recommend_pretraining_epochs,
)

DEFAULT_CONFIGS: tuple[str, ...] = ()


def _configure_hub_auth() -> None:
    from seiso.models.hf_env import configure_hf_hub_auth

    configure_hf_hub_auth()


def _load_stream(repo_id: str, *, config: str | None = None, token: str | None = None):
    from datasets import load_dataset

    kwargs: dict = {"split": "train", "streaming": True}
    if token:
        kwargs["token"] = token
    if config:
        return load_dataset(repo_id, config, **kwargs)
    return load_dataset(repo_id, **kwargs)


def _hub_token() -> str | None:
    from seiso.models.hf_env import _read_hub_token

    return _read_hub_token()


def _iter_repo_rows(
    repo_id: str,
    *,
    configs: tuple[str, ...] | None,
    max_samples: int | None,
    skip_metadata_only: bool,
    token: str | None,
):
    configs = configs or (None,)
    kept = 0
    for config in configs:
        label = f"{repo_id}" if config is None else f"{repo_id}/{config}"
        print(f"Streaming {label}", file=sys.stderr)
        stream = _load_stream(repo_id, config=config, token=token)
        for row in stream:
            if skip_metadata_only and is_metadata_only_row(row):
                continue
            sample = normalize_code_row(row, source=label)
            if sample is None:
                continue
            kept += 1
            yield sample.to_record()
            if max_samples is not None and kept >= max_samples:
                return


def _iter_normalized(
    repos: list[str],
    *,
    configs: tuple[str, ...] | None,
    max_samples: int | None,
    skip_metadata_only: bool,
    token: str | None,
):
    kept = 0
    for repo_id in repos:
        try:
            for record in _iter_repo_rows(
                repo_id,
                configs=configs,
                max_samples=None if max_samples is None else max(0, max_samples - kept),
                skip_metadata_only=skip_metadata_only,
                token=token,
            ):
                kept += 1
                yield record
                if max_samples is not None and kept >= max_samples:
                    return
        except Exception as exc:
            print(f"SKIP {repo_id}: {exc}", file=sys.stderr)
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default="~/.seiso/datasets/code-pretrain.jsonl",
        help="Normalized JSONL output path",
    )
    parser.add_argument(
        "--repos",
        required=True,
        help="Comma-separated HF dataset repos (e.g. org/code-pretrain-v2)",
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated dataset configs/subsets (empty = default split)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap rows written")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--model-id",
        default="",
        help="Optional base model id for epoch recommendation (uses active params when set)",
    )
    parser.add_argument(
        "--model-params-b",
        type=float,
        default=None,
        help="Override active parameter count (billions) for epoch recommendation",
    )
    parser.add_argument(
        "--include-metadata-only",
        action="store_true",
        help="Keep metadata-only rows (not useful for training)",
    )
    args = parser.parse_args()

    _configure_hub_auth()
    token = _hub_token()
    if not token:
        print(
            "No HF token found — set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated datasets.",
            file=sys.stderr,
        )
        return 1

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = tuple(config_names) if config_names else None

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with out.open("w", encoding="utf-8") as handle:
            for record in _iter_normalized(
                repos,
                configs=configs,
                max_samples=args.max_samples,
                skip_metadata_only=not args.include_metadata_only,
                token=token,
            ):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "gated" in msg.lower():
            print(
                "\nAccess denied for a gated dataset. Accept the dataset terms on Hugging Face "
                "with the same account as your HF token.",
                file=sys.stderr,
            )
            return 1
        raise

    if args.model_params_b is not None:
        params_b = float(args.model_params_b)
    elif args.model_id.strip():
        params_b = infer_active_params_b(args.model_id.strip(), trust_remote_code=True)
    else:
        params_b = 7.0

    epoch_info = recommend_pretraining_epochs(
        sample_count=written,
        max_seq_length=args.max_seq_length,
        model_params_b=params_b,
    )
    manifest = {
        "output": str(out.resolve()),
        "repos": repos,
        "configs": list(configs or ()),
        "rows_written": written,
        **epoch_info,
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    if written == 0:
        print("\nNo training rows written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
