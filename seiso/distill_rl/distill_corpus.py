"""Optional corpus override when aligning distillation with rollout prompts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def override_distill_corpus(texts: list[str]) -> Iterator[None]:
    """Feed fixed texts into seiso.codellama_compress distillation instead of Hub datasets."""
    from unittest.mock import patch

    with patch(
        "seiso.codellama_compress.distill.iter_dataset_texts", lambda _cfg: iter(texts)
    ):
        yield
