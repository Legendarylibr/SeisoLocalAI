from __future__ import annotations

from pathlib import Path

import pytest


def test_bundled_result_allows_user_scoped_research_roots(tmp_path: Path):
    from forge.orchestrators._bundled_job import (
        BundledJobContract,
        validate_bundled_result,
    )

    user_id = "user-1"
    run_dir = tmp_path / "compress" / user_id / "runs" / "run-a"
    run_dir.mkdir(parents=True)

    validate_bundled_result(
        tmp_path,
        user_id,
        {"run_dir": str(run_dir), "stage_results": {"distilled": str(run_dir)}},
        BundledJobContract(),
    )


def test_bundled_result_rejects_artifact_outside_user_scope(tmp_path: Path):
    from forge.orchestrators._bundled_job import (
        BundledJobContract,
        validate_bundled_result,
    )

    user_id = "user-1"
    escaped = tmp_path / "compress" / "other-user" / "run"
    escaped.mkdir(parents=True)

    with pytest.raises(PermissionError, match="outside sandbox"):
        validate_bundled_result(
            tmp_path,
            user_id,
            {"run_dir": str(escaped)},
            BundledJobContract(),
        )
