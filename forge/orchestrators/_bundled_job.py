"""Shared executor pattern for bundled pipeline orchestrators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path, is_local_filesystem_path
from seiso.security import SecurityError


@dataclass(frozen=True)
class BundledJobContract:
    """Boundary contract for research-code runners integrated into Forge."""

    artifact_keys: tuple[str, ...] = (
        "output_dir",
        "output_root",
        "run_dir",
        "model_dir",
        "recommendation_path",
        "paper_bundle",
    )
    nested_artifact_keys: tuple[str, ...] = ("stage_results", "artifacts", "summary")
    requires_manifest: bool = False


def _validate_artifact_value(
    sandbox_root: Path,
    user_id: str,
    key: str,
    value: Any,
) -> None:
    if isinstance(value, str) and is_local_filesystem_path(value):
        try:
            assert_user_path(sandbox_root, user_id, value)
        except SecurityError as exc:
            raise PermissionError(f"Bundled artifact {key!r} is outside sandbox: {exc}") from exc
    elif isinstance(value, dict):
        for nested_key, nested_value in value.items():
            _validate_artifact_value(
                sandbox_root, user_id, f"{key}.{nested_key}", nested_value
            )
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_artifact_value(sandbox_root, user_id, f"{key}[{idx}]", item)


def validate_bundled_result(
    sandbox_root: Path,
    user_id: str,
    result: dict[str, Any],
    contract: BundledJobContract,
) -> None:
    """Validate runner-returned artifact paths before routes persist them."""
    if contract.requires_manifest and not result.get("manifest"):
        raise RuntimeError("Bundled job did not return a manifest")
    for key in contract.artifact_keys:
        if key in result and result[key]:
            _validate_artifact_value(sandbox_root, user_id, key, result[key])
    for key in contract.nested_artifact_keys:
        if key in result and result[key]:
            _validate_artifact_value(sandbox_root, user_id, key, result[key])


async def run_bundled_job(
    orchestrator: Orchestrator,
    job_id: str,
    payload: dict[str, Any],
    *,
    user_id_error: str,
    path_keys: tuple[str, ...],
    start_message: str,
    runner: Callable[..., dict[str, Any]],
    result_log: Callable[[dict[str, Any]], str],
    contract: BundledJobContract = BundledJobContract(),
) -> dict[str, Any]:
    from forge.services.memory_release import prepare_for_gpu_task, release_after_task

    user_id = payload.get("user_id")
    if not user_id:
        raise PermissionError(user_id_error)

    for key in path_keys:
        if path := payload.get(key):
            try:
                assert_user_path(orchestrator.sandbox_root, user_id, path)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

    prepare_for_gpu_task(
        task=orchestrator.kind,
        job_id=job_id,
        log=lambda msg: orchestrator._emit_log(job_id, msg),
    )
    policy = {
        "trust_remote_code": bool(payload.get("trust_remote_code", False)),
        "external_tools": True,
    }
    orchestrator._emit_event(job_id, "policy", {"bundled": policy})
    orchestrator._emit_log(job_id, start_message)
    loop = asyncio.get_running_loop()

    def on_log(msg: str) -> None:
        loop.call_soon_threadsafe(orchestrator._emit_log, job_id, msg)

    try:
        from forge.services.executors import GPU_EXECUTOR

        result = await loop.run_in_executor(
            GPU_EXECUTOR,
            lambda: runner(
                job_id=job_id,
                user_id=user_id,
                data_dir=Path(orchestrator.sandbox_root),
                payload=payload,
                on_log=on_log,
            ),
        )
    finally:
        release_after_task(
            reason=f"{orchestrator.kind} complete",
            log=lambda msg: orchestrator._emit_log(job_id, msg),
            job_id=job_id,
        )

    validate_bundled_result(orchestrator.sandbox_root, user_id, result, contract)
    orchestrator._emit_log(job_id, result_log(result))
    return result


def bundled_orchestrator(
    *,
    class_name: str,
    kind: str,
    user_id_error: str,
    path_keys: tuple[str, ...],
    start_message: str,
    runner: Callable[..., dict[str, Any]],
    result_log: Callable[[dict[str, Any]], str],
    contract: BundledJobContract = BundledJobContract(),
) -> type[Orchestrator]:
    """Build a thin Orchestrator subclass for a bundled pipeline runner."""

    class _BundledOrchestrator(Orchestrator):
        resource_key = "gpu"

        async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            return await run_bundled_job(
                self,
                job_id,
                payload,
                user_id_error=user_id_error,
                path_keys=path_keys,
                start_message=start_message,
                runner=runner,
                result_log=result_log,
                contract=contract,
            )

    _BundledOrchestrator.kind = kind
    _BundledOrchestrator.__name__ = class_name
    _BundledOrchestrator.__qualname__ = class_name
    return _BundledOrchestrator
