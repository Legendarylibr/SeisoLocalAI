"""Regression tests for the 2026-07-26 codebase bug-review fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_strip_attributed_think_blocks():
    from seiso.chat.sanitize import strip_leaked_reasoning

    attributed = '<think channel="analysis">API_KEY=x</think>\nVisible'
    assert strip_leaked_reasoning(attributed) == "Visible"
    bare = "<think>secret</think>\nok"
    assert strip_leaked_reasoning(bare) == "ok"


def test_format_sample_uses_code_column():
    from seiso.training.datasets import DatasetFormat, format_sample

    assert format_sample({"code": "print(1)"}, DatasetFormat.TEXT, None) == "print(1)"


def test_bundled_result_rejects_failed_manifest(tmp_path: Path):
    from forge.orchestrators._bundled_job import (
        BundledJobContract,
        validate_bundled_result,
    )

    user_id = "user-1"
    run_dir = tmp_path / "compress" / user_id / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Manifest verification failed"):
        validate_bundled_result(
            tmp_path,
            user_id,
            {
                "run_dir": str(run_dir),
                "manifest": {"ok": False, "error": "hash mismatch"},
            },
            BundledJobContract(requires_manifest=True),
        )


@pytest.mark.asyncio
async def test_job_log_event_gen_skips_result_when_cancelled(tmp_path: Path):
    from forge.api.routes._stream import job_log_event_gen
    from forge.orchestrators.base import JobRecord, JobStatus, Orchestrator

    class _Orch(Orchestrator):
        kind = "test"

        async def execute(self, job_id: str, payload: dict) -> dict:
            return {}

    orch = _Orch(tmp_path)
    job_id = "job-cancel-result"
    rec = JobRecord(id=job_id, kind="test", user_id="u1")
    rec.status = JobStatus.CANCELLED
    rec.result = {"model_dir": "/tmp/x"}
    orch._jobs[job_id] = rec

    events = [event async for event in job_log_event_gen(orch, job_id)]
    assert not any(e.get("event") == "result" for e in events)


def test_restore_registry_keeps_modules_on_failure(monkeypatch):
    from seiso.kernels import lifecycle as life

    class Mod:
        pass

    m1, m2 = Mod(), Mod()
    m1._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m2._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m1.forward = lambda x: 1  # type: ignore[attr-defined]
    m2.forward = lambda x: 2  # type: ignore[attr-defined]

    orig_clear = life._clear_patch_markers

    def flaky(module: object) -> None:
        if module is m2:
            raise RuntimeError("restore boom")
        orig_clear(module)

    monkeypatch.setattr(life, "_clear_patch_markers", flaky)
    life._PATCH_REGISTRY.clear()
    life._PATCH_REGISTRY[42] = [m1, m2]
    with pytest.raises(RuntimeError, match="restore boom"):
        life._restore_registry_key(42)
    assert life._PATCH_REGISTRY[42] == [m2]
    life._PATCH_REGISTRY.clear()


def test_hf_token_no_host_fallback_for_user(monkeypatch, tmp_path: Path):
    from forge.db.crypto import generate_encryption_key
    from forge.services.hf_auth import resolve_hf_token

    monkeypatch.setenv("HF_TOKEN", "hf_host_secret")
    monkeypatch.delenv("SEISO_HF_ALLOW_HOST_TOKEN", raising=False)
    monkeypatch.setattr("forge.services.hf_auth._read_cli_token", lambda: "hf_cli")
    key = generate_encryption_key()
    token, source = resolve_hf_token(
        user_id="bob",
        data_dir=tmp_path,
        encryption_key=key,
    )
    assert token is None
    assert source == "none"

    monkeypatch.setenv("SEISO_HF_ALLOW_HOST_TOKEN", "1")
    token, source = resolve_hf_token(
        user_id="bob",
        data_dir=tmp_path,
        encryption_key=key,
    )
    assert token == "hf_host_secret"
    assert source == "env_hf"


def test_export_checksum_hashes_weight_files(tmp_path: Path):
    from seiso.research.provenance import directory_checksum_manifest

    (tmp_path / "model.safetensors").write_bytes(b"weights" * 1000)
    (tmp_path / "readme.txt").write_bytes(b"x" * 32)
    manifest = directory_checksum_manifest(
        tmp_path,
        max_files=None,
        max_file_bytes=8,
        always_hash_suffixes=(".safetensors",),
    )
    assert manifest["readme.txt"] == "skipped-large-file"
    assert manifest["model.safetensors"] not in {"skipped-large-file", "error"}
    assert len(manifest["model.safetensors"]) == 64


def test_example_vllm_yaml_enables_ddp():
    from seiso.training.config import DistributedStrategy, TrainConfig
    from seiso.training.multi_gpu import distributed_requested

    cfg = TrainConfig.from_yaml("configs/example_training_slime_vllm.yaml")
    assert cfg.multi_gpu is True
    assert cfg.distributed_strategy == DistributedStrategy.DDP
    assert distributed_requested(cfg) is True


def test_example_lora_uses_hub_dataset_not_toy_sample():
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/example_lora.yaml")
    assert "sample.jsonl" not in str(cfg.dataset)
    assert "/" in str(cfg.dataset)  # Hub id


def test_smoke_slime_max_steps_projects_via_train_config(monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.from_yaml("configs/smoke_slime_cpu.yaml")
    assert cfg.max_steps == 1
    slime = cfg.to_single_gpu_slime_config()
    assert slime.max_steps == 1


def test_slime_distributed_context_ignores_stale_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def set_device(_idx):
            raise AssertionError("stale DDP must not set_device")

    class _Torch:
        cuda = _Cuda()
        distributed = None

    from seiso.slime.config import SingleGpuSlimeConfig
    from seiso.slime.distributed import _distributed_context

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=Path("data/slime_sample.jsonl"),
        output_dir=Path("/tmp/out"),
        device="cuda",
        rollouts_per_prompt=2,
        policy_micro_batch_size=2,
        require_held_out_eval=False,
    )
    ctx = _distributed_context(_Torch(), cfg)
    assert ctx.enabled is False
    assert ctx.world_size == 1


def test_resolve_distributed_env_keeps_multi_node(monkeypatch):
    """Global WORLD_SIZE > local GPU count is multi-node, not stale."""
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    from seiso.training.multi_gpu import resolve_distributed_env

    env = resolve_distributed_env(device_count=8)
    assert env.enabled is True
    assert env.world_size == 16
    assert env.local_rank == 0


def test_resolve_distributed_env_rejects_master_leftover_without_worker_proof(
    monkeypatch,
):
    """Parent shell with MASTER_* + in-range ranks must not false-enable DDP."""
    monkeypatch.delenv("SEISO_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.delenv("GROUP_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    from seiso.training.multi_gpu import resolve_distributed_env

    env = resolve_distributed_env(device_count=2)
    assert env.enabled is False
    assert env.stale is True


def test_resolve_distributed_env_rejects_empty_group_rank(monkeypatch):
    """Empty GROUP_RANK leftover must not count as worker proof."""
    monkeypatch.delenv("SEISO_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.setenv("GROUP_RANK", "")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    from seiso.training.multi_gpu import resolve_distributed_env

    env = resolve_distributed_env(device_count=2)
    assert env.enabled is False
    assert env.stale is True


def test_resolve_distributed_env_rejects_empty_master_marker(monkeypatch):
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "")
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("GROUP_RANK", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

    from seiso.training.multi_gpu import resolve_distributed_env

    assert resolve_distributed_env(device_count=2).enabled is False


def test_resolve_distributed_env_requires_rank(monkeypatch):
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    from seiso.training.multi_gpu import resolve_distributed_env

    assert resolve_distributed_env(device_count=2).enabled is False


def test_is_main_process_follows_resolver_when_stale(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.delenv("SEISO_DISTRIBUTED_WORKER", raising=False)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("GROUP_RANK", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

    from seiso.training.metrics import is_main_process

    # Stale → single process → rank 0 for save/manifest gates.
    assert is_main_process() is True


def test_resolve_training_device_map_ignores_stale_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.memory.protection.device_map import resolve_training_device_map

    assert resolve_training_device_map("cuda") == "auto"


def test_resolve_training_device_map_pins_multi_node_rank(monkeypatch):
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 4

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.memory.protection.device_map import resolve_training_device_map

    assert resolve_training_device_map("cuda") == {"": "cuda:2"}


def test_slime_manifest_uses_runtime_world_size(monkeypatch, tmp_path: Path):
    for key in (
        "WORLD_SIZE",
        "LOCAL_RANK",
        "RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_WORLD_SIZE",
        "SEISO_DISTRIBUTED_WORKER",
        "GROUP_RANK",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    from seiso.training.config import TrainConfig, _write_slime_manifest

    cfg = TrainConfig.model_validate(
        {
            "model_id": "m",
            "dataset": "open-r1/OpenR1-Math-220k",
            "output_dir": tmp_path / "out",
            "method": "slime",
            "multi_gpu": True,
            "distributed_strategy": "ddp",
            "data_gen": True,
            "data_gen_source": "dataset",
            "dataset_ref": "open-r1/OpenR1-Math-220k",
            "data_gen_count": 2048,
            "require_held_out_eval": True,
        }
    )
    _write_slime_manifest(cfg, tmp_path)
    payload = json.loads((tmp_path / "seiso_manifest.json").read_text(encoding="utf-8"))
    assert payload["post_training_algorithm"] == "single_gpu_slime_grpo"

    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    _write_slime_manifest(cfg, tmp_path)
    payload = json.loads((tmp_path / "seiso_manifest.json").read_text(encoding="utf-8"))
    assert payload["post_training_algorithm"] == "distributed_slime_grpo"


def test_forge_pipeline_defaults_are_product_presets():
    from forge.api.routes.compress import CompressStartRequest
    from forge.api.routes.distill_rl import DistillRLStartRequest

    assert DistillRLStartRequest().preset == "reproducible"
    assert CompressStartRequest().preset == "full"


def test_rl_quant_keeps_gguf_export_when_gguf_path_promotes_backend():
    """gguf_path implies llama_cpp — do not clear export using default simulator."""
    from forge.api.routes.rl_quant import (
        RLQuantStartRequest,
        _effective_rl_quant_backend,
    )

    config = RLQuantStartRequest(
        gguf_path="/tmp/model.gguf",
        gguf_export=True,
        backend="simulator",
    ).model_dump()
    assert _effective_rl_quant_backend(config) == "llama_cpp"
    if _effective_rl_quant_backend(config) == "simulator":
        config["gguf_export"] = False
    assert config["gguf_export"] is True


def test_emit_standard_artifacts_skips_orphan_distill_by_default(tmp_path: Path):
    from seiso.rl_verify.synth_code import emit_standard_artifacts

    stats = emit_standard_artifacts(
        data_dir=tmp_path, seed=0, verify=True, limit=2, include_variants=False
    )
    assert "distill_code_synth" not in stats
    assert not (tmp_path / "distill_code_synth.jsonl").exists()
    assert (tmp_path / "slime_code_sample.jsonl").is_file()


def test_knowledge_upload_refuses_symlink_destination(tmp_path: Path):
    import asyncio

    from fastapi import HTTPException

    from forge.api.routes import knowledge as kb
    from forge.config import ForgeSettings

    data = tmp_path / "data"
    uploads = data / "uploads" / "alice"
    uploads.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("secret", encoding="utf-8")
    dest = uploads / "planted.txt"
    dest.symlink_to(victim)

    class FakeUpload:
        filename = "planted.txt"

        async def read(self) -> bytes:
            return b"new-content"

    settings = ForgeSettings(data_dir=data)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            kb.upload_file(
                user_id="alice",
                settings=settings,
                file=FakeUpload(),  # type: ignore[arg-type]
            )
        )
    assert exc.value.status_code == 400
    assert victim.read_text(encoding="utf-8") == "secret"
    assert dest.is_symlink()


def test_resolve_distributed_artifact_prefers_slime_final(tmp_path: Path):
    from forge.orchestrators.training import TrainingOrchestrator

    out = tmp_path / "run"
    out.mkdir()
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "adapter_model.safetensors").write_bytes(b"weights")
    older = out / "checkpoint-1"
    older.mkdir()
    (older / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "slime_training_state.json").write_text(
        json.dumps(
            {
                "final_checkpoint_dir": str(out),
                "best_checkpoint_dir": str(older),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = TrainingOrchestrator._resolve_distributed_artifact(out)
    assert resolved == out


def test_resolve_distributed_artifact_falls_back_to_checkpoint_glob(tmp_path: Path):
    from forge.orchestrators.training import TrainingOrchestrator

    out = tmp_path / "run"
    out.mkdir()
    ckpt = out / "checkpoint-3"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"w")

    assert TrainingOrchestrator._resolve_distributed_artifact(out) == ckpt


def test_resolve_distributed_artifact_prefers_newer_root_over_old_checkpoint(
    tmp_path: Path,
):
    import os
    import time

    from forge.orchestrators.training import TrainingOrchestrator

    out = tmp_path / "run"
    out.mkdir()
    older = out / "checkpoint-1"
    older.mkdir()
    (older / "adapter_config.json").write_text("{}", encoding="utf-8")
    time.sleep(0.02)
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "adapter_model.safetensors").write_bytes(b"final")
    # Ensure root mtime is newer even on coarse filesystems.
    os.utime(out, None)

    assert TrainingOrchestrator._resolve_distributed_artifact(out) == out


def test_broadcast_vllm_full_resumes_after_update_failure(monkeypatch):
    from seiso.slime import rollout_sync
    from seiso.slime.config import SingleGpuSlimeConfig

    events: list[str] = []

    class FakeClient:
        def __init__(self) -> None:
            self.base_url = ""

        @classmethod
        def from_config(cls, _config):
            return cls()

        def pause(self) -> None:
            events.append("pause")

        def update_weights_from_disk(self, *_a, **_k) -> None:
            events.append("update")
            raise RuntimeError("disk sync failed")

        def resume(self) -> None:
            events.append("resume")

    monkeypatch.setattr(rollout_sync, "VLLMRolloutClient", FakeClient)
    monkeypatch.setattr(
        rollout_sync, "vllm_engine_urls", lambda *_a, **_k: ["http://127.0.0.1:8000"]
    )
    monkeypatch.setattr(rollout_sync, "resolve_vllm_base_url", lambda *_a, **_k: None)

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset="d.jsonl",
        output_dir="/tmp/out",
        vllm_base_url="http://127.0.0.1:8000",
    )
    with pytest.raises(RuntimeError, match="vLLM full weight sync failed"):
        rollout_sync._broadcast_vllm_full(cfg, model_path="/tmp/w", weight_version="v1")
    assert events == ["pause", "update", "resume"]


def test_keep_rollout_group_uses_sample_std():
    """Filter std must match GRPO advantage unbiased std (n-1)."""
    from seiso.slime.config import SingleGpuSlimeConfig
    from seiso.slime.policy import _keep_rollout_group
    from seiso.slime.types import Rollout

    # Two rewards: mean 0.5, sample std = sqrt(0.5) ≈ 0.707; population ≈ 0.5.
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset="d.jsonl",
        output_dir="/tmp/out",
        dynamic_sampling_filter="reward_nonzero_std",
        dynamic_sampling_min_reward_std=0.6,
        rollouts_per_prompt=2,
    )

    def _r(reward: float) -> Rollout:
        return Rollout(
            None,
            None,
            None,
            None,
            None,
            reward=reward,
            outcome_reward=reward,
            status="stop",
        )

    assert _keep_rollout_group([_r(0.0), _r(1.0)], cfg) is True
