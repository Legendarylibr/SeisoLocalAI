from pathlib import Path
import json

import pytest

from seiso.training.config import TrainConfig
from seiso.training.multi_gpu import (
    GpuLayout,
    configure_distributed_training_args,
    detect_training_layout,
    launch_worker_command,
    resolve_distributed_plan,
)


def test_detect_training_layout():
    layout = detect_training_layout()
    assert layout.world_size >= 1
    assert layout.local_rank >= 0


def test_launch_worker_command():
    cmd = launch_worker_command("/tmp/cfg.yaml", 2)
    assert cmd[:4] == ["accelerate", "launch", "--multi_gpu", "--num_processes=2"]
    assert "--module" in cmd
    assert "seiso.training.worker" in cmd


def test_resolve_distributed_plan_defaults_to_single_process():
    cfg = TrainConfig(model_id="m", dataset="/tmp/data.jsonl")
    layout = GpuLayout(
        world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
    )

    plan = resolve_distributed_plan(cfg, layout)

    assert plan.enabled is False
    assert plan.strategy == "none"
    assert plan.world_size == 1


def test_distributed_strategy_none_keeps_single_process_when_multi_gpu_set():
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_strategy="none",
    )
    layout = GpuLayout(
        world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
    )

    plan = resolve_distributed_plan(cfg, layout)

    assert plan.enabled is False
    assert plan.strategy == "none"


def test_resolve_distributed_plan_uses_requested_gpu_count():
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_nproc_per_node=2,
    )
    layout = GpuLayout(
        world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=4
    )

    plan = resolve_distributed_plan(cfg, layout)

    assert plan.enabled is True
    assert plan.strategy == "ddp"
    assert plan.nproc_per_node == 2
    assert plan.world_size == 2


def test_resolve_distributed_plan_rejects_more_processes_than_visible_gpus():
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_nproc_per_node=3,
    )
    layout = GpuLayout(
        world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
    )

    with pytest.raises(ValueError, match="exceeds visible GPU count"):
        resolve_distributed_plan(cfg, layout)


def test_launch_worker_command_includes_multinode_args(monkeypatch: pytest.MonkeyPatch):
    from seiso.research.nostr.keys import generate_keypair

    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", generate_keypair().nsec)
    monkeypatch.setenv("SEISO_MESH_JOB_ID", "job-test-multinode")
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_nproc_per_node=2,
        distributed_num_nodes=2,
        distributed_node_rank=1,
        distributed_master_addr="10.0.0.2",
        distributed_master_port=29555,
    )
    plan = resolve_distributed_plan(
        cfg,
        GpuLayout(
            world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
        ),
    )

    cmd = launch_worker_command("/tmp/cfg.yaml", plan)

    assert "--num_processes=4" in cmd
    assert "--num_machines=2" in cmd
    assert "--machine_rank=1" in cmd
    assert "--main_process_ip=10.0.0.2" in cmd
    assert "--main_process_port=29555" in cmd


def test_launch_worker_command_refuses_multinode_without_mesh_job(
    monkeypatch: pytest.MonkeyPatch,
):
    from seiso.research.nostr.keys import generate_keypair

    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", generate_keypair().nsec)
    monkeypatch.delenv("SEISO_MESH_JOB_ID", raising=False)
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_nproc_per_node=2,
        distributed_num_nodes=2,
        distributed_node_rank=0,
        distributed_master_addr="10.0.0.2",
    )
    plan = resolve_distributed_plan(
        cfg,
        GpuLayout(
            world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
        ),
    )
    with pytest.raises(ValueError, match="SEISO_MESH_JOB_ID"):
        launch_worker_command("/tmp/cfg.yaml", plan)


def test_resolve_distributed_plan_refuses_multinode_without_buzz_mesh(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        distributed_nproc_per_node=2,
        distributed_num_nodes=2,
        distributed_master_addr="10.0.0.2",
    )
    layout = GpuLayout(
        world_size=1, local_rank=0, device="cuda", use_ddp=False, device_count=2
    )
    with pytest.raises(ValueError, match="Buzz-agent/mesh-only"):
        resolve_distributed_plan(cfg, layout)


def test_configure_distributed_training_args_honors_ddp_options(monkeypatch):
    monkeypatch.setattr(
        "seiso.memory.protection.training_pin_memory",
        lambda: True,
    )
    cfg = TrainConfig(
        model_id="m",
        dataset="/tmp/data.jsonl",
        multi_gpu=True,
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
    )
    layout = GpuLayout(
        world_size=2, local_rank=1, device="cuda:1", use_ddp=True, device_count=2
    )

    args = configure_distributed_training_args({}, layout, cfg, enabled=True)

    assert args["local_rank"] == 1
    assert args["ddp_backend"] == "nccl"
    assert args["ddp_find_unused_parameters"] is True
    assert args["dataloader_pin_memory"] is True


def test_cloud_gpu_config_requires_provider_and_instance_type():
    with pytest.raises(ValueError, match="cloud_gpu_provider is required"):
        TrainConfig(
            model_id="m",
            dataset="/tmp/data.jsonl",
            cloud_gpu_enabled=True,
        )

    with pytest.raises(ValueError, match="cloud_gpu_instance_type is required"):
        TrainConfig(
            model_id="m",
            dataset="/tmp/data.jsonl",
            cloud_gpu_enabled=True,
            cloud_gpu_provider="aws",
        )


def test_cloud_gpu_config_rejects_secret_like_labels():
    with pytest.raises(ValueError, match="cannot contain secrets"):
        TrainConfig(
            model_id="m",
            dataset="/tmp/data.jsonl",
            cloud_gpu_enabled=True,
            cloud_gpu_provider="aws",
            cloud_gpu_region="token=abc",
            cloud_gpu_instance_type="p5.48xlarge",
        )

def test_example_vllm_yaml_enables_ddp():
    from seiso.training.config import DistributedStrategy, TrainConfig
    from seiso.training.multi_gpu import distributed_requested

    cfg = TrainConfig.from_yaml("configs/example_training_slime_vllm.yaml")
    assert cfg.multi_gpu is True
    assert cfg.distributed_strategy == DistributedStrategy.DDP
    assert distributed_requested(cfg) is True

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

def test_detect_training_layout_ignores_stale_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    class _Torch:
        cuda = _Cuda()

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.training.multi_gpu import detect_training_layout

    layout = detect_training_layout()
    assert layout.world_size == 1
    assert layout.use_ddp is False

def test_detect_training_layout_keeps_multi_node(monkeypatch):
    monkeypatch.setenv("SEISO_DISTRIBUTED_WORKER", "1")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 8

    import sys
    from types import ModuleType

    fake = ModuleType("torch")
    fake.cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", fake)

    from seiso.training.multi_gpu import detect_training_layout

    layout = detect_training_layout()
    assert layout.world_size == 16
    assert layout.local_rank == 1
    assert layout.use_ddp is True

