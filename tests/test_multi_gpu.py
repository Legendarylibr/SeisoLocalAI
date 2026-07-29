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
