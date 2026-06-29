from seiso.training.multi_gpu import detect_training_layout, launch_worker_command


def test_detect_training_layout():
    layout = detect_training_layout()
    assert layout.world_size >= 1
    assert layout.local_rank >= 0


def test_launch_worker_command():
    cmd = launch_worker_command("/tmp/cfg.yaml", 2)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=2" in cmd
