"""Distributed training worker entrypoint (Accelerate launch)."""

from __future__ import annotations

import argparse
from pathlib import Path

from seiso.training.config import TrainConfig, run_training
from seiso.training.multi_gpu import mark_distributed_worker


def main() -> None:
    # Must run before layout / is_main_process / slime DDP resolve env.
    mark_distributed_worker()
    parser = argparse.ArgumentParser(description="Seiso distributed training worker")
    parser.add_argument("--config", required=True, help="Path to training YAML/JSON config")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(Path(args.config))
    run_training(cfg)


if __name__ == "__main__":
    main()
