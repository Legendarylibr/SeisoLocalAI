# Multi-GPU training

Requires multiple NVIDIA (or ROCm) GPUs visible to PyTorch.

## Forge

Open **Training Studio → Distributed** and enable local multi-GPU. Forge launches:

```bash
torchrun --nproc_per_node=<N> -m seiso.training.worker --config <temp.yaml>
```

## CLI / manual

```bash
seiso train --config configs/example_lora.yaml
```

When `multi_gpu: true` or `distributed_strategy: ddp` is set, the CLI launches
`torchrun` for you. You can still invoke the worker directly for custom schedulers:

```bash
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs/example_lora.yaml
```

## Config

```yaml
multi_gpu: true
distributed_strategy: auto        # auto, none, ddp
distributed_nproc_per_node: 2     # null = all visible GPUs
distributed_num_nodes: 1
distributed_node_rank: 0
distributed_master_addr: 127.0.0.1
distributed_master_port: 29500
ddp_backend: null                 # e.g. nccl or gloo
ddp_find_unused_parameters: false
```

Single-GPU training remains the default. Leave `multi_gpu: false` and
`distributed_strategy: auto` (or set `distributed_strategy: none`) to run the normal
single-process trainer.

## Behavior

- Rank 0 writes checkpoint and manifest
- Non-rank workers call `release_training_memory()` and exit without saving
- DDP via HuggingFace `TrainingArguments`
- Forge and CLI use the same distributed plan resolution
- Multi-node launches add `--nnodes`, `--node_rank`, `--master_addr`, and `--master_port`

## Cloud GPU metadata

The **Distributed** tab also lets you record a cloud GPU target:

```yaml
cloud_gpu_enabled: true
cloud_gpu_provider: aws          # aws, gcp, azure, lambda, runpod, coreweave, custom
cloud_gpu_region: us-east-1
cloud_gpu_instance_type: p5.48xlarge
cloud_gpu_count: 8
cloud_gpu_project: finetune-prod
```

This is metadata for a secure external launcher or scheduler. Seiso does not accept
cloud API keys, SSH keys, shell commands, or provider tokens in training configs.
Secret-looking labels and URLs are rejected before the job is stored.

## macOS / single GPU

Multi-GPU checkbox is disabled when `training_defaults.multi_gpu_available` is false.
