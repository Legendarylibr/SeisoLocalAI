# Distributed training

Requires multiple NVIDIA (or ROCm) GPUs visible to PyTorch.

Seiso uses Hugging Face Accelerate for distributed launches. See the
[huggingface/accelerate repository](https://github.com/huggingface/accelerate).

## Forge

Open **Training Studio → Distributed** and enable local multi-GPU. Forge launches:

```bash
accelerate launch --multi_gpu --num_processes=<N> --module seiso.training.worker --config <temp.yaml>
```

The Distributed tab can also override the current setup tab training knobs for
distributed runs only: per-device batch size, gradient accumulation, learning rate,
sequence length, epochs, logging cadence, checkpoint cadence, and eval sample cap.
Those overrides are applied only when distributed launch is enabled; single-GPU
training continues to use the normal setup tab values.

## CLI / manual

```bash
seiso train --config configs/example_lora.yaml
```

When `multi_gpu: true` or `distributed_strategy: ddp` is set, the CLI launches
Accelerate for you. You can still invoke the worker directly for custom schedulers:

```bash
accelerate launch --multi_gpu --num_processes=2 --module seiso.training.worker --config configs/example_lora.yaml
```

## Config

```yaml
multi_gpu: true
distributed_strategy: auto        # auto, none, ddp
distributed_nproc_per_node: 2     # null = all visible GPUs on this machine
distributed_num_nodes: 1
distributed_node_rank: 0
distributed_master_addr: 127.0.0.1
distributed_master_port: 29500
ddp_backend: null                 # e.g. nccl or gloo
ddp_find_unused_parameters: false
```

Single-GPU training is unchanged outside distributed mode. Leave `multi_gpu: false`
or set `distributed_strategy: none` to run the existing single-process trainer.

## Behavior

- Rank 0 writes checkpoint and manifest
- Non-rank workers call `release_training_memory()` and exit without saving
- DDP via HuggingFace `TrainingArguments`
- Forge and CLI use the same distributed plan resolution
- Multi-node launches add Accelerate `--num_machines`, `--machine_rank`,
  `--main_process_ip`, and `--main_process_port`
- Keep `ddp_find_unused_parameters: false` unless the model graph really has unused trainable parameters

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
