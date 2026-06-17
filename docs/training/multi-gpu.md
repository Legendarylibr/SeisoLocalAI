# Multi-GPU training

Requires multiple NVIDIA (or ROCm) GPUs visible to PyTorch.

## Forge

Enable **Multi-GPU** in Training Studio. Forge launches:

```bash
torchrun --nproc_per_node=<N> -m seiso.training.worker --config <temp.yaml>
```

## CLI / manual

```bash
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs/example_lora.yaml
```

## Config

```yaml
multi_gpu: true
```

## Behavior

- Rank 0 writes checkpoint and manifest
- Non-rank workers call `release_training_memory()` and exit without saving
- DDP via HuggingFace `TrainingArguments` (`ddp_find_unused_parameters: false`)

## macOS / single GPU

Multi-GPU checkbox is disabled when `training_defaults.multi_gpu_available` is false.
