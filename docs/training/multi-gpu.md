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

## Slime multi-GPU rollouts (vLLM or SGLang)

`method: slime` can keep **policy updates on Accelerate DDP** while sending
**generation** to a multi-GPU inference server:

| Backend | Config | Launch helper |
|---------|--------|---------------|
| **vLLM** (tensor parallel) | `rollout_backend: vllm` + `configs/example_training_slime_vllm.yaml` | `scripts/run_slime_vllm_ddp.sh` |
| **SGLang** | `rollout_backend: sglang` + `configs/example_training_slime_ddp.yaml` | `scripts/run_slime_ddp.sh` |

vLLM weight sync defaults to dynamic LoRA (`/v1/load_lora_adapter`) when
`slime_use_lora: true`. Start the server with `--enable-lora`, or set
`SEISO_MANAGED_VLLM_ENABLE_LORA=true` for Seiso-managed multi-GPU. Single-GPU
`rollout_backend: hf` and existing LoRA/SFT multi-GPU paths are unchanged.

**Logprobs:** remote engines sample tokens; Seiso recomputes `old_logprobs` on the
local actor for GRPO (engine sampling logprobs unused). Keep weight sync on to
limit off-policyness — see [quickstart § Slime](quickstart.md#slime-post-training).

### Data (RLVR defaults)

Prefer a grounded `dataset` + frozen `eval_dataset`. Synth is opt-in only
(`data_gen_source` defaults to `off`). For Data Designer set `data_gen: true`,
`data_gen_source: data_designer`, `data_designer: "on"`, and an explicit
`vllm_base_url` ([NVIDIA NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)).
No silent localhost / package auto-select. Code RL uses HF/operator JSONL.

```bash
pip install -e '.[data-designer]'
# then run with data_gen_source=data_designer and a live vLLM endpoint
```

## Behavior

- Rank 0 writes checkpoint and manifest
- Non-rank workers call `release_training_memory()` and exit without saving
- DDP via HuggingFace `TrainingArguments`
- Forge and CLI use the same distributed plan resolution
- Multi-node launches add Accelerate `--num_machines`, `--machine_rank`,
  `--main_process_ip`, and `--main_process_port`
- Keep `ddp_find_unused_parameters: false` unless the model graph really has unused trainable parameters

## Cloud access

The **Cloud access** tab stores provider credentials separately from training
configs. API keys, provider tokens, SSH private keys, temporary session tokens,
and optional bootstrap commands are encrypted in Seiso's local provider store.
Training jobs reference the saved credential by id and keep only non-secret target
metadata in the job config:

```yaml
cloud_gpu_enabled: true
cloud_gpu_provider: aws          # aws, gcp, azure, lambda, runpod, coreweave, custom
cloud_gpu_region: us-east-1
cloud_gpu_instance_type: p5.48xlarge
cloud_gpu_count: 8
cloud_gpu_project: finetune-prod
cloud_gpu_credential_id: <encrypted-credential-id>
```

Keep cloud API keys, SSH material, and bootstrap commands in the **Cloud access**
tab. They are never echoed back to the UI after save, and they are not copied into
training job history.

## macOS / single GPU

Multi-GPU checkbox is disabled when `training_defaults.multi_gpu_available` is false.
