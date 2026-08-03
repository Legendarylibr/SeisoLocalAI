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
| **vLLM** (tensor parallel) | `rollout_backend: vllm` + `multi_gpu: true` in `configs/example_training_slime_vllm.yaml` | `seiso train -c …` or `scripts/run_slime_vllm_ddp.sh` |
| **SGLang** | `rollout_backend: sglang` + `multi_gpu: true` in `configs/example_training_slime_ddp.yaml` | `seiso train -c …` or `scripts/run_slime_ddp.sh` |

`seiso slime` is single-process only and ignores DDP launch flags — use `seiso train` for policy DDP.

vLLM weight sync defaults to dynamic LoRA (`/v1/load_lora_adapter`) when
`slime_use_lora: true`. Start the server with `--enable-lora`, or set
`SEISO_MANAGED_VLLM_ENABLE_LORA=true` for Seiso-managed multi-GPU. Prefer
`vllm_weight_mode: auto` (or `lora`); `full` with LoRA is refused unless
`SEISO_SLIME_ALLOW_VLLM_FULL_WITH_LORA=1`. Single-GPU `rollout_backend: hf`
and existing LoRA/SFT multi-GPU paths are unchanged.

Multiple engine URLs (`vllm_base_url` comma-separated or `vllm_engine_urls`)
round-robin **generation** and fan-out **weight sync** to every engine.

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

**Important:** `cloud_gpu_*` is provisioning metadata only. It does **not** start
a remote vLLM server. For slime on cloud multi-GPU:

1. Provision the instance (or use your orchestrator / bootstrap command)
2. Start vLLM with tensor parallel + `--enable-lora` on that host
3. Set `rollout_backend: vllm` and `vllm_base_url: http://<host>:8000`
4. Launch policy training (`scripts/run_slime_vllm_ddp.sh` or Accelerate)

Configs that enable `cloud_gpu` with `method: slime` and `rollout_backend: vllm`
must still provide `vllm_base_url`.

## macOS / single GPU

Multi-GPU checkbox is disabled when `training_defaults.multi_gpu_available` is false.

## Buzz mesh (experimental secondary)

Opt-in Buzz-agent multi-node path (`SEISO_ALLOW_MESH=1`). Local `nnodes=1`
remains the default. See [mesh.md](mesh.md) — plans map onto the same
`distributed_*` / Accelerate knobs documented above.
