# CLI reference

All commands run from the **repository root** with your virtualenv active after `pip install -e ".[forge,train,...]"`.

## `seiso forge`

Launch the Forge web server (API + built UI).

```bash
seiso forge
seiso forge --reload          # auto-reload Python on code changes
seiso forge --port 8766       # custom port
```

Requires `forge-ui/dist` for the web UI — build with `cd forge-ui && npm run build`.

## `seiso train`

Fine-tune from a YAML config.

```bash
seiso train --config configs/example_lora.yaml
```

Example config: `configs/example_lora.yaml` (dataset: `data/sample.jsonl`).

## `seiso chat`

Terminal chat with a local model.

```bash
seiso chat --model meta-llama/Llama-3.2-3B-Instruct --prompt "Hello"
seiso chat --model /path/to/model.gguf   # interactive mode (no --prompt)
```

## `seiso export`

Export a training checkpoint to merged weights, LoRA, full fine-tune, or GGUF.

```bash
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --formats merged,gguf
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --profile inference
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --hub-repo user/my-model
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --hub-repo user/my-model --precheck-only
seiso export --checkpoint ./outputs/lora-run/checkpoint-<ts> --profile list
```

Outputs land under `{SEISO_DATA_DIR}/exports/` by default.

## `seiso inference`

One-shot inference (alias for single-turn chat).

```bash
seiso inference --model meta-llama/Llama-3.2-3B-Instruct --prompt "Summarize Seiso in one sentence."
```

## `seiso compress`

Code Llama compression pipeline (vendored `third_party/codellama-compress`).

```bash
# Presets: smoke | full | distill_only | prune_recover | quantize
seiso compress run --preset smoke
seiso compress run --preset full --teacher-model codellama/CodeLlama-13b-hf --student-model codellama/CodeLlama-7b-hf

# Verify hash-chained manifest
seiso compress manifest-verify --run-dir ~/.seiso/compress/<user>/<run>

# Speculative decoding benchmark
seiso compress speculative --target-model ./finetuned --draft-model ./distilled --prompt "def fib(n):"
```

Requires `.[train]` for GPU stages. Optional `.[compress-quant]` for GPTQ/AWQ, `.[compress-eval]` for lm-eval.

See [compression.md](compression.md).

## `seiso-bench-kernels`

Benchmark fused training kernels (NVIDIA CUDA or AMD Triton).

```bash
seiso-bench-kernels --op all --rows 4096 --hidden 4096 --vocab 32000
```

## Multi-GPU training (manual)

```bash
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs/example_lora.yaml
```

See [training/multi-gpu.md](training/multi-gpu.md).
