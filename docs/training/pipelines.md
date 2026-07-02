# Training Pipelines

Step-by-step runbook for the local training and research flows referenced by
the quickstart. Use smoke presets first, then scale settings once the path is
healthy on your machine.

## Before you run

From the repository root:

```bash
source .venv/bin/activate
seiso doctor
```

For memory-sensitive runs, unload chat models in Forge or with:

```bash
seiso forge
```

Then use the Chat or Model Hub **Free memory** action before starting a heavy
training, compression, or export job.

## Supervised training

Start with the CPU smoke config for a fast end-to-end check:

```bash
seiso train --config configs/smoke_train_cpu.yaml
```

Then move to the example LoRA config and adjust dataset/model paths:

```bash
seiso train --config configs/example_lora.yaml
```

See [quickstart.md](quickstart.md) for UI-driven training and preset guidance,
and [multi-gpu.md](multi-gpu.md) for distributed runs.

## Kernel and low-VRAM training

Review fused-kernel behavior before changing CUDA kernel paths:

```bash
pytest tests/test_e2e_gpu_training.py -m slow
```

Use the smoke or small dataset configs when validating cleanup paths. Kernel
patches must restore model methods on both success and exception paths. See
[kernels.md](kernels.md).

## Compression

Compression flows use the shared job/orchestrator pattern and should be tested
with a small local model first:

```bash
seiso compress --config configs/example_compress.json
```

For Forge behavior, verify logs stream and artifacts resolve under the active
data directory. See [../compression.md](../compression.md).

## Distill-RL and preference data

Generate preference rows with a small prompt set before scaling teacher/student
runs. Keep generated datasets under a throwaway `SEISO_DATA_DIR` when iterating:

```bash
SEISO_DATA_DIR=/tmp/seiso-distill-smoke seiso distill-rl --help
```

Use the Forge job page when validating streamed logs and cancel behavior.

## RL quant and quant regression

Run the documented quant regression study with the example config:

```bash
seiso experiment quant-regression -c configs/examples/quant_regression_study.yaml
```

For adaptive quant flows, prefer smoke-sized request counts until reports and
sidecars are generated correctly.

## Quality gate

Before opening a PR or after significant pipeline changes:

```bash
make ci-fast
```

Run the full gate for frontend changes or broader refactors:

```bash
make ci
```
