# Native Linux KV pipeline

Seiso uses one active local generation and one bounded Torch prefix entry. It
does not duplicate KV caches or share them across users. Long Torch prompts are
prefilled sequentially through one `past_key_values` chain; short prompts keep
the one-shot path. Reuse requires an exact token prefix and is cleared on model
switch, unload, cancellation, OOM, or decode failure.

The defaults preserve output behavior:

- dynamic fp16/bf16 KV is the default;
- static, offloaded, and quantized caches use Transformers-native generation;
- quantized KV and decode compilation require explicit opt-in;
- unsupported cache modes retry with the dynamic cache before streamed output;
- GGUF remains isolated in Ollama/llama-swap on native NVIDIA Linux.

## Controls

| Variable | Default | Purpose |
|---|---:|---|
| `SEISO_TORCH_LOAD_PRECISION` | `auto` | Use BF16/FP16 only when weights plus runtime reserve fit; otherwise 4-bit |
| `SEISO_TORCH_LOAD_RESERVE_MB` | `2048` | Minimum load-time VRAM reserve (also keeps 25% free) |
| `SEISO_TORCH_PREFILL_CHUNK_THRESHOLD` | `2048` | Minimum prompt length for chunking |
| `SEISO_TORCH_PREFILL_CHUNK_SIZE` | automatic | Override the 512/1024/2048 headroom tiers |
| `SEISO_TORCH_PREFIX_CACHE` | `true` on Linux | Enable exact-prefix reuse |
| `SEISO_TORCH_PREFIX_CACHE_MAX_TOKENS` | `32768` | Bound retained token IDs and KV |
| `SEISO_TORCH_CACHE_IMPLEMENTATION` | `dynamic` | `dynamic`, `static`, `offloaded`, or `quantized` |
| `SEISO_TORCH_QUANTIZED_KV` | `false` | Explicitly permit quantized KV |
| `SEISO_TORCH_KV_BITS` | `8` | Quantized KV width |
| `SEISO_TORCH_KV_HEADROOM_MB` | `768` | VRAM left outside the estimated cache |
| `SEISO_TORCH_DECODE_GRAPHS` | `false` | Warm, then compile guarded token-step decode |
| `SEISO_OLLAMA_NUM_KEEP` | unset | Send Ollama `num_keep` explicitly |

Sidecar request options are capability-gated through
`sidecar_capabilities`. With no capability data, request bodies are unchanged.
Ollama manages prompt reuse internally; llama-swap receives `cache_prompt` only
when that capability is advertised. KV precision for these servers is a
server-launch setting and is intentionally not guessed per request.

Preload performs a real one-token Torch kernel warmup. Ollama preload sends a
native empty-prompt load request with the pinned context and active-session
keep-alive, so first chat does not pay model-load latency. Interactive sessions
retain Ollama residency for 15 minutes; explicit Free Memory still unloads it
immediately.

## Rollback

Set `SEISO_TORCH_KV_STREAM=0`, `SEISO_TORCH_PREFIX_CACHE=0`, and
`SEISO_TORCH_CACHE_IMPLEMENTATION=dynamic` to use the established Transformers
generation path with dynamic KV. Set `SEISO_TORCH_LOAD_PRECISION=4bit` to
restore forced bitsandbytes loading. Leave `SEISO_TORCH_DECODE_GRAPHS=0` and
`SEISO_TORCH_QUANTIZED_KV=0` for strict numerical compatibility.

Generation updates expose additive metadata for cache mode, fallback reason,
prefill chunks/backoffs, prefix hits, TTFT, decode tokens/second, and measured
headroom. The inference benchmark stores the same values under `kv_metadata`.
