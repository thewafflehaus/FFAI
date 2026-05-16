# FFAI — Roadmap

The high-level view of what's shipped and what's coming. For the
fully phased build-out (deliverables, kernels, tests per phase) see
[`plan.md`](plan.md). For the user-facing pitch see the top-level
[`README`](../README.md).

## Shipped today

- **Apple Silicon Metal-only inference.** No CPU fallback in the hot
  path. Single `MTLCommandBuffer` per token, single
  `commit + waitUntilCompleted` per token, GPU-side argmax — the only
  4 bytes that cross CPU↔GPU per token are the sampled token id.
- **Pre-compiled metallib.** Kernels are generated from the metaltile
  Rust DSL at build time and shipped as `kernels.metallib` resources.
  No runtime JIT; no Metal compile latency on first call.
- **mlx-format quantization.** 3 / 4 / 5 / 6 / 8-bit affine group quant
  with sub-group split dispatch. Same `*.safetensors` files mlx-lm
  uses; no conversion step.
- **HuggingFace Hub integration.** `Model.load("org/repo")` resolves,
  downloads (resumable, glob-filtered), caches under
  `~/.cache/huggingface/hub/`, and shares cache with Python's
  `huggingface_hub`.
- **Capability-driven loading.** `Capability` enum (`.textIn`,
  `.textOut`, `.visionIn`, `.audioIn`, `.audioOut`, `.toolCalling`)
  declared per family; user picks which to enable at load time.
  Disabled modalities skip weight allocation entirely. Phase 6
  exercises vision; the API surface is in place from Phase 2.
- **Async lifecycle.** `Model` exposes
  `AsyncStream<ModelLifecycleEvent>` — `idle → downloading → loading
  → loaded → ready`, plus `failed(Error)` from any state. Hot
  `enable(_:)` / `disable(_:)` for capabilities ships in Phase 6.
- **Single-stream KV cache** (raw fp16 / bf16). Append + slice on the
  GPU via the `kv_cache_update` kernel — no per-layer CPU sync.
  Quantized + TurboQuant + SSM/GDN caches land in Phase 5+.
- **Full sampling pipeline** (Phase 5a). `temperature`, `top-K`,
  `top-P`, `min-P`, `repetition penalty`, seeded reproducible
  sampling — all wired through `GenerationParameters` + CLI flags
  (`--temperature`, `--top-k`, `--top-p`, `--min-p`,
  `--repetition-penalty`, `--seed`). Greedy stays on the GPU argmax
  fast path (no logits readback); non-greedy reads logits to CPU
  and runs the pipeline there (~30% decode-tok/s tax). A fully-on-GPU
  sample kernel is on the `ek/sampling-kernels` metaltile branch
  for a later commit.

## Planned

| Capability | Phase | Notes |
|---|---|---|
| **GPU softmax + categorical sample kernel** | 5b | Eliminates the per-token logits readback when sampling. Branch: `ek/sampling-kernels` in metaltile. |
| Quantized KV cache (affine 4 / 6 / 8-bit) | 5c | ~3.5× memory at 4-bit; modest decode-tok/s tax. Needs quantize-on-append + bulk-dequant kernels + cache-abstraction refactor (protocol over the concrete `KVCache` class). |
| TurboQuant compressed-domain attention | 5d | ~6-8× memory. Block-wise MSE codec with asymmetric K/V bits. Substantial research-grade codec port — multiple sessions. |
| SSM / GatedDeltaNet hybrid models (Qwen 3.5, NemotronH, Mamba) | 5e | New `SSMStateCache` + `gated_delta_step` / `ssm_kernel` kernels. Requires Mamba selective-scan port. |
| Vision encoders + multi-modal capability matrix | 6 | First targets Qwen 2.5-VL / Qwen 3.5-VL. Depends on Phase 5e for the text backbone if going hybrid. |
| Audio (`.audioIn` for STT, `.audioOut` for TTS) | 8+ | First audio target TBD (Whisper, Qwen-Omni, …). |
| Speculative decoding (n-gram + draft model) | 8+ | Requires the batched KV cache. |
| Argument buffers / ICB dispatch modes | 8+ | If profiles continue to show encoding cost matters. |
| Autotuner over kernel parameters | 7 | Grid search over `(tile_dims, threads, unroll, simd_matrix, async_copy)`. |
| GGUF format support | 8+ | If community demand justifies a per-arch name mapper. |
| Chat-template auto-application in `generate(...)` | 6 | Lands alongside the first instruct-tuned VL model. |
| Multi-stream / batched serving | 8+ | `BatchedKVCache` + multi-stream decode. |
| Presence penalty in sampling | 5+ | Field is on `GenerationParameters` already; pipeline integration is a small follow-up to Phase 5a. |

## Potential Future Work

These aren't on the current roadmap but are considered once core functionality is stable and general feature parity and speed is 
caught up to leading inference engines like vllm, llama.cpp, ollama, omlx, mlx-vlm, mlx-swift-lm, etc. Different projects, or hard
technical mismatches with the static-kernel approach.

- **CoreML / ANE backend** — realistic only for boring kernels
  (RMSNorm, RoPE, plain GEMV at fp16/int8). TurboQuant, FWHT,
  online softmax, recurrent SSM/GDN do not fit ANE constraints.
  Add a `mil/` codegen sibling to `msl/` in metaltile-codegen
  when v0.3 demand justifies it.
- **Swift macro frontend** for kernel authoring. metaltile IR is
  serde-serializable; a Swift `@kernel` macro emitting IR JSON could
  feed the same backend later. Don't build it preemptively — wait
  for demand.
- **Training / autograd.** Different project.
- **CUDA / Linux backends.** Different project.
- **ONNX format.** Graph format with embedded weights — would need a
  graph executor, which doesn't align with the static-kernel
  approach.

## See also

- [`plan.md`](plan.md) — phased build-out, deliverables per phase.
- [`architecture.md`](architecture.md) — visual reference for the
  build pipeline and dispatch loop.
- [`../documentation/`](../documentation/README.md) — user-facing
  docs (installation, quickstart, models, kv-cache, quantization,
  performance, capabilities).
