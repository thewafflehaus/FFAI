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
  TurboQuant + SSM/GDN caches land in Phase 5d/e.
- **Affine-quantized KV cache (int8 + int4)** (Phase 5c). Activate via
  `LoadOptions.kvCache = .affineQuantized(bits: N, groupSize: ...)`
  or CLI `--kv-cache int8` / `--kv-cache int4`. Measured on Qwen3 1.7B
  at maxSeq=40960: int8 saves 47% KV (4.38 → 2.32 GB) at −7% tok/s;
  int4 (group_size=32) saves 69% KV (4.38 → 1.37 GB) at −3% tok/s.
  int6 + fused-dequant-into-SDPA are 5c follow-ups.
- **Full sampling pipeline** (Phase 5a + 5b). `temperature`, `top-K`,
  `top-P`, `min-P`, `repetition penalty`, seeded reproducible
  sampling — all wired through `GenerationParameters` + CLI flags
  (`--temperature`, `--top-k`, `--top-p`, `--min-p`,
  `--repetition-penalty`, `--seed`). Three execution paths:
  *greedy-GPU* (T==0, no filters — argmax kernel); *gpu-categorical*
  (T>0, no filters — new `softmax_categorical_sample` kernel,
  logits stay on GPU); *cpu-sample* (any filter — CPU readback +
  full pipeline). Per-family `forwardSampleCategorical` fusion and
  GPU filter kernels (top-K / top-P / min-P sort) are follow-ups.

## Planned

| Capability | Phase | Notes |
|---|---|---|
| Parallel-prefix CDF walk in `softmax_categorical_sample` | 5b+ | Per-family fusion shipped (Llama + Qwen 3 override `forwardSampleCategorical` to use one cmdbuf), but perf is bottlenecked by the single-thread CDF walk inside the kernel (~150µs at vocab=152K). Parallel-prefix replacement is the remaining lever. |
| GPU filter kernels (top-K / top-P / min-P sort) | 5b+ | Today's filter-bearing paths fall back to `cpu-sample`. GPU filters need a sort or radix-select kernel. |
| Parallel prefix-scan CDF walk | 5b+ | Replaces the single-thread CDF walk in `softmax_categorical_sample` (~150µs at vocab=152K today). |
| Affine KV cache int6 | 5c+ | int4 + int8 shipped (47%/69% memory savings). int6 is a byte-packed follow-up between them. |
| Fused `bulk_dequant + sdpa_decode` | 5c+ | Today each attention step queues a separate dequant kernel into the shared working buffer before SDPA. Fusing removes the working-buffer materialisation entirely. |
| TurboQuant compressed-domain attention | 5d | ~6-8× memory. Block-wise MSE codec with asymmetric K/V bits. Substantial research-grade codec port — multiple sessions. |
| Mamba 2 hybrid models (NemotronH, GraniteMoeHybrid, FalconH1) | 5e | **Foundation shipped**: `ssm_step` kernel + `SSMStateCache` + `Ops.ssmStep` + tests. Still needed: chunked-prefill scan kernel, depthwise-conv state buffer, Mamba 2 family file. |
| GatedDeltaNet hybrid (Qwen 3.5) | 5e+ | Needs `gated_delta_step` + `gated_delta_step_record` + `state_replay` kernels for speculative-decoding rollback. Builds on the 5e SSM foundation. |
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
