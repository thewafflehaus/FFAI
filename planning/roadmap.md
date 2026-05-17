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
  Disabled modalities skip weight allocation entirely. Phase 6.5
  exercises vision; the API surface is in place from Phase 2.
- **Async lifecycle.** `Model` exposes
  `AsyncStream<ModelLifecycleEvent>` — `idle → downloading → loading
  → loaded → ready`, plus `failed(Error)` from any state. Hot
  `enable(_:)` / `disable(_:)` for capabilities ships in Phase 6.5.
- **Single-stream KV cache** (raw fp16 / bf16). Append + slice on the
  GPU via the `kv_cache_update` kernel — no per-layer CPU sync.
  GigaQuant + SSM/GDN caches land in Phase 5d/e.
- **Affine-quantized KV cache (`affine8` + `affine4`)** (Phase 5c).
  Activate via
  `LoadOptions.kvCache = .affineQuantized(bits: N, groupSize: ...)`
  or CLI `--kv-cache affine8` / `--kv-cache affine4`. Measured on
  Qwen3 1.7B at maxSeq=40960: `affine8` saves 47% KV (4.38 → 2.32 GB)
  at −7% tok/s; `affine4` (group_size=32) saves 69% KV
  (4.38 → 1.37 GB) at −3% tok/s. `affine6` + fused-dequant-into-SDPA
  are 5c follow-ups (now under Planned).
- **Mamba 2 dense (Phase 5e — initial drop).** End-to-end decode of
  `mlx-community/mamba2-130m` (130M / 370M / 780M / 1.3B / 2.7B all
  share `Mamba2Dense`). Per-token forward = one MTLCommandBuffer:
  RMSNorm → in_proj → conv1d-causal-step + SiLU → softplus(dt) →
  `ssm_step` → `D·x` skip → SiLU(z) gate → mixer norm → out_proj.
  Constant-memory recurrent state via `Mamba2LayerCache`
  (`SSMStateCache` + `ConvStateCache`). `LayerCacheProtocol` introduced
  so SSM caches don't have to bolt on no-op attention methods.
  Limitations today: `n_groups = 1` only; no chunked-prefill scan
  (decode-only — prefill walks tokens one at a time); Mamba 2 hybrids
  (NemotronH / GraniteMoeHybrid / FalconH1) are 5e+ follow-ups.
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
| Parallel-prefix CDF walk in `softmax_categorical_sample` | 5b+ | Per-family fusion shipped, but perf is bottlenecked by the single-thread CDF walk inside the kernel (~150µs at vocab=152K). Parallel-prefix replacement is the remaining lever. |
| GPU filter kernels (top-K / top-P / min-P sort) | 5b+ | Today's filter-bearing paths fall back to `cpu-sample`. GPU filters need a sort or radix-select kernel. |
| Affine KV cache `affine6` | 5c+ | `affine4` + `affine8` shipped (47% / 69% memory savings). `affine6` is a byte-packed follow-up between them. |
| Fused `bulk_dequant + sdpa_decode` | 5c+ | Today each attention step queues a separate dequant kernel into the shared working buffer before SDPA. Fusing removes the working-buffer materialisation entirely. |
| GigaQuant codec + `GigaQuantizedKVCache` | 5d | Renamed from TurboQuant. `giga{kb}v{vb}` schemes (`giga3` / `giga4` / `giga6` / `giga8` + asymmetric `giga8v4`, `giga4v2`, `giga3v2`). ~6-8× memory at `giga4v2`. Block-wise MSE codec with DC-bias correction + asymmetric K/V bits; substantial research-grade codec port. |
| GDN kernels (`gated_delta_step` + `_record` + `state_replay`) | 5e | Recurrent gated DeltaNet step (state in fp32) + tape-record + partial-accept replay for speculative decoding rollback. Unblocks Qwen 3.5 GDN. |
| Mamba 2 chunked prefill + `n_groups > 1` | 5e | Parallel-scan variant of `ssm_step` for long-prompt prefill; grouped B / C tensors; `conv1d_causal_prefill`; `ssm_step_record` / `ssm_replay` for speculative rollback. |
| Hybrid family files (NemotronH, Jamba, GraniteMoeHybrid, FalconH1) | 5e | Mamba 2 + attention / MoE / MLP interleavings driven by layer-type strings. Share the SSM/GDN kernel surface; marginal cost per family is the family file + integration test. |
| Attention sinks + symbolic sliding-window mask | 5f | Sliding window computed per-step from `(seq_offset, window_size)`; sinks fold via numerically-stable softmax clamp inside `giga_flash_sdpa_v`. |
| GPT-OSS-20B family | 5f | Hybrid sliding-FP16 + full-attention layer schedule with sinks. Full layers on `GigaQuantizedKVCache(useBias: true)`; sliding layers cap at 128 tokens raw FP16. |
| Qwen 3.5 / 3.6 dense (0.8B / 2B / 4B / 9B / 27B) | 6 | `Qwen35Dense` variant. Scaffolded by 5e for the hybrid path. |
| Qwen 3.5 / 3.6 MoE (35B-A3B) | 6 | `Qwen35MoE` — sparse top-K gating + shared expert + per-expert dequant. |
| Gemma 3 | 6 | `Gemma3Dense` variant; reuses Llama-style backbone. |
| Gemma 4 dense + MoE + E-series PLE | 6 | `Gemma4Dense` (incl. 31B), `Gemma4E` (E2B / E4B with per-layer embedding + soft-cap), `Gemma4MoE` (26B-A4B). Sliding window every other layer; 4096-token prefill chunk. |
| Nemotron Cascade 2 | 6 | `NemotronCascade2` variant inside `NemotronH.swift` — cascade scheduling is data-driven through the layer-type string. |
| Mistral, Phi | 6 | `Mistral.swift` (Mistral 7B + Mixtral 8x7B/8x22B), `Phi.swift` (Phi-3 / Phi-3.5). Both reuse the Llama-style GQA backbone. |
| VLM wave (Qwen 2.5-VL, Qwen 3.5-VL, Qwen 3.5-VL MoE, Gemma 3 VL, Gemma 4 VL) | 6.5 | `VisionEncoder` + `ImagePreprocessing` + `conv2d` / `patch_embed` / 2D RoPE kernels. VL variants in their existing family files. `Capability.visionIn` exercised end-to-end. |
| Audio wave (Whisper, Kokoro and/or Bark, Qwen-Omni) | 7 | `AudioEncoder` + `AudioPreprocessing` + `mel_spectrogram` / `audio_conv1d` kernels. Whisper STT (tiny → large-v3), one TTS family (pick whichever has the cleaner mlx-audio-swift ref), Qwen 3.5-Omni for text + vision + audio. |
| Bench per-prompt sweep (`ngram-spot` / `ngram-sweep-summary`) | 8.0 (spec 018) | Foundation for spec 013 — measure speculative accept rates per prompt before turning the feature on. |
| Leviathan accept/reject sampler | 8.0 (spec 023) | Non-greedy accept-reject for any speculative-decoding path (ngram, MTP, EAGLE). |
| `StateReplayCache` protocol | 8.0 (spec 020) | Formalised parent of `SSMStateCache` + `GDNStateCache`. Partly shipped in 5e; final protocol surface lands here. |
| N-gram speculative decoding | 8.1 (spec 013) | `NGramSpeculativeTokenIterator` + multi-size hash lookup + min-hits filter + auto-disengage on regressive regimes. |
| Prefix KV cache (in-mem LRU + L2 disk) | 8.2 (spec 017) | `PrefixKVCache` + `PrefixKey` + per-cache `serialise()` / `hydrate(from:)` on every shipped cache. Opt-in `FFAI_PREFIX_CACHE_DISK=1`. Warm-turn TTFT speedup. |
| Compressed-domain prefix KV cache | 8.3 (spec 039) | Reuses `GigaQuantizedKVCache.fusedEncodeDispatch` at snapshot time; bumps `PrefixKey.formatVersion`. |
| Batched decode (`generateBatched`) + `BatchedKVCache` + `BatchedHybridCache` | 8.4 | Variable-length prompts + per-sequence EOS + continuous batching. Hybrid variant for GDN + attention. |
| Cross-request n-gram cache | 8.5 (spec 016) | Three-tier (`nc_context` / `nc_dynamic` / `nc_static`) per llama.cpp. |
| Deterministic-stretch acceleration | 8.6 (spec 022) | `ChatTemplateGrammar` protocol + `BigramTable` + per-family grammars. Biggest win on GPT-OSS harmony channel transitions. |
| MTP / EAGLE-3 draft heads | 8.7 (spec 030) | Variant A: native `mtp.*` weights + `MTPSelfSpeculativeTokenIterator`. Variant B: companion EAGLE assistant draft + `AssistantDraftRegistry`. |
| Tree attention | 8.8 (spec 014) | K=2 root branches in phase 1; tree mask in SDPA. |
| PLD+ attention-weighted span selection | 8.9 (spec 019) | Pick the span the verifier wanted to copy. |
| DuoAttention retrieval/streaming head split | 8.10 (spec 036) | Calibration on synthetic NIAH; two-cache-per-layer dispatch. |
| Block-sparse SDPA Metal kernel | 8.10 (spec 033) | Consumed by DuoAttention phase 5. |
| Quest decode top-k | 8.11 (spec 034) | Decode-side K-side top-k. |
| Quest K_max / K_min refinement | 8.11 (spec 035) | Refinement on top of the V1 Quest kernel. |
| TEAL activation thresholding | 8.12 (spec 037) | `threshold_and_mask` hook + `teal_calibrate.py` + block-sparse `(masked_act, down_proj) → out` Metal kernel. |
| Vertical-slash sparse prefill | 8.13 (spec 031) | Prefill-side sparsity for long prompts. |
| Speculative prefill | 8.13 (spec 032) | Draft + verify on the prefill stage. |
| DFlash on GPU | 8.14 (spec 015) | Phases 1–3 — including the `z-lab/Qwen3.5-*-DFlash` draft model + refactor onto `StateReplayCache`. |
| ANE + GPU concurrency primitives | 8.15 (spec 025) | Land first; required by Mirror SD + ANE LM head. Decision point: ANE + GPU truly concurrent on Apple Silicon? |
| Mirror SD | 8.15 (spec 021) | `MirrorSpeculativeLoop` on top of the ANE primitives. |
| KV cache write fusion | 8.16 (spec 024) | Eliminate `copy_bfloat16` dispatches per decode token. |
| Profile-guided Morton-order expert reorder | 8.17 (spec 026) | MoE locality optimisation. |
| Adaptive per-layer mixed-precision | 8.18 (spec 027) | JSON sidecar + glob-pattern matching for layer-class → precision recipes. |
| Quadratic / chunkwise WY GDN prefill | 8.19 (spec 028) | Highest research bet. Could regress if it doesn't work. |
| ANE-offloaded LM head + Gemma 4 PLE projection | 8.20 (spec 029) | Blocked on spec 025 + Mirror SD measurement. |
| Active KV cache SSD offload | 8.21 (spec 038) | Long-context single-request memory reduction. Multi-month. |
| Flash-quantized SDPA | 8.22 (spec 041) | Drop-in Flash-tiled fused kernel for the affine quantized SDPA path. |
| Metal kernel SIMD audit | 8.22 (spec 042) | Cross-kernel `simdgroup_matrix_multiply_accumulate` MMA conversion (GigaFlash + affine flash + `giga_dequant_rotated` + `mse_*`). |
| GigaFlash decode-time uplift | 8.23 (spec 043) | Renamed from TurboFlash. Per-simdgroup bit-unpack reuse + bf16 V accumulator + headDim-aware tile autotune + bias-aware kernel. |
| Argument buffers / ICB dispatch modes | 9 | Mode 2 + Mode 3 from `architecture.md §4a`. Land once feature surface is stable. |
| Metaltile autotuner | 9 | Grid search over `(tile_dims, threads, unroll, simd_matrix, async_copy)`. Persist to `~/.cache/metaltile/tuning_cache.json`. CI: nightly autotune on a reference machine. |
| GGUF format support | 10 | Per-architecture name mapper. Worth doing if community gguf quants are valuable to users. |
| Chat-template auto-application in `generate(...)` | 8.2 | Lands alongside the prefix KV cache (needs `LastAssistantOpenerPolicy`) and the first instruct-tuned VL model. |
| Presence penalty in sampling | 5+ | Field is on `GenerationParameters` already; pipeline integration is a small follow-up to Phase 5a. |

## Potential Future Work

These aren't on the current roadmap. Different projects, or hard
technical mismatches with the static-kernel approach.

- **CoreML / ANE backend.** Realistic only for boring kernels
  (RMSNorm, RoPE, plain GEMV at fp16/int8). GigaQuant, FWHT,
  online softmax, recurrent SSM/GDN do not fit ANE constraints.
  Add a `mil/` codegen sibling to `msl/` in metaltile-codegen
  when v0.3 demand justifies it. (Note: spec 025 ANE primitives +
  spec 029 ANE-offloaded LM head are listed under Planned for the
  specific LM-head / PLE-projection use case where ANE *does* fit.)
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
