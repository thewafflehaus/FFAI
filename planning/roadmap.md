# FFAI — Roadmap

The high-level view of what's shipped and what's coming. For the
fully phased build-out (deliverables, kernels, tests per phase) see
[`plan.md`](plan.md). For the user-facing pitch see the top-level
[`README`](../README.md).

## Shipped today

**Inference engine.** Apple Silicon Metal-only, no CPU hot-path
fallback. One `MTLCommandBuffer` per token, one `commit + wait` per
token, GPU-side sampling — only the 4-byte sampled token id crosses
CPU↔GPU. Kernels are generated from the metaltile Rust DSL at build
time and shipped as a pre-compiled `kernels.metallib` resource — no
runtime JIT, no first-call Metal compile latency.

**Loading.** HuggingFace Hub integration (`Model.load("org/repo")` —
resumable, glob-filtered download; shared `~/.cache/huggingface`
cache). Capability-driven loading (`Capability` enum — `.textIn`,
`.textOut`, `.visionIn`, `.audioIn`, `.audioOut`, `.toolCalling`;
disabled modalities skip weight allocation). Async
`AsyncStream<ModelLifecycleEvent>` lifecycle. mlx-format 3/4/5/6/8-bit
affine group quant, with **per-tensor bit-width derivation** so
mixed-precision checkpoints (e.g. Gemma 4 26B-A4B) load correctly.

**Models — full text-LLM coverage.** Llama (+ the Llama-compatible
zoo — SmolLM, OLMo, Starcoder2, internlm2, DeepSeek-R1 distills),
Qwen 2, Qwen 3, Qwen 3.5 (dense / MoE / GDN-hybrid), Mistral, Phi,
Gemma 3, Gemma 4 (dense / E-series PLE / MoE), GPT-OSS-20B, Mamba 2,
the hybrid families (NemotronH, Jamba, GraniteMoeHybrid, FalconH1),
and Nemotron-Labs-Diffusion. Each ships with a coherent-output
integration test.

**KV caches.** Raw fp16/bf16; affine-quantized (`affine4` / `affine8`
— GPU append + bulk-dequant); AURA compressed (`aura{kb}v{vb}`
schemes — per-layer SRHT rotation + Lloyd-Max codebook + per-position
norm correction). Per-layer eviction policy (unbounded / sliding
window with attention-sink retention). The
`ModelKVCacheMatrixIntegrationTests` cross-product covers model
family × weight-bitwidth × KV-cache scheme.

**Kernels.** Full sampling pipeline (greedy-GPU / GPU-categorical /
CPU-sample paths — `temperature`, `top-K`, `top-P`, `min-P`,
`repetition penalty`, seeded sampling). AURA codec kernels. GDN
(`gated_delta_step`) + Mamba 2 (`ssm_step`, `conv1d_causal_step`)
recurrent kernels. Attention sinks + sliding-window mask. MoE router
+ per-expert dispatch. SDPA decode at head_dim {64, 128, 256, 512}.

**Tooling.** `ffai inspect` (architecture + tokens + logits), the
`tile` metaltile CLI, a GPU-correctness test layer (naive-CPU
oracle), `insta` MSL snapshots.

## Planned

The roadmap is a high-level view; per-phase deliverables, kernels,
and tests live in [`plan.md`](plan.md).

| Capability | Phase | Notes |
|---|---|---|
| Sliding-window SDPA fast path | 6.1 | Thread `sink_end` / `window_start` through `Ops.sdpaDecode` for the kernel fast path — ~4–8× decode at long context. |
| AURA MSL snapshot tests | 6.2 | `insta` MSL fixtures pinning AURA-kernel codegen. |
| AURA performance (Stage 1b + 3) | 6.3 | Two independent K/V codecs, two-phase prefill, compressed-domain `aura_flash` as the default decode path, strided-output encode + cache-layout flip. Perf/architecture only — AURA correctness is shipped. |
| Profile injectable | 6.4 | `Profile` passed per `generate(...)` call instead of a singleton; per-sequence telemetry prereq for batched decode. |
| Chunked (batched) prefill | 6.6 | `forwardMulti` over prompt chunks via the `sdpa_decode_batched_prefill` kernel — large TTFT win on long prompts. Prioritized **before** the Vision wave; also the Phase 8 speculative-decode prereq. |
| VLM wave (Qwen 2.5-VL, Qwen 3.5-VL / -VL-MoE, Gemma 3-VL, Gemma 4-VL) | 6.5 | `VisionEncoder` + `ImagePreprocessing` + new `conv2d` / `patch_embed` / `rope_2d` metaltile kernels. `Capability.visionIn` exercised end-to-end. |
| Audio wave (Whisper, Kokoro, Qwen-Omni) | 7 | `AudioEncoder` + `AudioPreprocessing` + new `mel_spectrogram` / `audio_conv1d` / `vocoder` kernels. Whisper STT, Kokoro TTS, Qwen-Omni text+vision+audio. |
| Speculative decoding + cache + serving (specs 013–043) | 8 | ngram / MTP / EAGLE speculative decode, prefix KV cache (in-mem + disk), batched / continuous decode, tree attention, sparse prefill, DFlash, KV-cache write fusion, flash-quantized SDPA, AURAFlash uplift. Sub-phases 8.0–8.23 — see `plan.md`. |
| Argument-buffer / ICB dispatch modes + autotuner | 9 | Dispatch Mode 2 / 3 (`architecture.md §4a`); metaltile grid-search autotuner persisting to `tuning_cache.json`. |
| GGUF support, Homebrew formula, full bench sweep, docs-site polish | 10 | |

## Potential Future Work

These aren't on the current roadmap. Different projects, or hard
technical mismatches with the static-kernel approach.

- **CoreML / ANE backend.** Realistic only for boring kernels
  (RMSNorm, RoPE, plain GEMV at fp16/int8). AURA, FWHT, online
  softmax, recurrent SSM/GDN do not fit ANE constraints. (Spec 025
  ANE primitives + spec 029 ANE-offloaded LM head are on the Phase 8
  plan for the specific LM-head / PLE-projection use case where ANE
  *does* fit.)
- **Swift macro frontend** for kernel authoring. metaltile IR is
  serde-serializable; a Swift `@kernel` macro emitting IR JSON could
  feed the same backend later. Wait for demand.
- **Training / autograd.** Different project.
- **CUDA / Linux backends.** Different project.
- **ONNX format.** Graph format with embedded weights — needs a graph
  executor, which doesn't align with the static-kernel approach.

## See also

- [`plan.md`](plan.md) — phased build-out, deliverables per phase.
- [`architecture.md`](architecture.md) — visual reference for the
  build pipeline and dispatch loop.
- [`../documentation/`](../documentation/README.md) — user-facing
  docs (installation, quickstart, models, kv-cache, quantization,
  performance, capabilities).
