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
disabled modalities skip weight allocation). Capability sets:
`.textOnly` / `.textWithTools` / `.speechToText` / `.textToSpeech` /
`.omniAudio` / `.speechToSpeech` (audio enhancement, source separation,
audio segmentation). Async `AsyncStream<ModelLifecycleEvent>`
lifecycle. mlx-format 3/4/5/6/8-bit affine group quant, with
**per-tensor bit-width derivation** so mixed-precision checkpoints
(e.g. Gemma 4 26B-A4B) load correctly.

**Models — full text-LLM coverage.** Llama (+ the Llama-compatible
zoo — SmolLM, OLMo, Starcoder2, internlm2, DeepSeek-R1 distills),
Qwen 2, Qwen 3, Qwen 3.5 (dense / MoE / GDN-hybrid), Mistral, Phi,
Gemma 3, Gemma 4 (dense / E-series PLE / MoE), GPT-OSS-20B, Mamba 2,
the hybrid families (NemotronH, Jamba, GraniteMoeHybrid, FalconH1,
LFM2 / LFM2.5 — conv+attention, dense / MoE), and
Nemotron-Labs-Diffusion. Each ships with a coherent-output
integration test.

**Models — Vision (VLM).** Qwen 2-VL, Qwen 2.5-VL, Qwen 3-VL,
Qwen 3-VL-MoE, Gemma 3-VL, Gemma 4-VL, LFM2-VL, MiniCPM-V 4.6,
NemotronVL, SmolVLM2, Pixtral (Mistral 2D-RoPE ViT), Mistral 3 (small
3.1 with vision), FastVLM (Apple FastViTHD), GlmOcr (Zhipu GLM-OCR),
Idefics3 (HuggingFace), Paligemma (Google).

**Models — Audio.**
- *STT:* Whisper, SenseVoice, Parakeet, FireRedASR2, Qwen 3-ASR,
  Voxtral-Realtime (Mistral), GLM-ASR, CohereTranscribe,
  GraniteSpeech.
- *TTS:* Kokoro, LlamaTTS, Marvis, Qwen 3-TTS / Qwen3TTSBase,
  EchoTTS, Chatterbox, MossTTS / MossTTSNano, PocketTTS, Soprano,
  StyleTTS2, FishSpeech (dual-AR + FishS1DAC codec).
- *Omni:* Qwen-Omni (text + vision + audio), LFMAudio (Liquid).
- *VAD:* SileroVAD, SmartTurn, Sortformer (diarization), TenVAD
  (TEN-framework), FireRedVAD.
- *STS / audio enhancement:* DeepFilterNet, MossFormer2-SE,
  SAMAudio (segmentation).

**Audio codecs.** BigVGAN, Vocos, DACVAE, DescriptDAC, Encodec, Mimi,
SNAC, FishS1DAC. Each has its own loader/decoder; family TTS files
wire to whichever the checkpoint uses.

**KV caches.** Raw fp16/bf16; affine-quantized (`affine4` / `affine8`
— GPU append + bulk-dequant); AURA compressed (`aura{kb}v{vb}`
schemes — per-layer SRHT rotation + Lloyd-Max codebook + per-position
norm correction). Per-layer eviction policy (unbounded / sliding
window with attention-sink retention). The
`ModelKVCacheMatrixIntegrationTests` cross-product covers model
family × weight-bitwidth × KV-cache scheme.

**Generation features.** EOS-list stop tokens (Gemma 3+ multi-EOS
families). Parallelized CPU attention across every VLM vision tower
and audio encoder (`DispatchQueue.concurrentPerform` — the
single-threaded scalar attention was the VLM "image hang" and Whisper
empty-output bug). VLM image+text fixture (`dog.jpeg`) with shared
preprocessing helpers (`VLMTestSupport.dogImageCHW(...)`,
`dogImageCHWNormalized(...)`).

**Kernels.** Full sampling pipeline (greedy-GPU / GPU-categorical /
CPU-sample paths — `temperature`, `top-K`, `top-P`, `min-P`,
`repetition penalty`, seeded sampling). AURA codec kernels. GDN
(`gated_delta_step`) + Mamba 2 (`ssm_step`, `conv1d_causal_step`)
recurrent kernels. Attention sinks + sliding-window mask. MoE router
+ per-expert dispatch. SDPA decode at head_dim {64, 128, 256, 512}.
Patch-embed, conv2d, audio conv1d, mel spectrogram.

**Tooling.** `ffai inspect` (architecture + tokens + logits), the
`tile` metaltile CLI, a GPU-correctness test layer (naive-CPU
oracle), `insta` MSL snapshots.

## Planned

The roadmap is a high-level view; per-phase deliverables, kernels,
and tests live in [`plan.md`](plan.md).

| Capability | Phase | Notes |
|---|---|---|
| Sliding-window SDPA fast path | 6.1 | Thread `sink_end` / `window_start` through `Ops.sdpaDecode` — ~4–8× decode at long context. |
| AURA MSL snapshot tests | 6.2 | `insta` MSL fixtures pinning AURA-kernel codegen. |
| AURA performance (Stage 1b + 3) | 6.3 | Two independent K/V codecs, two-phase prefill, compressed-domain `aura_flash` as the default decode path, strided-output encode + cache-layout flip. Perf/architecture only — AURA correctness is shipped. |
| Profile injectable | 6.4 | `Profile` passed per `generate(...)` call instead of a singleton; per-sequence telemetry prereq for batched decode. |
| Chunked (batched) prefill | 6.6 | `forwardMulti` over prompt chunks via the `sdpa_decode_batched_prefill` kernel — large TTFT win on long prompts. Also the Phase 8 speculative-decode prereq. |
| GPU vision attention + depthwise conv | 6.5b | Move the Idefics3 / Paligemma / GlmOcr / FastVLM CPU bidirectional attention + depthwise conv2d onto metaltile. FastVLM cold inference at 1024px is the loudest signal. |
| Speculative decoding + cache + serving (specs 013–043) | 8 | ngram / MTP / EAGLE speculative decode, prefix KV cache (in-mem + disk), batched / continuous decode, tree attention, sparse prefill, DFlash, KV-cache write fusion, flash-quantized SDPA, AURAFlash uplift. Sub-phases 8.0–8.23 — see `plan.md`. |
| Argument-buffer / ICB dispatch modes + autotuner | 9 | Dispatch Mode 2 / 3 (`architecture.md §4a`); metaltile grid-search autotuner persisting to `tuning_cache.json`. |
| GGUF support, Homebrew formula, full bench sweep, docs-site polish | 10 | |
| `ffai bench --mactop` thermal-aware bench harness | 10 | Spawn `mactop` alongside `ffai bench`; capture CPU / GPU / memory / power / temperature samples to a sidecar; `--mactop-pin-fans` pins fans high for the bench window (requires sudo) so steady-state numbers aren't measured under thermal throttle. Design: [`bench-mactop-integration-design.md`](bench-mactop-integration-design.md). |

## Open performance & testing debt (flagged 2026-05-23)

Concrete gaps in the shipped code, ranked by user-visible impact.
Tracked in `planning/session-plan.md` "Performance gaps" + "Testing
gaps" tables; here just the headline:

- **Long-prompt TTFT** — one-token-per-dispatch prefill. Closed by
  Phase 6.6.
- **Long-context decode** — sliding-window SDPA falls through to full
  attention. Closed by Phase 6.1.
- **VLM cold inference** — Idefics3 / PaliGemma / GlmOcr / FastVLM
  vision towers run depthwise + bidirectional attention on CPU.
  Closed by a new Phase 6.5b (GPU vision attention kernel + depthwise
  conv2d).
- **GPU 100% pin** — deferred per the open issue in
  `known-issues.md`; needs a Metal System Trace to localise.
- **Integration tests written but unrun** for every family from the
  Phase 6.5 / 7 wave — coherence verdict lands the first time
  `make test-integration --filter <Family>` runs against a cached
  checkpoint.
- **No GPU correctness tests** for the (CPU-resident) VLM vision
  kernels — pair them with `*_gpu_correctness.rs` when the GPU port
  lands.
- **No per-layer forward tests** for the new families — most ship
  config + registry unit tests only.

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
- [`session-plan.md`](session-plan.md) — working edge: open phase
  items + perf/testing gap tables.
- [`known-issues.md`](known-issues.md) — open bugs caught during port.
- [`architecture.md`](architecture.md) — visual reference for the
  build pipeline and dispatch loop.
- [`../documentation/`](../documentation/README.md) — user-facing
  docs (installation, quickstart, models, kv-cache, quantization,
  performance, capabilities).
