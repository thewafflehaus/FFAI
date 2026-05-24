# Known Issues — to investigate after the model-implementation wave lands

Open bugs and rough notes captured during the model-family port. Each
entry has enough context that a future session can pick it up cold.

## GPU pin at 100% — long-running

**Symptom.** During (some) model loads / runs the GPU pegs at 100 %
utilisation and stays there. The machine does NOT freeze — `FFAI_MAX_COMMAND_BUFFERS`
caps the in-flight queue so we don't OOM the GPU — but the pin is
real and the application becomes effectively unresponsive on that
model.

**Status.** Cause unknown. Could be:
- A metaltile kernel with a dispatch shape that never converges
  (a reduction kernel encoded as `elementwiseGrid`, or an indirect
  dispatch reading a still-uninitialised threadgroup-count buffer).
- An FFAI-side path that re-encodes the same buffer in a busy loop
  without checking the prior commit's `status`.
- A specific kernel emitted by metaltile that has a per-thread loop
  parameterised on a stride that goes to zero / negative at certain
  shapes.

**Next steps when this lands as a priority.**
1. Reproduce deterministically — capture the exact model + prompt that
   pins the GPU.
2. Attach Metal System Trace (Instruments) while the pin is live.
   The trace shows which command buffer is in flight and which kernel
   is hung — that tells us metaltile-side vs FFAI-side immediately.
3. If metaltile-side: pull the kernel name from the trace, find the
   dispatch in `Ops.*`, audit the grid / threadgroup shape against
   the kernel's `## DISPATCH INVARIANTS` block.
4. If FFAI-side: check for a busy-loop that re-encodes the same
   compute pass without a `cmd.status` guard, or a missing
   `waitUntilCompleted` that lets the runtime queue grow unbounded.

**Mitigation today.** `FFAI_MAX_COMMAND_BUFFERS=16` (the production
default since the wrong-dispatch GPU-pin post-mortem in
`papers/optimizing-kernels-for-apple-m-series-architecture.md`)
prevents the runaway queue from OOMing the device — that's why the
pin is annoying but not fatal.

### 2026-05-23 status — post-Phase 6.5 / 7 wave + recent metaltile churn

**Hypothesis stack (ranked).** The Phase 6.5 (Vision) + Phase 7 (Audio)
wave landed 18 commits on `ek/aura-port` without re-running any kernel
or integration tests — every sub-agent prompt was "swift build only,
don't run swift test on ModelTests". On the metaltile side, the
`Sources/MetalTileSwift/Resources/kernels.metallib` was regenerated
2026-05-23 01:24 against `metaltile@840d281`. That regen pulled in:

- **#150** `mt_gated_delta_prep_chunk` — new chunked GDN prep+recurrence kernel.
- **#149** "remove all hand-written InlineMsl — port the MPP kernels to the coop_tile DSL" — the int4/int8/fp16 spread the user has been working on in another session.
- **#147** "All the remaining kernels" — large surface, broad.
- **#144** bm8 MoE BGEMM + dynamic-M qmm + naming standardisation.

Most likely cause: a kernel in #147 or #149 with a dispatch shape that
goes pathological at production sizes but isn't caught by the small-N
unit tests. The MPP coop_tile DSL port is the highest-suspicion bucket
because cooperative-tensor type IDs drift across SDK versions and the
DSL refactor reshuffled the lowering path.

**Test plan to localise (when the machine has been rebooted and the
pin cleared).**

1. **Spot-check unit suite first.** `make test-unit` runs FFAITests +
   MetalTileSwiftTests at small problem sizes against the real GPU. If
   it fails, the failing test name pins down the broken kernel
   immediately. ~3 minutes.
2. **If unit suite passes, drop in a small integration.**
   `make test-integration --filter LlamaIntegrationTests` — Llama 3.2 1B
   exercises the core decode path (RMSNorm, RoPE, SDPA d128, gemv,
   argmax) without the multi-modal or hybrid kernels. If this pins
   the GPU, the bug is on a core kernel.
3. **Bisect by family if step 2 passes.**
   `make test-integration --filter <Qwen3 | Gemma3 | NemotronH |
   Jamba | GraniteMoeHybrid | GPTOSS | Pixtral | Mistral3 |
   Whisper | Marvis>` — pick the families that cover the most
   distinct kernels (MoE, GDN, SSM, AURA, VLM, audio).
4. **Once a family pins reliably:**
   - **Path A — capture in Instruments (Metal System Trace).**
     Launch the failing test under Instruments → Metal System Trace.
     The trace shows the in-flight command buffer + its current
     dispatch. Read off the kernel name. If it's specialised
     (e.g. `mt_qmm_int4_f16_bm8_*`), that's our suspect.
   - **Path B — `MTLCaptureManager` Metal frame capture from FFAI.**
     Wrap the suspect dispatch with `MTLCaptureManager.shared()`
     `.startCapture` / `.stopCapture`. Saves a `.gputrace` you can
     open in Xcode and step through.
5. **Localised? Audit the kernel's `## DISPATCH INVARIANTS` block in
   metaltile vs the FFAI `Ops.*` wrapper that calls it.**
   Reduction-mode kernel encoded as `elementwiseGrid` is the
   canonical freeze pattern (see
   `papers/optimizing-kernels-for-apple-m-series-architecture.md`).

**Capturing the pin live (if it can be done without reboot).** It's
possible to attach Instruments to an already-pinned process and see
which command buffer is in flight — `xcrun xctrace record --template
"Metal System Trace" --attach <PID>`. Stop after ~5 seconds; the
trace shows the queue depth and the hung dispatch. This is preferable
to a reboot+rerun because we get the actual culprit rather than a
candidate from rerunning.

**Live signal at the moment.** The user reports the pin reproduces
right now (2026-05-23) after a fresh reboot — i.e. it's not a
sticky-from-previous-session situation. Some path the OS goes through
on this branch's `kernels.metallib` is enough to trigger it. That
makes the kernel-side hypothesis (one of the freshly-regenerated
kernels) substantially more likely than an FFAI-side busy-loop.

## CPU pin — vision / audio models burn all CPU cores

**Symptom.** During some integration tests (observed via `mactop`),
all CPU cores spike to 100 % while the GPU stays well below saturation.
GPU never sitting at 100 % during these runs is the giveaway — work
that should be on the GPU is silently running on the CPU instead.

**Status.** Cause unknown. The vision (Phase 6.5) and audio (Phase 7)
model families landed without a CPU-fallback audit, so the suspect
surface is anything those pipelines touch where:

- A model op silently falls through to the parallel-CPU
  `concurrentPerform` core because no GPU wrapper was wired up
  (e.g. unknown SDPA `head_dim`, missing `Ops.conv2dDepthwise`,
  a custom audio op).
- A test or harness allocates / keeps tensors on CPU instead of
  dispatching to GPU.
- A test fixture / oracle is running its CPU reference path
  (parallelised across cores) and we're misreading that as the
  model itself pinning the CPU.

**Live signal.** `mactop` shows full CPU saturation across all cores
during some currently-running tests; GPU has not been seen pegged
during the same window. Indicates CPU is doing real work the GPU
should be doing — or that a test-side CPU oracle dwarfs the GPU
dispatch.

**Next steps when this lands as a priority.** Deferred until the
current integration-test sweep finishes — that pass will surface
more failures and give a fuller picture of which model families need
attention. Then:

1. Re-run vision / audio model tests under `mactop` (or
   `powermetrics --samplers cpu_power,gpu_power`) to confirm the
   CPU pin reproduces and pin down which families.
2. For each pinning family, instrument the forward pass to log every
   op dispatch — find the calls that hit `concurrentPerform` /
   `cpuFallback` paths instead of `Ops.*` GPU dispatch.
3. Cross-reference against the FastVLM CPU depthwise-conv outlier
   in [§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage) —
   likely the same class of gap, just not yet enumerated for the
   other vision / audio families.
4. File missing `Ops.*` wrappers as separate metaltile / FFAI
   follow-ups (kernel exists in metaltile but no Swift wrapper, or
   kernel itself missing).

## Vision tower SDPA — head_dim coverage

VLM vision towers run bidirectional multi-head attention; FFAI now
ships GPU kernels for every head_dim in the cached zoo:

| head_dim | Kernel | Models |
|---|---|---|
| 32  | `Ops.sdpaBidirectional` (d32) | FastVLM (FastViT-HD) |
| 64  | `Ops.sdpaBidirectional` (d64) | SigLIP-base, CLIP-L, Mistral3, Gemma 4 E2B/E4B, Qwen3-VL 2B/4B |
| 72  | `Ops.sdpaBidirectional` (d72, ragged) | SigLIP-So400m → Paligemma, Gemma 3 VL, Gemma 4 26B/31B, Idefics3, Qwen3-VL-30B-A3B |
| 80  | `Ops.sdpaBidirectional` (d80, ragged) | Qwen2.5-VL |
| 96  | `Ops.sdpaBidirectional` (d96) | Qwen2-VL |
| 128 | `Ops.sdpaMulti(causal: false)` | Pixtral, Mistral3 Pixtral-base, GlmOcr |

`Sources/FFAI/VisionEncoder.swift::forward` dispatches based on the
tower's `headDim`. Unknown head_dims (e.g. d80 at SigLIP-So-400m
variants not yet seen) fall back to the parallel CPU `concurrentPerform`
core. No tracked head_dim gaps as of 2026-05-24.

The remaining known cold-inference outlier is FastVLM at 1024px: its
FastViT-HD stem has 256×256×96 + 128×128×192 depthwise convs that
still run CPU because no `Ops.conv2dDepthwise(...)` wrapper exists.
Tracked as a separate metaltile follow-up.

## Integration bisect — 2026-05-23/24 first-run findings

Per-suite serialised bisect run via `make integration-bisect`
(commit `6047340`+) surfacing failures across all 74 integration
tests. Fixed during the run:

- **ChatterboxIntegrationTests** — was failing on `unsupported
  dtype I64 for tensor "s3gen.speaker_encoder.xvector.block1.tdnnd6.
  nonlinear2.batchnorm.num_batches_tracked"`. Fixed in commit
  `af74ce7` by adding `.i64` / `.u64` to `DType` + making
  `SafeTensorsBundle` skip tensors with unsupported dtypes (Debug
  log only). Now PASSes in 21s.

- **CohereTranscribeIntegrationTests** — HF 404. Original
  `mlx-community/c4ai-aya-expanse-transcribe-mlx` archived; Cohere
  re-released as `cohere-transcribe-03-2026`. Fixed in commit
  `6ad9e73` pointing test at `mlx-community/cohere-transcribe-03-2026-mlx-8bit`.

Triage queue from same run (root-cause TBD — most look pre-existing):

- **DeepSeekR1DistillIntegrationTests** — R1-Distill-Llama-8B
  PASSes; R1-Distill-Qwen-1.5B produces degenerate output
  (token 15 repeated). Manually verified `mt_add_rms_norm` at
  hidden=1536 with `maxResidErr=0`, `maxNormedErr=1.4e-6` against
  CPU reference — fusion is NOT the regressor. Likely a pre-
  existing Qwen2 (R1-Distill base architecture) issue.

- **FastVLMIntegrationTests** — `Swift/ContiguousArrayBuffer.swift:
  692: Fatal error: Index out of range` during the `load` test.
  Crash happens before the GPU-SDPA path runs, so the VLM agent's
  migration is not the cause. Suspect the SafeTensors-skip change
  or a pre-existing array index in the FastViT-HD loader.

- **FireRedASR2IntegrationTests** — `FFAI/Layers.swift:19:
  Precondition failed: Linear: weight must be 2D`. Some weight
  tensor is non-2D where a `Linear` is expected — checkpoint shape
  vs loader-side reshape mismatch.

- **FishSpeechIntegrationTests** — `safetensors file not found:
  models--mlx-community--fish-audio-s2-pro-8bit/snapshots/.../
  model.safetensors`. Snapshot directory exists, file is missing —
  incomplete cache. Needs re-download or upstream check.

- **GLMASRIntegrationTests** — `Ops.swift:1072 Precondition failed:
  dequantGemv: input 65280 ≠ in_dim 1280`. Caller is feeding the
  wrong-rank tensor (looks like 51 rows of in_dim=1280 flat). Pre-
  existing shape bug in the GLMASR encoder/decoder.

Bisect continues; this list will be appended-to as more suites
complete.

## Audio model CPU bottlenecks — Whisper + QwenOmni timeout

**Symptom.** `WhisperIntegrationTests` and `QwenOmniIntegrationTests`
both hit the 900 s `gtimeout` cap during the integration bisect; the
log captures only the build banner — the test process was SIGKILL'd
before any test boundary was reported.

**Root cause — shared audio-encoder CPU attention.** Both families
route through `Sources/FFAI/AudioEncoder.swift::cpuAttention`
(line ~165). The Whisper-style encoder runs N encoder blocks (24 for
Whisper-large-v3 / Qwen2.5-Omni) of multi-head bidirectional
attention over `nAudioCtx = 1500` frames; each block performs
`O(nFrames² · nHeads · headDim)` work on the CPU via
`DispatchQueue.concurrentPerform`. With 1500² · 20 · 64 ≈ 2.9 G FLOPs
per layer × 24 layers ≈ 70 G FLOPs of pure CPU work per `encodeAudio`
call. That dominates the wall clock and pegs all cores (matches the
broader CPU-pin symptom in [§ CPU pin](#cpu-pin--vision--audio-models-burn-all-cpu-cores)).
The Q/K/V projections + LayerNorm + GELU + final GEMM are already on
GPU; only the SDPA-equivalent attention core falls back to CPU.

**Whisper additional site — decoder.** `Sources/FFAI/Models/Whisper.swift::cpuAttention`
(line ~424) runs the SAME CPU multi-head attention for the
**decoder's self- AND cross-attention**, once per generated token.
`generateTranscript` then loops `decoderLogits` for up to 224 tokens.
The cross-attention K/V is recomputed against the full 1500-frame
audio every step (no KV cache on the cross side), so each generated
token re-pays a `nQuery · nKV · headDim` matmul that should be a
single GPU SDPA dispatch. Whisper's transcribe test pays the encoder
cost once + the decoder cost ≈ 224 times.

**QwenOmni site.** `QwenOmniModel.encodeAudio`
(Sources/FFAI/Models/QwenOmni.swift:163) calls into the same shared
`AudioEncoder`, so it inherits the encoder-side bottleneck. QwenOmni
has no decoder cpuAttention of its own (it splices feature tokens
into the Qwen3 text backbone, which is already on GPU). So the
expected wall-clock improvement after migration is proportional to
the time spent in the audio tower alone — but for a multi-second
input that's still tens of seconds of pure CPU at the encoder.

**Migration needed.** Replace both `cpuAttention` cores
(`AudioEncoder.swift::cpuAttention` and
`Whisper.swift::cpuAttention`) with `Ops.sdpaBidirectional`
(encoder) and `Ops.sdpaMulti(causal: true)` (decoder self-attn,
already cached) + a cross-attention GPU dispatch. Whisper's head_dim
is 64 (small) / 80 (medium) / 80 (large-v3) — all already covered by
[§ Vision tower SDPA](#vision-tower-sdpa--head_dim-coverage). The
cross-attention path also needs a precomputed-K/V variant: encoder
K/V should be projected once per utterance, then reused for every
decoder step (eliminates the per-token K/V recompute). Mirrors the
VLM splice pattern.

**Status.** Diagnosed only; not in scope for the current bisect
pass. Tracked here so the audio-tower migration can pick this up as
the unified "audio CPU-attention → GPU SDPA" port.

## Quantization — missing 2-bit support + mixed-precision schemes

**Gap.** FFAI's quantized weight surface (`QuantizedLinear` /
`QuantizedEmbedding` + the `int4` / `int8` MetalTile kernels) covers
the symmetric per-group `affineQuantized` cases we ship today:
`bits ∈ {3, 4, 5, 6, 8}`. The integration matrix
(`Tests/ModelTests/Quantized{3,4,5,6,8}bitIntegrationTests.swift`)
exercises each, and all five passed in the 2026-05-24 bisect.

What's NOT covered yet:

1. **2-bit quantization.** No kernel variant, no `QuantizedLinear`
   path. `mlx-community` ships 2-bit conversions of several models
   (Qwen 3 32B, Llama 3 70B, etc.) that we can't load. Need:
   - `dequant_gemv_int2_{f16,bf16}` and `mt_qmm_int2_*` kernels in
     metaltile (mirror the int4 codegen — same per-group
     `(scale, bias)` layout, just 2 bits per index).
   - `Ops.dequantGemv` + `Ops.qmm` dispatch table extended with
     `bits == 2` cases.
   - `QuantizedLinear` accepts `bits == 2` (currently the validator
     rejects it).
   - `Tests/ModelTests/Quantized2bitIntegrationTests.swift` mirroring
     the existing pattern.

2. **Mixed-precision (per-tensor-class) quantization.** Some
   `mlx-community` checkpoints quantize each weight class at a
   different bit-width — e.g. attention projections at int4, MLP
   gate/up/down at int3, embeddings at int8, lm_head at int6.
   These are advertised as `mixed_3_6`, `mixed_4_8` etc. in the
   model card. The current loader (`AnyLinear.load(...)`) assumes
   a single uniform `bits` value from the config's `quantization`
   block; it can't decode a heterogeneous scheme.

   Sketch: extend the config decoder to recognise per-key
   bit-widths (probably a `quantization.weight_specs` dict in the
   config), thread the per-tensor `bits` through `loadLinear` /
   `loadEmbedding`, and let each call pick the matching kernel
   variant. No new kernels required — every selected bit-width
   already has one (modulo gap (1) for int2).

**Why it matters.** 2-bit is the only path to fit the 70B-class
dense models (Llama 3 70B, Qwen 3 70B) in a single Apple Silicon
device's wired memory; mixed-precision is what the better
`mlx-community` conversions are starting to ship (preserves quality
on the attention projections + lm_head while keeping the MLP slim).
Without these we can't load 30-40 % of the recent zoo.

**Test bar.** When implemented:
- `make test-unit` must add a 2-bit dequant_gemv GPU correctness
  test mirroring the existing int4/int8 ones, against a CPU
  reference.
- `Quantized2bitIntegrationTests` + `MixedPrecisionIntegrationTests`
  added to the bisect runner.

