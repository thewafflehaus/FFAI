# Known Issues

Open bugs + rough notes from the model-family port. Each entry carries
enough context to pick up cold. Resolved-work history lives in
`planning/session-plan.md` / `plan.md`, not here.

## GPU pin — vision / audio bidirectional SDPA at production shape

**Symptom.** Some VLM / audio runs peg the GPU at 100 % and never
return (worst case seen: WindowServer freeze → hard reboot). Text decode
is unaffected — `ffai inspect` confirms the same checkpoints load +
text-decode cleanly, so the pin is isolated to the **vision-tower /
audio-encoder forward**.

**Leading hypothesis.** `mt_sdpa_bidirectional_d64_*` (and likely the
other bidirectional dims) has an inner-loop / bounds pathology at
production geometry — first hit on GlmOcr + MiniCPM-V vision towers at
`(nQHeads=16, nQuery=576, kvStride=576)`. No metaltile GPU-correctness
test has ever exercised these production shapes. Same class as the
wrong-dispatch freeze in
`papers/post-mortem-2026-05-19-dispatch-shape-gpu-freeze.md`.

**Mitigation in place.** Every `Ops.sdpaBidirectional` consumer at
d ∈ {32,64,80,96} routes through a pure-CPU `concurrentPerform` softmax
loop; `VisionEncoder.forward` takes the GPU path only at the one
confirmed-safe dim (d72). Qwen2.5-VL d80 GPU path is behind
`FFAI_QWEN25VL_GPU_SDPA=1`. Slow but no pin. (`defaultMaxCommandBufferCount`
is 64 since PR #14 / J.7; the old `FFAI_MAX_COMMAND_BUFFERS=16` is still
an override.)

**Next step.** Reproduce under Metal System Trace / `MTLCaptureManager`
to read the hung dispatch, OR add per-dim
`sdpa_bidirectional_d{N}_gpu_correctness.rs` in metaltile-std at the real
`(nQHeads,nQuery,kvStride)` shapes vs a CPU oracle, fix the kernel
arithmetic, then drop the CPU guards (`VisionEncoder.swift` + per-family
`*Vision.swift`). Use `make integration-bisect` + `ffai inspect` to
localise which family/shape pins.

## CPU saturation on vision / audio towers — mostly the intentional guard now

**Status.** Largely EXPECTED today: vision/audio bidirectional attention
runs on CPU **on purpose** (the GPU-pin guard above). So "all cores at
100 %, GPU idle" during VLM/audio tests is the guard working, not a new
bug. The genuine perf cost that remains:

- **Audio encoders (Whisper, Qwen-Omni)** — `AudioEncoder.swift::cpuAttention`
  runs ~24 blocks of bidirectional attention over 1500 frames on CPU
  (~70 GFLOP/utterance); `WhisperIntegrationTests` /
  `QwenOmniIntegrationTests` can hit the test timeout. Whisper's decoder
  also re-pays cross-attention CPU cost per token (no cross-side KV cache).
- **FastVLM 1024px** — FastViT-HD depthwise convs run CPU (no
  `Ops.conv2dDepthwise`).

**Next step.** Same fix as the GPU-pin entry: once `sdpaBidirectional`
is GPU-safe at production shape, vision towers move back to GPU. Audio
encoder/decoder migrate to `Ops.sdpaBidirectional` (encoder) +
`Ops.sdpaMulti(causal:true)` (decoder self-attn) + a precomputed-K/V
cross-attention dispatch.

## SDPA head-dim + AURA kernel coverage

What's wired vs outstanding (verified 2026-05-28).

**Text decode — `Ops.sdpaDecode`** (`OpsValidation.supportedSdpaHeadDims`):

| head_dim | Status | Notes |
|---|---|---|
| 64, 96, 128, 256, 512 | ✅ wired | d96 added this session (Phi-3); d128 generic; d256 two-phase; d512 dedicated |

**Sliding-window decode** (`slidingWindowSdpaHeadDims`):

| head_dim | Status |
|---|---|
| 128 | ✅ |
| 64, 256, … | ❌ sliding-window only supported at d128 today |

**Vision bidirectional — `Ops.sdpaBidirectional`** (`sdpaBidirectionalSupportedHeadDims = {32,64,72,80,96}`):

| head_dim | Kernel | GPU-safe? |
|---|---|---|
| 72 | ✅ | ✅ only confirmed-safe dim |
| 32, 64, 80, 96 | ✅ | ❌ pins GPU at production shape → CPU-guarded (see GPU-pin entry) |
| 128 | n/a | routed through `Ops.sdpaMulti(causal:false)` (Pixtral, Mistral3, GlmOcr) |

**AURA KV (Π-rotated quantized cache):**

- Family wiring: ✅ **universal** — every attention family rotates Q by Π
  / un-rotates the output (generalised this session).
- Schemes: ✅ `AURACodebook.supportedBits = {2,3,4,8}`; any `aura{kb}v{vb}`
  combo (aura4v4 default, aura4v2, aura8v8, …); encode kernels exist for
  int2/3/4/8.
- head_dim: ⚠️ **power-of-two only** (factory guard) — works at
  64/128/256, excludes 96 (Phi-3) / 80.
- Performance: ❌ decode still uses the dequant B-path; compressed-domain
  `aura_flash` fast decode is Phase H (H.1/H.2 open in session-plan).

## Qwen3.6-27B-4bit degenerate output — fix landed, pending test confirmation

**Was:** greedy decode emitted `"!"` every step at 27B (hidden=5120)
while Qwen3.5-0.8B on the same engine was coherent. **Fix:**
verified-shape gates now exclude the 27B shape from both fast paths —
`Ops.batchedQkvQmmFast` (`Qwen3xText.swift:2032-2049`) and
`Ops.rmsNormQgemvInt4Fast` (`:2759-2773`) — routing to the safe path.
Env overrides `FFAI_NO_FUSED_QKV` / `FFAI_NO_FUSED_LM_HEAD` retained.

**Confirmed 2026-05-29.** `Qwen36TextIntegrationTests` PASSED (123 s) in
the full bisect — the fix holds. Remaining cleanup: delete the stale
`KNOWN FAILURE (2026-05-27)` comment at `Qwen36TextIntegrationTests.swift:122-129`.

## LFM2-MoE — router-shape mismatch CONFIRMED (MoE crashes; dense is fine)

**Confirmed 2026-05-29 bisect.** `LFM2TextIntegrationTests`: the **dense
LFM2-350M** case passes coherently (58 % diversity), but the **LFM2-MoE**
case crashes — `Ops.swift:406: Precondition failed: gemv: in_dim mismatch
512 vs 2048`. Quantized expert slicing works (`buildLFM2MoE` →
`sliceStackedExperts`, `LFM2Text.swift:466-486`); the router is loaded as
a plain `Linear` from `feed_forward.gate.weight` with no hidden(2048)→512
projection, so the router gemv mismatches at decode. Fix: add the
router-input projection LFM2-MoE actually uses (or correct the gate-weight
orientation). Dense LFM2 is unaffected. (Stale comment at
`LFM2Text.swift:198-200` still claims quantized MoE throws
`unsupportedConfig` — no longer true.)

## GPT-OSS-20B-4bit + Gemma4-31B-4bit degenerate output — newly surfaced (2026-05-29)

Surfaced once the `FFAI_BUILD_MACHINE` gate was removed and these heavy
models ran in the bisect for the first time:
- **GPT-OSS-20B-4bit** (`loan-star/gpt-oss-20b-mlx-4Bit`) —
  `GPTOSSTextIntegrationTests` greedy-decodes 200 tokens at **12 %
  diversity** (24 unique): real tokens, then loops. Default raw KV cache +
  host-side sink correction.
- **Gemma4-31B-4bit** (`mlx-community/gemma-4-31b-it-4bit`) — the
  `Gemma4Dense (31B)` case at **10 % diversity**. Gemma4 **E2B / E4B /
  26B-A4B pass** — only the 31B dense degenerates.

Same class as the Qwen3.6-27B-4bit case above (large hidden / fused-path
kernel-shape sensitivity at 4-bit). **NOT a regression** from the
2026-05-29 cache / spec-decode / ungating work — the default raw-decode
paths these hit were untouched; the suites had simply never run before
(build-machine-gated). Investigation: separate a genuine 4-bit
quant-quality floor from a fast-path kernel-shape bug by A/B-ing the
fused-path env overrides (`FFAI_NO_FUSED_QKV` / `FFAI_NO_FUSED_LM_HEAD`
for GPT-OSS; the analogous Gemma4 fast paths), exactly as was done for
Qwen3.6-27B.

## Qwen3-1.7B-3bit degenerate output — model quality floor, NOT a kernel bug

`Quantized3bitIntegrationTests` fails coherence (19 unique tokens / 200,
real-but-repetitive). **Confirmed not a kernel bug:** the only int3
kernels on the path (`dequant_gather_int3`, `dequant_gemv_int3`) pass new
production-shape GPU-correctness cells (metaltile `5cca5ac`), and the
emitted MSL is behaviorally identical before/after the `1522fbd` codegen
change. Pure 3-bit on a 1.7B model just sits at its greedy quality floor
— same class as Qwen3.5-0.8B (which relaxed its floor to 0.05) and the
2-bit test (no coherence assert).

**Plan (to rule out false positives).** Migrate the test off the tiny
1.7B-3bit checkpoint — convert + use **Qwen3.5-2B** quantized so the
suite asserts real coherence on a model that can actually hold up at low
bit-width. (Real-usage note: even Qwen3.5-0.8B-4bit is marginal, so
very-small + very-low-bit is expected to degrade.) Alternative: relax
the 3-bit floor to the 0.8B precedent. Left failing-on-purpose until
decided.

## Open integration-test triage (carried, unverified this session)

Brief — full per-item diagnoses are in `plan.md` / conversation history:

- **DeepSeekR1Distill-Qwen-1.5B** — greedy loop at temp=0; likely a
  Qwen2-base model property, not a kernel bug (needs mlx-lm baseline
  compare). R1-Distill-Llama-8B passes.
- **FastVLM** — 4-bit vision tower unsupported in `loadPW2D`; test points
  at the bf16 checkpoint.
- **FireRedASR2** — non-2D weight where a `Linear` is expected (depthwise
  axis pick at load).
- **GLMASR** — audio adapter feeds quantized weights into `Ops.gemm`
  without dequant.
- **FishSpeech** — codec Conv1d (dilated/transposed) still CPU until
  metaltile kernels land.

## Quantization — mixed-precision + format gaps

2-bit affine + per-tensor `QuantSpec` shipped (Phase E): `bits ∈
{2,3,4,5,6,8}` load + `ffai convert --bits/--embedding-bits/--lm-head-bits
/--vision-bits`; the loader derives each tensor's width from its packed
shape. Remaining gaps:

- **Config-advertised mixed-precision** (`mixed_3_6`, Unsloth UD recipes)
  — shape-derived loading works; needs verification on real checkpoints.
  `Qwen36UnslothMixedTextIntegrationTests` exists, gated by
  `IntegrationGroupGating.enableMixedPrecisionSuites` (default off) — flip
  + bisect.
- **Per-LAYER bit budgets** — `ffai convert` can't yet drive a
  layer-by-layer recipe (per-role only); loader-side per-tensor dispatch
  already supports arbitrary widths.
- **String-typed `bits`** — `dolphin-2.9.3-qwen2-*-2bit` ship `"bits":"2"`
  (string); `ModelConfig` parser expects Int → silently drops the quant
  path.
- **GGUF** — no parser (Phase 10).

## Tokenizers — InternLM2Tokenizer not registered

Loading InternLM2 logs `InternLM2Tokenizer is not registered, falling
back to BPE`. Low priority — the BPE fallback decodes coherently; only
special-token / chat-template edges are at risk. Fix: register
`InternLM2Tokenizer` against the SentencePiece/BPE handler where
`LlamaTokenizer` / `Qwen2Tokenizer` etc. are mapped. Surfaced 2026-05-28
with the InternLM2 loader work.
