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

## CPU attention single-threaded sweep — in flight

`VisionEncoder.cpuAttention` and `AudioEncoder.cpuAttention` were
single-threaded scalar attentions over `O(nHeads · nTokens² · headDim)`.
At Whisper's 1500 audio-context rows and SigLIP-896's 4096 patch
tokens those loops took 15-60+ minutes per encoder pass — what
surfaced as the VLM image+text "hang" and the Whisper `transcribe`
empty-output "bug" (the decoder timed out before any token came back).
Both fixed by parallelising over `(head, query-row)` with
`DispatchQueue.concurrentPerform` (commits `9bb4e5f`, `3fdb81b`).

The remaining single-threaded sites being swept now:
- `Whisper.cpuAttention` (decoder self + cross attention)
- `SenseVoice.cpuAttention`
- `Qwen3ASR` — the nested for-head MHA in the decoder
- `Gemma4VL.cpuAttention` — vision tower; per-head RMSNorm + multi-dim
  RoPE + GQA, two-stage parallelization needed
- `Qwen25VL`, `Qwen2VL`, `Qwen3VL` — vision tower `cpuAttention(qkv:nTokens:)`

The same fix likely applies to the Granite-H / Jamba decode-hang
the owner flagged — sweep their decode path next pass.

## Marvis TTS — loader rejects quantized weights

**Fixed in commit `6fd6a87`.** `MarvisConfig.quantization` is now
plumbed through `build` → `buildTransformer` → `loadLinear` /
`loadEmbedding`, and the `MarvisModel` fields switched from
`Embedding`/`Linear` to `AnyEmbedding`/`AnyLinear`. Every
`mlx-community` `-4bit`/`-8bit` Marvis checkpoint now binds to
`QuantizedLinear`/`QuantizedEmbedding`. Suite re-enabled.

## NemotronVL — no MLX checkpoint exists upstream

**Self-skipping in commit `69306a6`.** The suite no longer carries
`.disabled`; instead each test calls `nemotronVLIsCached()` which
probes the HF cache for the candidate snapshot names and exits
early when none is present. Drop an mlx-style Nemotron-VL
conversion into the cache and the suite auto-enables on the next
run — no code change needed.

## Idefics3 + Paligemma + GlmOcr — landed; full VLModel adapter pending

All three were ported by background agents and landed (commits
`0f33924`, `777cf63`). Each ships as a `LanguageModel` engine — the
test casts `m.engine` to the concrete family type to access the vision
APIs (`encodeImage(...)` / `setImagePixels(_:)` / etc). Conformance gaps
fixed inline:

- Idefics3 + GlmOcr: added the missing `forward(...:on:device:)` overload
  that `LanguageModel` requires for command-buffer chaining.
- Paligemma: same `forward(...:on:device:)` overload, plus the missing
  `.auraQuantized` case in `makeLayerCaches` (falls back to raw KV cache),
  plus renamed `bfloat16ToFloat` → `paligemmaBfloat16ToFloat` to dodge
  the file-scope collision with FishSpeechLayers' helper.
- GlmOcr: renamed its custom `SafeTensorsBundle.prefixed(...)` extension
  to `glmOcrPrefixed(...)` to dodge the public-API collision, dropped a
  shadowing `Tensor.toFloatArray()` extension, made `textEmbedding(...)`
  public.

**Design note — engine downcast is intentional for these three.**
FFAI's `VLModel` adapter wraps `VisionEncoder + LanguageModel + splice`
for families that cleanly factor into those three layers (Pixtral,
Mistral3, Qwen3VL, SmolVLM2, MiniCPM-V, Gemma3/4-VL …). Idefics3,
PaliGemma, and GLM-OCR all inline image substitution *into* the
engine's `forward(tokenId:...)`: at every image-token position the
forward swaps the text embedding for a precomputed vision feature.
That keeps their forward path identical to a plain text decode (no
adapter intermediation needed for the splice), so the `Model.engine`
downcast is the supported access pattern. `VLMTestSupport` now ships
`dogImageCHW(targetSize:)` + `dogImageCHWNormalized(targetSize:
normalization:)` so each of these tests can run the full image+text
dog assertion at its native resolution.

## Indirect-dispatch test gap on metaltile

`Ops.dequantGemvIndirect` exists; metaltile generates the
`dequant_gemv_int4_{f16,bf16}_indirect` Swift wrappers via the opt-in
`Kernel.wants_indirect_variant` field on `dev`. Integration tests
already exercise this. No outstanding gap, just noting it lives in the
GPU-router code path and is default-off pending the host-side router.
