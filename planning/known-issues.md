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

## Idefics3 + Paligemma — agent-ported, awaiting LanguageModel-protocol surgery

Both family files (`Sources/FFAI/Models/Idefics3.swift`,
`Sources/FFAI/Models/Paligemma.swift`) were ported by background agents
but stashed at `/tmp/*.stale` because the agent versions:
- Don't conform fully to FFAI's `LanguageModel` protocol (missing
  required methods).
- Paligemma duplicates the existing `bfloat16ToFloat` helper.
- Paligemma has a non-exhaustive `switch` over dtype.
- Both expose `loadModel(...)` returning the concrete model class
  instead of FFAI's `Loaded` tuple shape.

**Fix.** Restore from `/tmp/Idefics3.swift.stale` and
`/tmp/Paligemma.swift.stale`, rename their `loadModel` to match the
established `static func load(config:weights:options:device:) throws
-> VLModel` shape used by Pixtral / Mistral3 / SmolVLM2, drop the
duplicate `bfloat16ToFloat`, and complete the `LanguageModel`
conformance. Both have cached checkpoints
(`mlx-community/paligemma-3b-mix-448-8bit`; no Idefics3 cached
but the loader should still build).

## Indirect-dispatch test gap on metaltile

`Ops.dequantGemvIndirect` exists; metaltile generates the
`dequant_gemv_int4_{f16,bf16}_indirect` Swift wrappers via the opt-in
`Kernel.wants_indirect_variant` field on `dev`. Integration tests
already exercise this. No outstanding gap, just noting it lives in the
GPU-router code path and is default-off pending the host-side router.
