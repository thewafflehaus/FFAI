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

Every published Marvis checkpoint on HuggingFace is mlx-affine-quantized
(`-4bit` / `-8bit`); FFAI's `MarvisModel.buildTransformer` hard-codes
`loadLinear(..., quantization: nil)`. Loading those through the
nil-quantization path binds a U32-packed tensor as a dense `Linear`
weight → broken model. The integration test is `@Suite(.disabled(...))`
with a precise reason.

**Fix.** Plumb `config.quantization` from `MarvisModel.load` into
`buildTransformer`, OR find an unquantized Marvis checkpoint.

## NemotronVL — no MLX checkpoint exists upstream

The integration test is `@Suite(.disabled)` because there is no
`mlx-community` conversion of `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1`
on HuggingFace today. Loader is wired and ready; re-enable when a
checkpoint lands.

## Indirect-dispatch test gap on metaltile

`Ops.dequantGemvIndirect` exists; metaltile generates the
`dequant_gemv_int4_{f16,bf16}_indirect` Swift wrappers via the opt-in
`Kernel.wants_indirect_variant` field on `dev`. Integration tests
already exercise this. No outstanding gap, just noting it lives in the
GPU-router code path and is default-off pending the host-side router.
