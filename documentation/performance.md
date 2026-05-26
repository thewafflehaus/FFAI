# Performance

Phase 4 (perf) closed the gap between the correctness-first Phase 0-3 implementation and what the M-series GPU can actually deliver on single-stream decode. This page captures the current numbers, what each wave changed, and where the remaining headroom is.

All numbers are **single-stream decode tokens/sec** on **Apple M1 Max**, measured at batch 1 with ~32-token prompts and `maxNewTokens = 64`. They're regression-tracked by `Tests/PerfTests/`; CI publishes the numbers per commit.

## Headline numbers

| Model | Quant | Phase 3 baseline | Phase 4 (current) | Speedup |
|---|---|---|---|---|
| Llama 3.2 1B | bf16 | 5.45 tok/s | **64.6 tok/s** | 11.9× |
| Qwen 3 4B | bf16 | 5.0 tok/s | **28.0 tok/s** | 5.6× |
| Qwen 3 4B | 8-bit | 4.7 tok/s | **27.5 tok/s** | 5.9× |
| Qwen 3 4B | 6-bit | 4.2 tok/s | **26.1 tok/s** | 6.2× |
| Qwen 3 4B | 5-bit | 4.0 tok/s | **25.4 tok/s** | 6.4× |
| Qwen 3 4B | 4-bit | 5.0 tok/s | **29.8 tok/s** | 6.0× |
| Qwen 3 4B | 3-bit | 3.6 tok/s | **24.1 tok/s** | 6.7× |

Llama 3.2 1B sees the biggest win because it's small enough to be encoder-bound at Phase 3 — eliminating the per-layer `commit + waitUntilCompleted` was almost pure profit.

## What each Phase 4 wave changed

Phase 4 was sequenced into two waves; this is what each one bought.

### Wave 1 — encoder-bound wins

The Phase 0-3 dispatch loop looked roughly like:

```
for each layer:
  rms_norm                      ──┐
  Q/K/V projections               │   ~30-70 MTLCommandBuffers
  rope                            │   per token, each with its own
  KV append (CPU memcpy)        ──┤   commit + waitUntilCompleted.
  sdpa_decode                     │
  ...                           ──┘
final rms_norm + lm-head + argmax (CPU readback of logits)
```

Wave 1 collapsed this:

1. **`kv_cache_update` Metal kernel.** Replaces the CPU memcpy + mid-layer sync with a GPU kernel that appends K/V into the per-layer cache buffer. No CPU↔GPU traffic during a layer.
2. **Single `MTLCommandBuffer` per token.** All layers + LM head + sampling enqueue onto one command buffer; one `commit + waitUntilCompleted` per token (down from ~30-70).
3. **GPU-side `argmax`.** Logits never leave the GPU; the kernel writes a single uint32 (the sampled token id). 4 bytes cross CPU↔GPU per token instead of `vocab_size × 4`.
4. **`forwardSample(...)` on `LanguageModel`.** Single entry point that runs the forward pass + GPU argmax in one trip; the `Generate.swift` loop uses it exclusively now.

Wave 1 alone took Llama 3.2 1B from 5.45 to ~50 tok/s and Qwen 3 4B from ~5 to ~22 tok/s.

### Wave 2 — kernel-bound wins

Once the encoder overhead was gone, the GPU kernels themselves became the bottleneck:

5. **Cooperative-thread `gemv`.** Replaces the naive one-thread-per- output-row gemv with the strided-reduce-dot pattern (one threadgroup per row, `simd_sum` reduction across the in_dim axis). Brought bf16 / fp16 matvec close to the M-series memory bandwidth ceiling.
6. **Multi-row `RMSNorm`.** Qwen 3's per-head q_norm / k_norm was dispatching one `rmsNorm` per head (32 + 8 = 40 launches per layer × 36 layers = **1440 launches per token**). Replaced with a single multi-row dispatch — 36 launches per token total. Significant on Qwen 3, no-op on Llama.
7. **Sub-group split per-pack `dequant_gemv`.** For 3 / 5 / 6-bit (byte-packed widths), assign one SIMD subgroup per pack within a row. Closes the perf gap between byte-packed widths and the uint32-aligned 4-bit / 8-bit. Without this, 3-bit was 30% slower than 4-bit; after, it's within 20%.
8. **RoPE → `KernelMode::Grid3D`.** Previously `Elementwise` (1D), forced redundant program-id math. Grid3D matches the `(seq, n_heads, head_dim)` shape directly.

Wave 2 lifted Qwen 3 4B 4-bit from ~22 to **29.8 tok/s** and brought the byte-packed widths within striking distance of the uint32-aligned ones.

## Where the remaining headroom is

The main outstanding gaps vs MLX's hand-tuned fused kernels are:

- **Fused `RMSNorm + gemv`.** MLX folds the RMSNorm scale into the pre-matmul rescale of the next gemv. We dispatch them separately. Expected ~10-15% on the hot QKV path.
- **Fused QKV projection.** MLX dispatches one `gemv` for `Q`, `K`, `V` concatenated, instead of three. Expected ~5-10%.
- **Online-softmax SDPA decode.** Our `sdpa_decode` is correct but not yet using the simdgroup-cooperative online-softmax pattern from metaltile-bench. Largest remaining headroom on long-context decode.
- **Argument-buffers / ICB dispatch modes.** Pre-bind weights into a per-layer argument buffer (or pre-record the entire forward pass via Indirect Command Buffers), so per-token only the activations + KV offset get bound. ~5× fewer `setBuffer` calls. Phase 8+ if profiles continue to show encoding cost matters.
- **Autotuner.** Per-shape selection of `(tile_dims, threads, unroll, simd_matrix, async_copy)`. Phase 7.

## How to reproduce

```bash
make test                               # full suite, includes PerfTests
swift test --filter PerfTests           # just perf
```

`PerfTests` are deterministic for a fixed seed but their timing is machine-dependent; the recorded `tok/s` thresholds in CI are gentle floors (regression detection, not hard targets).

For a manual sanity check:

```bash
ffai --model mlx-community/Qwen3.5-0.8B-MLX-4bit --prompt "Once upon a time" --max-tokens 64
ffai --model mlx-community/Qwen3-4B-4bit --prompt "Once upon a time" --max-tokens 64
```

The CLI prints `prompt: N tokens (Xs prefill)` + `generated: N tokens in Ys (Z tok/s)` at the end of each run.

## Methodology notes

- **Single-stream decode.** No batching, no speculative decoding.
- **Greedy.** Greedy-argmax sampling on the GPU. No temperature / top-k / top-p in the hot path.
- **bf16 / fp16.** Inputs and accumulators per the kernel; final reduction in fp32.
- **Cold prefill.** Caches start empty; the prefill column is the first-N-token slow loop, the decode column is everything after.
- **`tok/s = generated_tokens / decode_time_s`.** Excludes prefill, excludes load. Same convention as mlx-lm and llama.cpp.
- **No prewarm in the timing.** The PSO cache is populated by `LoadOptions.prewarm = true` during `Model.load` (default on); the per-token timer starts after that.

## See also

- [Architecture](architecture.md) — the per-token dispatch loop the perf work optimized.
- [KV cache](kv-cache.md) — Phase 5 will add quantized cache variants on top of the current raw cache.
- [Quantization](quantization.md) — per-bit-width perf table.
- [`planning/plan.md` § Phase 4](../planning/plan.md) — the original perf targets and prioritization rationale.
