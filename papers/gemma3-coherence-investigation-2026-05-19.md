# Gemma 3 coherence bug investigation — 2026-05-19

## Status: RESOLVED 2026-05-19

**Root cause:** bf16 tanh overflow inside the GELU template.

Metal's native `tanh<bfloat>` evaluates as `(exp(2x)-1)/(exp(2x)+1)`.
For x ≈ ±54 (which Gemma 3 layer-1 gate values produce as the tanh
argument), `exp(108)` overflows bf16's 8-bit exponent (max ≈ 88.7),
yielding inf/inf = NaN.

**Fix:** Cast to fp32 inside `mt_gelu`, compute the tanh argument in
fp32, clamp to ±15 (tanh saturates to ±1 to within fp32 precision
well before that), cast back. Single edit in
`crates/metaltile-codegen/src/msl/preamble.rs`.

**Regression test:** `Tests/FFAITests/OpsTests.swift::geluBf16ExtremeInputs`
hammers bf16 GELU across [-15, 15] in 0.25 steps and asserts every
output is finite.

Gemma 3 1B-it integration test now passes with hard coherence
assertion. The original tap infrastructure remains in
`Sources/FFAI/Models/Gemma3.swift::dumpAndRestart` for future
debugging (gated by `GEMMA3_DEBUG_TAPS=1`).

## Symptom

Gemma 3 1B (`mlx-community/gemma-3-1b-it-bf16`) loads + dispatches
through `Gemma3Model.forward` without crashing. The KV cache scheme,
sliding-window per-layer eviction, RoPE base alternation, and shape
metadata all match the reference. But greedy decode degenerates:

```
ffai generate --model mlx-community/gemma-3-1b-it-bf16 \
              --prompt "Once upon a time" --verbose
top-5 next tokens:
  0 (nan)  "<pad>"
  1 (nan)  "<eos>"
  2 (nan)  "<bos>"
  3 (nan)  "<unk>"
  4 (nan)  "<mask>"
```

**All logits are NaN.** Argmax falls back to index 0 (`<pad>`), and
because the cache then carries NaN through every subsequent step, the
generated stream is `[0, 0, 0, 0, ...]`.

## What's been ruled out

Each of these would be a plausible NaN source, and each was verified
correct in isolation:

| Suspect | Test | Result |
|---|---|---|
| `head_dim=256` SDPA decode kernel | `sdpa_decode_d256_gpu_correctness.rs` — naive CPU SDPA vs the kernel on a 4×1×256 GQA shape | max \|diff\| < 1e-3 |
| `Ops.gelu` (tanh-approx) | `Tests/FFAITests/OpsTests.swift::geluF32` — closed-form formula at x ∈ {0, 1, -1, 2} | passes |
| `Ops.mul` (bf16 elementwise) | `Gemma3WeightFoldTests::bf16EmbedScaleMul` — 0.5 × 34 = 17 across all 1152 slots | passes |
| `fillScalar` (uniform bf16 fill) | `Gemma3WeightFoldTests::fillScalarBf16` — every slot holds `bf16(sqrt(1152))` ≈ 34 | passes |
| `+1.0` RMSNorm weight fold (bf16) | `Gemma3WeightFoldTests::bf16FoldGemmaTypicalValues` + `bf16FoldNegativeAndOutliers` — first 8 of `input_layernorm.weight` and the post_attn range incl. the 51.25 outlier | passes within bf16 rounding tolerance |
| RoPE convention | Compared metaltile `rope_llama.rs` (split-pair) vs `traditional=False` MLX → matches | matches |
| LM head tying | `lm_head.weight` and `embed_tokens.weight` byte-identical via Python safetensors dump | confirmed identical |

## What's known

- The integration test (`Gemma3IntegrationTests`) passes with the
  coherence assert gated behind `GEMMA3_COHERENCE_EXPECTED=1`. Set
  the env var to enforce.
- `GEMMA3_DISABLE_EMBED_SCALE=1` switches the symptom from
  all-`<pad>` to varied-degenerate (e.g.
  `[236864, 108, 108, 108, ...]`). Embed scale is therefore on the
  fault path but skipping it doesn't fix coherence — there's at
  least one other bug.
- `GEMMA3_SKIP_QK_NORM=1` (with embed scale on) keeps the all-`<pad>`
  symptom. With both knobs on, output is also all-`<pad>`. The
  combination tells us q/k norm is *necessary* for the varied-but-
  degenerate output the embed-scale-off path produces.

## The most likely remaining sources

Ranked by my current best guess:

1. **Residual-stream NaN cascade.** `input_layernorm.weight` has
   mean = 4.55 and max = 55.75 (per Python dump). After the `+1` fold
   that goes to mean = 5.55, max = 56.75. The RMSNorm-normalized
   residual stream therefore gets multiplied by ~5 per layer (and
   the outlier dim by ~56). Q/K then pass through `q_norm` /
   `k_norm` so the attention path stays bounded. But V is **not**
   normalized — `v_proj(x_norm)` carries the full ~5x scale through
   to the SDPA output and back into the residual via `o_proj`. After
   26 layers something is hitting Inf, and the next layer's RMSNorm
   sees `rsqrt(Inf) = 0`, producing NaN.

   Counter-argument: mlx_lm runs the same checkpoint coherently. Their
   forward path is identical to ours (we traced both Python and Swift
   line-by-line). The mlx-swift-lm Swift implementation runs MLX
   underneath, but MLX's `rms_norm` does the same computation as ours.

2. **A subtle dtype / cast slip.** Production Gemma 3 implementations
   in PyTorch upcast residual streams to fp32 for the modulating
   norms and back to bf16 for the projections, all transparently via
   autograd. Our pipeline stays bf16 throughout. With Gemma's
   outlier RMSNorm weights, bf16's 7-bit mantissa might be losing
   too much precision on the residual additions, eventually
   accumulating Inf in one element.

3. **An off-by-something in the per-head Q/K norm dispatch.**
   `Ops.rmsNormRows` expects `[nRows, rowSize]` flat layout; we
   reshape `q` (which has shape `[nHeads * headDim]` after `qProj`)
   without explicitly checking that the underlying memory is
   contiguous in head-major order. If the Linear output happened to
   land with head-minor strides we'd be normalizing across heads
   instead of within. (FFAI `Tensor` is contiguous-only today, so
   this *shouldn't* be possible — but the symptom matches.)

## How to proceed

The minimum next step is **GPU intermediate-value inspection**: add a
debug knob to `Gemma3Layer.forward` that, after each named operation
(`xNorm`, `q`, `qNorm2D`, `qRotated`, `attnOut`, `oOut`, `normedAttn`,
`postAttn`, `mlpNorm`, `gate`, `up`, `geluGate`, `mlpInner`, `mlpOut`,
`normedMLP`, `returned residual`), syncs the command buffer and
reads back the first 4 elements + max/min/has-NaN. Run once with the
BOS token; the first operation whose output contains Inf or NaN is
the bug location.

Mlx-swift-lm's MLX path takes care of similar diagnostics by way of
the lazy graph + `mx.eval()` — we don't have that affordance. A
2-3 hour focused debug session with the inspection plumbing is the
expected cost.

## Out of scope here

- **Gemma 4** — extends Gemma 3 with PLE (per-layer embedding) and
  alternating-sliding-window. Implementation requires Gemma 3 first.
- **Gemma 3 4B / 12B / 27B** — same architecture as 1B (head_dim=256,
  same norm + RoPE conventions), so the fix here unblocks all of them.
- **VL Gemma 3** — Phase 6.5; deferred.

The Gemma 3 family is the single largest blocked-by-this-bug surface.
Fixing it is the highest-priority Phase 6 follow-up.
