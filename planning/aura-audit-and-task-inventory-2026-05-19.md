# AURA audit + task inventory — 2026-05-19

Two parts. **Part 1** audits where FFAI's AURA codec diverges from the
working `mlx-swift-lm` TurboQuant reference. **Part 2** dumps every task
the current session has been tracking (in-session TaskCreate list, #1–#61)
so the real remaining work can be lifted into `plan.md` / `session-plan.md`.

---

## Part 1 — AURA implementation audit vs `mlx-swift-lm` TurboQuant

Reference: `~/Development/personal/ai/mlx-swift-lm@alpha`
`Libraries/MLXLMCommon/TurboQuantKVCache.swift` (+ `TurboQuantKernels.swift`).
The reference's own header calls the algorithm **GigaQuant** — a hybrid of
TurboQuant_mse + QuaRot + PolarQuant + llama.cpp k_quants, NOT a faithful
TurboQuant-paper implementation. It is the *working* implementation; FFAI's
AURA is a port of it.

### Headline finding

**The AURA model-level failure is an implementation bug, not a missing
algorithm step.** Two facts establish this:

- **The codebook is byte-identical.** FFAI `AURACodebook` (2/3/4/8-bit
  centroids, midpoints, the `√(128/dim)` scaling) matches reference
  `TurboQuantCodebook` value-for-value. → **Stage 2 "codebook
  recalibration" is NOT needed.**
- **The reference runs `useBias: false` on Qwen3 and produces coherent
  text.** DC-bias correction is opt-in in the reference and is only
  switched on for GPT-OSS-style `Linear(bias=True)` projections. → **DC-bias
  correction is NOT required for the Qwen3 AURA coherence bug.**

The background agent's Stage 1a report diagnosed the failure as the
"Stage 2 frontier — DC-bias + codebook recalibration." **That diagnosis is
refuted by the reference.** The reference works on Qwen3 with the same
codebook and no DC-bias. Whatever breaks FFAI is a bug to find by
differential debugging.

### Rotation math — matches (verified)

- Encode: reference `matmul(unit, rotationT)` == FFAI encode-kernel
  `Σ rotation[d,d']·unit[d']` — both compute `whtRot @ unit`.
- Q-prep: reference `matmul(q, rotationT)` == FFAI `auraRotatePerHead(q, Π)`
  — both `whtRot @ q`.
- Output un-rotation: reference `matmul(out, rotation)` == FFAI
  `auraRotatePerHead(out, Πᵀ)` — both `whtRotᵀ @ out`.
- FFAI's `AURARotation.srhtMatrix` == reference's `whtRot = H·diag(s)/√d`.
- Both internally consistent: `(Π·Q)·(Π·K)ᵀ = Q·K` for orthogonal Π.

No rotation-direction bug. Stage 1a's rotation wiring is correct.

### Structural divergences (alignment opportunities — not all are bugs)

1. **One codec vs two.** FFAI uses a single rotation/codec for both K and V
   per layer. Reference uses **two separate `MSECodec`s** (key + value, with
   different seeds) → decorrelated K/V quantization noise. Quality choice,
   not correctness — but worth matching.
2. **No two-phase prefill.** Reference: raw fp16 prefill buffer →
   `compressRawCache()` batch-compress at first decode → compressed decode
   thereafter (hides encode cost in TTFT). FFAI compresses every token on
   arrival. Perf divergence; *probably* not the coherence bug, but the
   batch-compress path is worth ruling out.
3. **Dequant+SDPA vs compressed-domain.** FFAI: `prepareForAttention`
   dequants the whole cache → `Ops.sdpaDecode`. Reference: compressed-domain
   `turbo_flash` kernels by default; dequant+SDPA is an opt-in fallback
   (`TURBO_DEQUANT_SDPA=1` / forced by `useBias`). Both paths exist and work
   in the reference → FFAI's dequant+SDPA path is not inherently broken.
   (This is FFAI Stage 1b / task #58.)
4. **Norm correction.** FFAI's `aura_encode` *always* stores
   `corrected_norm = ‖x‖/‖recon‖`. Reference skips correction on the
   orthogonal WHT path and stores raw `‖x‖`. FFAI's is arguably *more*
   accurate; not a bug, but a divergence to be aware of.
5. **Q-rotation dispatch.** FFAI: per-head `Ops.gemv` loop at activation
   dtype. Reference: single MLX `matmul` (fp32-accumulate). Confirm FFAI's
   `gemv` accumulates in fp32 — `strided_reduce_dot`/`reduce_sum` appear to,
   but worth pinning.

### The "collapse at index 50–55" clue

Stage 1a output is coherent for ~50 tokens, then degenerates into a repeat
run. Coherent-then-collapse is the signature of **error accumulation** (a
small systematic per-token error compounding) or a **context-length- /
cache-position-dependent bug** (rotating-buffer index, norm drift, an
off-by-one that only triggers past N tokens). It is *not* the signature of
a flat-wrong codebook (that fails from token 1).

### Recommended AURA next step

**Differential debugging, not Stage 2 research.** Capture FFAI's
intermediate tensors (rotated Q, dequanted K/V, attention scores, attention
output) on a fixed prompt and compare against the reference's on the same
prompt + checkpoint. Walk forward until the first divergence. The bug is in
there. Task #59 is **re-scoped** from "DC-bias + codebook recalibration" to
"differential-debug AURA vs TurboQuant reference."

DC-bias correction remains a real *feature* — but for GPT-OSS
(`Linear(bias=True)`), tracked separately, not a Qwen3 blocker.

---

## Part 2 — Full task inventory (in-session tracker #1–#61)

These are the ephemeral in-session `TaskCreate` items, not the
`planning/issues/` system. They accumulated across a long multi-compaction
session — many are granular completed sub-steps (rebases, per-kernel GPU
correctness tests, Phase C concurrency fixes). Listed here so the genuinely
*remaining* work can be promoted into `plan.md` / `session-plan.md` with
proper phase assignment, and the noise can be dropped.

### Completed — rebase / infra / tooling

- #1 ✅ Rebase onto upstream dev 6eccf00 (PR #25/#38/#41)
- #2 ✅ Regenerate FFAI kernels + swift test after rebase
- #3 ✅ Serialize ModelTests in Makefile (`--num-workers 1`)
- #4 ✅ Verify ek/kernel-port builds on 6eccf00
- #5 ✅ Verify ek/aura-port builds on 6eccf00
- #6 ✅ Update CLAUDE.md for new metaltile tooling
- #9 ✅ Rebase onto origin/dev def11ae (PR #43/#44/#45)
- #10 ✅ Diagnose make test-unit parallel GPU-freeze
- #11 ✅ Normalize metaltile invocations on `make`
- #13 ✅ Normalize FFAI CLAUDE.md on `make`
- #12 ✅ Drop mlx-lm/mlx-vlm goldens → `expectCoherentOutput`

### Completed — dispatch-shape / GPU-freeze post-mortem

- #14 ✅ rms_norm kernel OOB guards
- #15 ✅ Fix Ops.sdpaDecode dispatch + preconditions
- #16 ✅ Audit all Ops wrappers for kernel-invariant violations
- #17 ✅ GPU-pin / wrong-dispatch post-mortem in papers/
- #18 ✅ Reduction-mode wrapper pattern in FFAI/CLAUDE.md
- #19 ✅ DISPATCH INVARIANTS blocks in metaltile reduction kernels
- #20 ✅ metaltile GPU correctness tests — rms_norm + aura_encode
- #30 ✅ Extract OpsValidation + wrapper-precondition tests
- #33 ✅ papers/optimizing-kernels-for-apple-m-series-architecture.md

### Completed — GPU correctness test wave (A2.1–A2.6)

- #22 ✅ argmax
- #23 ✅ softmax_categorical_sample
- #24 ✅ mt_gemv + dequant_gemv + mt_qmv
- #25 ✅ kv_cache_update + conv1d_causal_step + ssm_step
- #26 ✅ aura_dequant_rotated + aura_score
- #27 ✅ aura_flash_p1 + aura_flash_pass2
- #28 ✅ Fix LayersTests RMSNorm forward (n=4 → n=128)
- #29 ✅ Fix failing aura_encode GPU correctness tests

### Completed — Phase C concurrency / cache-readiness fixes

- #34 ✅ Debug.swift setenv/getenv race (POSIX UB on Darwin)
- #35 ✅ Wire BufferPool into Ops
- #36 ✅ SafeTensors: drop mmap reference after upload
- #37 ✅ Model.events bounded buffer + dropOldest
- #38 ✅ KVCache.length lock (concurrent-decode prep)
- #39 ✅ forwardSampleCategorical 1-cmdbuf default path

### Completed — Phase 5d/5f/6 model + AURA build-out

- #41 ✅ aura_encode dtype-generic input
- #42 ✅ AURAQuantizedKVCache (identity rotation, first-light)
- #43 ✅ Wire AURA cache into Qwen3 + Llama
- #44 ✅ KV-cache-scheme smoke + integration test
- #45 ✅ Phase 5d.B — close bit-width gap (bits=6) + flash_p1 variants
- #47 ✅ Sliding-window KV cache (max-size + FIFO)
- #48 ✅ Phase 5f — attention sinks + GPT-OSS-20B
- #50 ✅ Phase 6 — dense text models (Mistral, Phi, Gemma3/4, …)
- #51 ✅ Gemma 3 family (text-only)
- #52 ✅ Per-layer KV cache eviction policy
- #53 ✅ Gemma 3 first-light coherence bug
- #54 ✅ ffai inspect command
- #55 ✅ Per-model integration test coverage audit
- #56 ✅ Standardize InspectTap + special-token surface
- #57 ✅ AURA 8-bit codec quality regression (const_fold empty-body fix)

### Pending — AURA Phase 5d.E (the live thread)

- #46 🔄 Phase 5d.E umbrella — AURA kernel follow-ups (in progress)
- #58 ⬜ **Stage 1b** — wire `aura_flash_p1`/`pass2` compressed-domain
  attention (replaces dequant+sdpaDecode). Perf.
- #59 ⬜ **Stage 2 — RE-SCOPED.** Was "DC-bias + codebook recalibration";
  the audit (Part 1) shows the codebook is identical to the working
  reference and DC-bias isn't needed for Qwen3. Real task: **differential-
  debug AURA vs the TurboQuant reference** to find the implementation bug
  behind the index-50 collapse. DC-bias stays a separate GPT-OSS feature.
- #60 ⬜ **Stage 3** — strided-output encode kernel + cache-layout flip
  `[maxSeq, nKVHeads, packedWidth]`. Perf. (User OK'd deferring this.)

  Stage 1a ✅ landed (`f067a9a`, `301e9ef`): per-layer SRHT, `auraRotatePerHead`,
  Q/output rotation in Qwen3 + dense-host forward. Infrastructure correct,
  verified by unit tests; model output still collapses → blocked on #59.

### Pending — speculative-decoding + cache foundation (audit roadmap)

Not yet in the task tracker as discrete items — these come from
`papers/concurrency-and-cache-readiness-audit-2026-05-19.md` §5 and map to
session-plan Phase 8.0–8.4:

- ⬜ `KVCacheProtocol.truncate(to:)` — unblocks all speculative shapes
- ⬜ `forwardMulti(tokenIds:positions:caches:)` — unblocks ngram + draft
- ⬜ n-gram speculative driver — first end-to-end speculative feature
- ⬜ `KVCacheProtocol.snapshot/restore` — unblocks prefix caching
- ⬜ `BatchedKVCache<C>` — unblocks batched + continuous decode
- #40 ⬜ Make `Profile` injectable (per-sequence telemetry; batched prereq)
- ⬜ Per-sequence `BufferPool` sub-allocators (only if profiling shows
  contention)

### Pending — metaltile / test-infra follow-ups

- #7 ⬜ MSL snapshot tests for AURA kernels (PR #25 pattern)
- #8 ⬜ GPU correctness tests for AURA kernels (PR #35 pattern) — partly
  covered by #26/#27; gap is SRHT (non-identity) rotation coverage, which
  metaltile's `aura_encode_gpu_correctness` does NOT exercise (only the
  FFAI-side `AURASRHTRoundTripTests` does — port upstream).
- #21 ⬜ metaltile runtime dispatch-shape validator
- #31 ⬜ metaltile codegen — preserve i32 signedness in lowering
- #32 ⬜ Flaky test: `matches_cpu_reference_f16_chained_resident_gqa`
- #61 ⬜ Plumb `sink_end` + `window_start` through `Ops.sdpaDecode`
  (kernel-level sliding-window fast path; 4–8× decode at long context)

### Pending — larger phases

- #49 ⬜ Phase 5e — GDN + SSM hybrid foundations (deferred behind 5d/5f/6)

---

## Suggested promotion into plan.md / session-plan.md

- **Phase 5d.E** — keep #46/#58/#59/#60 here. Reorder: #59 (debug, the
  blocker) → #58 (Stage 1b perf) → #60 (Stage 3 perf, deferrable).
- **Phase 8.0–8.4** — promote the 7 audit-roadmap items as discrete
  session-plan steps (truncate → forwardMulti → ngram → snapshot/restore →
  BatchedKVCache → Profile-injectable → per-seq BufferPool).
- **metaltile follow-ups** (#7/#8/#21/#31/#32/#61) — these belong in a
  metaltile-side backlog, not the FFAI phase plan; #61 is the only one with
  a user-visible payoff and could ride Phase 9 (perf).
- **Phase 5e** (#49) — unchanged; still queued behind the 5d.E close-out.
- Drop #1–#57 from active tracking — all completed; the work is in git
  history + work-session logs.
