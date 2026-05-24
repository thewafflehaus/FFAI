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

## Gap 3 (GPU vision attention) — blocked on metaltile

Closing this gap was scoped as wiring the four CPU-bound vision
towers (Idefics3, PaliGemma, GlmOcr, FastVLM) through
`Ops.sdpaMulti(causal: false)` for bidirectional attention plus
adding `Ops.conv2dDepthwise(...)` for FastVLM's FastViTHD chain.

**Audit finding (2026-05-23):** `Ops.sdpaMulti` is **head_dim-128
only** by hard kernel constraint — each lane owns 4 elements
unconditionally (`OpsValidation.validateSdpaMulti`:158). SigLIP /
CLIP / FastViT vision towers use head_dim in {64, 72, 80, …}; none
match. Migrating the four families to GPU SDPA therefore needs a new
metaltile kernel with parametric head_dim (`ffai/sdpa_bidirectional_*`
or extending `sdpa_multi` per-head-dim like the decode kernel does
for d64 / d128 / d256 / d512). FFAI-side, none of this can be
unblocked yet.

FastVLM additionally needs `conv2d_depthwise_*` (depthwise +
pointwise conv chain in FastViTHD's RepMixerBlock — `Ops.gemm`
covers pointwise, but no depthwise wrapper exists).

What's already mitigated in FFAI today: all four families' CPU
attention is parallelised across `(head, token)` with
`DispatchQueue.concurrentPerform`. That collapsed Whisper /
SigLIP-896 from minutes to seconds. The remaining cold-inference
tail is FastVLM at 1024px specifically — its early stages run
256×256×96 + 128×128×192 depthwise convs that need the GPU port.

**Re-scoped Phase 6.5b (metaltile work):**
1. `ffai_sdpa_bidirectional_{d64,d72,d80,d128}_{f16,bf16}` — single
   kernel parametric over head_dim via constexpr lane geometry, or
   per-head-dim specialised variants matching the SDPA decode pattern.
2. `conv2d_depthwise_{kh}_{kw}_{stride}_{f16,bf16}` — direct
   sliding-window MAC, no im2col blow-up.
3. FFAI Ops wrappers + migrate the four families.

Tracked, but blocked outside this session.
