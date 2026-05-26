# Post-mortem: wrong-dispatch-shape GPU freeze (2026-05-18 / 2026-05-19)

**Authors:** Eric Kryski **Hardware:** Apple Silicon (M-series) **Software:** [FFAI](https://github.com/ekryski/FFAI) + sibling [metaltile](https://github.com/0xClandestine/metaltile) **Date:** 2026-05-19 **Severity:** P0 — full WindowServer freeze, lost work session, ~1 day of debugging **Status:** Mitigated; layered prevention landed (this document, [FFAI/CLAUDE.md](../CLAUDE.md), kernel headers, GPU correctness tests). Underlying class still possible; metaltile-side dispatch validator queued.

---

## TL;DR

A Swift wrapper around a Metal reduction-mode kernel passed a threadgroup size of **4 threads** to a kernel hard-coded to run with **1024 threads**. The kernel's main loop step degree (`n_simd = TPG / 32`) integer-divided to **zero**. Every thread spun in an infinite GPU loop, the command buffer never retired, the WindowServer compositor was starved of GPU time, and the system froze. There were no preconditions in the wrapper to catch the mismatch and no kernel-side guard to fail loudly.

The kernel itself was correct. The kernel's `tile bench` harness was correct. The kernel's own GPU correctness test was correct — because all three control the dispatch shape from the kernel-author side. Only the Swift wrapper layer, where caller-supplied dimensions become the dispatch geometry, was vulnerable. And it had no guard rails.

This post-mortem documents the technical failure mode, the blast radius (why a tight-loop kernel can take down macOS), why the existing test layers didn't catch it, and the five layers of defense we've now added to make it not recur.

---

## 1. Timeline

| Time (UTC, 2026-05-18 → 2026-05-19) | Event |
|---|---|
| ~T-8 h | `make test-unit` parallel run on a clean checkout. Multiple test bundles dispatch Metal kernels concurrently. WindowServer GPU share drops near zero. |
| ~T-7 h | First system freeze. Hard power-cycle required. Initial hypothesis: command-queue depth saturation under parallel test load. |
| T-6 h → T-2 h | Layered mitigations: cap `MTLCommandQueue.maxCommandBufferCount`, single-flight PSO compile, autoreleasepool around every `runAndWait`, `ModelLoadLock` actor for cross-suite model loads, suite-level `.serialized` traits, `FFAI_MAX_COMMAND_BUFFERS=1` env override. All of these reduce probability of recurrence but do not address the root cause — they make the bad dispatch less likely to be reached, not less broken. |
| T-2 h | Targeted bisect: `.disabled("bisect")` trait applied to all `OpsTests` methods; tests re-enabled one at a time. |
| T-1 h | `sdpaDecode f32 — single position attends to itself` identified as the bad actor. Test pins GPU at 100%, never completes. |
| T0 | Root cause located: `Ops.sdpaDecode` wrapper used the elementwise sizing helper for a reduction-mode kernel. Test invoked it with `head_dim = 4`, dispatching 4 threads to a kernel that requires 1024. |
| T+1 h | Wrapper rewritten with kernel-aligned dispatch + invariant preconditions; test updated to `head_dim = 128`. GPU clean. |
| T+2 h | Audit of all other `Ops` wrappers; one more (`auraEncode`) found missing preconditions; fixed. |
| T+3 h | This post-mortem and the layered prevention land. |

---

## 2. The kernel

The kernel under investigation is `ffai_sdpa_decode` from [`metaltile/crates/metaltile-std/src/ffai/sdpa_decode.rs`](https://github.com/0xClandestine/metaltile/blob/dev/crates/metaltile-std/src/ffai/sdpa_decode.rs). It's the production single-token SDPA decode for FFAI — a reduction-mode kernel with strict dispatch invariants:

- **`head_dim == 128`.** One threadgroup is laid out as 32 simdgroups × 32 lanes, and each lane owns 4 consecutive Q/K/V elements (128 / 32 = 4). Loads index by `lane * 4 + {0..3}`, unconditionally.
- **One threadgroup per Q head.** `tgid_x = q_head`. Grid in the threadgroup dimension is `n_q_heads`.
- **1024 threads per threadgroup** (32 simdgroups × 32 lanes). The cross-simdgroup online-softmax reduction is hard-coded to this layout.

The hot loop walks K positions with a per-simdgroup stride:

```rust
for _t in range(sg, n_kv, ns) {     // ns = n_simd = threads_per_group / 32
    // … load K[t], Q·K via simd_sum, online-softmax update …
}
```

`ns` is the number of simdgroups in the threadgroup — 32 in the design dispatch. Critically, **`ns` is computed at runtime as integer division by 32**.

---

## 3. The wrapper bug

The Swift wrapper looked like this (before the fix):

```swift
public static func sdpaDecode(q: Tensor, k: Tensor, v: Tensor,
                              nQHeads: Int, nKVHeads: Int, headDim: Int,
                              nKV: Int, kvStride: Int,
                              scale: Float, on cmd: MTLCommandBuffer,
                              into out: Tensor? = nil) -> Tensor {
    let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)
    let totalThreads = nQHeads * headDim
    let (grid, tg) = elementwiseGrid(totalThreads)    // ← THE BUG
    let headsPerGroup = nQHeads / nKVHeads
    // … dispatch …
}
```

`elementwiseGrid` is a helper for elementwise kernels — one thread per output element, threadgroup size capped at 256. Its signature is:

```swift
private static func elementwiseGrid(_ n: Int) -> (MTLSize, MTLSize) {
    let tg = MTLSize(width: min(256, n), height: 1, depth: 1)
    let grid = MTLSize(width: n, height: 1, depth: 1)
    return (grid, tg)
}
```

The unit test invoked `sdpaDecode` with `nQHeads = 1, headDim = 4`. That meant:

- `totalThreads = 4`
- `tg = MTLSize(width: 4, height: 1, depth: 1)`
- `grid = MTLSize(width: 4, height: 1, depth: 1)`

The kernel was dispatched with **4 threads in 1 threadgroup**, not the 1024 it was designed for.

### Why this hangs the GPU

Inside the kernel:

```rust
let ns = n_simd;                      // = threads_per_group / 32
                                       // = 4 / 32 = 0 (integer div)
for _t in range(sg, n_kv, ns) {       // step = 0
    // body never advances _t
}
```

Apple Metal's `for _t in range(start, end, step)` with `step = 0` becomes a tight infinite loop on the GPU. Every one of the 4 threads spins forever. The command buffer never retires. The GPU stays at 100% utilization. The system compositor (WindowServer) — which itself uses the GPU for every display frame — is starved. After a few seconds, macOS gives up trying to schedule WindowServer GPU time and the screen locks at the last rendered frame. Touchpad and keyboard input continue to be queued by the kernel but the display never updates. The only recovery is a hard power-cycle.

### Two related issues, same root cause

The test also passed `head_dim = 4` rather than the kernel-required 128. Each lane is coded to load `q[lane*4 + {0..3}]` and likewise for K/V. With our 4-element Q buffer, the 0th lane's loads happened to land in-bounds; subsequent lanes would have read past the end. On the GPU that's undefined behavior — sometimes a page fault, sometimes garbage data, never the dot product the kernel was written to compute.

The GPU pin and the OOB reads come from the same root: **the wrapper translated caller-supplied dimensions into a dispatch geometry the kernel was not designed to handle, and nothing in the path noticed**.

---

## 4. The same class of bug elsewhere

This wasn't a one-off. While auditing every `Ops` wrapper after the fix, one more instance was found:

- **`rmsNorm` / `rmsNormRows`** — kernel invariant `N = TPG * 4` (each thread owns 4 elements) plus `TPG` must be a multiple of 32 (cross-simdgroup reduction). The wrapper originally hard-coded `tgWidth = 256`. For `n = 4096` (a real Llama hidden dim) that ran the kernel with one thread doing 4 elements × 256 threads = 1024 covered, *missing 3072 elements* of the row. Output: silently miscomputed RMS, model generates plausible-looking gibberish, no GPU pin. We caught this one earlier when an integration test produced wrong output; it's the same bug class, just with a different failure mode (silent miscompute instead of infinite loop). Fixed by adding preconditions (`n % 128 == 0`, `n / 4 <= 1024`) and computing `tgWidth = n / 4`.

- **`auraEncode`** — kernel invariant `dim` must be a multiple of 32 (`simd_sum` reduction) and `dim <= 1024` (kernel statically allocates `shared_unit[1024]`). Wrapper passed caller-supplied `dim` straight through as TPG with no checks. No bug shipped (we caught it in the audit) but the same hole was open. Fixed by adding the same shape of preconditions.

Two kernels with strict TPG/simdgroup invariants; two wrappers that didn't enforce them. The pattern matters more than the specific kernels.

---

## 5. What the bug class is, precisely

There's a structural asymmetry in how dispatch shape is established for our kernels:

**Author-side dispatch surfaces** — `tile bench`, `Context::dispatch_with_grid` in GPU correctness tests, `make inspect-stats`:

> The author of the test / bench / inspection writes the dispatch geometry by hand, knowing the kernel's invariants. The grid and threadgroup come from a constant in the same file as the kernel. **It is structurally impossible for these surfaces to dispatch a malformed launch** — you'd have to type the wrong numbers, and the kernel author won't.

**Consumer-side dispatch surface** — the `Ops` wrapper layer in `FFAI/Sources/FFAI/Ops.swift`:

> The wrapper accepts caller-supplied dimensions and synthesizes the dispatch geometry. The wrapper author may or may not have read the kernel header. The caller (a model file, a unit test) certainly hasn't. **There is no structural barrier between caller input and the kernel's PSO** — only what the wrapper checks.

The wrapper is the only place where a user-supplied parameter can become a malformed dispatch. It must therefore be the place where the kernel's invariants are checked. Anywhere else (kernel-side runtime guards in MSL, dispatch-time validators in metaltile) is also valid as defense in depth, but the wrapper is the front line.

We had no front line.

---

## 6. Why existing tests didn't catch it

This bug had multiple test layers that you might expect to have caught it, and didn't. Walking through each one is worth doing because it explains the prevention strategy:

| Test layer | What it catches | Why it missed this |
|---|---|---|
| **Metaltile DSL unit tests** (`cargo test`, `crates/metaltile-codegen/tests/`) | Codegen pass correctness, MSL snapshot drift, trybuild error paths | Validates the kernel compiles to correct MSL. Has nothing to do with how the kernel is dispatched. |
| **Metaltile GPU correctness tests** (`crates/metaltile-std/tests/<kernel>_gpu_correctness.rs`) | Algorithm correctness on real Metal device against a CPU naive reference | Dispatches the kernel with the correct, author-written grid + threadgroup. The wrapper's bad dispatch never enters this layer. |
| **MLX side-by-side via `tile bench`** | Numerical parity vs the MLX reference implementation | Same — bench dispatch comes from the kernel's own `BenchSpec`, never from a caller. |
| **FFAI per-Op tests** (`Tests/FFAITests/OpsTests.swift`) | The `Ops.<fn>` Swift API, including its dispatch path | Should have caught this. The test we wrote happened to use a tiny `head_dim = 4` that triggered the bug, so the test itself was the trigger — but with no kernel-side guard and no wrapper-side precondition, the bug manifested as an infinite loop rather than a visible test failure. |
| **FFAI per-model integration tests** (`Tests/ModelTests/<Family>IntegrationTests.swift`) | End-to-end coherent output from real models | Real models use `head_dim = 128` always — the kernel's required value — so the wrapper's bad sizing for `head_dim = 4` was never exercised. Production code never hit it. The Op test that hit it was, in effect, the canary. |

So the bug was reachable only from an `Ops` test that passed degenerate dimensions — and once reached, it manifested as an infinite loop instead of a comparison failure. The test layer that *could* have caught it (`OpsTests`) was the layer that triggered it. With no guard upstream, the trigger was indistinguishable from a system crash.

---

## 7. Code: what NOT to do

```swift
// ❌ WRONG — silent infinite loop on small inputs; OOB on real inputs
public static func sdpaDecode(q: Tensor, k: Tensor, v: Tensor,
                              nQHeads: Int, nKVHeads: Int, headDim: Int,
                              nKV: Int, kvStride: Int,
                              scale: Float, on cmd: MTLCommandBuffer,
                              into out: Tensor? = nil) -> Tensor {
    let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)

    // 🚨 Reduction-mode kernel, but using the elementwise sizing helper.
    //    `elementwiseGrid` knows nothing about the kernel's
    //    head_dim / lane-quartile / simdgroup invariants.
    let totalThreads = nQHeads * headDim
    let (grid, tg) = elementwiseGrid(totalThreads)

    MetalTileKernels.ffai_sdpa_decode_f32(
        q: q.buffer, qOffset: q.offset,
        // … other args …
        gridSize: grid, threadgroupSize: tg, on: cmd)
    return result
}
```

Three things wrong, all symptoms of one root mistake (no kernel-invariant enforcement):

1. **No precondition on `headDim`.** The kernel hard-requires 128. The wrapper accepts anything.
2. **Elementwise sizing helper for a reduction kernel.** `elementwiseGrid` is fine when "one thread per output element" is the layout. Reduction kernels need fixed TPG values (here 1024) determined by the kernel's internal simdgroup arithmetic, not by output element count.
3. **No precondition on the dispatch shape relative to the kernel's required `n_simd`.** Even if `headDim` were validated, dispatching with `TPG < 32` makes `n_simd = TPG / 32 = 0` and the kernel's main loop becomes infinite. The dispatch geometry itself needs to be derived from the invariants, not from user input.

---

## 8. Code: what it SHOULD look like

```swift
// ✓ CORRECT — kernel invariants asserted up front, dispatch derived from them
public static func sdpaDecode(q: Tensor, k: Tensor, v: Tensor,
                              nQHeads: Int, nKVHeads: Int, headDim: Int,
                              nKV: Int, kvStride: Int,
                              scale: Float, on cmd: MTLCommandBuffer,
                              into out: Tensor? = nil) -> Tensor {
    // ─── KERNEL INVARIANTS (from sdpa_decode.rs §"DISPATCH INVARIANTS") ───
    // 1. head_dim == 128.  One threadgroup is 32 simdgroups × 32 lanes;
    //    each lane owns 4 consecutive Q/K/V elements (128 / 32 = 4).
    // 2. nQHeads % nKVHeads == 0.  GQA fan-out is integer.
    // 3. 1 threadgroup per Q head, 1024 threads per group.  tgid_x = q_head.
    precondition(headDim == 128,
                 "Ops.sdpaDecode: head_dim must be 128 (got \(headDim)); other specializations not yet emitted")
    precondition(nQHeads % nKVHeads == 0,
                 "Ops.sdpaDecode: nQHeads (\(nQHeads)) must be a multiple of nKVHeads (\(nKVHeads))")

    let result = out ?? Tensor.empty(shape: [nQHeads, headDim], dtype: q.dtype)

    // ─── DISPATCH derived from invariants, not from caller input ───
    // 1 threadgroup per q-head, 1024 threads per group.  Metal slices
    // `nQHeads * 1024` total threads into `nQHeads` groups of 1024.
    let threadsPerGroup = 1024
    let grid = MTLSize(width: nQHeads * threadsPerGroup, height: 1, depth: 1)
    let tg = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
    let headsPerGroup = nQHeads / nKVHeads

    MetalTileKernels.ffai_sdpa_decode_f32(
        q: q.buffer, qOffset: q.offset,
        // … other args …
        gridSize: grid, threadgroupSize: tg, on: cmd)
    return result
}
```

Three changes:

1. **Preconditions encode the kernel's invariants.** They live at the top of the wrapper, before any work happens. They reference the kernel-side source of truth (the `DISPATCH INVARIANTS` block in the `.rs` file) so a reader can verify alignment.
2. **Dispatch geometry is computed from invariants, not from caller input.** `threadsPerGroup = 1024` is a hard-coded constant matching the kernel's design TPG. The only caller-derived dimension is the grid's threadgroup count (`nQHeads`), which we've already validated.
3. **No elementwise sizing helper for a reduction kernel.** `elementwiseGrid` is reserved for kernels whose dispatch is genuinely "one thread per output element."

The wrapper is now the kernel's contract enforcer. If a caller hands it the wrong shape, they get a Swift `precondition` trap with a useful message, not an infinite GPU loop.

---

## 9. The same pattern for caller-controlled TPG

Some reduction kernels accept a caller-controlled TPG that is meant to scale with the input size — `rmsNorm` is the canonical example, where the row width `n` determines the TPG. The pattern is the same but the precondition shape is different:

```swift
// ✓ CORRECT — caller-controlled TPG, invariants asserted on the controlling dimension
public static func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float,
                           on cmd: MTLCommandBuffer,
                           into out: Tensor? = nil) -> Tensor {
    let n = x.elementCount

    // ─── KERNEL INVARIANTS (from mlx/rms_norm.rs §"DISPATCH INVARIANTS") ───
    // 1. N = TPG * 4 — each thread owns 4 consecutive elements.
    //    Therefore TPG = n / 4.
    // 2. TPG must be a multiple of 32 (cross-simdgroup reduction).
    //    Combined with (1): n must be a multiple of 128.
    // 3. TPG ≤ 1024 (Apple's max-threads-per-threadgroup cap).
    //    Combined with (1): n ≤ 4096.  Larger rows need rmsNormRows.
    precondition(n % 128 == 0,
                 "rmsNorm: n=\(n) must be a multiple of 128 (32-lane simdgroup × 4 elements/thread)")
    precondition(n / 4 <= 1024,
                 "rmsNorm: n=\(n) > 4096 — exceeds the 1024-thread cap of this kernel; use rmsNormRows or a chunked variant")
    let tgWidth = n / 4
    let grid = MTLSize(width: tgWidth, height: 1, depth: 1)
    let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
    // … dispatch …
}
```

The shape of the fix is identical — invariants documented at the call site, dispatch geometry derived from them — only the specific arithmetic differs.

---

## 10. Why a tight-loop kernel takes down macOS

The blast radius surprised us. A bug in user-mode code shouldn't be able to freeze the OS. The explanation lies in how Apple Silicon shares GPU time:

- **The GPU is shared between user code and the system compositor (WindowServer).** WindowServer composites every display frame using the GPU.
- **Metal scheduling is non-preemptive within a command buffer.** Once a command buffer is committed and the GPU starts executing a dispatch, that dispatch runs to completion. Apple's GPU does not preempt running threadgroups for other clients.
- **A kernel with an infinite loop never completes.** Every thread of every threadgroup of that dispatch is spinning. The dispatch never returns. The command buffer never retires.
- **WindowServer needs the GPU for every frame.** With our test process owning all GPU time and never yielding, WindowServer queues compositor work that never runs. After enough frames, macOS gives up trying to schedule it.
- **The system appears frozen because the display stops updating.** Input is still queued by the kernel — `Cmd+Tab` is still being typed into the system — but nothing rendered to screen reflects it. The only way out is a power-cycle.

This is the same general failure mode as Linux's pre-2.6.23 "while(1) hangs the desktop" on a non-preemptive kernel, except on the GPU side. macOS Sequoia did not, in our hands, recover gracefully.

The prevention is to never let an infinite-loop kernel reach the GPU. That has to happen above the kernel — the wrapper layer — because once the dispatch is committed, the OS can't save you.

---

## 11. Prevention — five layers, landed in this work

In order of how close they sit to the bug:

### Layer 1 — Wrapper preconditions (where we caught it; primary defense)

Every `Ops` wrapper around a reduction-mode kernel now encodes the kernel's TPG, simdgroup, and shape invariants as `precondition`s. The dispatch geometry is computed from those invariants, never from `elementwiseGrid`. Three wrappers have been hardened:

- `Ops.rmsNorm` / `Ops.rmsNormRows`
- `Ops.sdpaDecode`
- `Ops.auraEncode`

The remaining reduction-mode wrappers (`gemv`, `dequantGemv`, `argmax`, `softmaxCategoricalSample`) were audited and verified to already pass kernel-matching dispatch shapes (`tg = 256` matching the kernel's declared `tpg`); they need no change but were added to the test matrix.

The pattern is documented in [`FFAI/CLAUDE.md`](../CLAUDE.md) as required reading for new wrappers.

### Layer 2 — Kernel-header `DISPATCH INVARIANTS` blocks (source of truth)

Every metaltile kernel with strict dispatch invariants now carries a structured `## DISPATCH INVARIANTS` block at the top of its `.rs` source. Format:

```rust
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 1024 threads** (32 simdgroups × 32 lanes).
//! - **Grid: 1 threadgroup per q_head** (1D grid, tgid_x = q_head).
//! - **head_dim == 128.** Each lane owns 4 consecutive Q/K/V elements;
//!   loads are unconditional. Other head dims pin GPU.
//! - **kv_stride is the pre-allocated maxSeq capacity.** n_kv is the
//!   filled prefix and must satisfy `n_kv ≤ kv_stride`.
```

This is the single source of truth the wrappers cite in their preconditions. Updates here propagate to the wrapper via the next CLAUDE.md-described wrapper review.

### Layer 3 — Metaltile GPU correctness tests (algorithm validation)

The three kernels in question — `rms_norm`, `sdpa_decode`, `aura_encode` — each now have a paired `<kernel>_gpu_correctness.rs` test that dispatches on the real Metal device with the kernel's design TPG, compares against a naive CPU reference, and asserts agreement within tolerance. These do not directly prevent the wrapper bug (they dispatch with correct shape), but they pin the kernel's correct behavior so that future codegen changes can't silently regress it.

### Layer 4 — Documented audit cadence (process)

[FFAI/CLAUDE.md](../CLAUDE.md) now requires that every new `Ops` wrapper around a reduction-mode kernel cite the kernel's `DISPATCH INVARIANTS` block and encode each invariant as a `precondition` before any dispatch work. The review checklist includes:

1. Is the underlying kernel reduction-mode (uses `simd_*`, `threadgroup_alloc`, or `KernelMode::Reduction`)?
2. If yes, does the wrapper assert each invariant from the kernel's `DISPATCH INVARIANTS` block?
3. Is the dispatch geometry computed from the invariants, not from `elementwiseGrid`?

### Layer 5 — Metaltile runtime dispatch validator (deferred, the real fix)

The structural fix is to lift the invariant check into metaltile itself. Each `BenchSpec` would declare its required TPG (and any shape-dependent constraints) as data; `Context::dispatch_with_grid` and the codegen-emitted Swift wrappers would assert grid/tg compliance before launching. This makes the invariant impossible to violate from any consumer, not just FFAI.

It's deferred because the design has tradeoffs — some constraints depend on constexpr values (`head_dim` for `sdpa_decode`), others on dynamic dimensions (`n` for `rms_norm`) — and we want one cohesive design, not several ad-hoc additions. Filed as a metaltile-side follow-up.

---

## 12. Lessons

**The dispatch shape is the kernel's API.** Every reduction-mode kernel has a contract not just on its buffer arguments but on the threadgroup geometry. That contract has to be enforced somewhere. Our previous mental model treated the dispatch as a runtime parameter the wrapper could pick freely; the correct model is that it's part of the kernel signature, and the wrapper enforces it.

**Elementwise sizing helpers are dangerous as defaults.** `elementwiseGrid` is a fine function for the kernels it was designed for. It is a sharp tool to reach for as the default when wrapping a new kernel — it works often enough that the wrapper "looks reasonable" until the kernel's invariants quietly differ. The lesson isn't to delete the helper; it's to make calling it a deliberate choice, with the alternative path (kernel-invariant-derived dispatch) explicitly documented for the non-elementwise case.

**Test-trigger ≠ test-detect.** Our `OpsTests` were the only layer that could exercise wrong dispatch shapes, and they did exercise this one. Without an upstream guard, the trigger manifested as a system crash instead of an assertion failure — which made it look like a flaky parallel-test problem (the symptom we initially debugged) rather than a deterministic wrapper bug (the actual root cause). Several hours went into chasing the parallelism red herring before we bisected to a single test. A loud precondition would have made the bisect step zero.

**Non-preemptive GPU scheduling makes infinite loops a system-level failure.** This is worth internalizing because it changes how we think about defensive coding. A `while (true)` in CPU code wastes a core. A `while (true)` in a Metal kernel takes down the desktop. The right level of paranoia about runaway loops in shader code is higher than instinct suggests.

**Defense in depth is correct here.** Any one of the five layers above, by itself, prevents recurrence in some scenarios but not all. Together they make the failure mode unreachable: wrong dispatch can't slip past a wrapper precondition; if it somehow does, the kernel header documents what's correct so debugging is fast; if a future codegen change breaks the kernel internally, the GPU correctness test catches it; and (when layer 5 lands) the runtime validator catches it before launch regardless of which consumer dispatched it.

---

## 13. Action items (status as of this writing)

- ✅ Wrapper preconditions added to `rmsNorm` / `rmsNormRows` / `sdpaDecode` / `auraEncode`.
- ✅ Audit of all remaining `Ops` wrappers; no further wrapper bugs found.
- ✅ `DISPATCH INVARIANTS` blocks added to `mlx/rms_norm.rs` and `ffai/aura_encode.rs` kernel headers (`ffai/sdpa_decode.rs` already had one).
- ✅ Required reading section added to [`FFAI/CLAUDE.md`](../CLAUDE.md).
- ✅ GPU correctness tests landed for `rms_norm`, `aura_encode` (matching the pre-existing `sdpa_decode_gpu_correctness.rs` pattern).
- ✅ Kernel-side OOB guards in `mlx/rms_norm.rs` (metaltile `4b21136`). Lanes with `col + 3 >= n` clamp their load index to row[0..3], mask their `partial_ssq` contribution to 0 (so `reduce_sum` stays correct), and skip their stores. Belt-and-braces against wrong-TPG dispatch; the wrapper preconditions are still the front line.
- 📋 **Deferred:** metaltile runtime dispatch validator. Filed as a follow-up; design needs care (constexpr-dependent vs dynamic constraints, error path).

The first seven are the prevention in place today. The remaining deferred item is structural — once it lands, the invariant becomes impossible to violate from any consumer of metaltile's emitted Swift wrappers (not just FFAI).

## 14. Follow-up findings discovered while landing the fixes

This is the section we add as later layers of the same investigation surface related issues. Each entry includes how we caught it.

**`aura_encode` codegen signedness bug (metaltile `18c34c0`).** The `#[kernel]` body parser lowered `(uint + bits).cast::<i32>() - 32i32 > 0i32` to MSL `int v = (int)(uint+4) - 32; bool b = v > 0u;` — the `0i32` literal got demoted to `0u` in the comparison. C/MSL int-to-uint promotion made `-28` reinterpret as `~4e9`, so the cross-word-spill branch fired for every thread regardless of bit-width. Symptom: low nibbles of words `[word_idx+1 ..]` polluted with the previous thread's quantization index. Caught by the `aura_encode_gpu_correctness` test from the Phase A defensive coverage pass — the test was originally mine and looked like a wrong-test-expectation until the diff pattern (0x7 OR'd with 0x8 = 0xF) showed the kernel was systematically miscomputing low nibbles. **Lesson:** GPU correctness tests against a naive CPU reference catch entire classes of codegen bugs that no amount of wrapper precondition would have detected.

**Grid3D over-dispatch in tests (metaltile `ec9d170` test file fixes).** Several `*_gpu_correctness.rs` tests I authored during Phase A dispatched `grid_groups=[N, 1, 1]` with `tg=[N, 1, 1]` = N² threads instead of N. For Grid3D mode `program_id<i>()` lowers to `thread_position_in_grid.[xyz]`, so `dispatch_with_grid([N,1,1], [N,1,1])` spawns N×N threads, most of them garbage. The `conv1d` test caught it cleanly: illegitimate threads' OOB reads (returning 0) raced against legitimate writes of `x[d]` to `state[2*nC+d]`, and the max state diff was exactly `max(x)` magnitude. Same pattern silently "passed" `kv_cache_update` (Metal clamps OOB writes) and `aura_dequant_rotated` (kernel's internal `if d < dim` guard). **Lesson:** the two semantically-different dispatch shapes (Reduction = grid in threadgroups, Grid3D = grid in threads) have to be documented as loudly as the TPG invariants. Now added to `FFAI/CLAUDE.md` §"Wrapping kernels in FFAI" and to `papers/optimizing-kernels-for-apple-m-series-architecture.md` §5.2.

**Llama 3.2 1B's `head_dim=64` was producing wrong attention output silently (metaltile `b6edc9b`).** The original `ffai_sdpa_decode` was hard-coded to `head_dim=128` (4 elements per lane). The wrapper's `elementwiseGrid` dispatch was already wrong for head_dim=64 — 4 threadgroups of 256 threads instead of 1 threadgroup of 1024 — but `expectCoherentOutput` was lenient enough that Llama 3.2 1B's integration test passed with subtly-wrong attention. Phase B's OpsValidation precondition correctly trapped the wrong dispatch and flagged the gap; we shipped the head_dim=64 kernel specialization (`ffai_sdpa_decode_d64`) to actually fix the underlying bug rather than just relax the precondition. **Lesson:** loose integration tests (coherent output ≠ correct numerics) can hide real kernel bugs for the entire history of a model variant's support. Per-kernel GPU correctness tests catch what coherent-output tests can't.
