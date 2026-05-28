# Design — incremental growth for quantized KV caches (affine + AURA)

Extend the incremental KV-cache growth shipped for the raw `KVCache` (start small, double on demand, window grow-then-rotate) to the quantized caches: `AffineQuantizedKVCache` (`affine4`/`affine8`) and `AURAQuantizedKVCache` (`aura*`). Density over grammar; companion to the other `planning/*-design.md` docs.

---

## 1. Why

`KVCache` (raw, default) now grows incrementally: starts at `defaultInitialCapacity` (2048), doubles up to the context ceiling, and for `.window(maxSize:)` grows linearly to `maxSize` then rotates FIFO — instead of pre-allocating the full context. See `Sources/FFAI/KVCache/KVCache.swift` (`ensureCapacityLocked`) + `documentation/kv-cache.md` § Memory growth.

The two quantized caches still **pre-allocate** their (compressed) storage to the full ceiling. So an unbounded `affine4` cache with a 256K ceiling reserves ~0.31× the raw 256K footprint up front, and a quantized 16K window reserves the full 16K compressed ring. Growth would make those incremental too.

**Marginal value (smaller than raw growth was).** The over-allocation guard (`MemoryBudget`) + the generation-budget cap already bound the ceiling to `prompt+maxTokens` for a normal generation, and these caches are already compressed (31–55% of raw). So quantized growth only further helps the **large-ceiling** cases: an explicit big `maxContextLength`, or a quantized sliding-window at 16K+. Real, but narrower than the raw win. Sequenced accordingly — ship after the raw growth + budget/guard (done).

---

## 2. What makes the quantized caches harder than raw

Raw `KVCache` is 2 buffers (`kBuffer`, `vBuffer`), self-contained, contiguous per head → growth is one blit per head per buffer. The quantized caches add three complications:

### 2.1 Many per-layer buffers, different strides

`AffineQuantizedKVCache` (`Sources/FFAI/KVCache/KVCache.swift`) holds **6** per-layer buffers, each `[nKVHeads, maxSeq, width]` with a different inner width:

| Buffer | dtype | inner width |
|---|---|---|
| `kWeights` / `vWeights` | `u32` | `headDim / (32/bits)` (packs/row) |
| `kScales` / `vScales` | model dtype | `headDim / groupSize` (groups/row) |
| `kBiases` / `vBiases` | model dtype | `headDim / groupSize` (groups/row) |

All 6 restride identically (per head: move rows `[0, length)` from old stride → new stride) but with three distinct row-byte sizes. Same blit pattern as raw, ×6, with care on the u32-vs-dtype byte math.

### 2.2 Shared dequant scratch (the real blocker)

`AffineQuantizedKVCache` takes `sharedWorkingK` / `sharedWorkingV` — **one** full-precision `[nKVHeads, maxSeq, headDim]` pair **shared across every layer** (created once in `makeLayerCaches`). It's the bulk-dequant target: `prepareForAttention` dequants the live `length` rows of this layer's packed storage into the shared scratch, then SDPA reads it. Sharing is what keeps affine memory-efficient — per-layer full-precision scratch would be the size of a raw cache for every layer, negating the compression.

The dequant + quantize Ops (`Ops.bulkDequantKVAffine`, `Ops.quantizeKVAffine`) take a **single `maxSeq:` stride** used for *both* the packed buffer and the scratch. So the scratch stride must equal the packed capacity. Growing one layer's packed capacity therefore requires growing the shared scratch to match — and it's shared, so it can't be a plain per-cache `Tensor`.

**Resolution — `SharedDequantScratch` holder (reference type).** A small `final class` holding `var k`, `var v`, `var capacity`, created in `makeLayerCaches` and passed to every affine cache instead of two `Tensor`s. The scratch is pure *scratch* (rewritten every `prepareForAttention`, no persistent data), so growing it is a **realloc with no copy**.

Lockstep invariant that makes the single-stride Op safe: every layer appends exactly one token per forward, so all layers share the same `length` + `capacity` after each forward. When capacity crosses a growth boundary, each layer's `ensureCapacity` (a) restrides its own 6 packed buffers and (b) calls `scratch.ensureCapacity(newCapacity)` (idempotent — first layer grows it, the rest no-op). After the forward, `scratch.capacity == every cache.capacity`, so the `maxSeq` each Op passes (its cache's capacity) matches the scratch stride. Document this invariant at the holder + in `ensureCapacity`.

### 2.3 AURA: codebook + rotation + norms

`AURAQuantizedKVCache` (`Sources/FFAI/KVCache/AURAQuantizedKVCache.swift`) is another step up:
- Per-layer **packed codebook indices** `[nKVHeads, maxSeq, packedWidth]` u32 (K and V can have *different* bit widths → different packedWidths) + **norms** `[nKVHeads, maxSeq]` f32 — both restride on growth.
- Per-layer **SRHT rotation Π** state (fixed-size, capacity-independent — no restride).
- A shared working buffer in the `.dequantMirror` decode path (same holder treatment as affine); the `.compressed` path attends on packed codes directly (`aura_flash_p1`/`pass2`) and may not need the scratch — confirm per-path.
- The encode kernel `atomic_or`s into the packed row, so a grown (zeroed) row is already correct; the existing per-row zero-before-encode still applies.

AURA is the most complex + most niche → last.

---

## 3. Implementation plan

### Phase 1 — `SharedDequantScratch` + `AffineQuantizedKVCache` growth

1. **`SharedDequantScratch`** (new `final class`, `KVCache.swift` near the affine cache):
   ```
   final class SharedDequantScratch: @unchecked Sendable {
       private(set) var k, v: Tensor
       private(set) var capacity: Int
       let nKVHeads, headDim: Int; let dtype: DType; let device: Device
       private let lock = NSLock()
       func ensureCapacity(_ needed: Int)  // realloc (no copy), re-pin residency, idempotent
   }
   ```
2. **`AffineQuantizedKVCache`**: mirror the raw cache's growth structure —
   - 6 buffers → `private(set) var`; `maxSeq` computed = current `capacity`; add `contextCeiling`; store `device` + a retired-buffer list.
   - init: take `scratch: SharedDequantScratch` (replacing the two `Tensor`s) + `preallocate` + `initialCapacity`; start at `min(ceiling, initialCapacity-or-default)` (or `maxSize` for a small window, ceiling for preallocate).
   - `ensureCapacityLocked(_ needed:)`: same doubling + growth-ceiling logic as raw (unbounded → `contextCeiling`, window → `maxSize`); restride all 6 packed buffers (per-head blit, per-buffer row-bytes); then `scratch.ensureCapacity(newCapacity)`.
   - call `ensureCapacityLocked(length + 1)` at the top of `appendOnGPU` (inside the lock, before `reserveNextSlot`); the Ops already read `maxSeq` (now = capacity).
   - `effectiveMaxSize`: unbounded → `contextCeiling`, window → `maxSize` (same as raw).
3. **4 producing families** — `LlamaText`, `Qwen3Text`, `NemotronDiffusionText`, `GlmOcrVision` `makeLayerCaches`: build a `SharedDequantScratch` (at the same initial capacity logic) instead of two `Tensor`s; pass it to each cache. NemotronDiffusion keeps `preallocate: true` (free-tail staging — see raw cache).
4. **Tests** (`KVCacheTests`): affine growth preserves dequant round-trip across a boundary (quantize N rows spanning a growth, dequant, compare within int8/int4 tolerance); scratch grows with packed; window grow-then-rotate for affine; update the existing affine test call sites to the `SharedDequantScratch` init.

### Phase 2 — `AURAQuantizedKVCache` growth

Same shape: packed-indices + norms restride; reuse `SharedDequantScratch` for the dequant-mirror path; confirm the compressed path's buffer needs; rotation state untouched. Tests: aura round-trip across a growth boundary + window grow.

---

## 4. Call sites + blast radius

- `Sources/FFAI/KVCache/KVCache.swift` — `SharedDequantScratch`, `AffineQuantizedKVCache`.
- `Sources/FFAI/KVCache/AURAQuantizedKVCache.swift` — AURA growth.
- `makeLayerCaches` (affine): `Models/Text/LlamaText.swift`, `Models/Text/Qwen3Text.swift`, `Models/Text/NemotronDiffusionText.swift`, `Models/Vision/GlmOcrVision.swift`.
- `makeLayerCaches` (AURA): the 2 producers (grep `AURAQuantizedKVCache(`).
- Tests: `Tests/FFAITests/KVCache/KVCacheTests.swift` (affine + aura init sites + new growth tests).
- Docs: `documentation/kv-cache.md` — drop the "quantized caches still pre-allocate" caveat once both grow.

No `makeLayerCaches` protocol-signature change needed: the context ceiling flows through the existing `maxSeq:` param (resolved by `Model.makeManagedCaches`), and `preallocate`/`initialCapacity` ride the `KVCache.defaultInitialCapacity` knob the managed path already sets.

---

## 5. Risks

- **Restride correctness** — a wrong byte offset silently corrupts attention (the class of bug we've been burned by). Mitigate: per-head contiguous blits (same pattern as the verified raw cache), exhaustive round-trip tests across a growth boundary, distinct row-byte constants per buffer asserted in tests.
- **Scratch lockstep** — if a future code path appends to layers at *different* rates, `scratch.capacity` could lag a layer's capacity → stride mismatch. Mitigate: document the invariant; `prepareForAttention` can `precondition(scratch.capacity >= capacity)` as a cheap guard.
- **GPU-timeline safety** — same as raw: grow synchronously at append-time before queuing the token's work (the live data is from prior committed forwards). Scratch realloc has no copy, so no timeline hazard there.

---

## 6. Status

- ✅ Raw `KVCache` growth (unbounded + window grow-then-rotate) — shipped.
- ✅ Budget calculator + over-allocation guard + 8 GB OS reserve + LoadOptions overrides + `makeManagedCaches` chokepoint — shipped.
- ⏳ Phase 1 — `AffineQuantizedKVCache` growth — this spec.
- ⏳ Phase 2 — `AURAQuantizedKVCache` growth — this spec.
