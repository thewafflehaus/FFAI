# Concurrency + cache-readiness audit — 2026-05-19

Forward-looking audit before the next wave of features lands:

- **Prefix caching** — share a KV-cache prefix across multiple requests with the same conversation history (system prompt, few-shot examples).
- **Smarter eviction / retention** (H2O, StreamingLLM keep-+-roll, SnapKV, etc.) — beyond the current `.window(maxSize:keep:)` FIFO.
- **Speculative decoding** — n-gram, draft model, MTP. Decode N candidate tokens, verify in parallel, rewind on rejection.
- **Batch + continuous decoding** — multiple sequences sharing one model. Mid-batch admission, eviction.

Each section below: what we already have working in the right shape, where the current code is going to fight us, and what to do about it.

## 1. KV cache surface — current shape vs needs

### What we have

- `KVCacheProtocol` — one cache instance per (layer, sequence). The protocol surface today: `appendOnGPU`, `prepareForAttention`, `reset`, `length`, `absolutePosition`, `eviction` (the new `KVEviction` enum), `effectiveMaxSize`, plus bytes accounting.
- `KVEvictionState` — slot-allocator value type used by every cache impl. `.unbounded` and `.window(maxSize:keep:)` policies. Tracks `absoluteCount` independently of `length` (the latter saturates at `maxSize`).
- Three implementations: `KVCache` (raw bf16/fp16), `AffineQuantizedKVCache` (int4/int8), `AURAQuantizedKVCache` (Lloyd-Max codec).

### What's missing for the upcoming features

**A. Truncation / rewind to an arbitrary position.** Speculative decoding needs to call `cache.truncate(to: acceptedPrefix)` after a partial-accept. The current `reset()` only zeroes; no in-between option exists. Adding `truncate(to:)` is **trivial** for `KVCache` + `AffineQuantizedKVCache` (just call `_evictionState.truncate(to:)`); for `AURAQuantizedKVCache` we also need to clear the packed-u32 rows past the new tip so the next encode's `atomic_or` doesn't OR through stale bits (similar to the rotating-write zeroing we already do).

**B. Snapshot / restore.** Prefix caching needs to checkpoint a cache state after the shared prefix is filled, then restore it for each new request. Three sub-needs:

1. **In-memory snapshot** — just clone the live `length` rows of the backing K/V buffers. For raw KV this is a single blit; for affine + AURA it's a blit of the packed/scales/biases tensors. The eviction state also needs to be cloned.
2. **Disk snapshot** — same data, serialized. Out of scope for this audit; just observe that the in-memory shape is the precondition.
3. **Block / page reuse** — vLLM-style paged attention where multiple sequences point at the same physical pages for the shared prefix. **This is a much bigger refactor** (the cache becomes a virtual-address layer over a shared page pool). Out of scope this audit.

**C. Per-sequence parallelism inside one cache.** Batched decode needs N sequences sharing one cache *layout* (so we get one big matmul instead of N small ones in attention). Today, one `KVCache` instance = one sequence. To go batched we'd need either:

- N independent `KVCache` instances + a per-step gather kernel that stacks them before SDPA — easy plumbing, wastes attention parallelism, bad for B>4.
- A `BatchedKVCache` wrapper that owns one [B, nKVHeads, maxSeq, headDim] buffer with per-slot fill counts + per-slot eviction state.

The second shape is the one mlx-swift-lm uses (`BatchedKVCache.swift`). Recommend adopting that same shape. It's a new file, not a refactor of `KVCacheProtocol`.

**D. Eviction *policy* surface.** Beyond FIFO, the upcoming algorithms need information the current `KVEvictionState` doesn't track:

- **H2O / SnapKV** — pick the K slots with the *highest* accumulated attention score across some window. Needs per-position attention-mass accumulation (a separate GPU buffer the SDPA kernel adds to) + a reservoir-eviction policy.
- **StreamingLLM (keep+roll)** — what we already have at `.window(maxSize:keep:)`.
- **Sink-rolling with merge** — merge evicted slots into a sink-average rather than discarding. Needs an `evict(at:)` hook that runs before the slot is overwritten.

The `KVEviction` enum is the right place to grow. Add cases as needed; the enum-driven dispatch inside `KVEvictionState.reserveNextSlot()` localises the new logic.

### Concrete plan: extend `KVCacheProtocol`

```swift
public protocol KVCacheProtocol: LayerCacheProtocol {
    // ... existing surface ...

    /// Truncate the cache to the first `length` positions. The
    /// positions past `length` become available for write but
    /// retain whatever garbage was there. Cheap O(1) for raw +
    /// affine caches; AURA needs to zero the packed-u32 rows in
    /// the rolled tail.
    func truncate(to length: Int)

    /// Capture the *live* state for prefix-reuse. The result is
    /// opaque to the caller; pass back to `restore(from:)` on
    /// a fresh cache of the same shape to reload. Cost: O(length).
    func snapshot(on cmd: MTLCommandBuffer) -> KVCacheSnapshot

    /// Inverse of `snapshot`. Resets to whatever state the
    /// snapshot captured. The caller is responsible for the
    /// snapshot being from a compatible cache (same nKVHeads /
    /// headDim / dtype / scheme); we check + panic.
    func restore(from snapshot: KVCacheSnapshot, on cmd: MTLCommandBuffer)
}

public struct KVCacheSnapshot: Sendable {
    public let length: Int
    public let absolutePosition: Int
    // ... opaque buffer references the cache impl needs to restore ...
}
```

`truncate(to:)` is the unblocker for speculative decoding. `snapshot/restore` is the unblocker for prefix caching. Both can land independently; speculative is the higher priority.

## 2. Race-condition surface — current state

Audited every lock + `@unchecked Sendable` in `Sources/FFAI/`. Summary:

| Component | Lock | Risk under upcoming features |
|---|---|---|
| `KVCache.length` / `absolutePosition` | `NSLock` (`lengthLock`) | ✅ Safe for sequential decode. Batched decode contended — see § 2.A. |
| `AffineQuantizedKVCache.length` | `NSLock` | Same as above. |
| `AURAQuantizedKVCache.length` | `NSLock` | Same — plus the per-slot `zeroPackedSlot` BLIT runs INSIDE the lock; can't parallelise sequences sharing one cache instance. |
| `Model.currentState` / `events` | `NSLock` + bounded `AsyncStream` (Phase C #4) | ✅ Safe under any decoder. |
| `Profile.shared` | `OSAllocatedUnfairLock<ProfileLevel>`, `OSAllocatedUnfairLock<PhaseTimings>` | ✅ Per-field locks; safe under concurrent `recordPhase`. But singleton — multiple parallel sequences accumulate together. **Task #40 already tracks "make Profile injectable" — needed for batched.** |
| `BufferPool.shared` + `BufferPoolScope` | `NSLock` per pool / per scope | ✅ Safe. But the *shared* pool's freelist is contended under many parallel decoders. May want per-sequence sub-pools to avoid lock convoying — measure first. |
| `Debug.shared` setenv/getenv | `NSLock` (Phase C #1) | ✅ Safe. |
| `SafeTensorsBundle` | Read-only after load | ✅ Safe; no concurrent mutation. |
| `InspectTap.cachedFromEnvironment` | Swift `static let` (thread-safe lazy init) | ✅ Safe. |
| MTL command-buffer ordering | Implicit via `device.makeCommandBuffer()` ordering on a single `MTLCommandQueue` | ⚠️ See § 2.B. |
| AURA / affine `sharedWorkingK` / `sharedWorkingV` | None — shared across all layers in one model load | ⚠️ See § 2.C. |

### 2.A — KV-cache locks under batched decode

Today, each layer's KV cache has its own `lengthLock` (NSLock). One forward pass holds the lock for the duration of an `appendOnGPU` call (microseconds — it just queues a dispatch). Sequential decode pays a single uncontended lock per token per layer.

**Under batched decode**, multiple sequences share one `BatchedKVCache` instance. If we keep the current "one lock for the whole cache" shape and N concurrent task threads call `appendOnGPU`, every other thread waits. The fix is to thread the per-slot `KVEvictionState` (per-sequence) and lock per-slot. The `BatchedKVCache` design I sketch in § 1.C should carry one `NSLock` *per slot*, plus the cache's own bytes / capacity-allocation lock at the top.

Counter-argument: at first cut we can keep one lock per cache and serialise admission — the critical section is microseconds and N is small (4 / 8 / 16). Measure before optimizing.

### 2.B — MTLCommandBuffer + MTLCommandQueue ordering

Apple's `MTLCommandQueue` serialises submission order: cmdbufs commit in the order they were `commit()`'d on a given queue. **Within one queue this is safe for sequential decode**. For batched decode where N tasks each create their own cmdbuf and commit independently, we need to think about:

1. **Cross-sequence ordering of cache writes vs reads.** If task A queues `appendOnGPU(seq=0)` and task B queues `sdpaDecode(over the same buffer including seq=0's row)` on the same queue, the queue's FIFO order guarantees B sees A's write IFF B was committed after A.
2. **Per-sequence cmdbuf ownership.** The current `forward(...on cmd:)` API takes one cmdbuf and queues all layers + lm_head on it. With B>1, we'd want one cmdbuf per sequence, or one cmdbuf per layer-step covering all sequences. The mlx-swift-lm `EvaluateBatched` pattern is "one cmdbuf per layer-step, all B sequences" — this is the cleanest extension of our current code.

**Action:** when batched decode lands, the model's `forward(...)` becomes `forward(tokenIds: [Int], positions: [Int], caches: [[any LayerCacheProtocol]], ...)`. Backwards-compatible single-sequence overloads keep the existing surface for non-batched callers.

### 2.C — Shared working buffers in compressed caches

`AffineQuantizedKVCache` and `AURAQuantizedKVCache` share a single `sharedWorkingK` / `sharedWorkingV` buffer pair *across every layer* of one model load. The contract is: each layer's `prepareForAttention(on: cmd)` writes the dequantized live slice INTO the shared buffer, then SDPA reads from it. Metal's default hazard tracking serialises this WITHIN one cmdbuf.

**This breaks under batched decode** where N sequences want N independent dequantized views simultaneously. Two options:

1. Per-batch-slot working buffers — one pair per sequence. Memory cost: N × (one layer's dequantized KV) = N × ~10 MB for typical shapes. Acceptable up to B=16.
2. Sequential prepare-then-attend per slot inside one cmdbuf — works but kills the batching win.

Option (1) is the right shape. The `BatchedKVCache` wrapper allocates the working pool sized for B.

**Untouched by this:** raw `KVCache` doesn't have working buffers (it serves K/V from its own storage), so it's fine for batched as long as the underlying buffer has space for B sequences (which the `BatchedKVCache` shape provides).

### 2.D — `Profile.shared` singleton

Multiple parallel sequences hit `Profile.shared.recordPhase(...)` at end-of-generation. Per-field `OSAllocatedUnfairLock` makes the writes safe, but the *aggregation* is "all generations into one shared bucket" — useless for per-sequence telemetry under batched.

**Action:** Task #40 already tracks "make Profile injectable". Land before batched lands. Each `Model.generate(...)` accepts a `profile: Profile = .shared` parameter; batched dispatchers create per-sequence Profiles.

### 2.E — `Model.events` AsyncStream

Currently a per-`Model` instance bounded buffer. For batched / continuous decoding the events surface needs to discriminate per-sequence:

```swift
public struct ModelLifecycleEvent: Sendable {
    public let state: ModelLifecycleState
    public let sequenceId: SequenceId?   // NEW — nil for whole-model events
    public let timestamp: Date
    // ... existing payload ...
}
```

Adding the optional field is non-breaking. Continuous-decoding spec lands the consumer side.

## 3. Speculative decoding — readiness

Three flavours queued:

1. **n-gram (PLD)** — match incoming tokens against a sliding prefix dictionary; speculate based on the longest match. **No new kernels needed.** Just a `[Int]` history buffer + suffix-array lookup. The cache requirement is `truncate(to:)`.
2. **Draft model (vanilla speculative)** — run a small draft model decoding N tokens, then verify N tokens in parallel against the main model. Needs:
   - `truncate(to:)` on both main + draft KV caches.
   - **Parallel verification of N tokens in one forward pass.** Our `forward(...)` is single-token today. Need a `forward(tokenIds: [Int], positions: [Int], ...)` overload that runs N tokens through the attention stack in one cmdbuf. This is the same generalisation batched decode needs at the per-sequence level.
3. **MTP** (Medusa / EAGLE-3 / native MTP head) — main model emits N+1 candidate tokens per forward; rejection-sample. Same N-token verification need.

**Common dependency:** N-token forward. Recommend landing this **before** any speculative scheme; both n-gram and draft-model speculative trivially build on top.

Shape:
```swift
extension LanguageModel {
    /// Multi-token forward. `tokenIds` are the candidate next tokens
    /// (positions[i] gives each token's absolute sequence position).
    /// Returns logits for each of the N positions. Caller decides
    /// which prefix to accept; the cache holds all N candidates'
    /// K/V; caller `truncate(to:)`s past the accepted boundary.
    func forwardMulti(
        tokenIds: [Int], positions: [Int],
        caches: [any LayerCacheProtocol]
    ) -> [Tensor]
}
```

Default extension implements it as a per-token loop on top of `forward(...on cmd:)`. Concrete families override when there's a parallelism win (the obvious one: prefill is already this shape internally — just expose it via the protocol).

## 4. Inspect command — what to extend for the new features

The current `ffai inspect` surface is plenty for single-sequence model bring-up. The upcoming features need:

- **Per-slot KV occupancy** in batched mode — `inspect --batched` shows N slots' `length / maxSize / eviction-state`.
- **Speculative-decode hit rate** — `inspect --spec --tokens N` shows the empirical accept rate of the chosen speculative scheme on a probe prompt.
- **Active experts (MoE)** — once MoE lands, `inspect --moe-routing` shows which experts fire on a per-token basis. Not in scope today; queued.

Add these incrementally as the underlying features land. No pre-work needed.

## 5. Recommended order of operations

1. **`KVCacheProtocol.truncate(to:)`** — small, unblocks all three speculative-decoding shapes. Land first.
2. **`forwardMulti(tokenIds:positions:caches:)`** — small, unblocks n-gram + draft speculative. Land second.
3. **n-gram speculative driver** — pure-Swift logic on top of (1) + (2). First decoding feature, end-to-end.
4. **`KVCacheProtocol.snapshot/restore`** — unblocks prefix caching.
5. **`BatchedKVCache<C>`** — unblocks batched + continuous.
6. **Make `Profile` injectable** (Task #40) — prerequisite for per-sequence telemetry.
7. **Per-sequence `BufferPool` sub-allocators** — only after profiling shows lock contention.

Each item is a self-contained PR. Land them in this order and the surface for every upcoming feature is in place without backtracking.

## 6. Race-condition checklist before each merge

A short list of "did I think about this" questions for any PR that touches generation / cache:

- [ ] Does any mutable state escape its owner's lock?
- [ ] Does a new singleton (`*.shared`) prevent per-sequence isolation? (If yes, make it injectable.)
- [ ] Does a new cache field need an `absolutePosition` separate from `length` (RoPE math)?
- [ ] Are MTLCommandBuffer ordering assumptions documented at the boundary?
- [ ] Is the shared-working-buffer pattern still safe when N sequences hit the same layer concurrently? (See § 2.C.)
- [ ] Does the new feature need `truncate(to:)` / `snapshot/restore` that doesn't exist yet?

If any answer is "I'm not sure," it's a code-review blocker.

## Out of scope

- vLLM-style **paged attention** (virtual address layer over physical pages). Major refactor; defer until B > 16 makes the memory savings worth the complexity.
- **ANE offload of LM head / vision tower**. Separate axis; not affected by anything in this audit.
- **Dynamic adapter / LoRA swap** mid-decode. The cache + Profile changes here don't preclude it but don't enable it either; needs its own design pass.
