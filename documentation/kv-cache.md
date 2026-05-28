# KV Cache

The KV cache holds per-layer K and V tensors so subsequent decode steps don't re-compute attention over the entire prefix. FFAI ships raw fp16/bf16, affine-quantized, and AURA-compressed attention caches today — plus recurrent SSM/GDN state caches for the hybrid families and a sliding-window eviction policy that composes with all of them.

## What's supported today

| Algorithm | When to use | Memory ratio | Status |
|---|---|---|---|
| **Raw fp16 / bf16** (`KVCache`, default) | All current models. | 1× | ✅ Shipped — grows incrementally (starts at 2048, doubles on demand) rather than pre-allocating the full context. See [Memory growth](#memory-growth). |
| **`affine8`** (`AffineQuantizedKVCache`, 8-bit) | Memory-constrained; ~7% decode-tok/s tax. | ~0.55× (45% smaller) measured on Qwen3 1.7B | ✅ Shipped. |
| **`affine4`** (`AffineQuantizedKVCache`, 4-bit) | Tight memory; same speed as `affine8`. | ~0.31× (69% smaller, group_size=32) | ✅ Shipped. |
| **AURA** (Adaptive Unified Rotated Activations, `AURAQuantizedKVCache`) | Best memory ratio at minimal quality loss. | ~6–8× at `aura4v2` | ✅ Shipped — per-layer SRHT rotation; decodes coherently across `aura4v4` / `aura4v2` / `aura8v4` / `aura8v8`. See [`papers/aura-compression-algorithm.md`](../papers/aura-compression-algorithm.md). |
| **Sliding window / FIFO eviction** (`.window(maxSize:keep:)`) | Long-running streams; capped memory regardless of context length. | Caps each layer's KV at `maxSize` positions independent of model `max_position_embeddings`. | ✅ Shipped — composes with every cache scheme above. |
| **SSM / Hybrid** (`Mamba2LayerCache` / `GDNStateCache` / …) | Mamba / GatedDeltaNet families (Qwen 3.5, NemotronH, Jamba, Granite4, FalconH1) | n/a — stores recurrent + conv state | ✅ Shipped. |
| **Batched** | Multi-stream decode (speculative, B>1 serving) | linear in B | 🚧 planned — see [`planning/plan.md`](../planning/plan.md) for the batched cache + speculative-decode work. |

Every attention-cache scheme (raw, `affine8`, `affine4`, AURA — and any of them under sliding-window eviction) is exercised by the `ModelKVCacheMatrixIntegrationTests` cross-product; the recurrent SSM/GDN caches ride the hybrid families' integration tests.

## How the cache works

Each layer holds its own `KVCache` instance. During the forward pass:

1. `Q`, `K`, `V` are projected from the post-RMSNorm hidden state.
2. RoPE is applied to `Q` and `K`.
3. **`kv_cache_update`** kernel appends the new `K`/`V` rows into the per-layer cache buffer **on the GPU**. No CPU↔GPU sync — the append enqueues onto the same `MTLCommandBuffer` as the rest of the layer.
4. `sdpa_decode` kernel scores the single query row against the full cached `K`/`V` slice up to the current position.

The raw cache buffer **grows incrementally** — it starts at a small depth and doubles on demand as the conversation lengthens, rather than pre-allocating the model's full context window up front (see [Memory growth](#memory-growth) below). Appends bump an `offset` within the current allocation; a growth event reallocates to a larger buffer and copies the live region over.

```swift
let caches = model.engine.makeLayerCaches()  // [any LayerCacheProtocol], one per layer
```

`makeLayerCaches()` is on the `LanguageModel` protocol — `LlamaModel`, `Qwen3Model`, and `Mamba2Model` all implement it. The user owns the cache lifetime; keep it across `forward(...)` / `forwardSample(...)` calls for multi-turn or streaming.

## Memory growth

A model's advertised context window can be enormous — Qwen 3.6-27B publishes `max_position_embeddings = 262144` (256K). Pre-allocating a KV cache for the full window costs ~16 GB on that model (16 attention layers × 4 KV heads × 262144 × 256 × 2 (K+V) × 2 bytes) **before generating a single token** — enough to exhaust unified memory on its own. The raw `KVCache` avoids that by growing incrementally:

- **Starts small.** The backing `[nKVHeads, capacity, headDim]` K and V buffers are allocated to `KVCache.defaultInitialCapacity` (**2048**) — or the context ceiling, whichever is smaller.
- **Doubles on demand.** When the live length reaches the current `capacity`, the cache reallocates to `capacity × 2` (clamped to the ceiling) and copies the live region into the new buffer. Geometric growth means O(log N) reallocations and O(N) total copy across a full run — negligible next to the per-token matmul, and reads stay contiguous (no per-token gather penalty).
- **Never exceeds the ceiling.** `contextCeiling` is the maximum depth the cache may grow to — the chosen context (`maxContextLength`, or the model's `max_position_embeddings`). The generation driver caps the ceiling at the actual generation budget (`prompt + maxTokens`), so a short request never grows beyond what it will use.

Why 2048: decode throughput on Apple Silicon peaks in the ~2K–4K context band before the quadratic attention term and KV bandwidth start to dominate (past that, sparse decode + KV eviction are the levers — see below). Starting at 2048 means the entire common operating range incurs **zero reallocations**, while a 256K-context model still only allocates ~128 MB up front instead of ~16 GB.

### Capacity vs ceiling — the three sizes

| Property | Meaning |
|---|---|
| `capacity` | Current physical depth of the K/V buffers — the SDPA / append stride. Grows; never shrinks. |
| `maxSeq` | Returns the current `capacity` (the buffer stride every SDPA dispatch uses). |
| `contextCeiling` | The maximum depth growth may reach — the chosen context window. |
| `effectiveMaxSize` | The retained-window size for reporting + sliding-window masks: `contextCeiling` for unbounded, `maxSize` for `.window`. |

### Tuning the growth

```swift
// Global default — set once at startup, before any model load.
KVCache.defaultInitialCapacity = 1024   // lower the baseline allocation

// Per-cache overrides (on the KVCache initializer):
KVCache(
    nKVHeads: …, headDim: …, maxSeq: ceiling, dtype: …,
    eviction: .unbounded,
    preallocate: false,        // true → allocate the full ceiling up front (no growth)
    initialCapacity: 4096      // nil → use defaultInitialCapacity
)
```

- **`defaultInitialCapacity`** — process-global starting depth (2048). A startup tuning knob; not safe to mutate during concurrent inference.
- **`initialCapacity:`** — per-cache starting depth override (clamped to the ceiling).
- **`preallocate:`** — force full-ceiling allocation at init (the legacy fixed-allocation behaviour). Use when you want the entire context's memory reserved up front, or for callers that stage writes into the buffer's free tail (e.g. diffusion-block forwards). `.window` caches always preallocate to their `maxSize` ring.

> **Note.** Incremental growth applies to the raw `KVCache` (the default). The quantized `affine8` / `affine4` and AURA caches still pre-allocate their (already-compressed) storage; growing them is a separate follow-up. Sliding-window (`.window`) caches allocate exactly their `maxSize` ring up front and never grow.

## Choosing a configuration

Three schemes ship today, selectable via `LoadOptions.kvCache`:

```swift
public enum KVCacheKind: Sendable, Equatable {
    case raw                                                 // default
    case affineQuantized(bits: Int = 8, groupSize: Int = 64) // affine4 / affine8
    case auraQuantized(scheme: AURAScheme = .default)        // aura4v4 / aura4v2 / aura8v4 / …
}
```

Activating the 8-bit affine cache:

```swift
let model = try await Model.load(
    "mlx-community/Qwen3-1.7B-4bit",
    options: LoadOptions(kvCache: .affineQuantized(bits: 8, groupSize: 64))
)
```

Or via the CLI:

```bash
ffai --model mlx-community/Qwen3-1.7B-4bit --prompt "..." --kv-cache affine8
```

### How `AffineQuantizedKVCache` works

Per attention layer the cache holds three packed buffers per K (and V): `kWeights` (u32, 4 int8 values per word), `kScales` (fp16/bf16, per-group), `kBiases` (fp16/bf16, per-group). All layers in one `makeLayerCaches(...)` call share **one** pair of working buffers sized `[nKVHeads, maxSeq, headDim]` in the model dtype. On `appendOnGPU(...)` the `quantize_kv_int8` kernel writes the new row into the layer's compressed storage. On `prepareForAttention(...)` (called before SDPA) the `bulk_dequant_kv_int8` kernel materialises the live slice into the shared working buffer, which SDPA then reads. Metal's default hazard tracking serializes the working-buffer reuse across layers within a single command buffer.

### Measured on Qwen3 1.7B 4-bit at maxSeq=40960

|  | Raw | affine8 | affine4 | Δ vs raw |
|---|---|---|---|---|
| KV cache (alloc) | 4.38 GB | 2.32 GB | 1.37 GB | −47% / −69% |
| Peak GPU | 5.28 GB | 3.38 GB | 2.44 GB | −36% / −54% |
| Decode tok/s | 46.7 | 43.6 | 45.4 | −7% / −3% |
| Output quality | reference | first ~13 tokens match raw, then minor drift | coherent, simpler answers | both stay on-topic |

### Per-bit `groupSize` choice

| Scheme | Default `groupSize` | Why |
|---|---|---|
| `affine8` | 64 | Plenty of precision per group; matches mlx-format weight-quant convention. |
| `affine4` | **32** | 4 bits per element ÷ a wider group loses too much discriminative power on K/V — decode degenerates into repetition at group_size=64. AURA-style rotation lets larger groups work; that's the AURA cache. |

### Affine follow-ups

- **`affine6` variant** — byte-packed sub-byte storage (mirror the existing `dequant_gather_int6` pattern). Memory between `affine4` and `affine8`.
- **Fused dequant-into-SDPA** — today each attention step pays one extra dequant kernel dispatch. A fused `bulk_dequant + sdpa_decode` kernel removes the working-buffer materialisation. Tracked alongside the AURA-performance work in [`planning/plan.md`](../planning/plan.md).

## Sliding window / FIFO eviction

Every cache implementation supports a bounded "rolling" mode where the oldest non-sink positions are evicted as new tokens stream in. Useful for indefinitely-long sessions, agent loops that re-prompt with growing context, or any workload where the upper bound on context matters more than perfect recall.

Set `LoadOptions.kvEviction` (or `--kv-window-size` / `--kv-window-keep` on the CLI):

```swift
let model = try await Model.load(
    "mlx-community/Qwen3-1.7B-bf16",
    options: LoadOptions(
        kvCache: .auraQuantized(scheme: .aura4v2),
        kvEviction: .window(maxSize: 2048, keep: 4)
    )
)
```

```bash
ffai --model mlx-community/Qwen3-1.7B-bf16 \
     --kv-cache aura4v2 \
     --kv-window-size 2048 \
     --kv-window-keep 4 \
     --prompt "..."
```

What that does:

- The cache buffer is allocated for exactly `maxSize` positions (the ring is bounded, so there's no reason to reserve the model's full `max_position_embeddings`); `maxSize` is both the physical depth and how many *positions* the cache reports as live to SDPA. Window caches do not grow — the ring rotates in place.
- The first `keep` positions are pinned — they're written linearly and never evicted. These map to the **attention-sink** tokens of Xiao et al. (2023): inputs the model relies on as anchor points, typically the first 4 tokens (BOS + tokenizer-special).
- After the cache fills, the next `maxSize - keep` slots act as a FIFO ring: slot `keep + ((absolute - keep) % (maxSize - keep))` is overwritten by the new token's K/V.
- Each K row was RoPE'd at its absolute insertion position, so softmax stays correct regardless of buffer order — the kernels see the same `[nKVHeads, maxSeq, headDim]` shape with `nKV = length`.

The AURA cache also zeroes the packed-u32 row before re-encoding, since the encode kernel `atomic_or`s into shared memory and stale bits from a prior token would OR through into the new codebook indices.

`KVCacheProtocol.absolutePosition` keeps growing monotonically across rotations — use it (instead of `length`) when computing RoPE for the next decode token.

### Caveats

- **Prefill larger than the window**: prefill writes tokens one at a time, so a prompt longer than `maxSize` rotates older tokens out before generation starts. The model has whatever K/V was last written for those positions. Practically: pick `maxSize ≥ prompt tokens you care about`.
- **No partial-recall fast path**: there's no "summary token" or attention-sink merge yet (Xiao et al. propose averaging older positions into the sinks). The current implementation is the vanilla rotating-buffer baseline that downstream perf-papers build on.

## Multi-turn / streaming

For multi-turn or streaming UIs, drive the loop yourself and reuse the cache across calls (see [quickstart.md § Lower-level API](quickstart.md#lower-level-api)):

```swift
let caches = model.engine.makeLayerCaches()

func respond(_ prompt: String, position: inout Int) -> String {
    var pos = position
    var nextToken = 0
    for t in model.tokenizer.encode(text: prompt) {
        nextToken = model.engine.forwardSample(tokenId: t, position: pos, caches: caches)
        pos += 1
    }
    var generated: [Int] = []
    while !isStop(nextToken) {
        generated.append(nextToken)
        nextToken = model.engine.forwardSample(tokenId: nextToken, position: pos, caches: caches)
        pos += 1
    }
    position = pos
    return model.tokenizer.decode(tokens: generated)
}
```

`pos` keeps advancing across calls; the cache holds every K / V row appended so far.

## What's coming

From [`planning/plan.md`](../planning/plan.md):

- **AURA performance** — compressed-domain attention (`aura_flash_p1` / `aura_flash_pass2`) as the *default* decode path, dropping the persistent dequant working-buffer; two independent K/V codecs; two-phase prefill. Perf only — AURA correctness ships today.
- **Compressed-domain prefix KV cache** — reuse the AURA encode at snapshot time for cross-request prefix caching.
- **Batched / hybrid batched cache** — `BatchedKVCache` + `BatchedHybridCache` (slot-based admission) for speculative decoding and multi-stream / continuous-batch serving.

## See also

- [Architecture](architecture.md) — where the cache sits in the per-token dispatch loop.
- [Performance](performance.md) — current `tok/s` numbers, including what `kv_cache_update` bought us vs the original CPU-memcpy append.
- [Quantization](quantization.md) — weight quantization (a different axis from KV cache compression).
