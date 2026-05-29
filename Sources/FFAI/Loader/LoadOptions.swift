// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// LoadOptions — what the user requests when calling Model.load(...).

import Foundation

public enum KVCacheKind: Sendable, Equatable {
    /// Unquantized fp16 / bf16. The default. Per-layer `KVCache` holds
    /// raw K/V tensors directly; SDPA reads them with no per-step
    /// preprocessing.
    case raw

    /// Affine group-quantized K/V at `bits` per element (4 + 8 today;
    /// 6 is a follow-up). All layers share a single working buffer
    /// pair (≈ one layer's worth of fp16 K/V) into which the
    /// `bulk_dequant_kv` kernel writes before each SDPA step.
    /// Net memory vs `.raw`: ~45% less at 8-bit, ~65% less at 4-bit for
    /// typical models. CLI: `--kv-cache affine8` or `--kv-cache affine4`.
    ///
    /// `groupSize` interacts strongly with `bits`. Affine min-max int4
    /// only has 16 quant levels, so a single per-head outlier channel
    /// inflates a wide group's range and collapses the rest of the
    /// group onto 1-2 levels — degenerate decode. The CLI uses
    /// `groupSize: 16` for `affine4` and `groupSize: 64` for `affine8`;
    /// when constructing this case directly, pass `groupSize: 16` (or
    /// smaller) for `bits: 4`. int8's 256 levels tolerate `groupSize:
    /// 64` fine.
    case affineQuantized(bits: Int = 8, groupSize: Int = 64)

    /// AURA-compressed K/V. Rotated + Lloyd-Max scalar
    /// quantized + bit-packed, with optional asymmetric K/V
    /// bit-widths. See `AURAScheme` for the recipe surface and
    /// `papers/aura-compression-algorithm.md` for the codec design.
    /// CLI: `--kv-cache aura4v2`, `--kv-cache aura4`, etc.
    case auraQuantized(scheme: AURAScheme = .default)
}

public enum DispatchMode: Sendable {
    case eager
    // .argumentBuffers — planned
    // .icb             — planned
}

/// How an AURA-quantized KV cache services attention at decode time.
/// Only relevant when `LoadOptions.kvCache == .auraQuantized(...)` —
/// raw / affine caches ignore this setting.
public enum AURADecodePath: Sendable, Equatable {
    /// **Default.** Compressed-domain attention via the
    /// `aura_flash_p1` + `aura_flash_pass2` kernel pair. Q is rotated,
    /// scored directly against the packed K codes (no full-precision
    /// dequant), then combined with the packed V codes — the kernel
    /// dequantises per-tile on chip, never materialising a maxSeq-sized
    /// f16 mirror buffer. Realises AURA's full memory savings (~4× at
    /// `aura4v2`).
    case compressed

    /// Stage 1a behaviour. `prepareForAttention(on:)` dequantises the
    /// full compressed K/V cache into per-layer shared working buffers
    /// (`sharedWorkingK` / `sharedWorkingV`, sized
    /// `[nKVHeads, maxSeq, headDim]`), and the standard
    /// `Ops.sdpaDecode` reads those. Preserves AURA's quality but
    /// **gives back the memory savings** — the mirror is the same size
    /// as a raw fp16 cache. Kept as an opt-in path for A/B benching
    /// (`compressed` vs `dequantMirror` speed at production shapes)
    /// and for callers with the memory headroom who want
    /// matrix-engine SDPA.
    case dequantMirror
}

public struct LoadOptions: Sendable {
    /// Which capabilities to load. textIn + textOut implicitly always on.
    public var capabilities: Set<Capability>
    public var kvCache: KVCacheKind
    public var dispatchMode: DispatchMode
    /// Per-layer KV cache eviction policy. `.unbounded` (default)
    /// preserves the legacy growth-up-to-maxSeq behaviour. `.window`
    /// caps the cache at `maxSize` positions with FIFO eviction past
    /// that, with optional attention-sink retention via `keep`.
    /// Applies uniformly to every attention layer in non-hybrid models
    /// (hybrid families like GPT-OSS may override per-layer).
    public var kvEviction: KVEviction
    /// Run prewarm() before transitioning to .ready. Default true.
    public var prewarm: Bool
    /// Allow runtime enable/disable of capabilities after load.
    public var lazyCapabilities: Bool
    /// Override revision for HF download. Defaults to "main".
    public var revision: String
    /// Override the HuggingFace cache root for this load. `nil` (the
    /// default) honors the standard discovery order:
    /// `HF_HOME` env var → `~/.cache/huggingface/hub/`. Set to a fixed
    /// `URL` to point at a different location (e.g. an external SSD).
    /// Has no effect when `idOrPath` resolves to a local directory.
    public var cacheDirectory: URL?

    /// Maximum context length the KV cache may grow to (the growth
    /// *ceiling*, not the up-front allocation). `nil` (the default)
    /// uses the model's `max_position_embeddings`. The cache still
    /// grows incrementally from `KVCache.defaultInitialCapacity` up to
    /// this ceiling (unless `preallocateKVCache` is set), and the
    /// memory-budget guard clamps the effective ceiling so weights +
    /// max-KV + a working-memory margin never exceed the device's
    /// wired-memory budget. Set this to bound a model that advertises
    /// an extreme context (e.g. a 262144 window) to a context you'll
    /// actually use.
    public var maxContextLength: Int?

    /// Pre-allocate the KV cache to its full context ceiling at load
    /// instead of growing incrementally. `false` (default) starts the
    /// cache small (`KVCache.defaultInitialCapacity`, or
    /// `initialKVCacheCapacity`) and doubles on demand — lower peak
    /// memory for short sessions. Set `true` to reserve the entire
    /// context's memory up front (no realloc/copy during decode, at the
    /// cost of allocating the worst-case footprint immediately). Still
    /// subject to the over-allocation guard.
    public var preallocateKVCache: Bool

    /// Override the starting physical depth of an incrementally-grown
    /// KV cache. `nil` (default) uses `KVCache.defaultInitialCapacity`
    /// (2048). Ignored when `preallocateKVCache` is true. Clamped to
    /// the resolved context ceiling.
    public var initialKVCacheCapacity: Int?

    /// Override the wired-memory budget (in bytes) used by the
    /// over-allocation guard. `nil` (default) uses the device's
    /// `recommendedMaxWorkingSetSize` (Apple's ~75%-of-unified-memory
    /// ticket). Set a higher value to let large models keep more
    /// resident before the guard clamps the context — bounded by a hard
    /// machine ceiling so a load can never request more than the box
    /// can physically back. Set a lower value to leave more headroom
    /// for other processes.
    public var wiredLimitBytes: Int?

    /// Selects the AURA decode path. Defaults to `.compressed` (Stage
    /// 1b: attend on packed K/V codes directly via the `aura_flash_*`
    /// kernel pair — full ~4× memory savings). Set to `.dequantMirror`
    /// for the Stage 1a path that maintains a full-precision
    /// `[nKVHeads, maxSeq, headDim]` mirror buffer and runs the
    /// standard `Ops.sdpaDecode` against it — useful for A/B speed
    /// benching. Has no effect when `kvCache != .auraQuantized(...)`.
    public var auraDecodePath: AURADecodePath

    /// Default decode strategy for a NemotronDiffusion model when
    /// `generate(prompt:)` is called without an explicit `mode:`. Defaults
    /// to `.selfSpeculative` (the reference `linear_spec_generate`
    /// default). An explicit `mode:` at the call site always overrides
    /// this. Ignored by non-diffusion families.
    public var diffusionMode: DiffusionMode

    public init(
        capabilities: Set<Capability> = Capability.textOnly,
        kvCache: KVCacheKind = .raw,
        kvEviction: KVEviction = .unbounded,
        dispatchMode: DispatchMode = .eager,
        prewarm: Bool = true,
        lazyCapabilities: Bool = true,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        maxContextLength: Int? = nil,
        preallocateKVCache: Bool = false,
        initialKVCacheCapacity: Int? = nil,
        wiredLimitBytes: Int? = nil,
        auraDecodePath: AURADecodePath = .compressed,
        diffusionMode: DiffusionMode = .selfSpeculative
    ) {
        self.capabilities = capabilities.union(Capability.textOnly)
        self.kvCache = kvCache
        self.kvEviction = kvEviction
        self.dispatchMode = dispatchMode
        self.prewarm = prewarm
        self.lazyCapabilities = lazyCapabilities
        self.revision = revision
        self.cacheDirectory = cacheDirectory
        self.maxContextLength = maxContextLength
        self.preallocateKVCache = preallocateKVCache
        self.initialKVCacheCapacity = initialKVCacheCapacity
        self.wiredLimitBytes = wiredLimitBytes
        self.auraDecodePath = auraDecodePath
        self.diffusionMode = diffusionMode
    }
}
