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

    /// AURA-compressed K/V (Phase 5d). Rotated + Lloyd-Max scalar
    /// quantized + bit-packed, with optional asymmetric K/V
    /// bit-widths. See `AURAScheme` for the recipe surface and
    /// `papers/aura-compression-algorithm.md` for the codec design.
    /// CLI: `--kv-cache aura4v2`, `--kv-cache aura4`, etc.
    case auraQuantized(scheme: AURAScheme = .default)
}

public enum DispatchMode: Sendable {
    case eager
    // .argumentBuffers — Phase 5
    // .icb             — Phase 5+
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

    /// Maximum context length the KV cache is sized for. `nil` (the
    /// default) lets the model family pick a sane default — useful for
    /// checkpoints that advertise an extreme context (e.g. a YaRN
    /// 262144 window), where allocating the full depth would need tens
    /// of GB. Set this to size the cache for a specific context: pass
    /// the checkpoint's full `max_position_embeddings` to use the
    /// entire advertised window, or a smaller value to bound memory.
    public var maxContextLength: Int?

    public init(
        capabilities: Set<Capability> = Capability.textOnly,
        kvCache: KVCacheKind = .raw,
        kvEviction: KVEviction = .unbounded,
        dispatchMode: DispatchMode = .eager,
        prewarm: Bool = true,
        lazyCapabilities: Bool = true,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        maxContextLength: Int? = nil
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
    }
}
