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
    /// **Default.** Compressed-domain attention via the 2-pass FA-2
    /// kernel pair (`aura_flash_p1` + `aura_flash_pass2`) when emitted
    /// for the (keyBits, valueBits, headDim, dtype) combo, with the
    /// single-pass `aura_flash_sdpa` as fallback for cells the 2-pass
    /// kernel hasn't been emitted for. Q is rotated + pre-scaled,
    /// scored directly against the packed K codes (no full-precision
    /// dequant), and the V codes are dequanted per-tile on chip. The
    /// `[nKVHeads, maxSeq, headDim]` mirror buffer never materialises,
    /// realising AURA's memory savings (~1.88× at aura4v4, ~3.7× at
    /// aura4v2 on Qwen3 d=128).
    ///
    /// AURA-dtype unification (metaltile + FFAI joint change) put the
    /// per-token norms and per-scheme codebook into the activation
    /// dtype, so encode + both decode kernel paths consume the cache
    /// buffers directly — no per-call f32 cast on the decode hot path
    /// and no parallel f32 mirror storage.
    case compressed

    /// Dequant-mirror path. `prepareForAttention(on:)` materialises
    /// the full compressed K/V cache into per-layer shared working
    /// buffers (`sharedWorkingK` / `sharedWorkingV`, sized
    /// `[nKVHeads, maxSeq, headDim]`) and `Ops.sdpaDecode` reads those.
    /// Same quality as `.compressed`, **gives back the memory
    /// savings** — the mirror is the same size as a raw fp16 cache.
    /// Useful as an A/B baseline against the compressed path.
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

    /// Maximum context length the KV cache is sized for. `nil` (the
    /// default) lets the model family pick a sane default — useful for
    /// checkpoints that advertise an extreme context (e.g. a YaRN
    /// 262144 window), where allocating the full depth would need tens
    /// of GB. Set this to size the cache for a specific context: pass
    /// the checkpoint's full `max_position_embeddings` to use the
    /// entire advertised window, or a smaller value to bound memory.
    public var maxContextLength: Int?

    /// Selects the AURA decode path. Defaults to `.compressed` — the
    /// 2-pass FA-2 kernel pair gives token-parallel attention over the
    /// packed K/V codes directly, with no f16/f32 mirror materialised.
    /// Set to `.dequantMirror` for an A/B baseline that dequants the
    /// cache into a per-layer working buffer and runs the standard
    /// `Ops.sdpaDecode` against it. Has no effect when
    /// `kvCache != .auraQuantized(...)`.
    public var auraDecodePath: AURADecodePath

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
        auraDecodePath: AURADecodePath = .compressed
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
        self.auraDecodePath = auraDecodePath
    }
}
