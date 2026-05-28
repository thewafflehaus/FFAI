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
// AttentionCacheFactory — one place that turns a `KVCacheKind` into
// per-layer attention caches (raw / affine-quantized / AURA).
//
// Every attention family's `makeLayerCaches` builds caches through here
// rather than re-implementing the raw/affine/AURA switch, so every
// family honours `LoadOptions.kvCache` (affine4 / affine8 / aura*)
// identically — previously only Llama / Qwen3 wired affine + AURA and
// every other family silently built raw.
//
// Three entry points, layered:
//   1. `makeAttentionCaches(kind:count:…)` — `count` identical layers
//      (uniform dense decoders). The common case.
//   2. `makeAttentionCaches(kind:specs:…)` — heterogeneous stacks
//      (sliding-window families with per-layer eviction, Gemma 4's
//      per-layer head geometry). Layers are grouped by geometry so
//      same-shape layers share one dequant scratch.
//   3. `makeAttentionScratch` + `makeAttentionCache` — the primitives
//      hybrids use: build one scratch for the attention geometry, then
//      build a cache per attention layer while interleaving their own
//      SSM / conv / GDN state caches.
//
// Shape guard. affine needs `head_dim % group_size == 0`; AURA needs a
// power-of-two `head_dim` (SRHT). When the requested scheme can't run on
// a layer's geometry the factory warns once and falls back to the raw
// cache for that geometry — a coherent uncompressed model beats a
// quantized cache the kernels would feed garbage.

import Foundation
import Metal

/// One attention layer's geometry + eviction, for heterogeneous stacks.
struct AttentionCacheSpec {
    let nKVHeads: Int
    let headDim: Int
    let contextLength: Int
    let eviction: KVEviction
    /// Layers that never append to their own cache (Gemma 4 KV-shared
    /// layers use a 1-slot placeholder) always get a raw cache —
    /// quantizing an unused slab is pointless.
    let placeholder: Bool

    init(
        nKVHeads: Int, headDim: Int, contextLength: Int,
        eviction: KVEviction = .unbounded, placeholder: Bool = false
    ) {
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.contextLength = contextLength
        self.eviction = eviction
        self.placeholder = placeholder
    }
}

/// Shared dequant working buffers (+ AURA codebooks) for one geometry,
/// reused across every layer of that geometry. Sharing is what keeps
/// affine / AURA memory-efficient vs per-layer full-precision scratch.
struct AttentionCacheScratch {
    let sharedK: Tensor
    let sharedV: Tensor
    // AURA only — codebooks are dim-only so they're shared too.
    let kCodebook: Tensor?
    let kBoundaries: Tensor?
    let vCodebook: Tensor?
    let vBoundaries: Tensor?
}

/// Build the shared scratch for `kind` at one geometry, or `nil` when
/// the scheme is raw or the geometry can't support it (caller then
/// builds raw). Emits one warning on the unsupported-geometry path.
func makeAttentionScratch(
    kind: KVCacheKind,
    nKVHeads: Int, headDim: Int, contextLength: Int,
    dtype: DType, device: Device
) -> AttentionCacheScratch? {
    switch kind {
    case .raw:
        return nil

    case .affineQuantized(let bits, let groupSize):
        guard headDim % groupSize == 0 else {
            warnCacheFallback(
                "affine\(bits) (group_size \(groupSize))",
                reason: "head_dim \(headDim) is not divisible by group_size \(groupSize)")
            return nil
        }
        return AttentionCacheScratch(
            sharedK: Tensor.empty(shape: [nKVHeads, contextLength, headDim], dtype: dtype, device: device),
            sharedV: Tensor.empty(shape: [nKVHeads, contextLength, headDim], dtype: dtype, device: device),
            kCodebook: nil, kBoundaries: nil, vCodebook: nil, vBoundaries: nil)

    case .auraQuantized(let scheme):
        guard headDim > 0, headDim & (headDim - 1) == 0 else {
            warnCacheFallback(
                "AURA (\(scheme))",
                reason: "head_dim \(headDim) is not a power of two (SRHT requirement)")
            return nil
        }
        return AttentionCacheScratch(
            sharedK: Tensor.empty(shape: [nKVHeads, contextLength, headDim], dtype: dtype, device: device),
            sharedV: Tensor.empty(shape: [nKVHeads, contextLength, headDim], dtype: dtype, device: device),
            kCodebook: auraConstantTensor(AURACodebook.centroids(dim: headDim, bits: scheme.keyBits), device),
            kBoundaries: auraConstantTensor(AURACodebook.boundaries(dim: headDim, bits: scheme.keyBits), device),
            vCodebook: auraConstantTensor(AURACodebook.centroids(dim: headDim, bits: scheme.valueBits), device),
            vBoundaries: auraConstantTensor(AURACodebook.boundaries(dim: headDim, bits: scheme.valueBits), device))
    }
}

/// Build one attention cache. `scratch == nil` (raw kind, unsupported
/// geometry, or a placeholder layer) yields a raw `KVCache`. `layerIndex`
/// seeds the per-layer AURA SRHT rotation.
func makeAttentionCache(
    kind: KVCacheKind,
    scratch: AttentionCacheScratch?,
    nKVHeads: Int, headDim: Int, contextLength: Int,
    dtype: DType, eviction: KVEviction,
    auraDecodePath: AURADecodePath = .compressed,
    layerIndex: Int,
    preallocate: Bool = false,
    device: Device
) -> any LayerCacheProtocol {
    guard let scratch else {
        return KVCache(
            nKVHeads: nKVHeads, headDim: headDim, contextLength: contextLength,
            dtype: dtype, eviction: eviction, preallocate: preallocate, device: device)
    }
    switch kind {
    case .raw:
        return KVCache(
            nKVHeads: nKVHeads, headDim: headDim, contextLength: contextLength,
            dtype: dtype, eviction: eviction, preallocate: preallocate, device: device)

    case .affineQuantized(let bits, let groupSize):
        return AffineQuantizedKVCache(
            nKVHeads: nKVHeads, headDim: headDim, contextLength: contextLength,
            dtype: dtype, bits: bits, groupSize: groupSize,
            sharedWorkingK: scratch.sharedK, sharedWorkingV: scratch.sharedV,
            eviction: eviction, device: device)

    case .auraQuantized(let scheme):
        let rot = AURAQuantizedKVCacheRotations.build(
            headDim: headDim, layerIndex: layerIndex,
            activationDtype: dtype, device: device)
        return AURAQuantizedKVCache(
            nKVHeads: nKVHeads, headDim: headDim, contextLength: contextLength,
            dtype: dtype, scheme: scheme,
            rotation: rot.rotation, rotationT: rot.rotationT,
            rotationDtype: rot.rotationDtype, rotationDtypeT: rot.rotationDtypeT,
            kCodebook: scratch.kCodebook!, kBoundaries: scratch.kBoundaries!,
            vCodebook: scratch.vCodebook!, vBoundaries: scratch.vBoundaries!,
            sharedWorkingK: scratch.sharedK, sharedWorkingV: scratch.sharedV,
            eviction: eviction, decodePath: auraDecodePath, device: device)
    }
}

/// `count` identical layers — the uniform dense case.
///
/// - `preallocate` only affects the raw cache (quantized caches
///   pre-allocate their compressed storage regardless).
func makeAttentionCaches(
    kind: KVCacheKind,
    count: Int,
    nKVHeads: Int,
    headDim: Int,
    contextLength: Int,
    dtype: DType,
    eviction: KVEviction = .unbounded,
    auraDecodePath: AURADecodePath = .compressed,
    preallocate: Bool = false,
    device: Device
) -> [any LayerCacheProtocol] {
    let scratch = makeAttentionScratch(
        kind: kind, nKVHeads: nKVHeads, headDim: headDim,
        contextLength: contextLength, dtype: dtype, device: device)
    return (0 ..< count).map { i in
        makeAttentionCache(
            kind: kind, scratch: scratch,
            nKVHeads: nKVHeads, headDim: headDim, contextLength: contextLength,
            dtype: dtype, eviction: eviction, auraDecodePath: auraDecodePath,
            layerIndex: i, preallocate: preallocate, device: device)
    }
}

/// Heterogeneous stack — one cache per spec. Layers are grouped by
/// geometry so same-shape layers share a single dequant scratch; each
/// layer keeps its own eviction (sliding window vs global). Placeholder
/// layers always get a raw cache.
func makeAttentionCaches(
    kind: KVCacheKind,
    specs: [AttentionCacheSpec],
    dtype: DType,
    auraDecodePath: AURADecodePath = .compressed,
    preallocate: Bool = false,
    device: Device
) -> [any LayerCacheProtocol] {
    // One scratch per distinct (nKVHeads, headDim, contextLength) so
    // every same-geometry layer shares it. Keyed by a string to keep the
    // dictionary simple.
    var scratchByGeometry: [String: AttentionCacheScratch?] = [:]
    func scratch(for s: AttentionCacheSpec) -> AttentionCacheScratch? {
        let key = "\(s.nKVHeads)x\(s.headDim)x\(s.contextLength)"
        if let cached = scratchByGeometry[key] { return cached }
        let made = makeAttentionScratch(
            kind: kind, nKVHeads: s.nKVHeads, headDim: s.headDim,
            contextLength: s.contextLength, dtype: dtype, device: device)
        scratchByGeometry[key] = made
        return made
    }

    return specs.enumerated().map { (i, s) in
        makeAttentionCache(
            kind: s.placeholder ? .raw : kind,
            scratch: s.placeholder ? nil : scratch(for: s),
            nKVHeads: s.nKVHeads, headDim: s.headDim, contextLength: s.contextLength,
            dtype: dtype, eviction: s.eviction, auraDecodePath: auraDecodePath,
            layerIndex: i, preallocate: preallocate, device: device)
    }
}

// ─── AURA per-layer Π rotation ───────────────────────────────────────
//
// AURA stores K/V in a per-layer SRHT-rotated space (Π·K, Π·V). To keep
// attention scores + the residual stream correct, every attention layer
// must rotate Q into that space before SDPA and un-rotate the SDPA output
// before o_proj:
//
//     score = (Πq)·(Πk) = qᵀΠᵀΠk = qᵀk        (Π orthogonal)
//     out   = Πᵀ · softmax(score)·(Πv) = softmax(score)·v
//
// Raw / affine caches store K/V un-rotated, so both calls are no-ops for
// them — the helpers below branch on the cache type so every family can
// call them unconditionally around its `Ops.sdpaDecode`.

/// Rotate the query into the cache's Π space (AURA only). `q` is
/// `[nHeads, headDim]`; returns the same shape, ready for SDPA. Raw /
/// affine return `q` unchanged.
func auraRotatedQuery(
    _ q: Tensor, cache: any KVCacheProtocol,
    nHeads: Int, headDim: Int, on cmd: MTLCommandBuffer
) -> Tensor {
    guard let aura = cache as? AURAQuantizedKVCache else { return q }
    return Ops.auraRotatePerHead(
        q.reshaped(to: [nHeads * headDim]),
        rotation: aura.rotationDtype, nHeads: nHeads, headDim: headDim, on: cmd
    ).reshaped(to: [nHeads, headDim])
}

/// Un-rotate the SDPA output back to the original activation space before
/// o_proj (AURA only). Returns flat `[nHeads * headDim]`.
func auraUnrotatedOutput(
    _ attnOut: Tensor, cache: any KVCacheProtocol,
    nHeads: Int, headDim: Int, on cmd: MTLCommandBuffer
) -> Tensor {
    let flat = attnOut.reshaped(to: [nHeads * headDim])
    guard let aura = cache as? AURAQuantizedKVCache else { return flat }
    return Ops.auraRotatePerHead(
        flat, rotation: aura.rotationDtypeT,
        nHeads: nHeads, headDim: headDim, on: cmd)
}

/// `head_dim` supports the requested scheme (affine: divisible by group
/// size; AURA: power of two). Exposed for callers that gate inline.
func attentionCacheGeometrySupports(kind: KVCacheKind, headDim: Int) -> Bool {
    switch kind {
    case .raw:
        return true
    case .affineQuantized(_, let groupSize):
        return headDim % groupSize == 0
    case .auraQuantized:
        return headDim > 0 && headDim & (headDim - 1) == 0
    }
}

private func auraConstantTensor(_ data: [Float], _ device: Device) -> Tensor {
    let t = Tensor.empty(shape: [data.count], dtype: .f32, device: device)
    t.copyIn(from: data)
    return t
}

private func warnCacheFallback(_ scheme: String, reason: String) {
    let line =
        "Warning: requested \(scheme) KV cache, but \(reason) — "
        + "falling back to the raw fp16/bf16 cache for these layers.\n"
    FileHandle.standardError.write(Data(line.utf8))
}
