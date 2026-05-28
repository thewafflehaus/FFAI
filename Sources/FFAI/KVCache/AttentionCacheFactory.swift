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
// Every attention family's `makeLayerCaches` should call this rather
// than re-implementing the raw/affine/AURA switch. Centralizing it means
// every family honours `LoadOptions.kvCache` (affine4 / affine8 / aura*)
// identically — previously only Llama / Qwen3 wired affine + AURA and
// every other family silently built raw — and the shared dequant
// scratch + AURA codebooks + per-layer SRHT rotations live in exactly
// one spot.
//
// Shape guard. The quantized schemes have geometry requirements the
// kernels can't satisfy for every model:
//   - affine: `head_dim` must be divisible by `group_size`.
//   - AURA:   `head_dim` must be a power of two (SRHT rotation).
// When the requested scheme can't run on this geometry the factory warns
// once and falls back to the raw cache — a coherent (uncompressed) model
// beats a quantized cache the kernels would feed garbage.

import Foundation
import Metal

/// Build `count` per-layer attention KV caches for `kind`.
///
/// - `preallocate` only affects the raw cache (the quantized caches
///   pre-allocate their compressed storage regardless). Callers that
///   stage writes into the buffer's free tail — diffusion block forwards
///   — pass `true`.
/// - Affine + AURA share one dequant working-buffer pair across all
///   `count` layers (the memory win vs per-layer scratch).
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
    let cap = contextLength

    func rawCaches() -> [any LayerCacheProtocol] {
        (0 ..< count).map { _ in
            KVCache(
                nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                dtype: dtype, eviction: eviction,
                preallocate: preallocate, device: device)
        }
    }

    switch kind {
    case .raw:
        return rawCaches()

    case .affineQuantized(let bits, let groupSize):
        guard headDim % groupSize == 0 else {
            warnCacheFallback(
                "affine\(bits) (group_size \(groupSize))",
                reason: "head_dim \(headDim) is not divisible by group_size \(groupSize)")
            return rawCaches()
        }
        let sharedK = Tensor.empty(
            shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
        let sharedV = Tensor.empty(
            shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
        return (0 ..< count).map { _ in
            AffineQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                dtype: dtype, bits: bits, groupSize: groupSize,
                sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                eviction: eviction, device: device)
        }

    case .auraQuantized(let scheme):
        // SRHT needs a power-of-2 head_dim. `headDim & (headDim - 1) == 0`
        // is the standard power-of-two test (false for 0).
        guard headDim > 0, headDim & (headDim - 1) == 0 else {
            warnCacheFallback(
                "AURA (\(scheme))",
                reason: "head_dim \(headDim) is not a power of two (SRHT requirement)")
            return rawCaches()
        }
        // Codebooks are shared across layers (dim-only Lloyd-Max levels);
        // each layer gets its own SRHT rotation seeded by its index.
        let kCodebook = auraConstantTensor(
            AURACodebook.centroids(dim: headDim, bits: scheme.keyBits), device)
        let kBoundaries = auraConstantTensor(
            AURACodebook.boundaries(dim: headDim, bits: scheme.keyBits), device)
        let vCodebook = auraConstantTensor(
            AURACodebook.centroids(dim: headDim, bits: scheme.valueBits), device)
        let vBoundaries = auraConstantTensor(
            AURACodebook.boundaries(dim: headDim, bits: scheme.valueBits), device)
        let sharedK = Tensor.empty(
            shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
        let sharedV = Tensor.empty(
            shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
        return (0 ..< count).map { i in
            let rot = AURAQuantizedKVCacheRotations.build(
                headDim: headDim, layerIndex: i,
                activationDtype: dtype, device: device)
            return AURAQuantizedKVCache(
                nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                dtype: dtype, scheme: scheme,
                rotation: rot.rotation, rotationT: rot.rotationT,
                rotationDtype: rot.rotationDtype, rotationDtypeT: rot.rotationDtypeT,
                kCodebook: kCodebook, kBoundaries: kBoundaries,
                vCodebook: vCodebook, vBoundaries: vBoundaries,
                sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                eviction: eviction, decodePath: auraDecodePath, device: device)
        }
    }
}

/// `head_dim` is a power of two — the geometry affine (any) + AURA (SRHT)
/// need. Exposed so families that build caches inline (hybrids) can run
/// the same guard before requesting a quantized scheme.
func attentionCacheGeometrySupports(
    kind: KVCacheKind, headDim: Int
) -> Bool {
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
        + "falling back to the raw fp16/bf16 cache for this model.\n"
    FileHandle.standardError.write(Data(line.utf8))
}
