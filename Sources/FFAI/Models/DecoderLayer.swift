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
// DecoderLayer — the per-layer mixer abstraction for hybrid models.
//
// A homogeneous model (Llama, Qwen3, Mamba 2) has one concrete layer
// type repeated `nLayers` times, so it can hold a typed array and call
// the layer's `forward` directly. A hybrid model — Qwen 3.5 (GDN ↔
// attention), NemotronH / Jamba / Granite4 / FalconH1 (Mamba 2 +
// attention + MoE/MLP) — interleaves several layer kinds in one stack.
//
// `DecoderLayer` lets a hybrid model hold a heterogeneous
// `[any DecoderLayer]` and drive a uniform decode loop:
//
//   for (i, layer) in layers.enumerated() {
//       h = layer.decode(h, position: pos, cache: caches[i],
//                        cmd: cmd, device: device)
//   }
//
// `caches` is the `[any LayerCacheProtocol]` from `makeLayerCaches`,
// indexed in lockstep with `layers`. Each layer downcasts its cache
// slot to the concrete type it needs (`KVCacheProtocol`,
// `Mamba2LayerCache`, `GDNStateCache`, …) — the model is responsible
// for `makeLayerCaches` having produced a matching cache per index.
// A layer with no recurrent / attention state (a pure MLP or MoE
// feed-forward block) gets a `StatelessLayerCache` slot and ignores it.

import Foundation
import Metal

/// One layer of a (possibly hybrid) decoder stack.
public protocol DecoderLayer: Module {
    /// Layer-local single-token decode. All work is queued onto `cmd`;
    /// the method does not commit. Returns the post-layer hidden state.
    ///
    /// - `position`: absolute sequence position. Attention layers use
    ///   it for RoPE; recurrent layers (Mamba 2, GDN) ignore it — their
    ///   cache already tracks recurrent state.
    /// - `cache`: this layer's slot from `makeLayerCaches`. The layer
    ///   downcasts to its concrete cache type. Pure feed-forward layers
    ///   receive a `StatelessLayerCache` and ignore it.
    func decode(_ h: Tensor, position: Int,
                cache: any LayerCacheProtocol,
                cmd: MTLCommandBuffer, device: Device) -> Tensor

    /// Layer-local **multi-token** decode — process `nRows` consecutive
    /// positions in one logical call. `h` is `[nRows, hidden]`; the
    /// returned tensor is `[nRows, hidden]`. The starting absolute
    /// position is `position` (rows take `position + 0 ..< position +
    /// nRows`).
    ///
    /// The default implementation loops `decode(...)` per row on the
    /// supplied `cmd` (correct-but-slow; commit-count win only).
    /// Attention-style layers override this to collapse the N SDPA
    /// dispatches into one `Ops.sdpaMulti(causal: true)` + batched
    /// `Ops.gemm` projections (the TTFT win). Recurrent
    /// layers (Mamba 2 selective scan, GDN delta) keep the default
    /// because their recurrence is inherently sequential — they can't
    /// batch state updates without giving up correctness.
    func decodeMulti(_ h: Tensor, startingAt position: Int,
                     cache: any LayerCacheProtocol,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor
}

public extension DecoderLayer {
    /// Default `decodeMulti`: loop `decode(...)` per row on the same
    /// `cmd`. Correct + commit-count-batched, but every dispatch
    /// inside `decode(...)` still runs per-token. Attention layers
    /// override with the chunked path; recurrent layers (Mamba 2 /
    /// GDN) inherit this default because their step is intrinsically
    /// sequential.
    ///
    /// Result is allocated up front and zero-initialised host-side
    /// (`Tensor.zero()` is a synchronous `memset`, safe because no
    /// GPU work has touched the fresh buffer yet). Each row's
    /// `decode(...)` result is written into the matching `result`
    /// slice via `Ops.add(...into: dst)` — the only "row-copy onto
    /// the same `cmd`" primitive available without a dedicated
    /// `Ops.copy` kernel.
    func decodeMulti(_ h: Tensor, startingAt position: Int,
                     cache: any LayerCacheProtocol,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let nRows = h.shape[0]
        let hidden = h.shape.last!
        let result = Tensor.empty(shape: h.shape, dtype: h.dtype, device: device)
        result.zero()
        for i in 0..<nRows {
            let row = h.slicedRows(start: i, count: 1).reshaped(to: [hidden])
            let out = decode(row, position: position + i,
                             cache: cache, cmd: cmd, device: device)
            let dst = result.slicedRows(start: i, count: 1).reshaped(to: [hidden])
            _ = Ops.add(out, dst, on: cmd, into: dst)
        }
        return result
    }
}

/// A no-op `LayerCacheProtocol` for layers that hold no per-token state
/// (pure MLP / MoE feed-forward blocks in a hybrid stack). Lets
/// `makeLayerCaches` return one cache per layer index — dense and 1:1
/// with `layers` — without making the cache slot optional.
public final class StatelessLayerCache: LayerCacheProtocol {
    public init() {}

    /// Stateless: never consumes a timestep.
    public var length: Int { 0 }
    /// Not length-bound — mirrors the SSM caches' `.max` convention.
    public var maxSeq: Int { .max }
    public var bytesAllocated: Int { 0 }
    public var bytesInUse: Int { 0 }

    public func reset() {}
}
