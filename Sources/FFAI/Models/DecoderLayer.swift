// DecoderLayer — the per-layer mixer abstraction for hybrid models.
//
// A homogeneous model (Llama, Qwen3, Mamba 2) has one concrete layer
// type repeated `nLayers` times, so it can hold a typed array and call
// the layer's `forward` directly. A hybrid model — Qwen 3.5 (GDN ↔
// attention), NemotronH / Jamba / GraniteMoeHybrid / FalconH1 (Mamba 2 +
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
