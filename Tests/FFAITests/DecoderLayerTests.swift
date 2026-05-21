// DecoderLayer + StatelessLayerCache — the Phase 5e.A hybrid-stack
// scaffolding. Verifies a heterogeneous `[any DecoderLayer]` stack
// decodes in lockstep with its cache array, that `position` threads
// through, and that `StatelessLayerCache` is inert.

import Foundation
import Metal
import Testing
@testable import FFAI

// ─── Mock layers ─────────────────────────────────────────────────────

/// Returns the hidden state unchanged.
private final class IdentityDecoderLayer: DecoderLayer {
    func parameters() -> [(String, Tensor)] { [] }
    func decode(_ h: Tensor, position _: Int, cache _: any LayerCacheProtocol,
                cmd _: MTLCommandBuffer, device _: Device) -> Tensor { h }
}

/// Adds a fixed per-element bias vector to the hidden state.
private final class AddBiasDecoderLayer: DecoderLayer {
    let bias: Tensor
    init(bias: Tensor) { self.bias = bias }
    func parameters() -> [(String, Tensor)] { [("bias", bias)] }
    func decode(_ h: Tensor, position _: Int, cache _: any LayerCacheProtocol,
                cmd: MTLCommandBuffer, device _: Device) -> Tensor {
        Ops.add(h, bias, on: cmd)
    }
}

/// Adds `Float(position)` to every element — proves `position` threads
/// through the decode loop.
private final class PositionAddDecoderLayer: DecoderLayer {
    func parameters() -> [(String, Tensor)] { [] }
    func decode(_ h: Tensor, position: Int, cache _: any LayerCacheProtocol,
                cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let n = h.elementCount
        let posVec = Tensor.empty(shape: [n], dtype: .f32, device: device)
        posVec.copyIn(from: [Float](repeating: Float(position), count: n))
        return Ops.add(h, posVec, on: cmd)
    }
}

@Suite("DecoderLayer — hybrid-stack scaffolding")
struct DecoderLayerTests {

    @Test("heterogeneous [any DecoderLayer] stack decodes in lockstep")
    func heterogeneousDecodeStack() {
        autoreleasepool {
            let n = 4
            let input = Tensor.empty(shape: [n], dtype: .f32)
            input.copyIn(from: [Float(1), 2, 3, 4])

            let bias1 = Tensor.empty(shape: [n], dtype: .f32)
            bias1.copyIn(from: [Float](repeating: 1, count: n))
            let bias10 = Tensor.empty(shape: [n], dtype: .f32)
            bias10.copyIn(from: [Float](repeating: 10, count: n))

            // Heterogeneous stack: +1 → identity → +10  ⇒  net +11.
            let layers: [any DecoderLayer] = [
                AddBiasDecoderLayer(bias: bias1),
                IdentityDecoderLayer(),
                AddBiasDecoderLayer(bias: bias10),
            ]
            let caches: [any LayerCacheProtocol] = layers.map { _ in StatelessLayerCache() }

            var h = input
            runAndWait { cb in
                for (i, layer) in layers.enumerated() {
                    h = layer.decode(h, position: 0, cache: caches[i],
                                     cmd: cb, device: .shared)
                }
            }
            #expect(h.toArray(as: Float.self) == [12, 13, 14, 15])
        }
    }

    @Test("position threads through the decode loop")
    func positionThreadsThrough() {
        autoreleasepool {
            let n = 3
            let input = Tensor.empty(shape: [n], dtype: .f32)
            input.copyIn(from: [Float(0), 0, 0])

            let layer = PositionAddDecoderLayer()
            let cache = StatelessLayerCache()

            var h = input
            runAndWait { cb in
                h = layer.decode(h, position: 7, cache: cache,
                                 cmd: cb, device: .shared)
            }
            #expect(h.toArray(as: Float.self) == [7, 7, 7])
        }
    }

    @Test("StatelessLayerCache is inert")
    func statelessLayerCacheIsInert() {
        let cache = StatelessLayerCache()
        #expect(cache.length == 0)
        #expect(cache.maxSeq == Int.max)
        #expect(cache.bytesAllocated == 0)
        #expect(cache.bytesInUse == 0)
        cache.reset()   // no-op, must not crash
        #expect(cache.length == 0)
    }
}
