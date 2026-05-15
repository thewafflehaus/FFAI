import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Layers")
struct LayersTests {
    func runAndWait(_ block: (MTLCommandBuffer) -> Void) {
        let cb = Device.shared.makeCommandBuffer()
        block(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    @Test("Linear forward — matmul against gemv reference")
    func linearForward() {
        let w = Tensor.empty(shape: [3, 2], dtype: .f32)
        w.copyIn(from: [Float(1), 2, 3, 4, 5, 6])
        let layer = Linear(weight: w)
        let x = Tensor.empty(shape: [2], dtype: .f32)
        x.copyIn(from: [Float(7), 8])
        var out: Tensor!
        runAndWait { cb in out = layer(x, on: cb) }
        #expect(out.toArray(as: Float.self) == [23, 53, 83])
        let params = layer.parameters().map { $0.0 }
        #expect(params == ["weight"])
    }

    @Test("Embedding forward — gather rows")
    func embeddingForward() {
        let w = Tensor.empty(shape: [3, 2], dtype: .f32)
        w.copyIn(from: [Float(10), 11, 20, 21, 30, 31])
        let embed = Embedding(weight: w)
        let ids = Tensor.empty(shape: [2], dtype: .u32)
        ids.copyIn(from: [UInt32(2), 0])
        var out: Tensor!
        runAndWait { cb in out = embed(ids, on: cb) }
        #expect(out.toArray(as: Float.self) == [30, 31, 10, 11])
        #expect(embed.parameters().map { $0.0 } == ["weight"])
    }

    @Test("RMSNorm forward — y = x / rms(x) * weight")
    func rmsNormForward() {
        let w = Tensor.empty(shape: [4], dtype: .f32)
        w.copyIn(from: [Float(1), 1, 1, 1])
        let rms = RMSNorm(weight: w, eps: 1e-6)
        let x = Tensor.empty(shape: [4], dtype: .f32)
        x.copyIn(from: [Float(1), 2, 3, 4])
        var out: Tensor!
        runAndWait { cb in out = rms(x, on: cb) }
        let r = out.toArray(as: Float.self)
        let expectedRms = Float((30.0 / 4.0).squareRoot())
        for i in 0..<4 {
            let expected = Float(i + 1) / expectedRms
            #expect(abs(r[i] - expected) < 1e-3)
        }
        #expect(rms.parameters().map { $0.0 } == ["weight"])
    }
}
