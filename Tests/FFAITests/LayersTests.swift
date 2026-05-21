import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Layers")
struct LayersTests {
    @Test("Linear forward — matmul against gemv reference")
    func linearForward() {
        autoreleasepool {
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
    }

    @Test("Linear forward with bias — y = Wx + b")
    func linearForwardWithBias() {
        autoreleasepool {
            // W = [[1, 2], [3, 4], [5, 6]] @ [7, 8] = [23, 53, 83]
            // + bias [0.5, -1, 100] = [23.5, 52, 183]
            let w = Tensor.empty(shape: [3, 2], dtype: .f32)
            w.copyIn(from: [Float(1), 2, 3, 4, 5, 6])
            let b = Tensor.empty(shape: [3], dtype: .f32)
            b.copyIn(from: [Float(0.5), -1, 100])
            let layer = Linear(weight: w, bias: b)
            let x = Tensor.empty(shape: [3], dtype: .f32) // 3 elements; bias add reads matching shape
            // gemv produces [3]; we want the layer to broadcast bias [3] over the output [3].
            x.copyIn(from: [Float(7), 8, 0])  // last element wasted (gemv reads only 2)
            // Use a 2-element x to match weight in_features=2.
            let xReal = Tensor.empty(shape: [2], dtype: .f32)
            xReal.copyIn(from: [Float(7), 8])
            var out: Tensor!
            runAndWait { cb in out = layer(xReal, on: cb) }
            #expect(out.toArray(as: Float.self) == [Float(23.5), 52, 183])
            // parameters() must surface both weight + bias for SafeTensors binding.
            #expect(layer.parameters().map { $0.0 } == ["weight", "bias"])
        }
    }

    @Test("Embedding forward — gather rows")
    func embeddingForward() {
        autoreleasepool {
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
    }

    @Test("deriveAffineQuantBits — recovers per-tensor bit-width from packed shapes")
    func deriveAffineQuantBitsFromShapes() {
        // MLX affine quantization: a linear with `inFeatures` inputs
        // packs `weight` to `inFeatures * bits / 32` columns and
        // `scales` to `inFeatures / groupSize` columns.
        let groupSize = 64
        for (inFeatures, bits) in [(256, 4), (256, 8), (4096, 4), (4096, 8),
                                   (1024, 3), (1024, 6)] {
            let weightCols = inFeatures * bits / 32
            let scaleCols = inFeatures / groupSize
            let derived = deriveAffineQuantBits(
                weightPackedCols: weightCols, scaleCols: scaleCols,
                groupSize: groupSize)
            #expect(derived == bits,
                    "in=\(inFeatures) bits=\(bits): derived \(derived)")
        }
    }

    @Test("RMSNorm forward — y = x / rms(x) * weight")
    func rmsNormForward() {
        autoreleasepool {
            // Underlying Ops.rmsNorm kernel requires n % 128 == 0 (32-lane
            // simdgroup × 4 elements/thread). n=128 is the smallest legal
            // size — see Ops.rmsNorm preconditions / mlx/rms_norm.rs.
            let n = 128
            let xs: [Float] = (0..<n).map { Float($0 + 1) }
            let ws: [Float] = Array(repeating: Float(1), count: n)
            let w = Tensor.empty(shape: [n], dtype: .f32)
            w.copyIn(from: ws)
            let rms = RMSNorm(weight: w, eps: 1e-6)
            let x = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: xs)
            var out: Tensor!
            runAndWait { cb in out = rms(x, on: cb) }
            let r = out.toArray(as: Float.self)
            // CPU reference: rms = sqrt(mean(x²)); y = x / rms * weight.
            let ssq = xs.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n)).squareRoot()
            for i in 0..<n {
                let expected = xs[i] / expectedRms
                #expect(abs(r[i] - expected) < 1e-2,
                        "i=\(i) got \(r[i]) expected \(expected)")
            }
            #expect(rms.parameters().map { $0.0 } == ["weight"])
        }
    }
}
