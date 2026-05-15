import Foundation
import Testing
@testable import FFAI

@Suite("Sampling")
struct SamplingTests {
    @Test("argmax over f32 picks the largest")
    func argmaxF32() {
        let t = Tensor.empty(shape: [5], dtype: .f32)
        t.copyIn(from: [Float(0.1), 0.5, -2.0, 4.2, 1.0])
        #expect(Sampling.argmax(t) == 3)
    }

    @Test("argmax over f16 picks the largest")
    func argmaxF16() {
        let t = Tensor.empty(shape: [4], dtype: .f16)
        t.copyIn(from: [Float16(1), 7, 3, -1])
        #expect(Sampling.argmax(t) == 1)
    }

    @Test("argmax over bf16 picks the largest")
    func argmaxBf16() {
        let t = Tensor.empty(shape: [4], dtype: .bf16)
        // bf16 = top 16 bits of f32; build by shifting our floats down.
        let bits: [UInt16] = [0x3F80, 0x40C0, 0x3FC0, 0xBF80] // 1.0, 6.0, 1.5, -1.0
        t.copyIn(from: bits)
        #expect(Sampling.argmax(t) == 1)
    }

    @Test("topN returns N entries sorted descending")
    func topN() {
        let t = Tensor.empty(shape: [6], dtype: .f32)
        t.copyIn(from: [Float(1), 5, 2, 9, 0, 7])
        let top3 = Sampling.topN(t, n: 3)
        #expect(top3.count == 3)
        #expect(top3[0].0 == 3)   // value 9
        #expect(top3[1].0 == 5)   // value 7
        #expect(top3[2].0 == 1)   // value 5
        #expect(top3[0].1 == 9)
    }

    @Test("topN works for f16 + bf16 too")
    func topNDtypes() {
        let f16 = Tensor.empty(shape: [3], dtype: .f16)
        f16.copyIn(from: [Float16(1), 5, 2])
        #expect(Sampling.topN(f16, n: 1)[0].0 == 1)

        let bf16 = Tensor.empty(shape: [3], dtype: .bf16)
        bf16.copyIn(from: [UInt16(0x3F80), 0x40A0, 0x4000]) // 1, 5, 2
        #expect(Sampling.topN(bf16, n: 1)[0].0 == 1)
    }
}
