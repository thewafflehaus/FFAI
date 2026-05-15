import Testing
@testable import FFAI

@Suite("KVCache")
struct KVCacheTests {
    @Test("init creates zeroed K/V buffers of the right shape")
    func initShape() {
        let c = KVCache(nKVHeads: 2, headDim: 4, maxSeq: 8, dtype: .f32)
        #expect(c.kBuffer.shape == [2, 8, 4])
        #expect(c.vBuffer.shape == [2, 8, 4])
        #expect(c.length == 0)
        // Both buffers initialize to zero.
        let zeros = [Float](repeating: 0, count: 64)
        #expect(c.kBuffer.toArray(as: Float.self) == zeros)
        #expect(c.vBuffer.toArray(as: Float.self) == zeros)
    }

    @Test("append writes per-head slabs at current position")
    func appendOnePosition() {
        let c = KVCache(nKVHeads: 2, headDim: 4, maxSeq: 8, dtype: .f32)
        let kFlat = Tensor.empty(shape: [2, 4], dtype: .f32)
        let vFlat = Tensor.empty(shape: [2, 4], dtype: .f32)
        kFlat.copyIn(from: [Float(1), 2, 3, 4, 5, 6, 7, 8])     // head0=[1..4], head1=[5..8]
        vFlat.copyIn(from: [Float(11), 12, 13, 14, 15, 16, 17, 18])

        c.append(kFlat: kFlat, vFlat: vFlat)
        #expect(c.length == 1)

        // Layout: [n_kv_heads=2, max_seq=8, head_dim=4]
        // head0 position 0 should be [1,2,3,4], head1 position 0 should be [5,6,7,8]
        let kAll = c.kBuffer.toArray(as: Float.self)
        #expect(kAll[0..<4] == [1, 2, 3, 4][0..<4])
        // head1 starts at index 8 * 4 = 32
        #expect(kAll[32..<36] == [5, 6, 7, 8][0..<4])
        let vAll = c.vBuffer.toArray(as: Float.self)
        #expect(vAll[0..<4] == [11, 12, 13, 14][0..<4])
        #expect(vAll[32..<36] == [15, 16, 17, 18][0..<4])
    }

    @Test("multiple appends advance length and use the right offsets")
    func appendMultiple() {
        let c = KVCache(nKVHeads: 1, headDim: 2, maxSeq: 4, dtype: .f32)
        for p in 0..<3 {
            let kFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
            let vFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
            kFlat.copyIn(from: [Float(p * 10), Float(p * 10 + 1)])
            vFlat.copyIn(from: [Float(p * 100), Float(p * 100 + 1)])
            c.append(kFlat: kFlat, vFlat: vFlat)
        }
        #expect(c.length == 3)
        let k = c.kBuffer.toArray(as: Float.self)
        // Layout [1, 4, 2]: pos0=[0,1], pos1=[10,11], pos2=[20,21], pos3=zero (unused)
        #expect(k[0..<6] == [0, 1, 10, 11, 20, 21][0..<6])
        #expect(k[6..<8] == [0, 0][0..<2])
    }

    @Test("reset zeros length but keeps allocation")
    func reset() {
        let c = KVCache(nKVHeads: 1, headDim: 2, maxSeq: 4, dtype: .f32)
        let kFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
        let vFlat = Tensor.empty(shape: [1, 2], dtype: .f32)
        kFlat.copyIn(from: [Float(1), 2])
        vFlat.copyIn(from: [Float(3), 4])
        c.append(kFlat: kFlat, vFlat: vFlat)
        #expect(c.length == 1)
        c.reset()
        #expect(c.length == 0)
    }
}
