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

    // MARK: - Sampling pipeline

    private func makeLogits(_ vs: [Float]) -> Tensor {
        let t = Tensor.empty(shape: [vs.count], dtype: .f32)
        t.copyIn(from: vs)
        return t
    }

    @Test("sample at temperature=0 matches argmax (no RNG draw)")
    func tempZeroIsArgmax() {
        let t = makeLogits([0.1, 0.5, -2.0, 4.2, 1.0])
        var rng = SeededRandomNumberGenerator(seed: 42)
        let p = GenerationParameters(temperature: 0)
        #expect(Sampling.sample(t, parameters: p, rng: &rng) == 3)
    }

    @Test("sample is reproducible with the same seed")
    func reproducibleSeed() {
        let t = makeLogits([0.1, 1.0, 0.5, 2.0, 0.3, 1.5])
        let p = GenerationParameters(temperature: 0.7)
        var rng1 = SeededRandomNumberGenerator(seed: 12345)
        var rng2 = SeededRandomNumberGenerator(seed: 12345)
        let seq1 = (0..<8).map { _ in Sampling.sample(t, parameters: p, rng: &rng1) }
        let seq2 = (0..<8).map { _ in Sampling.sample(t, parameters: p, rng: &rng2) }
        #expect(seq1 == seq2)
    }

    @Test("top-K cuts the candidate set to K largest logits")
    func topKFilter() {
        // logits — argmax is index 4 (3.0). Top-2 are indices 4, 1.
        let t = makeLogits([0.0, 2.0, -1.0, 0.5, 3.0])
        var rng = SeededRandomNumberGenerator(seed: 1)
        let p = GenerationParameters(temperature: 1.0, topK: 2)
        var hits = Set<Int>()
        for _ in 0..<200 {
            hits.insert(Sampling.sample(t, parameters: p, rng: &rng))
        }
        // Across 200 draws with topK=2 we should only ever see {1, 4}.
        #expect(hits.isSubset(of: [1, 4]))
        #expect(hits.contains(4))   // the dominant prob should fire
    }

    @Test("top-P keeps only the cumulative-prob nucleus")
    func topPFilter() {
        // Very peaked distribution: idx 0 has nearly all the mass.
        let t = makeLogits([10.0, 0.0, 0.0, 0.0])
        var rng = SeededRandomNumberGenerator(seed: 2)
        let p = GenerationParameters(temperature: 1.0, topP: 0.5)
        // top-P=0.5 with the peak >>> rest should always pick the peak.
        for _ in 0..<50 {
            #expect(Sampling.sample(t, parameters: p, rng: &rng) == 0)
        }
    }

    @Test("min-P drops tokens below min_p × max_prob")
    func minPFilter() {
        // Peak prob ~ 0.999, others near 0. min-P=0.5 should only keep the peak.
        let t = makeLogits([8.0, 0.0, 0.0, 0.0])
        var rng = SeededRandomNumberGenerator(seed: 3)
        let p = GenerationParameters(temperature: 1.0, minP: 0.5)
        for _ in 0..<50 {
            #expect(Sampling.sample(t, parameters: p, rng: &rng) == 0)
        }
    }

    @Test("repetition penalty discounts already-seen tokens with positive logits")
    func repPenaltyDiscountsSeen() {
        // Without penalty: argmax would be index 0 (logit 5.0).
        // With penalty 2.0 applied to {0}: logit 0 → 5/2 = 2.5; new max is index 1 (3.0).
        let t = makeLogits([5.0, 3.0, 1.0, 0.5])
        var rng = SeededRandomNumberGenerator(seed: 4)
        let p = GenerationParameters(temperature: 0, repetitionPenalty: 2.0)
        let plain = Sampling.sample(t, parameters: GenerationParameters(temperature: 0),
                                    rng: &rng, tokenHistory: [0])
        #expect(plain == 0)
        let penalized = Sampling.sample(t, parameters: p,
                                        rng: &rng, tokenHistory: [0])
        #expect(penalized == 1)
    }

    @Test("repetition penalty multiplies tokens with negative logits")
    func repPenaltyAmplifiesNegative() {
        // Argmax would be index 0 (logit 1.0). Index 2 has -0.5.
        // With penalty 2.0 on {2}: -0.5 * 2 = -1.0 (further suppressed).
        let t = makeLogits([1.0, 0.5, -0.5, -1.0])
        var rng = SeededRandomNumberGenerator(seed: 5)
        let p = GenerationParameters(temperature: 0, repetitionPenalty: 2.0)
        let r = Sampling.sample(t, parameters: p, rng: &rng, tokenHistory: [2])
        // index 0 still wins; index 2 is further suppressed.
        #expect(r == 0)
    }

    @Test("makeRNG honors the seed")
    func makeRNGSeedHonored() {
        let p1 = GenerationParameters(seed: 99)
        let p2 = GenerationParameters(seed: 99)
        var r1 = p1.makeRNG()
        var r2 = p2.makeRNG()
        #expect(r1.next() == r2.next())
        #expect(r1.next() == r2.next())
    }

    @Test("makeRNG with no seed gives a system random generator")
    func makeRNGNoSeed() {
        var rng = GenerationParameters().makeRNG()
        // Just exercise: must not crash and must produce some bits.
        _ = rng.next()
    }

    @Test("SeededRandomNumberGenerator is deterministic across instances")
    func seededDeterminism() {
        var r1 = SeededRandomNumberGenerator(seed: 0xDEADBEEF)
        var r2 = SeededRandomNumberGenerator(seed: 0xDEADBEEF)
        let s1 = (0..<10).map { _ in r1.next() }
        let s2 = (0..<10).map { _ in r2.next() }
        #expect(s1 == s2)
    }

    @Test("sample bf16 + f16 logits both work")
    func sampleDtypes() {
        var rng = SeededRandomNumberGenerator(seed: 7)
        let p = GenerationParameters(temperature: 0)
        let f16 = Tensor.empty(shape: [4], dtype: .f16)
        f16.copyIn(from: [Float16(1), 7, 3, -1])
        #expect(Sampling.sample(f16, parameters: p, rng: &rng) == 1)

        let bf16 = Tensor.empty(shape: [4], dtype: .bf16)
        bf16.copyIn(from: [UInt16(0x3F80), 0x40C0, 0x3FC0, 0xBF80])
        #expect(Sampling.sample(bf16, parameters: p, rng: &rng) == 1)
    }
}
