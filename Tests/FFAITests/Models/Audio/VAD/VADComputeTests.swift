// VADComputeTests — unit tests for the shared CPU compute primitives
// in VADCompute.swift (used by SileroVAD, SmartTurn, and Sortformer).
//
// These primitives are plain `[Float]` kernels — no GPU, no Tensor — so
// every test runs in microseconds and is offline-safe. The CPU pieces
// covered here form the load-bearing foundation of every VAD forward
// pass, so they get more aggressive numeric coverage than the family
// files above them.
//
// Covers:
//   * VADMath — sigmoid (positive / negative / zero), GELU symmetry,
//     ReLU in place, tanh shape, softmax over a range.
//   * VADLinear — single-row and multi-row identity + numeric apply.
//   * VADConv1d — outputLength formula, identity passthrough.
//   * VADLayerNorm — zero-mean unit-variance output on a constant row.
//   * VADLSTM — output shape over a sequence, zero-input identity-ish.
//   * VADAudioFrontend — Hann window endpoints, mel filterbank shape.
//   * vadMultiHeadAttention — output shape + V-projection invariance
//     under uniform Q/K.

import Foundation
import Testing
@testable import FFAI

@Suite("VADCompute")
struct VADComputeTests {

    // ─── VADMath ─────────────────────────────────────────────────────────

    @Test("VADMath.sigmoid — sigmoid(0) is 0.5")
    func sigmoidZero() {
        #expect(abs(VADMath.sigmoid(0) - 0.5) < 1e-6)
    }

    @Test("VADMath.sigmoid — large positive saturates near 1")
    func sigmoidLargePositive() {
        #expect(VADMath.sigmoid(20) > 0.999)
    }

    @Test("VADMath.sigmoid — large negative saturates near 0")
    func sigmoidLargeNegative() {
        #expect(VADMath.sigmoid(-20) < 0.001)
    }

    @Test("VADMath.sigmoid — vector form maps elementwise")
    func sigmoidVector() {
        let out = VADMath.sigmoid([0, 1, -1])
        #expect(abs(out[0] - 0.5) < 1e-6)
        #expect(out[1] > 0.7 && out[1] < 0.8)
        #expect(out[2] > 0.2 && out[2] < 0.3)
    }

    @Test("VADMath.reluInPlace — clips negatives to zero, leaves positives unchanged")
    func reluInPlace() {
        var xs: [Float] = [-1, 0, 1, -2, 3]
        VADMath.reluInPlace(&xs)
        #expect(xs == [0, 0, 1, 0, 3])
    }

    @Test("VADMath.gelu — gelu(0) is 0")
    func geluZero() {
        #expect(abs(VADMath.gelu(0)) < 1e-6)
    }

    @Test("VADMath.gelu — for large positive x, gelu(x) ≈ x")
    func geluLargePositive() {
        let v = VADMath.gelu(5)
        #expect(abs(v - 5) < 0.001)
    }

    @Test("VADMath.gelu — for large negative x, gelu(x) ≈ 0")
    func geluLargeNegative() {
        #expect(abs(VADMath.gelu(-5)) < 0.001)
    }

    @Test("VADMath.tanhActivation — tanh(0) is 0, tanh(big) is ±1")
    func tanhActivation() {
        let out = VADMath.tanhActivation([0, 10, -10])
        #expect(abs(out[0]) < 1e-6)
        #expect(abs(out[1] - 1) < 1e-3)
        #expect(abs(out[2] + 1) < 1e-3)
    }

    @Test("VADMath.softmaxInPlace — uniform input produces uniform distribution")
    func softmaxUniform() {
        var xs: [Float] = [1, 1, 1, 1]
        VADMath.softmaxInPlace(&xs, range: 0..<4)
        for v in xs { #expect(abs(v - 0.25) < 1e-6) }
    }

    @Test("VADMath.softmaxInPlace — output sums to 1 over the range")
    func softmaxSumsToOne() {
        var xs: [Float] = [0, 1, 2, 3]
        VADMath.softmaxInPlace(&xs, range: 0..<4)
        let s = xs.reduce(0, +)
        #expect(abs(s - 1.0) < 1e-5)
    }

    // ─── VADLinear ───────────────────────────────────────────────────────

    @Test("VADLinear.apply — identity weight + zero bias passes the input through")
    func linearIdentity() {
        // Identity 3×3: weight[o, i] = 1 if o == i else 0.
        var w = [Float](repeating: 0, count: 9)
        for i in 0..<3 { w[i * 3 + i] = 1 }
        let lin = VADLinear(weight: w, bias: nil, inFeatures: 3, outFeatures: 3)
        let x: [Float] = [1, 2, 3]
        #expect(lin.apply(x) == x)
    }

    @Test("VADLinear.apply — bias is added to the dot product")
    func linearBias() {
        let w = [Float](repeating: 0, count: 4)  // [out=2, in=2]
        let bias: [Float] = [1.5, -2.5]
        let lin = VADLinear(weight: w, bias: bias, inFeatures: 2, outFeatures: 2)
        let y = lin.apply([7, 9])
        #expect(y[0] == 1.5)
        #expect(y[1] == -2.5)
    }

    @Test("VADLinear.applyRows — multi-row apply matches row-wise apply")
    func linearApplyRows() {
        // Simple 2×2 linear: weight = [[1, 2], [3, 4]], no bias.
        let w: [Float] = [1, 2, 3, 4]
        let lin = VADLinear(weight: w, bias: nil, inFeatures: 2, outFeatures: 2)
        let x: [Float] = [1, 0, 0, 1]  // two rows: [1,0], [0,1]
        let out = lin.applyRows(x, rows: 2)
        // Row 0: [1, 3]; Row 1: [2, 4].
        #expect(out == [1, 3, 2, 4])
    }

    // ─── VADConv1d ───────────────────────────────────────────────────────

    @Test("VADConv1d.outputLength — K=3, stride=1, pad=1 keeps length")
    func conv1dOutputLengthSame() {
        let w = [Float](repeating: 0, count: 1 * 1 * 3)
        let c = VADConv1d(weight: w, bias: nil, inChannels: 1, outChannels: 1,
                          kernelSize: 3, stride: 1, padding: 1)
        #expect(c.outputLength(forInputLength: 10) == 10)
    }

    @Test("VADConv1d.outputLength — K=3, stride=2, pad=1 halves length")
    func conv1dOutputLengthStride2() {
        let w = [Float](repeating: 0, count: 1 * 1 * 3)
        let c = VADConv1d(weight: w, bias: nil, inChannels: 1, outChannels: 1,
                          kernelSize: 3, stride: 2, padding: 1)
        // (10 + 2*1 - 3) / 2 + 1 = 9 / 2 + 1 = 4 + 1 = 5.
        #expect(c.outputLength(forInputLength: 10) == 5)
    }

    @Test("VADConv1d.apply — single 1-tap identity passes the input through")
    func conv1dIdentity() {
        // 1 in, 1 out, kernel=1, weight=1 → output = input.
        let w: [Float] = [1]
        let c = VADConv1d(weight: w, bias: nil, inChannels: 1, outChannels: 1,
                          kernelSize: 1, stride: 1, padding: 0)
        let (vals, len) = c.apply([1, 2, 3, 4], inLength: 4)
        #expect(len == 4)
        #expect(vals == [1, 2, 3, 4])
    }

    // ─── VADLayerNorm ────────────────────────────────────────────────────

    @Test("VADLayerNorm.apply — constant input row maps to bias (zero variance)")
    func layerNormConstantInput() {
        let dim = 4
        let weight = [Float](repeating: 1, count: dim)
        let bias = [Float](repeating: 0, count: dim)
        let ln = VADLayerNorm(weight: weight, bias: bias, dim: dim)
        // A constant input row has zero variance; output is all-zero (the bias).
        let y = ln.apply([3, 3, 3, 3])
        for v in y { #expect(abs(v) < 1e-3) }
    }

    @Test("VADLayerNorm.applyRows — output has matching shape")
    func layerNormShape() {
        let dim = 4
        let ln = VADLayerNorm(weight: [Float](repeating: 1, count: dim),
                              bias: [Float](repeating: 0, count: dim),
                              dim: dim)
        let x = [Float](repeating: 1, count: 3 * dim)
        let y = ln.applyRows(x, rows: 3)
        #expect(y.count == 3 * dim)
    }

    // ─── VADLSTM ─────────────────────────────────────────────────────────

    @Test("VADLSTM.run — output sequence has shape [seqLen, hiddenSize]")
    func lstmRunShape() {
        let inSize = 3
        let hidden = 4
        let wih = [Float](repeating: 0, count: 4 * hidden * inSize)
        let whh = [Float](repeating: 0, count: 4 * hidden * hidden)
        let lstm = VADLSTM(weightIH: wih, weightHH: whh, biasIH: nil,
                           biasHH: nil, inputSize: inSize, hiddenSize: hidden)
        let x = [Float](repeating: 0.1, count: 5 * inSize)
        let (seq, h, c) = lstm.run(x, seqLen: 5)
        #expect(seq.count == 5 * hidden)
        #expect(h.count == hidden)
        #expect(c.count == hidden)
    }

    @Test("VADLSTM.run — zero weights + zero input produce zero output")
    func lstmZeroPath() {
        let inSize = 3
        let hidden = 4
        let wih = [Float](repeating: 0, count: 4 * hidden * inSize)
        let whh = [Float](repeating: 0, count: 4 * hidden * hidden)
        let lstm = VADLSTM(weightIH: wih, weightHH: whh, biasIH: nil,
                           biasHH: nil, inputSize: inSize, hiddenSize: hidden)
        let x = [Float](repeating: 0, count: 3 * inSize)
        let (seq, h, c) = lstm.run(x, seqLen: 3)
        for v in seq { #expect(abs(v) < 1e-6) }
        for v in h { #expect(abs(v) < 1e-6) }
        for v in c { #expect(abs(v) < 1e-6) }
    }

    // ─── VADAudioFrontend ───────────────────────────────────────────────

    @Test("VADAudioFrontend.hannWindow — endpoints are 0, midpoint is 1")
    func hannWindow() {
        let w = VADAudioFrontend.hannWindow(size: 256)
        #expect(w.count == 256)
        #expect(abs(w[0]) < 1e-6)
        // Period Hann (size as divisor) — w[size] would be 0; w[128] is 1.
        #expect(abs(w[128] - 1.0) < 1e-6)
    }

    @Test("VADAudioFrontend.hannWindow — size = 0 yields empty window")
    func hannWindowZeroSize() {
        let w = VADAudioFrontend.hannWindow(size: 0)
        #expect(w.isEmpty)
    }

    @Test("VADAudioFrontend.melFilterbank — has the right shape and is non-negative")
    func melFilterbankShape() {
        let nFft = 400
        let nMels = 80
        let fb = VADAudioFrontend.melFilterbank(sampleRate: 16_000,
                                                nFft: nFft, nMels: nMels)
        // Shape: [nBins, nMels] where nBins = nFft/2 + 1.
        let nBins = nFft / 2 + 1
        #expect(fb.count == nBins * nMels)
        // Every filterbank weight is non-negative.
        for v in fb { #expect(v >= 0) }
    }

    @Test("VADAudioFrontend.powerSpectrogram — output rows match nFft/2 + 1 bins")
    func powerSpectrogramShape() {
        let nFft = 64
        let hopLength = 16
        let window = VADAudioFrontend.hannWindow(size: nFft)
        let audio = (0..<1024).map { Float(sin(Double($0) * 0.1)) }
        let (spec, nFrames, nBins) =
            VADAudioFrontend.powerSpectrogram(audio, window: window,
                                              nFft: nFft, hopLength: hopLength)
        #expect(nBins == nFft / 2 + 1)
        #expect(nFrames > 0)
        #expect(spec.count == nFrames * nBins)
        // Power is non-negative.
        for v in spec { #expect(v >= 0) }
    }

    @Test("VADAudioFrontend.applyMelFilterbank — shape match")
    func applyMelFilterbank() {
        let nFft = 64
        let nBins = nFft / 2 + 1
        let nMels = 8
        let nFrames = 5
        let power = [Float](repeating: 0.5, count: nFrames * nBins)
        let fb = VADAudioFrontend.melFilterbank(sampleRate: 16_000,
                                                nFft: nFft, nMels: nMels)
        let mel = VADAudioFrontend.applyMelFilterbank(
            power: power, numFrames: nFrames, nBins: nBins,
            filterbank: fb, nMels: nMels)
        #expect(mel.count == nFrames * nMels)
    }

    // ─── vadMultiHeadAttention ──────────────────────────────────────────

    @Test("vadMultiHeadAttention — output shape matches input")
    func mhaOutputShape() {
        let seqLen = 4
        let numHeads = 2
        let headDim = 3
        let dim = numHeads * headDim
        let q = [Float](repeating: 1, count: seqLen * dim)
        let k = [Float](repeating: 1, count: seqLen * dim)
        let v = [Float](repeating: 0.5, count: seqLen * dim)
        let out = vadMultiHeadAttention(
            q: q, k: k, v: v, seqLen: seqLen,
            numHeads: numHeads, headDim: headDim,
            scale: Float(headDim).squareRoot())
        #expect(out.count == seqLen * dim)
    }

    @Test("vadMultiHeadAttention — uniform Q/K yields V-average per head")
    func mhaUniformAttention() {
        // With uniform Q and K, every score is equal → softmax yields a
        // uniform attention distribution → output = mean(V) per head/dim.
        let seqLen = 4
        let numHeads = 1
        let headDim = 2
        let q = [Float](repeating: 1, count: seqLen * headDim)
        let k = [Float](repeating: 1, count: seqLen * headDim)
        // V varies per row so we can check the average.
        let v: [Float] = [
            0, 1,
            2, 3,
            4, 5,
            6, 7,
        ]
        let out = vadMultiHeadAttention(
            q: q, k: k, v: v, seqLen: seqLen,
            numHeads: numHeads, headDim: headDim,
            scale: Float(headDim).squareRoot())
        // Expected: each row equals the mean of V rows: [(0+2+4+6)/4, (1+3+5+7)/4] = [3, 4].
        for r in 0..<seqLen {
            #expect(abs(out[r * headDim + 0] - 3) < 1e-3)
            #expect(abs(out[r * headDim + 1] - 4) < 1e-3)
        }
    }
}
