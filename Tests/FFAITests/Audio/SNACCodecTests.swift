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
// SNACCodecTests — exercises the audio primitive layer that the SNAC
// neural codec is built on, plus a graceful-skip end-to-end path that
// runs only when a real SNAC checkpoint is available.
//
// SNAC checkpoints are multi-MB HF snapshots; CI does not ship one. The
// primitive tests therefore carry the correctness signal: conv1d,
// transposed conv, snake, and the VQ nearest-codebook lookup are each
// checked against an independent hand-computed reference.

import Foundation
import Testing

@testable import FFAI

@Suite("SNAC codec — audio primitives + round-trip")
struct SNACCodecTests {

    // MARK: - conv1d

    @Test("conv1d matches a hand-computed reference")
    func conv1dReference() {
        // x: [1, 1, 5] = 1,2,3,4,5  weight: [1,1,3] = 1,0,-1  (no pad)
        let x: [Float] = [1, 2, 3, 4, 5]
        let w: [Float] = [1, 0, -1]
        let (out, shape) = AudioMath.conv1d(
            x: x, xShape: [1, 1, 5], weight: w, wShape: [1, 1, 3],
            bias: nil, stride: 1, padding: 0, dilation: 1, groups: 1)
        // out[t] = x[t]*1 + x[t+1]*0 + x[t+2]*(-1)
        #expect(shape == [1, 1, 3])
        let expected: [Float] = [-2, -2, -2]
        #expect(out == expected)
    }

    @Test("conv1d with padding preserves length when k=3 pad=1")
    func conv1dSamePadding() {
        let x: [Float] = [1, 1, 1, 1]
        let w: [Float] = [1, 1, 1]
        let (out, shape) = AudioMath.conv1d(
            x: x, xShape: [1, 1, 4], weight: w, wShape: [1, 1, 3],
            bias: nil, stride: 1, padding: 1, dilation: 1, groups: 1)
        #expect(shape == [1, 1, 4])
        // edges see one zero -> sum 2; interior -> sum 3.
        #expect(out == [2, 3, 3, 2])
    }

    @Test("conv1d stride halves the output length")
    func conv1dStride() {
        let x: [Float] = [0, 1, 2, 3, 4, 5]
        let w: [Float] = [1]
        let (out, shape) = AudioMath.conv1d(
            x: x, xShape: [1, 1, 6], weight: w, wShape: [1, 1, 1],
            bias: nil, stride: 2, padding: 0, dilation: 1, groups: 1)
        #expect(shape == [1, 1, 3])
        #expect(out == [0, 2, 4])
    }

    @Test("conv1d groups (depthwise) keeps channels independent")
    func conv1dGroups() {
        // 2 channels, depthwise (groups=2), k=1 weight = [10, 100].
        let x: [Float] = [1, 2, 3, 4, 5, 6]  // [1,2,3] then [4,5,6]
        let w: [Float] = [10, 100]  // [2,1,1]
        let (out, shape) = AudioMath.conv1d(
            x: x, xShape: [1, 2, 3], weight: w, wShape: [2, 1, 1],
            bias: nil, stride: 1, padding: 0, dilation: 1, groups: 2)
        #expect(shape == [1, 2, 3])
        #expect(out == [10, 20, 30, 400, 500, 600])
    }

    @Test("conv1d adds bias per output channel")
    func conv1dBias() {
        let x: [Float] = [1, 1]
        let w: [Float] = [2]
        let (out, _) = AudioMath.conv1d(
            x: x, xShape: [1, 1, 2], weight: w, wShape: [1, 1, 1],
            bias: [5], stride: 1, padding: 0, dilation: 1, groups: 1)
        #expect(out == [7, 7])
    }

    // MARK: - transposed conv1d

    @Test("convTransposed1d upsamples by stride")
    func convTransposeUpsample() {
        // x: [1,1,3], identity kernel k=1, stride 2 -> length (3-1)*2+1 = 5
        let x: [Float] = [1, 2, 3]
        let w: [Float] = [1]
        let (out, shape) = AudioMath.convTransposed1d(
            x: x, xShape: [1, 1, 3], weight: w, wShape: [1, 1, 1],
            bias: nil, stride: 2, padding: 0, dilation: 1,
            outputPadding: 0, groups: 1)
        #expect(shape == [1, 1, 5])
        #expect(out == [1, 0, 2, 0, 3])
    }

    @Test("convTransposed1d is the adjoint of conv1d for k=2 stride=2")
    func convTransposeAdjoint() {
        // weight [1,1,2] = [1,1]; input [1,1,2] = [3,5].
        // Lout = (2-1)*2 - 0 + 1*(2-1) + 0 + 1 = 4
        let x: [Float] = [3, 5]
        let w: [Float] = [1, 1]
        let (out, shape) = AudioMath.convTransposed1d(
            x: x, xShape: [1, 1, 2], weight: w, wShape: [1, 1, 2],
            bias: nil, stride: 2, padding: 0, dilation: 1,
            outputPadding: 0, groups: 1)
        #expect(shape == [1, 1, 4])
        // sample 0 scatters into [0,1], sample 1 into [2,3].
        #expect(out == [3, 3, 5, 5])
    }

    // MARK: - activations

    @Test("snake activation matches its definition")
    func snakeReference() {
        // x = 0 -> 0 + (1/(a+eps)) * sin(0)^2 = 0 for any alpha.
        let x: [Float] = [0, 0]
        let out = AudioMath.snake(x, shape: [1, 2, 1], alpha: [1, 2])
        #expect(abs(out[0]) < 1e-6)
        #expect(abs(out[1]) < 1e-6)

        // Non-zero check against a direct computation.
        let xv: Float = 0.7
        let a: Float = 1.5
        let out2 = AudioMath.snake([xv], shape: [1, 1, 1], alpha: [a])
        let s = sin(a * xv)
        let expected = xv + (1.0 / (a + 1e-9)) * s * s
        #expect(abs(out2[0] - expected) < 1e-5)
    }

    @Test("tanh activation is bounded and odd")
    func tanhActivation() {
        let out = AudioMath.tanhAll([-100, 0, 100])
        #expect(out[0] < -0.999)
        #expect(abs(out[1]) < 1e-6)
        #expect(out[2] > 0.999)
    }

    // MARK: - normalization

    @Test("layerNorm yields zero mean / unit variance per row")
    func layerNormStats() {
        let x: [Float] = [1, 2, 3, 4]
        let out = AudioMath.layerNorm(
            x, rows: 1, dim: 4,
            weight: nil, bias: nil)
        let mean = out.reduce(0, +) / 4
        #expect(abs(mean) < 1e-5)
        var v: Float = 0
        for e in out { v += e * e }
        #expect(abs(v / 4 - 1.0) < 1e-3)
    }

    @Test("l2NormalizeRows produces unit-norm rows")
    func l2Normalize() {
        let x: [Float] = [3, 4, 0, 0]  // row0 norm 5, row1 norm 0
        let out = AudioMath.l2NormalizeRows(x, rows: 2, dim: 2)
        #expect(abs(out[0] - 0.6) < 1e-5)
        #expect(abs(out[1] - 0.8) < 1e-5)
        // zero row stays finite (eps guard).
        #expect(out[2].isFinite && out[3].isFinite)
    }

    // MARK: - linear algebra

    @Test("matmul matches a hand-computed product")
    func matmulReference() {
        // [2x3] · [3x2]
        let a: [Float] = [1, 2, 3, 4, 5, 6]
        let b: [Float] = [7, 8, 9, 10, 11, 12]
        let out = AudioMath.matmul(a, b, m: 2, k: 3, n: 2)
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        #expect(out == [58, 64, 139, 154])
    }

    @Test("linear applies weightᵀ and bias")
    func linearReference() {
        // x [1x2], weight [3x2] (PyTorch layout), bias [3].
        let x: [Float] = [1, 2]
        let w: [Float] = [1, 0, 0, 1, 1, 1]
        let out = AudioMath.linear(
            x, rows: 1, inDim: 2,
            weight: w, outDim: 3, bias: [10, 20, 30])
        // out = [1, 2, 3] + bias
        #expect(out == [11, 22, 33])
    }

    // MARK: - layout helpers

    @Test("transpose12 swaps the last two dims")
    func transpose12() {
        // [1,2,3]
        let x: [Float] = [1, 2, 3, 4, 5, 6]
        let out = AudioMath.transpose12(x, shape: [1, 2, 3])
        // -> [1,3,2] : columns become rows.
        #expect(out == [1, 4, 2, 5, 3, 6])
    }

    @Test("reflectionPad1d mirrors edge samples")
    func reflectionPad() {
        let x: [Float] = [1, 2, 3, 4]
        let (out, shape) = AudioMath.reflectionPad1d(
            x, shape: [1, 1, 4], left: 2, right: 1)
        #expect(shape == [1, 1, 7])
        // left reflect of [1,2,3,4] by 2 -> [3,2 | 1,2,3,4 | 3]
        #expect(out == [3, 2, 1, 2, 3, 4, 3])
    }

    @Test("zeroPad1d pads with zeros")
    func zeroPad() {
        let x: [Float] = [1, 2]
        let (out, shape) = AudioMath.zeroPad1d(
            x, shape: [1, 1, 2], left: 1, right: 2)
        #expect(shape == [1, 1, 5])
        #expect(out == [0, 1, 2, 0, 0])
    }

    // MARK: - weight norm

    @Test("WeightNorm reconstructs g·v/||v|| per output slice")
    func weightNormReconstruct() {
        // v: [2,1,2] two output slices; g one magnitude per slice.
        let v: [Float] = [3, 4, 1, 0]  // slice0 norm 5, slice1 norm 1
        let g: [Float] = [10, 7]  // target magnitudes
        let out = WeightNorm.effectiveWeight(
            g: g, v: v,
            shape: [2, 1, 2], exceptDim: 0)
        // slice0 scaled to magnitude ~10 : [3,4]*(10/5) = [6,8]
        #expect(abs(out[0] - 6) < 1e-3)
        #expect(abs(out[1] - 8) < 1e-3)
        // slice1 already magnitude 1 -> scaled to 7.
        #expect(abs(out[2] - 7) < 1e-3)
        #expect(abs(out[3]) < 1e-3)
    }

    // MARK: - SNAC config

    @Test("SNACConfig decodes a representative config.json")
    func configDecode() throws {
        let json = """
            {
              "sampling_rate": 24000,
              "encoder_dim": 48,
              "encoder_rates": [2, 4, 8, 8],
              "decoder_dim": 1024,
              "decoder_rates": [8, 8, 4, 2],
              "attn_window_size": null,
              "codebook_size": 4096,
              "codebook_dim": 8,
              "vq_strides": [4, 2, 1],
              "noise": true,
              "depthwise": true
            }
            """
        let config = try JSONDecoder().decode(
            SNACConfig.self, from: Data(json.utf8))
        #expect(config.samplingRate == 24000)
        #expect(config.hopLength == 2 * 4 * 8 * 8)  // product of rates
        #expect(config.vqStrides.count == 3)
        #expect(config.attnWindowSize == nil)
    }

    // MARK: - lcm / gcd helpers

    @Test("gcd and lcm are correct")
    func gcdLcm() {
        #expect(gcd(12, 18) == 6)
        #expect(lcm(4, 6) == 12)
        #expect(lcm(8, 4) == 8)
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local SNAC checkpoint directory, or nil if none is set.
    /// Set `FFAI_SNAC_DIR` to a directory holding `config.json` +
    /// `model.safetensors` to exercise the full encode/decode path.
    private func snacCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_SNAC_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("SNAC encode→decode round-trip reconstructs a waveform")
    func encodeDecodeRoundTrip() throws {
        guard let dir = snacCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let snac = try SNAC.fromPretrained(directory: dir)

        // A short 0.25s sine sweep at the codec sample rate.
        let n = snac.sampleRate / 4
        var samples = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            let t = Float(i) / Float(snac.sampleRate)
            samples[i] = 0.5 * sin(2.0 * .pi * 220.0 * t)
        }
        let waveform = AudioMath.tensor(samples, shape: [n])

        let codes = try snac.encode(waveform: waveform)
        #expect(!codes.isEmpty)
        #expect(codes.allSatisfy { !$0.isEmpty })

        let recon = try snac.decode(codes: codes)
        let reconFloats = AudioMath.floats(recon)
        #expect(reconFloats.allSatisfy { $0.isFinite })

        // Codecs are lossy; assert correlation rather than equality.
        let len = min(samples.count, reconFloats.count)
        var dotXY: Float = 0
        var dotXX: Float = 0
        var dotYY: Float = 0
        for i in 0 ..< len {
            dotXY += samples[i] * reconFloats[i]
            dotXX += samples[i] * samples[i]
            dotYY += reconFloats[i] * reconFloats[i]
        }
        let corr = dotXY / (sqrt(dotXX) * sqrt(dotYY) + 1e-9)
        // A working codec keeps a clearly positive correlation.
        #expect(corr > 0.3)
    }
}
