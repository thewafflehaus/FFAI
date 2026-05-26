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
// MimiCodecTests — exercises the Mimi neural-codec building blocks
// plus a graceful-skip end-to-end path that runs only when a real Mimi
// checkpoint is available.
//
// Mimi checkpoints are multi-hundred-MB HF snapshots; CI does not ship
// one. The structural tests therefore carry the correctness signal: the
// `mimi_202407` preset geometry, the sanitize() key rewriting, the
// traditional RoPE rotation, and the Euclidean-codebook lookup are each
// checked against an independent hand-computed reference.

import Foundation
import Testing
@testable import FFAI

@Suite("Mimi codec — structure + round-trip")
struct MimiCodecTests {

    // MARK: - config preset

    @Test("mimi_202407 preset has the expected geometry")
    func presetGeometry() {
        let c = MimiConfig.mimi202407
        #expect(c.sampleRate == 24_000)
        #expect(c.hopLength == 8 * 6 * 5 * 4)        // 960
        // encoder fps = 24000 / 960 = 25; frameRate 12.5 -> stride 2.
        #expect(c.encoderFPS == 25.0)
        #expect(c.downsampleStride == 2)
        #expect(c.headDim == 64)                     // 512 / 8
        #expect(c.quantizerNQ == 32)
    }

    // MARK: - sanitize key rewriting

    @Test("sanitizeKey rewrites encoder/decoder sequential indices")
    func sanitizeKeys() {
        // Leading underscores stripped, encoder.model. prefix removed.
        #expect(MimiWeights.sanitizeKey("_encoder._model._0._conv._weight")
                == "encoder.init_conv1d.conv.weight")
        // encoder.1 -> encoder.layers.0.residuals.0
        #expect(MimiWeights.sanitizeKey("encoder.1.conv.weight")
                == "encoder.layers.0.residuals.0.conv.weight")
        // encoder.3 -> encoder.layers.0.downsample (encoderIdx+2).
        #expect(MimiWeights.sanitizeKey("encoder.3.conv.weight")
                == "encoder.layers.0.downsample.conv.weight")
        // decoder.14 -> decoder.final_conv1d.
        #expect(MimiWeights.sanitizeKey("decoder.14.conv.bias")
                == "decoder.final_conv1d.conv.bias")
        // in_proj_weight -> in_proj.weight.
        #expect(MimiWeights.sanitizeKey("layers.0.self_attn.in_proj_weight")
                == "layers.0.self_attn.in_proj.weight")
    }

    // MARK: - RoPE

    @Test("traditional RoPE leaves position 0 unchanged")
    func ropeZeroPosition() {
        let x: [Float] = [1, 2, 3, 4]   // one row, headDim 4
        let out = MimiRoPE.apply(x, t: 1, headDim: 4, base: 10_000)
        // theta = 0 at position 0 -> identity rotation.
        for i in 0..<4 { #expect(abs(out[i] - x[i]) < 1e-5) }
    }

    @Test("traditional RoPE rotates a pair by the expected angle")
    func ropeRotation() {
        // headDim 2, position 1, base 10000 -> freq = base^0 = 1,
        // theta = 1. Pair (1, 0) rotates to (cos1, sin1).
        let x: [Float] = [1, 0,   1, 0]   // two rows
        let out = MimiRoPE.apply(x, t: 2, headDim: 2, base: 10_000)
        #expect(abs(out[0] - 1) < 1e-5)        // pos 0 unchanged
        #expect(abs(out[1] - 0) < 1e-5)
        #expect(abs(out[2] - cosf(1)) < 1e-4)  // pos 1 rotated
        #expect(abs(out[3] - sinf(1)) < 1e-4)
    }

    // MARK: - Euclidean codebook

    @Test("Mimi codebook picks the nearest entry by c2 - dot")
    func codebookNearest() {
        // 3 entries, dim 2. embedding rows are 2× the embedding_sum
        // since cluster_usage is 0.5 for every entry.
        let embSum: [Float] = [0, 0,   5, 0,   0, 5]
        let usage: [Float] = [0.5, 0.5, 0.5]
        let cb = MimiCodebook.testInstance(embeddingSum: embSum,
                                           clusterUsage: usage, dim: 2)
        // Effective embedding: [0,0], [10,0], [0,10].
        let queries: [Float] = [9, 1,   1, 9]   // -> idx 1, idx 2
        let idx = cb.encode(queries, rows: 2)
        #expect(idx == [1, 2])
        let decoded = cb.decode(codes: idx)
        #expect(decoded == [10, 0,  0, 10])
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local Mimi checkpoint directory, or nil if unset. Set
    /// `FFAI_MIMI_DIR` to a directory holding a `*.safetensors` weights
    /// file to exercise the full encode/decode path.
    private func mimiCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_MIMI_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    @Test("Mimi encode→decode round-trip reconstructs a waveform")
    func encodeDecodeRoundTrip() throws {
        guard let dir = mimiCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let codec = try Mimi.fromPretrained(directory: dir)

        // A short 0.5s sine tone at the codec sample rate.
        let n = codec.sampleRate / 2
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(codec.sampleRate)
            samples[i] = 0.5 * sin(2.0 * .pi * 220.0 * t)
        }
        let waveform = AudioMath.tensor(samples, shape: [n])

        let codes = try codec.encode(waveform: waveform)
        #expect(!codes.isEmpty)
        #expect(codes.allSatisfy { !$0.isEmpty })

        let recon = try codec.decode(codes: codes)
        let reconFloats = AudioMath.floats(recon)
        #expect(reconFloats.allSatisfy { $0.isFinite })

        // Codecs are lossy; assert correlation rather than equality.
        let len = min(samples.count, reconFloats.count)
        var dotXY: Float = 0, dotXX: Float = 0, dotYY: Float = 0
        for i in 0..<len {
            dotXY += samples[i] * reconFloats[i]
            dotXX += samples[i] * samples[i]
            dotYY += reconFloats[i] * reconFloats[i]
        }
        let corr = dotXY / (sqrt(dotXX) * sqrt(dotYY) + 1e-9)
        // A working codec keeps a clearly positive correlation.
        #expect(corr > 0.3)
    }
}

// MARK: - test helpers

extension MimiCodebook {
    /// Build a codebook from in-memory `embedding_sum` / `cluster_usage`
    /// tables — used by unit tests that have no checkpoint.
    static func testInstance(embeddingSum: [Float], clusterUsage: [Float],
                             dim: Int) -> MimiCodebook {
        MimiCodebook(embeddingSum: embeddingSum,
                     clusterUsage: clusterUsage, dim: dim)
    }
}
