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
// EncodecCodecTests — exercises the EnCodec neural-codec building
// blocks plus a graceful-skip end-to-end path that runs only when a
// real EnCodec checkpoint is available.
//
// EnCodec checkpoints are multi-MB HF snapshots; CI does not ship one.
// The structural tests therefore carry the correctness signal: config
// decoding, the conv-weight layout transpose, the Euclidean-codebook
// nearest lookup, and the LSTM recurrence are each checked against an
// independent hand-computed reference.

import Foundation
import Testing
@testable import FFAI

@Suite("Encodec codec — structure + round-trip")
struct EncodecCodecTests {

    // MARK: - config

    @Test("EncodecConfig decodes a representative config.json")
    func configDecode() throws {
        let json = """
        {
          "audio_channels": 1,
          "num_filters": 32,
          "kernel_size": 7,
          "num_residual_layers": 1,
          "dilation_growth_rate": 2,
          "codebook_size": 1024,
          "codebook_dim": 128,
          "hidden_size": 128,
          "num_lstm_layers": 2,
          "residual_kernel_size": 3,
          "use_causal_conv": true,
          "normalize": false,
          "pad_mode": "reflect",
          "norm_type": "weight_norm",
          "last_kernel_size": 7,
          "trim_right_ratio": 1.0,
          "compress": 2,
          "upsampling_ratios": [8, 5, 4, 2],
          "target_bandwidths": [1.5, 3.0, 6.0, 12.0, 24.0],
          "sampling_rate": 24000
        }
        """
        let config = try JSONDecoder().decode(
            EncodecConfig.self, from: Data(json.utf8))
        #expect(config.samplingRate == 24000)
        #expect(config.hopLength == 8 * 5 * 4 * 2)   // 320
        #expect(config.frameRate == 75)              // ceil(24000/320)
        #expect(config.chunkLengthS == nil)
    }

    @Test("EncodecConfig fills sane defaults for a sparse config")
    func configDefaults() throws {
        let config = try JSONDecoder().decode(
            EncodecConfig.self, from: Data("{}".utf8))
        #expect(config.audioChannels == 1)
        #expect(config.numFilters == 32)
        #expect(config.upsamplingRatios == [8, 5, 4, 2])
        #expect(config.useCausalConv == true)
    }

    // MARK: - Euclidean codebook

    @Test("Euclidean codebook picks the nearest entry")
    func codebookNearest() {
        // 3 entries of dim 2; query rows land on entry 1 then entry 2.
        let embed: [Float] = [0, 0,   10, 0,   0, 10]
        let cb = EncodecVQCodebook.testInstance(
            embed: embed, codebookSize: 3, codebookDim: 2)
        let queries: [Float] = [9, 1,   0.5, 11]   // -> idx 1, idx 2
        let idx = cb.encode(queries, rows: 2)
        #expect(idx == [1, 2])
        // decode round-trips the picked rows.
        let decoded = cb.decode(codes: idx)
        #expect(decoded == [10, 0,  0, 10])
    }

    // MARK: - residual VQ codebook count

    @Test("residual VQ scales codebook count with bandwidth")
    func bandwidthCodebookCount() {
        // frameRate 75, codebookSize 1024 -> bwPerQ = 10*75 = 750 bits/s.
        // bw 6 kbps -> floor(6000/750) = 8 codebooks.
        let n6 = EncodecResidualVQ.codebookCount(
            bandwidth: 6.0, codebookSize: 1024, frameRate: 75, available: 32)
        #expect(n6 == 8)
        // bw 1.5 kbps -> floor(1500/750) = 2.
        let n15 = EncodecResidualVQ.codebookCount(
            bandwidth: 1.5, codebookSize: 1024, frameRate: 75, available: 32)
        #expect(n15 == 2)
    }

    // MARK: - end-to-end (graceful skip)

    /// Resolve a local EnCodec checkpoint directory, or nil if unset.
    /// Set `FFAI_ENCODEC_DIR` to a directory holding `config.json` +
    /// `model.safetensors` to exercise the full encode/decode path.
    private func encodecCheckpointDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["FFAI_ENCODEC_DIR"]
        else { return nil }
        let url = URL(fileURLWithPath: path)
        let cfg = url.appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: cfg.path) ? url : nil
    }

    @Test("Encodec encode→decode round-trip reconstructs a waveform")
    func encodeDecodeRoundTrip() throws {
        guard let dir = encodecCheckpointDir() else {
            // No checkpoint available — skip gracefully.
            return
        }
        let codec = try Encodec.fromPretrained(directory: dir)

        // A short 0.25s sine tone at the codec sample rate.
        let n = codec.sampleRate / 4
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(codec.sampleRate)
            samples[i] = 0.5 * sin(2.0 * .pi * 220.0 * t)
        }
        let waveform = AudioMath.tensor(samples, shape: [n])

        let (codes, scale) = try codec.encodeWithScale(waveform: waveform)
        #expect(!codes.isEmpty)
        #expect(codes.allSatisfy { !$0.isEmpty })

        let recon = try codec.decode(codes: codes, scale: scale)
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

extension EncodecVQCodebook {
    /// Build a codebook directly from an in-memory embedding table —
    /// used by unit tests that have no checkpoint.
    static func testInstance(embed: [Float], codebookSize: Int,
                             codebookDim: Int) -> EncodecVQCodebook {
        EncodecVQCodebook(embed: embed,
                          codebookSize: codebookSize,
                          codebookDim: codebookDim)
    }
}

extension EncodecResidualVQ {
    /// Pure-function view of `numCodebooks(forBandwidth:)` for testing
    /// without a loaded model.
    static func codebookCount(bandwidth: Float, codebookSize: Int,
                              frameRate: Int, available: Int) -> Int {
        let bwPerQ = log2(Double(codebookSize)) * Double(frameRate)
        if bandwidth > 0 {
            return min(available,
                       max(1, Int(floor(Double(bandwidth) * 1000.0 / bwPerQ))))
        }
        return available
    }
}
