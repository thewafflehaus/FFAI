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
import Foundation
import Metal
import Testing
@testable import FFAI
import TestHelpers

// AudioEncoder — exercises the Whisper-style audio-encoder stack
// end-to-end on synthetic weights. The contract is structural: a
// forward through the conv stem + transformer must produce finite,
// correctly-shaped, non-degenerate audio-frame tokens.
@Suite("AudioEncoder")
struct AudioEncoderTests {

    /// Cheap deterministic LCG-filled tensor in `[-scale, scale]`.
    private func randTensor(_ shape: [Int], scale: Float = 0.05,
                            seed: Int) -> Tensor {
        let n = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: n)
        var s = UInt64(seed &+ 1)
        for i in 0..<n {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let u = Float(s >> 40) / Float(1 << 24)
            data[i] = (u - 0.5) * 2 * scale
        }
        let tensor = Tensor.empty(shape: shape, dtype: .f32)
        tensor.copyIn(from: data)
        return tensor
    }

    private func ones(_ n: Int) -> Tensor {
        let t = Tensor.empty(shape: [n], dtype: .f32)
        t.copyIn(from: [Float](repeating: 1, count: n))
        return t
    }

    private func zeros(_ n: Int) -> Tensor {
        let t = Tensor.empty(shape: [n], dtype: .f32)
        t.copyIn(from: [Float](repeating: 0, count: n))
        return t
    }

    /// Build a small AudioEncoder with deterministic synthetic weights.
    private func makeEncoder(
        nMels: Int, hidden: Int, intermediate: Int,
        nLayers: Int, nHeads: Int, maxAudioCtx: Int
    ) -> AudioEncoder {
        let config = AudioEncoderConfig(
            nMels: nMels, hidden: hidden, intermediate: intermediate,
            nLayers: nLayers, nHeads: nHeads, maxAudioCtx: maxAudioCtx,
            layerNormEps: 1e-5)

        let conv1W = randTensor([hidden, nMels, 3], seed: 1)
        let conv1B = zeros(hidden)
        let conv2W = randTensor([hidden, hidden, 3], seed: 2)
        let conv2B = zeros(hidden)
        let posEmb = randTensor([maxAudioCtx, hidden], seed: 3)

        var layers: [AudioEncoderLayer] = []
        for l in 0..<nLayers {
            let ln1 = LayerNorm(weight: ones(hidden), bias: zeros(hidden), eps: 1e-5)
            let ln2 = LayerNorm(weight: ones(hidden), bias: zeros(hidden), eps: 1e-5)
            let qP = Linear(weight: randTensor([hidden, hidden], seed: 10 + l * 8),
                            bias: zeros(hidden))
            // Whisper's k_proj has no bias — exercise the optional path.
            let kP = Linear(weight: randTensor([hidden, hidden], seed: 11 + l * 8))
            let vP = Linear(weight: randTensor([hidden, hidden], seed: 12 + l * 8),
                            bias: zeros(hidden))
            let oP = Linear(weight: randTensor([hidden, hidden], seed: 13 + l * 8),
                            bias: zeros(hidden))
            let fc1 = Linear(weight: randTensor([intermediate, hidden], seed: 14 + l * 8),
                             bias: zeros(intermediate))
            let fc2 = Linear(weight: randTensor([hidden, intermediate], seed: 15 + l * 8),
                             bias: zeros(hidden))
            layers.append(AudioEncoderLayer(
                layerNorm1: ln1, qProj: qP, kProj: kP, vProj: vP, oProj: oP,
                layerNorm2: ln2, fc1: fc1, fc2: fc2,
                hidden: hidden, nHeads: nHeads, intermediate: intermediate))
        }
        let postLN = LayerNorm(weight: ones(hidden), bias: zeros(hidden), eps: 1e-5)

        return AudioEncoder(
            config: config, conv1Weight: conv1W, conv1Bias: conv1B,
            conv2Weight: conv2W, conv2Bias: conv2B,
            positionEmbedding: posEmb, layers: layers,
            postLayerNorm: postLN, dtype: .f32)
    }

    @Test("config — derived geometry is correct")
    func configGeometry() {
        // Whisper base: 80 mels, d_model 512, 8 heads → head_dim 64.
        let c = AudioEncoderConfig(
            nMels: 80, hidden: 512, intermediate: 2048,
            nLayers: 6, nHeads: 8)
        #expect(c.headDim == 64)
        #expect(c.maxAudioCtx == 1500)
    }

    @Test("encode — channel-major mel produces finite halved-length tokens")
    func encodeChannelMajor() {
        autoreleasepool {
            // head_dim 16 — a non-128 head dim, the case the CPU
            // attention core handles.
            let enc = makeEncoder(
                nMels: 8, hidden: 64, intermediate: 128,
                nLayers: 2, nHeads: 4, maxAudioCtx: 64)
            let nFrames = 40
            let mel = randTensor([8, nFrames], scale: 1.0, seed: 100)
            let out = enc.encode(mel: mel, melFrameMajor: false)
            // Stride-2 conv: nAudioCtx = (40 + 2 - 3)/2 + 1 = 20.
            #expect(out.shape == [20, 64])
            let vals = out.toArray(as: Float.self)
            #expect(vals.allSatisfy { $0.isFinite })
            // Post-LayerNorm output must not be degenerate.
            let variance = vals.map { $0 * $0 }.reduce(0, +) / Float(vals.count)
            #expect(variance > 1e-6)
        }
    }

    @Test("encode — frame-major mel is transposed internally")
    func encodeFrameMajor() {
        autoreleasepool {
            let enc = makeEncoder(
                nMels: 8, hidden: 48, intermediate: 96,
                nLayers: 1, nHeads: 4, maxAudioCtx: 64)
            let nFrames = 32
            // Frame-major [nFrames, nMels] — the Ops.melSpectrogram layout.
            let mel = randTensor([nFrames, 8], scale: 1.0, seed: 200)
            let out = enc.encode(mel: mel, melFrameMajor: true)
            // nAudioCtx = (32 + 2 - 3)/2 + 1 = 16.
            #expect(out.shape == [16, 48])
            #expect(out.toArray(as: Float.self).allSatisfy { $0.isFinite })
        }
    }

    @Test("encode — chains directly off the mel front-end")
    func encodeFromFrontEnd() {
        autoreleasepool {
            // A tiny synthetic front-end so the test stays fast.
            let cfg = AudioFrontEndConfig(
                sampleRate: 16_000, nFFT: 64, hopLength: 32, nMels: 8)
            let enc = makeEncoder(
                nMels: 8, hidden: 32, intermediate: 64,
                nLayers: 1, nHeads: 4, maxAudioCtx: 256)
            // 0.05 s of a sine.
            let n = 800
            var wave = [Float](repeating: 0, count: n)
            for i in 0..<n {
                wave[i] = 0.4 * sin(2.0 * Float.pi * 300.0 * Float(i) / 16_000.0)
            }
            var mel: Tensor!
            // `whisperNormalize: false` keeps the kernel only *queued*
            // on `cb` so `runAndWait` owns the commit (the normalised
            // path commits internally — that would double-commit here).
            runAndWait { cb in
                mel = AudioPreprocessing.logMelSpectrogram(
                    waveform: wave, cfg: cfg, whisperNormalize: false,
                    on: cb)
            }
            // mel is frame-major [nFrames, nMels].
            let out = enc.encode(mel: mel, melFrameMajor: true)
            #expect(out.shape[1] == 32)
            #expect(out.toArray(as: Float.self).allSatisfy { $0.isFinite })
        }
    }

    @Test("parameters — names follow the HF Whisper convention")
    func parameterNames() {
        let enc = makeEncoder(
            nMels: 8, hidden: 16, intermediate: 32,
            nLayers: 2, nHeads: 2, maxAudioCtx: 64)
        let names = Set(enc.parameters().map { $0.0 })
        #expect(names.contains("conv1.weight"))
        #expect(names.contains("conv2.bias"))
        #expect(names.contains("embed_positions.weight"))
        #expect(names.contains("layers.0.self_attn.q_proj.weight"))
        #expect(names.contains("layers.1.fc2.weight"))
        #expect(names.contains("layer_norm.weight"))
        // k_proj has no bias.
        #expect(!names.contains("layers.0.self_attn.k_proj.bias"))
    }
}
