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
// LFM2 — unit coverage for the LiquidAI LFM2 / LFM2.5 family.
//
// Pure, GPU-free testing:
//   1. `lfm2LayerKinds` — the conv / attention schedule resolver.
//      LFM2 checkpoints carry EITHER a `layer_types` string array OR a
//      `full_attn_idxs` index list; the resolver must accept both and
//      reject malformed schedules.
//   2. `MoERouter` expert-bias routing — LFM2-MoE adds a per-expert
//      `expert_bias` to the post-softmax gate values before top-K.
//   3. `lfm2HostPerHeadRMSNorm` — the host-side per-head Q/K RMSNorm
//      (LFM2's head_dim 64 is not 128-aligned, so the GPU kernel can't
//      run it), checked against an independent CPU reference.

import Foundation
import Metal
import Testing

@testable import FFAI

// ─── Layer-schedule resolver ─────────────────────────────────────────

@Suite("LFM2 Text Layer Schedule")
struct LFM2LayerScheduleTests {

    @Test("layer_types drives the conv / attention schedule")
    func layerTypesSchedule() throws {
        let kinds = try lfm2LayerKinds(
            layerTypes: ["conv", "conv", "full_attention", "conv"],
            fullAttnIdxs: nil, numLayers: 4)
        #expect(kinds == [.conv, .conv, .attention, .conv])
    }

    @Test("full_attn_idxs drives the schedule when layer_types is absent")
    func fullAttnIdxsSchedule() throws {
        let kinds = try lfm2LayerKinds(
            layerTypes: nil, fullAttnIdxs: [2, 5], numLayers: 6)
        #expect(kinds == [.conv, .conv, .attention, .conv, .conv, .attention])
    }

    @Test("layer_types takes precedence over full_attn_idxs")
    func layerTypesWins() throws {
        let kinds = try lfm2LayerKinds(
            layerTypes: ["full_attention", "conv"],
            fullAttnIdxs: [1], numLayers: 2)
        #expect(kinds == [.attention, .conv])
    }

    @Test("layer_types length mismatch throws")
    func layerTypesCountMismatch() {
        #expect(throws: LFM2Error.self) {
            _ = try lfm2LayerKinds(
                layerTypes: ["conv", "conv"], fullAttnIdxs: nil, numLayers: 4)
        }
    }

    @Test("unknown layer_types entry throws")
    func unknownLayerType() {
        #expect(throws: LFM2Error.self) {
            _ = try lfm2LayerKinds(
                layerTypes: ["conv", "linear_attention"],
                fullAttnIdxs: nil, numLayers: 2)
        }
    }

    @Test("neither layer_types nor full_attn_idxs throws")
    func missingSchedule() {
        #expect(throws: LFM2Error.self) {
            _ = try lfm2LayerKinds(
                layerTypes: nil, fullAttnIdxs: nil, numLayers: 4)
        }
    }

    @Test("empty layer_types falls back to full_attn_idxs")
    func emptyLayerTypesFallsBack() throws {
        let kinds = try lfm2LayerKinds(
            layerTypes: [], fullAttnIdxs: [0], numLayers: 2)
        #expect(kinds == [.attention, .conv])
    }
}

// ─── Family registration ─────────────────────────────────────────────

@Suite("LFM2 Text Family Registration")
struct LFM2RegistrationTests {

    @Test("LFM2 owns the lfm2 / lfm2_moe model_types and architectures")
    func familyStrings() {
        #expect(LFM2.modelTypes.contains("lfm2"))
        #expect(LFM2.modelTypes.contains("lfm2_moe"))
        #expect(LFM2.architectures.contains("Lfm2ForCausalLM"))
        #expect(LFM2.architectures.contains("Lfm2MoeForCausalLM"))
    }
}

// ─── LFM2-MoE expert-bias routing ────────────────────────────────────

@Suite("LFM2-MoE Text Expert-bias Routing")
struct LFM2MoERoutingTests {

    /// LFM2-MoE adds a per-expert `expert_bias` to the post-softmax gate
    /// values before top-K. With logits [1,3,2,0]:
    ///   softmax ≈ [0.0871, 0.6439, 0.2369, 0.0321]
    ///   + bias  [0.5, 0, 0, 0.5] → [0.5871, 0.6439, 0.2369, 0.5321]
    ///   top-2   → expert 1, expert 0 (expert 2 would win unbiased).
    @Test("expert_bias steers selection and combine weights")
    func biasedRouting() {
        let router = MoERouter(
            nExperts: 4, topK: 2, gatingMode: .softmaxThenTopK,
            normTopKProb: true, expertBias: [0.5, 0, 0, 0.5])
        let r = router.route(logits: [1, 3, 2, 0])

        #expect(Set(r.indices) == Set([0, 1]))
        #expect(abs(r.weights.reduce(0, +) - 1.0) < 1e-5)
        let w = Dictionary(uniqueKeysWithValues: zip(r.indices, r.weights))
        #expect(w[1]! > w[0]!)
    }

    /// A `nil` expert bias must leave `.softmaxThenTopK` byte-for-byte
    /// unchanged — every other MoE family relies on that.
    @Test("nil expert_bias leaves softmaxThenTopK unchanged")
    func noBiasUnchanged() {
        let biased = MoERouter(
            nExperts: 4, topK: 2, gatingMode: .softmaxThenTopK,
            normTopKProb: true, expertBias: nil)
        let plain = MoERouter(
            nExperts: 4, topK: 2, gatingMode: .softmaxThenTopK,
            normTopKProb: true)
        let a = biased.route(logits: [1, 3, 2, 0])
        let b = plain.route(logits: [1, 3, 2, 0])
        #expect(a.indices == b.indices)
        #expect(a.weights == b.weights)
    }
}

// ─── Host per-head RMSNorm ───────────────────────────────────────────

@Suite("LFM2 Text Host Per-head RMSNorm")
struct LFM2HostRMSNormTests {

    /// Two heads of width 4, unit weight. Checked against a hand-computed
    /// RMSNorm: out = x / sqrt(mean(x²) + eps).
    @Test("per-head RMSNorm matches a CPU reference")
    func perHeadNorm() {
        let device = Device.shared
        let headDim = 4
        let nHeads = 2

        // head0 = [1,2,3,4], head1 = [2,2,2,2].
        let xValues: [Float] = [1, 2, 3, 4, 2, 2, 2, 2]
        let x = Tensor.empty(shape: [nHeads * headDim], dtype: .f32, device: device)
        x.copyIn(from: xValues)
        let weight = Tensor.empty(shape: [headDim], dtype: .f32, device: device)
        weight.copyIn(from: [Float](repeating: 1, count: headDim))

        let eps: Float = 1e-5
        let out = lfm2HostPerHeadRMSNorm(
            x, weight: weight, eps: eps,
            nHeads: nHeads, headDim: headDim, device: device)
        let got = out.toArray(as: Float.self)

        // Independent CPU reference.
        var expected = [Float](repeating: 0, count: nHeads * headDim)
        for h in 0..<nHeads {
            let base = h * headDim
            var sumSq: Float = 0
            for d in 0..<headDim { sumSq += xValues[base + d] * xValues[base + d] }
            let inv = 1.0 / (sumSq / Float(headDim) + eps).squareRoot()
            for d in 0..<headDim { expected[base + d] = xValues[base + d] * inv }
        }

        #expect(got.count == expected.count)
        for i in 0..<got.count {
            #expect(abs(got[i] - expected[i]) < 1e-4,
                    "element \(i): got \(got[i]), expected \(expected[i])")
        }
    }

    /// The per-head norm is independent: a head of all-equal values
    /// normalises to all-ones (× weight).
    @Test("constant head normalises to unit magnitude")
    func constantHead() {
        let device = Device.shared
        let x = Tensor.empty(shape: [4], dtype: .f32, device: device)
        x.copyIn(from: [Float](repeating: 3, count: 4))
        let weight = Tensor.empty(shape: [4], dtype: .f32, device: device)
        weight.copyIn(from: [Float](repeating: 1, count: 4))

        let out = lfm2HostPerHeadRMSNorm(
            x, weight: weight, eps: 1e-5,
            nHeads: 1, headDim: 4, device: device)
        for v in out.toArray(as: Float.self) {
            #expect(abs(v - 1.0) < 1e-3)
        }
    }
}
