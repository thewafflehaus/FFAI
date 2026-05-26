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
// NemotronH unit coverage — Phase 5e stack-interleaved hybrid.
//
// The full load + decode path is exercised by
// Tests/ModelTests/NemotronHIntegrationTests.swift against a real
// checkpoint. These unit tests cover the parts that don't need GPU
// weights: the `hybrid_override_pattern` → layer-kind parsing, and the
// dense squared-ReLU MLP layer's decode against a CPU reference.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("NemotronH Text Hybrid Layer Parsing + Dense MLP Layer")
struct NemotronHTextTests {

    // ─── Layer-kind parsing ──────────────────────────────────────────

    @Test("hybrid_override_pattern characters map to the right layer kind")
    func layerKindParsing() throws {
        #expect(try NemotronHLayerKind(from: "M") == .mamba)
        #expect(try NemotronHLayerKind(from: "*") == .attention)
        #expect(try NemotronHLayerKind(from: "-") == .mlp)
        #expect(try NemotronHLayerKind(from: "E") == .moe)
    }

    @Test("an unknown pattern character throws unsupportedConfig")
    func layerKindRejectsUnknown() {
        #expect(throws: NemotronHError.self) {
            _ = try NemotronHLayerKind(from: "X")
        }
    }

    @Test("a real NemotronH-4B pattern parses to the expected kind counts")
    func realPatternParses() throws {
        // The published Nemotron-H-4B-Base hybrid_override_pattern.
        let pattern = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
        let kinds = try Array(pattern).map { try NemotronHLayerKind(from: $0) }
        #expect(kinds.count == 52)
        #expect(kinds.filter { $0 == .attention }.count == 4)
        #expect(kinds.filter { $0 == .mamba }.count == 24)
        #expect(kinds.filter { $0 == .mlp }.count == 24)
        #expect(kinds.filter { $0 == .moe }.count == 0)
    }

    // ─── Dense MLP layer decode ──────────────────────────────────────

    /// Build a `dim × dim` identity-matrix `Linear` scaled by `scale`,
    /// so `linear(x) = scale * x` — a per-element multiply, which lets a
    /// CPU reference stay closed-form even though the layer runs a gemv.
    private func scaledIdentityLinear(dim: Int, scale: Float) -> AnyLinear {
        // Weight is row-major [outFeatures, inFeatures]; gemv computes
        // out[i] = Σ_j W[i,j] x[j], so W[i,i] = scale, else 0.
        var w = [Float](repeating: 0, count: dim * dim)
        for i in 0..<dim { w[i * dim + i] = scale }
        let t = Tensor.empty(shape: [dim, dim], dtype: .f32)
        t.copyIn(from: w)
        return AnyLinear(Linear(weight: t))
    }

    /// `NemotronHMLPLayer.decode` computes `h + down(relu(up(norm(h)))^2)`.
    /// With scaled-identity up/down projections and an all-ones RMSNorm,
    /// every projection is a per-element multiply, so the layer reduces
    /// to a closed-form per-element CPU reference. `hidden = 128` keeps
    /// the RMSNorm kernel's row-size invariant satisfied.
    @Test("dense MLP layer decode matches the squared-ReLU CPU reference")
    func mlpLayerDecodeMatchesReference() {
        autoreleasepool {
            let dim = 128
            let normWeight = Tensor.empty(shape: [dim], dtype: .f32)
            normWeight.copyIn(from: [Float](repeating: 1, count: dim))
            let norm = RMSNorm(weight: normWeight, eps: 1e-6)

            let up: Float = 0.5
            let down: Float = 3.0
            let layer = NemotronHMLPLayer(
                norm: norm,
                upProj: scaledIdentityLinear(dim: dim, scale: up),
                downProj: scaledIdentityLinear(dim: dim, scale: down),
                hidden: dim, intermediate: dim)

            // A spread of positive + negative inputs so the ReLU clamp
            // is genuinely exercised.
            var hValues = [Float](repeating: 0, count: dim)
            for i in 0..<dim { hValues[i] = Float(i) - 64.0 }
            let h = Tensor.empty(shape: [dim], dtype: .f32)
            h.copyIn(from: hValues)

            let device = Device.shared
            var out: Tensor!
            let cmd = device.makeCommandBuffer()
            out = layer.decode(h, position: 0,
                               cache: StatelessLayerCache(),
                               cmd: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()

            // CPU reference: norm(h) = h / rms(h), rms = sqrt(mean(h²)+eps).
            let meanSq = hValues.map { $0 * $0 }.reduce(0, +) / Float(dim)
            let rms = (meanSq + 1e-6).squareRoot()
            let result = out.toArray(as: Float.self)
            for i in 0..<dim {
                let normed = hValues[i] / rms
                let upOut = up * normed
                let relu2 = max(upOut, 0) * max(upOut, 0)
                let expected = hValues[i] + down * relu2
                #expect(abs(result[i] - expected) < 1e-3)
            }
        }
    }
}
