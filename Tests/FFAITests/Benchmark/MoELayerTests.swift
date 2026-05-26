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
// MoERouter + MoELayer — Phase 5e.B mixture-of-experts coverage.
//
// Two layers of testing:
//   1. MoERouter — pure-CPU top-K + combine-weight math, checked
//      against hand-computed softmax / normalisation values for both
//      gating orders.
//   2. MoELayer — a tiny 4-expert / top-2 synthetic decode, with the
//      expected output computed by an independent CPU reference of the
//      same route → SwiGLU → weighted-combine pipeline.

import Foundation
import Metal
import Testing

@testable import FFAI

// ─── MoERouter ───────────────────────────────────────────────────────

@Suite("MoERouter — top-K gating math")
struct MoERouterTests {

    /// softmaxThenTopK with re-normalisation (Qwen3 / Qwen3.5 MoE).
    /// logits [1, 3, 2, 0], topK = 2.
    ///   softmax([1,3,2,0]) = [0.08714, 0.64391, 0.23688, 0.03206]
    ///   top-2 probs        = expert 1 (0.64391), expert 2 (0.23688)
    ///   normalised         = [0.73107, 0.26893]
    @Test("softmaxThenTopK + normTopKProb matches hand-computed softmax")
    func softmaxThenTopKNormalised() {
        let router = MoERouter(
            nExperts: 4, topK: 2,
            gatingMode: .softmaxThenTopK, normTopKProb: true)
        let r = router.route(logits: [1, 3, 2, 0])
        #expect(r.indices == [1, 2])
        #expect(abs(r.weights[0] - 0.73107) < 1e-4)
        #expect(abs(r.weights[1] - 0.26893) < 1e-4)
        // Normalised combine weights sum to 1.
        #expect(abs(r.weights.reduce(0, +) - 1) < 1e-5)
    }

    /// softmaxThenTopK WITHOUT re-normalisation — weights are the raw
    /// softmax probabilities of the picked experts (do NOT sum to 1).
    @Test("softmaxThenTopK without normTopKProb keeps raw softmax probs")
    func softmaxThenTopKUnnormalised() {
        let router = MoERouter(
            nExperts: 4, topK: 2,
            gatingMode: .softmaxThenTopK, normTopKProb: false)
        let r = router.route(logits: [1, 3, 2, 0])
        #expect(r.indices == [1, 2])
        #expect(abs(r.weights[0] - 0.64391) < 1e-4)
        #expect(abs(r.weights[1] - 0.23688) < 1e-4)
        // Unnormalised: sum is < 1.
        #expect(r.weights.reduce(0, +) < 0.9)
    }

    /// topKThenSoftmax (Granite4). logits [1, 3, 2, 0], topK = 2.
    ///   top-2 raw logits = expert 1 (3.0), expert 2 (2.0)
    ///   softmax([3,2])   = [0.73106, 0.26894]   (always normalised)
    @Test("topKThenSoftmax matches softmax over the picked logits")
    func topKThenSoftmax() {
        let router = MoERouter(
            nExperts: 4, topK: 2,
            gatingMode: .topKThenSoftmax)
        let r = router.route(logits: [1, 3, 2, 0])
        #expect(r.indices == [1, 2])
        #expect(abs(r.weights[0] - 0.73106) < 1e-4)
        #expect(abs(r.weights[1] - 0.26894) < 1e-4)
        // topKThenSoftmax is normalised by construction.
        #expect(abs(r.weights.reduce(0, +) - 1) < 1e-5)
    }

    /// Tie-breaking: equal logits resolve to the smaller expert index,
    /// matching argpartition / argmax semantics — keeps decode
    /// deterministic.
    @Test("equal logits break ties toward the smaller expert index")
    func tieBreakingSmallerIndexWins() {
        let router = MoERouter(
            nExperts: 4, topK: 2,
            gatingMode: .topKThenSoftmax)
        // All four logits equal — top-2 must be experts 0 and 1.
        let r = router.route(logits: [5, 5, 5, 5])
        #expect(r.indices == [0, 1])
        // Equal logits → equal softmax weights.
        #expect(abs(r.weights[0] - 0.5) < 1e-5)
        #expect(abs(r.weights[1] - 0.5) < 1e-5)
    }

    /// topK == nExperts selects every expert (the dense limit).
    @Test("topK == nExperts selects all experts")
    func topKEqualsNExperts() {
        let router = MoERouter(
            nExperts: 3, topK: 3,
            gatingMode: .softmaxThenTopK, normTopKProb: true)
        let r = router.route(logits: [0, 1, 2])
        #expect(Set(r.indices) == [0, 1, 2])
        #expect(abs(r.weights.reduce(0, +) - 1) < 1e-5)
    }
}

// ─── MoELayer ────────────────────────────────────────────────────────

@Suite("MoELayer — mixture-of-experts feed-forward block")
struct MoELayerTests {

    /// Independent CPU reference: one SwiGLU FFN over a scalar-weight
    /// linear chain. hidden = 1, intermediate = 1, so every projection
    /// is a 1×1 matrix.
    ///   gate(x) = g·x ; up(x) = u·x
    ///   silu(z) = z · sigmoid(z)
    ///   out     = d · ( silu(g·x) · (u·x) )
    private static func swiGLUReference(x: Float, g: Float, u: Float, d: Float) -> Float {
        let gx = g * x
        let silu = gx * (1 / (1 + Foundation.exp(-gx)))
        return d * (silu * (u * x))
    }

    /// Flush the shared device queue: submit + wait a no-op command
    /// buffer so every previously-committed buffer (including the
    /// internal `work` buffer `MoELayer.decode` commits without
    /// waiting) is GPU-complete before a host readback.
    private static func flushQueue() {
        let flush = Device.shared.makeCommandBuffer()
        flush.commit()
        flush.waitUntilCompleted()
    }

    /// Build a 1×1 `Linear` from a single scalar weight.
    private func scalarLinear(_ w: Float) -> AnyLinear {
        let t = Tensor.empty(shape: [1, 1], dtype: .f32)
        t.copyIn(from: [w])
        return AnyLinear(Linear(weight: t))
    }

    /// 4 experts, top-2, hidden = 1. Distinct scalar weights per expert
    /// so the routed combination is unambiguous, and a gate projection
    /// chosen to make experts 1 and 2 win.
    @Test("4-expert top-2 decode matches CPU reference combination")
    func fourExpertTopTwoDecode() {
        autoreleasepool {
            let hidden = 1
            let nExperts = 4
            let inputX: Float = 2.0

            // Gate weight [nExperts, hidden] = [4, 1]. With x = 2, the
            // gate logits are 2·[0.5, 1.5, 1.0, 0.0] = [1, 3, 2, 0].
            // softmaxThenTopK + norm picks experts 1 and 2 with
            // combine weights 0.73107 / 0.26893.
            let gateW = Tensor.empty(shape: [nExperts, hidden], dtype: .f32)
            gateW.copyIn(from: [Float(0.5), 1.5, 1.0, 0.0])
            let gate = AnyLinear(Linear(weight: gateW))

            // Per-expert SwiGLU scalar weights (gate, up, down).
            let gW: [Float] = [0.1, 0.7, 0.3, 0.9]
            let uW: [Float] = [1.0, 0.5, 2.0, 0.4]
            let dW: [Float] = [0.2, 1.1, 0.6, 0.8]
            let gateProj = gW.map { scalarLinear($0) }
            let upProj = uW.map { scalarLinear($0) }
            let downProj = dW.map { scalarLinear($0) }

            let router = MoERouter(
                nExperts: nExperts, topK: 2,
                gatingMode: .softmaxThenTopK, normTopKProb: true)
            let layer = MoELayer(
                gate: gate,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                router: router, hidden: hidden)

            let input = Tensor.empty(shape: [hidden], dtype: .f32)
            input.copyIn(from: [inputX])

            var out: Tensor!
            let cb = Device.shared.makeCommandBuffer()
            out = layer.decode(
                input, position: 0,
                cache: StatelessLayerCache(),
                cmd: cb, device: .shared)
            // `decode` commits its internal `work` buffer WITHOUT
            // waiting — the production caller hazard-tracks the read on
            // the next GPU cmd. A host readback must flush the queue
            // first: submit + wait a trailing no-op on the same
            // in-order MTLCommandQueue so `work` is guaranteed done.
            Self.flushQueue()

            // CPU reference: experts 1 and 2 win, weighted 0.73107 /
            // 0.26893.
            let w1: Float = 0.73107
            let w2: Float = 0.26893
            let e1 = Self.swiGLUReference(x: inputX, g: gW[1], u: uW[1], d: dW[1])
            let e2 = Self.swiGLUReference(x: inputX, g: gW[2], u: uW[2], d: dW[2])
            let expected = w1 * e1 + w2 * e2

            #expect(abs(out.toFloatArray()[0] - expected) < 1e-3)
        }
    }

    /// Shared (always-on) expert: its output is added to the routed
    /// combination unconditionally — Granite4 layout.
    @Test("shared expert output is added unconditionally")
    func sharedExpertContributes() {
        autoreleasepool {
            let hidden = 1
            let nExperts = 4
            let inputX: Float = 2.0

            let gateW = Tensor.empty(shape: [nExperts, hidden], dtype: .f32)
            gateW.copyIn(from: [Float(0.5), 1.5, 1.0, 0.0])
            let gate = AnyLinear(Linear(weight: gateW))

            let gW: [Float] = [0.1, 0.7, 0.3, 0.9]
            let uW: [Float] = [1.0, 0.5, 2.0, 0.4]
            let dW: [Float] = [0.2, 1.1, 0.6, 0.8]
            let gateProj = gW.map { scalarLinear($0) }
            let upProj = uW.map { scalarLinear($0) }
            let downProj = dW.map { scalarLinear($0) }

            // Shared expert scalar weights.
            let sg: Float = 0.4
            let su: Float = 0.6
            let sd: Float = 1.5

            let router = MoERouter(
                nExperts: nExperts, topK: 2,
                gatingMode: .softmaxThenTopK, normTopKProb: true)
            let layer = MoELayer(
                gate: gate,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                sharedGateProj: scalarLinear(sg),
                sharedUpProj: scalarLinear(su),
                sharedDownProj: scalarLinear(sd),
                router: router, hidden: hidden)

            let input = Tensor.empty(shape: [hidden], dtype: .f32)
            input.copyIn(from: [inputX])

            var out: Tensor!
            let cb = Device.shared.makeCommandBuffer()
            out = layer.decode(
                input, position: 0,
                cache: StatelessLayerCache(),
                cmd: cb, device: .shared)
            Self.flushQueue()  // see `decode` no-wait note above

            let w1: Float = 0.73107
            let w2: Float = 0.26893
            let e1 = Self.swiGLUReference(x: inputX, g: gW[1], u: uW[1], d: dW[1])
            let e2 = Self.swiGLUReference(x: inputX, g: gW[2], u: uW[2], d: dW[2])
            let shared = Self.swiGLUReference(x: inputX, g: sg, u: su, d: sd)
            let expected = w1 * e1 + w2 * e2 + shared

            #expect(abs(out.toFloatArray()[0] - expected) < 1e-3)
        }
    }

    /// `parameters()` exposes the gate, every per-expert projection, and
    /// the shared expert — under HF-style nested names.
    @Test("parameters() enumerates gate + experts + shared expert")
    func parametersEnumeration() {
        let hidden = 1
        let nExperts = 3
        let gateW = Tensor.empty(shape: [nExperts, hidden], dtype: .f32)
        gateW.copyIn(from: [Float(0), 0, 0])
        let gate = AnyLinear(Linear(weight: gateW))
        let gateProj = (0 ..< nExperts).map { _ in scalarLinear(1) }
        let upProj = (0 ..< nExperts).map { _ in scalarLinear(1) }
        let downProj = (0 ..< nExperts).map { _ in scalarLinear(1) }

        let router = MoERouter(
            nExperts: nExperts, topK: 1,
            gatingMode: .topKThenSoftmax)
        let layer = MoELayer(
            gate: gate,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            sharedGateProj: scalarLinear(1),
            sharedUpProj: scalarLinear(1),
            sharedDownProj: scalarLinear(1),
            router: router, hidden: hidden)

        let names = Set(layer.parameters().map { $0.0 })
        #expect(names.contains("gate.weight"))
        #expect(names.contains("experts.0.gate_proj.weight"))
        #expect(names.contains("experts.2.down_proj.weight"))
        #expect(names.contains("shared_expert.up_proj.weight"))
        // gate + 3 experts × 3 projections + 3 shared = 13.
        #expect(layer.parameters().count == 13)
    }
}
