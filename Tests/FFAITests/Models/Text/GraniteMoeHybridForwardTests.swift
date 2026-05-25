// GraniteMoeHybrid forward-path unit coverage.
//
// The slow integration test (Tests/ModelTests/GraniteMoeHybridIntegrationTests
// .swift) loads the published H-350M checkpoint, which is DENSE
// (`num_local_experts = 0`) — every layer's FFN is a plain SwiGLU MLP
// and no layer commits the command buffer. The MoE-bearing checkpoints
// (H-Tiny / H-Small) are 7B+ and ship only quantized, so there is no
// small raw checkpoint to integration-test the MoE feed-forward path.
//
// That left `GraniteMoeHybridModel.forward`'s MoE-commit path entirely
// unexercised in CI — and it carried a latent double-commit bug: the
// caller's `cmd` was handed straight to the layers, so the first
// MoE-bearing layer (whose `MoELayer` FFN commits `cmd`) committed the
// caller's buffer, and the caller's own later commit double-committed
// it. The fix keeps every per-layer dispatch on internal command
// buffers and queues only the final norm + lm_head onto the caller's
// `cmd` (the Jamba discipline).
//
// This test builds a tiny synthetic GraniteMoeHybrid-shaped model with a
// real MoE-bearing layer and runs `forward` end-to-end, asserting it
// completes without a command-buffer error and yields finite,
// correctly-shaped logits. It actually executes the previously
// CI-unexercised MoE-commit path in `forward`.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("GraniteMoeHybrid Forward MoE-commit Path")
struct GraniteMoeHybridForwardTests {

    // Synthetic geometry. `headDim == 128` satisfies the `sdpaDecode`
    // kernel invariant; `hidden = nHeads * headDim` and `hidden` is a
    // multiple of 128 ≤ 4096, satisfying the RMSNorm row-size invariant.
    private static let nHeads = 1
    private static let nKVHeads = 1
    private static let headDim = 128
    private static let hidden = 128          // nHeads * headDim
    private static let vocab = 64
    private static let nExperts = 4
    private static let topK = 2
    private static let moeIntermediate = 32
    private static let maxSeq = 8

    /// A `[rows, cols]` weight tensor filled with small deterministic
    /// values — a fixed-seed LCG keeps the test reproducible without
    /// pulling in a checkpoint. Values are scaled small so a multi-layer
    /// forward stays numerically tame (finite logits).
    private func smallWeight(rows: Int, cols: Int, seed: UInt64) -> Tensor {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var values = [Float](repeating: 0, count: rows * cols)
        for i in 0..<values.count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            // Map the top bits to a small symmetric range [-0.05, 0.05].
            let unit = Float(state >> 40) / Float(1 << 24)
            values[i] = (unit - 0.5) * 0.1
        }
        let t = Tensor.empty(shape: [rows, cols], dtype: .f32)
        t.copyIn(from: values)
        return t
    }

    /// A `[n]` RMSNorm weight, all ones (identity-scaling norm).
    private func onesNorm(_ n: Int) -> RMSNorm {
        let w = Tensor.empty(shape: [n], dtype: .f32)
        w.copyIn(from: [Float](repeating: 1, count: n))
        return RMSNorm(weight: w, eps: 1e-5)
    }

    /// Build one synthetic GraniteMoeHybrid *attention* layer. An
    /// attention mixer (rather than Mamba) keeps the synthetic weights
    /// minimal — no SSM parameter derivation needed. `moe == true`
    /// attaches an MoE FFN whose `MoELayer.decode` commits the command
    /// buffer, so the layer's `commitsCommandBuffer` flag is true.
    private func attentionLayer(seed: UInt64, moe: Bool) -> GraniteMoeHybridLayer {
        let H = Self.hidden
        let mixer = GraniteMoeHybridAttentionMixer(
            qProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: H, seed: seed))),
            kProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: H, seed: seed + 1))),
            vProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: H, seed: seed + 2))),
            oProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: H, seed: seed + 3))),
            nHeads: Self.nHeads, nKVHeads: Self.nKVHeads, headDim: Self.headDim,
            scale: 1.0 / Float(Double(Self.headDim).squareRoot()))

        let ffn: GraniteMoeHybridFFN
        if moe {
            // 4-expert top-2 MoE FFN plus an always-on shared expert —
            // the GraniteMoeHybrid block-sparse layout. `MoELayer.decode`
            // commits the command buffer, so this layer commits.
            let I = Self.moeIntermediate
            let gateProj = (0..<Self.nExperts).map {
                AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 10 + UInt64($0))))
            }
            let upProj = (0..<Self.nExperts).map {
                AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 20 + UInt64($0))))
            }
            let downProj = (0..<Self.nExperts).map {
                AnyLinear(Linear(weight: smallWeight(rows: H, cols: I, seed: seed + 30 + UInt64($0))))
            }
            let router = MoERouter(nExperts: Self.nExperts, topK: Self.topK,
                                   gatingMode: .topKThenSoftmax)
            let moeLayer = MoELayer(
                gate: AnyLinear(Linear(weight: smallWeight(rows: Self.nExperts, cols: H, seed: seed + 40))),
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                sharedGateProj: AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 50))),
                sharedUpProj: AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 51))),
                sharedDownProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: I, seed: seed + 52))),
                router: router, hidden: H)
            ffn = .moe(moeLayer)
        } else {
            // Dense SwiGLU MLP — no command-buffer commit.
            let I = Self.moeIntermediate
            ffn = .dense(GraniteMoeHybridDenseMLP(
                gateProj: AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 60))),
                upProj: AnyLinear(Linear(weight: smallWeight(rows: I, cols: H, seed: seed + 61))),
                downProj: AnyLinear(Linear(weight: smallWeight(rows: H, cols: I, seed: seed + 62)))))
        }

        return GraniteMoeHybridLayer(
            inputNorm: onesNorm(H), postNorm: onesNorm(H),
            mixer: .attention(mixer), ffn: ffn, hidden: H)
    }

    /// Assemble a synthetic GraniteMoeHybridModel from a list of layers.
    private func model(layers: [GraniteMoeHybridLayer]) -> GraniteMoeHybridModel {
        let H = Self.hidden
        let embedW = smallWeight(rows: Self.vocab, cols: H, seed: 1000)
        let embedTokens = AnyEmbedding(Embedding(weight: embedW))
        let lmHead = AnyLinear(Linear(weight: smallWeight(rows: Self.vocab, cols: H, seed: 2000)))
        return GraniteMoeHybridModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: onesNorm(H), lmHead: lmHead,
            hidden: H, nLayers: layers.count,
            nHeads: Self.nHeads, nKVHeads: Self.nKVHeads, headDim: Self.headDim,
            // Mamba geometry is unused (no mamba layer) — pass valid placeholders.
            mambaNHeads: 1, mambaHeadDim: 128, stateDim: 128,
            convDim: 128, convKernel: 4, nGroups: 1, dInner: 128,
            vocab: Self.vocab, maxSeq: Self.maxSeq,
            logitsScaling: 1.0, dtype: .f32)
    }

    /// `forward` over a stack containing an MoE-bearing layer must run to
    /// completion without a command-buffer error and produce finite,
    /// correctly-shaped logits. This is the path the dense H-350M
    /// integration test never reaches — and the one that carried the
    /// double-commit bug (the MoE layer committed the caller's `cmd`,
    /// then `forward`'s default wrapper committed it a second time).
    @Test("forward over an MoE-bearing layer completes without a double commit")
    func forwardWithMoELayerCompletes() {
        autoreleasepool {
            // A single MoE-bearing attention layer — `commitsCommandBuffer`
            // is true, so `forward` must keep the caller's `cmd` pristine.
            let layer = attentionLayer(seed: 1, moe: true)
            #expect(layer.commitsCommandBuffer)
            let m = model(layers: [layer])
            #expect(m.hasMoE)

            let caches = m.makeLayerCaches(maxSeq: Self.maxSeq, device: .shared)
            #expect(caches.count == 1)

            // `forward(...)` (no explicit cmd) wraps the primitive in its
            // own command buffer and commits it. Pre-fix, the MoE layer
            // committed that same buffer mid-forward, so this final
            // commit double-committed and Metal raised an error.
            let logits = m.forward(tokenId: 3, position: 0, caches: caches)

            #expect(logits.elementCount == Self.vocab)
            let values = logits.toFloatArray()
            #expect(values.count == Self.vocab)
            // Every logit must be finite — a dead/double-committed buffer
            // leaves the output tensor full of garbage / NaN.
            #expect(values.allSatisfy { $0.isFinite })
            // Logits must not be uniformly zero (the experts + lm_head
            // actually ran on a live buffer).
            #expect(values.contains { $0 != 0 })
        }
    }

    /// A mixed stack — a dense (non-committing) layer FOLLOWED by an
    /// MoE-bearing (committing) layer — exercises the `workCmd` refresh
    /// between layers and the post-loop "commit if the last layer did
    /// not" branch is skipped (last layer committed). Both layers'
    /// outputs feed the residual stream, and the final norm + lm_head
    /// queue onto the caller's pristine `cmd`.
    @Test("forward over a dense-then-MoE stack completes with valid logits")
    func forwardDenseThenMoECompletes() {
        autoreleasepool {
            let dense = attentionLayer(seed: 100, moe: false)
            let moe = attentionLayer(seed: 200, moe: true)
            #expect(!dense.commitsCommandBuffer)
            #expect(moe.commitsCommandBuffer)

            let m = model(layers: [dense, moe])
            #expect(m.hasMoE)

            let caches = m.makeLayerCaches(maxSeq: Self.maxSeq, device: .shared)
            #expect(caches.count == 2)

            let cmd = Device.shared.makeCommandBuffer()
            let logits = m.forward(tokenId: 7, position: 0,
                                   caches: caches, on: cmd, device: .shared)
            // The caller owns `cmd`'s single commit — `forward` must not
            // have committed it (the layers ran on internal buffers).
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)

            #expect(logits.elementCount == Self.vocab)
            let values = logits.toFloatArray()
            #expect(values.allSatisfy { $0.isFinite })
            #expect(values.contains { $0 != 0 })
        }
    }
}
