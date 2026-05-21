// MoELayer — a reusable mixture-of-experts feed-forward block.
//
// An MoE FFN replaces the dense SwiGLU MLP of a transformer layer. The
// substitution is:
//
//   dense:   y = down(silu(gate(x)) * up(x))
//   MoE:     logits = gateLinear(x)                  // [nExperts]
//            (indices, weights) = route(logits)      // top-K
//            y = Σ_{e ∈ indices} weights[e] · expert_e(x)
//            (+ sharedExpert(x) if present)
//
// Each expert is itself a SwiGLU FFN. The router selects the top-K
// experts by gate logit and combines their outputs weighted by the
// (softmax-normalised) gate values. Some architectures additionally run
// an always-on "shared" expert whose output is added unconditionally.
//
// ─── Gating math — the part that breaks silently ─────────────────────
//
// Two checkpoint families disagree on the order of softmax vs top-K:
//
//   .softmaxThenTopK  — Qwen3 / Qwen3.5 MoE. softmax over ALL nExperts
//                       logits first, then pick the top-K of the
//                       softmax probabilities, then (optionally)
//                       re-normalise the K picked probabilities so they
//                       sum to 1.
//   .topKThenSoftmax  — GraniteMoeHybrid. pick the top-K raw logits
//                       first, then softmax over just those K. The
//                       result is always normalised by construction.
//
// `normTopKProb` only applies to `.softmaxThenTopK` (Qwen's
// `norm_topk_prob` config flag). `.topKThenSoftmax` is intrinsically
// normalised so the flag is ignored.
//
// ─── Command-buffer contract ─────────────────────────────────────────
//
// The router needs the gate logits on the CPU to do top-K selection.
// FFAI's decode path batches a whole token's work onto one command
// buffer, so `MoELayer.decode` commits the caller's `cmd`, waits, reads
// the gate logits back, then runs the selected experts on its own
// private command buffer and returns a fully-resident tensor.
//
// A host model that places `MoELayer` in its `[any DecoderLayer]` stack
// MUST therefore obtain a fresh command buffer after the MoE layer's
// `decode` returns — the passed `cmd` has been committed. This mirrors
// the established `InspectTap.dumpLayerBoundary` pattern (commit + wait
// + hand back a new command buffer). See the integration note at the
// bottom of this file.

import Foundation
import Metal

// ─── Gating mode ─────────────────────────────────────────────────────

/// Order in which the router applies softmax relative to top-K
/// selection. The two production checkpoint families disagree; pick the
/// one matching the architecture being ported.
public enum MoEGatingMode: Sendable {
    /// softmax over all experts, then top-K of the probabilities, then
    /// optional re-normalisation (Qwen3 / Qwen3.5 MoE, `norm_topk_prob`).
    case softmaxThenTopK
    /// top-K of the raw logits, then softmax over just those K
    /// (GraniteMoeHybrid). Always normalised.
    case topKThenSoftmax
}

// ─── MoERouter — pure-CPU top-K + combine-weight computation ─────────

/// Stateless router: maps a `[nExperts]` gate-logit vector to the
/// top-K expert indices and their (normalised) combine weights.
///
/// Implemented entirely on the CPU — the gate vector is tiny (nExperts
/// 8–128) and the decode loop already has a sync point per token, so a
/// CPU readback of the gate gemv result is the simplest correct path.
public struct MoERouter: Sendable {
    public let nExperts: Int
    public let topK: Int
    public let gatingMode: MoEGatingMode
    /// Re-normalise the K combine weights to sum to 1. Honoured only for
    /// `.softmaxThenTopK` (`.topKThenSoftmax` is normalised by
    /// construction).
    public let normTopKProb: Bool

    /// Optional per-expert additive bias, applied to the post-softmax
    /// gate values *before* top-K selection. This is LFM2-MoE's
    /// load-balancing `expert_bias`: the biased value steers selection
    /// AND becomes the combine weight. `nil` (the default) leaves every
    /// other MoE family's routing byte-for-byte unchanged. Honoured only
    /// for `.softmaxThenTopK`.
    public let expertBias: [Float]?

    public init(nExperts: Int, topK: Int, gatingMode: MoEGatingMode,
                normTopKProb: Bool = true, expertBias: [Float]? = nil) {
        precondition(nExperts > 0, "MoERouter: nExperts must be positive")
        precondition(topK > 0 && topK <= nExperts,
                     "MoERouter: topK (\(topK)) must be in 1...nExperts (\(nExperts))")
        if let bias = expertBias {
            precondition(bias.count == nExperts,
                         "MoERouter: expertBias has \(bias.count) entries, "
                         + "nExperts is \(nExperts)")
        }
        self.nExperts = nExperts
        self.topK = topK
        self.gatingMode = gatingMode
        self.normTopKProb = normTopKProb
        self.expertBias = expertBias
    }

    /// Routing result: `indices[i]` is the i-th selected expert,
    /// `weights[i]` its combine weight. Both arrays have `topK` entries
    /// and are index-aligned.
    public struct Routing: Equatable, Sendable {
        public let indices: [Int]
        public let weights: [Float]
    }

    /// Compute the top-K expert indices + combine weights for one gate
    /// vector. `logits.count` must equal `nExperts`.
    ///
    /// Tie-breaking on equal logits: the smaller expert index wins —
    /// matches NumPy / PyTorch / MLX `argpartition` + `argmax`
    /// semantics, so output stays deterministic across runs.
    public func route(logits: [Float]) -> Routing {
        precondition(logits.count == nExperts,
                     "MoERouter.route: logits.count \(logits.count) ≠ nExperts \(nExperts)")

        switch gatingMode {
        case .softmaxThenTopK:
            // 1. softmax over ALL experts (numerically stable).
            let probs = Self.softmax(logits)
            // 2. optional per-expert additive bias (LFM2-MoE
            //    `expert_bias`). With no bias `gated == probs`, so the
            //    other softmaxThenTopK families are unaffected.
            let gated: [Float]
            if let bias = expertBias {
                gated = zip(probs, bias).map { $0 + $1 }
            } else {
                gated = probs
            }
            // 3. top-K of the (biased) gate values.
            let idx = Self.topKIndices(gated, k: topK)
            var weights = idx.map { gated[$0] }
            // 4. optional re-normalisation of the K picked weights.
            if normTopKProb {
                let sum = weights.reduce(0, +)
                if sum > 0 { weights = weights.map { $0 / sum } }
            }
            return Routing(indices: idx, weights: weights)

        case .topKThenSoftmax:
            // 1. top-K of the raw logits.
            let idx = Self.topKIndices(logits, k: topK)
            // 2. softmax over just the K picked logits — always
            //    normalised, so `normTopKProb` does not apply.
            let weights = Self.softmax(idx.map { logits[$0] })
            return Routing(indices: idx, weights: weights)
        }
    }

    // ─── CPU helpers ─────────────────────────────────────────────────

    /// Numerically-stable softmax over a small vector.
    static func softmax(_ x: [Float]) -> [Float] {
        guard let maxV = x.max() else { return [] }
        let exps = x.map { Foundation.exp($0 - maxV) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else {
            // Degenerate (all -inf): uniform fallback.
            return [Float](repeating: 1 / Float(x.count), count: x.count)
        }
        return exps.map { $0 / sum }
    }

    /// Indices of the K largest values, ordered largest-first. Ties
    /// resolved by smaller index (matches argpartition semantics).
    static func topKIndices(_ x: [Float], k: Int) -> [Int] {
        // Sort indices by (value desc, index asc). nExperts ≤ 128 so a
        // full sort is cheaper than maintaining a heap.
        let order = (0..<x.count).sorted { a, b in
            if x[a] != x[b] { return x[a] > x[b] }
            return a < b
        }
        return Array(order.prefix(k))
    }
}

// ─── MoELayer — DecoderLayer-conforming MoE feed-forward block ───────

/// A mixture-of-experts feed-forward layer. Holds the gate projection,
/// the per-expert SwiGLU weights, and an optional always-on shared
/// expert. Conforms to `DecoderLayer` so a hybrid model can place it in
/// a heterogeneous `[any DecoderLayer]` stack; it carries no per-token
/// state and is handed a `StatelessLayerCache`.
///
/// `decode` runs: gate gemv → CPU route → per-expert SwiGLU on the
/// selected experts → combine. See the command-buffer contract note in
/// the file header — `decode` commits the passed `cmd`.
public final class MoELayer: Module, DecoderLayer {
    /// Router projection: hidden → nExperts logits.
    public let gate: AnyLinear
    /// Per-expert SwiGLU projections. Each array has `nExperts` entries;
    /// `gateProj[e]` / `upProj[e]` / `downProj[e]` are expert `e`'s
    /// projections.
    public let gateProj: [AnyLinear]
    public let upProj: [AnyLinear]
    public let downProj: [AnyLinear]
    /// Optional always-on shared expert (GraniteMoeHybrid-style). Its
    /// output is added to the routed combination unconditionally.
    public let sharedGateProj: AnyLinear?
    public let sharedUpProj: AnyLinear?
    public let sharedDownProj: AnyLinear?

    public let router: MoERouter
    public let hidden: Int

    /// - gate: hidden → nExperts router projection.
    /// - gateProj/upProj/downProj: `nExperts`-long arrays of per-expert
    ///   SwiGLU projections, index-aligned with the expert id.
    /// - sharedGate/Up/DownProj: optional shared-expert SwiGLU; pass all
    ///   three or none.
    /// - router: the top-K + gating-math configuration.
    public init(gate: AnyLinear,
                gateProj: [AnyLinear], upProj: [AnyLinear], downProj: [AnyLinear],
                sharedGateProj: AnyLinear? = nil,
                sharedUpProj: AnyLinear? = nil,
                sharedDownProj: AnyLinear? = nil,
                router: MoERouter, hidden: Int) {
        precondition(gateProj.count == router.nExperts,
                     "MoELayer: gateProj has \(gateProj.count) experts, router expects \(router.nExperts)")
        precondition(upProj.count == router.nExperts,
                     "MoELayer: upProj has \(upProj.count) experts, router expects \(router.nExperts)")
        precondition(downProj.count == router.nExperts,
                     "MoELayer: downProj has \(downProj.count) experts, router expects \(router.nExperts)")
        let sharedCount = [sharedGateProj, sharedUpProj, sharedDownProj]
            .filter { $0 != nil }.count
        precondition(sharedCount == 0 || sharedCount == 3,
                     "MoELayer: shared expert needs all three of gate/up/down or none")
        self.gate = gate
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.sharedGateProj = sharedGateProj
        self.sharedUpProj = sharedUpProj
        self.sharedDownProj = sharedDownProj
        self.router = router
        self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gate.parameters() { out.append(("gate.\(k)", v)) }
        for (e, proj) in gateProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).gate_proj.\(k)", v)) }
        }
        for (e, proj) in upProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).up_proj.\(k)", v)) }
        }
        for (e, proj) in downProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).down_proj.\(k)", v)) }
        }
        if let sg = sharedGateProj {
            for (k, v) in sg.parameters() { out.append(("shared_expert.gate_proj.\(k)", v)) }
        }
        if let su = sharedUpProj {
            for (k, v) in su.parameters() { out.append(("shared_expert.up_proj.\(k)", v)) }
        }
        if let sd = sharedDownProj {
            for (k, v) in sd.parameters() { out.append(("shared_expert.down_proj.\(k)", v)) }
        }
        return out
    }

    /// `DecoderLayer` conformance. `cache` is a `StatelessLayerCache`
    /// and is ignored — an MoE FFN holds no per-token state.
    ///
    /// IMPORTANT: this commits the passed `cmd` (the router needs the
    /// gate logits on the CPU). The host model must obtain a fresh
    /// command buffer afterwards. See the file header.
    public func decode(_ h: Tensor, position _: Int,
                       cache _: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(h.elementCount == hidden,
                     "MoELayer.decode: input has \(h.elementCount) elements, expected hidden \(hidden)")

        // ── 1. Gate gemv on the caller's command buffer ──────────────
        // Queued onto `cmd` so it runs after whatever produced `h`.
        let logitsTensor = gate(h, on: cmd)

        // ── 2. Commit + wait so the router can read the logits ───────
        // The decode path batches a token onto one command buffer; the
        // router needs a CPU sync point. Commit here, then run the
        // experts on a private buffer below.
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 3. CPU routing — top-K + combine weights ─────────────────
        let logits = logitsTensor.toFloatArray()
        let routing = router.route(logits: logits)

        // ── 4. Per-expert SwiGLU on a private command buffer ─────────
        // Only the selected experts run (the non-selected ones would
        // contribute zero — skipping them is the same result, cheaper).
        let work = device.makeCommandBuffer()
        var accumulator: Tensor?
        for (slot, expertId) in routing.indices.enumerated() {
            // Broadcast the CPU combine weight into a [hidden] constant
            // tensor so the element-wise `Ops.mul` can scale the expert
            // output — avoids a dedicated scalar-multiply kernel.
            let weightTensor = Tensor.filled(routing.weights[slot],
                                             shape: [hidden], dtype: h.dtype,
                                             device: device)
            let expertOut = swiGLU(h,
                                   gateProj: gateProj[expertId],
                                   upProj: upProj[expertId],
                                   downProj: downProj[expertId],
                                   on: work)
            let scaled = Ops.mul(expertOut, weightTensor, on: work)
            accumulator = accumulator.map { Ops.add($0, scaled, on: work) } ?? scaled
        }

        // ── 5. Optional always-on shared expert ──────────────────────
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            let sharedOut = swiGLU(h, gateProj: sg, upProj: su, downProj: sd, on: work)
            accumulator = accumulator.map { Ops.add($0, sharedOut, on: work) } ?? sharedOut
        }

        // topK ≥ 1 so `accumulator` is always non-nil here.
        let result = accumulator!
        work.commit()
        work.waitUntilCompleted()
        return result
    }

    /// One SwiGLU FFN: down(silu(gate(x)) * up(x)).
    private func swiGLU(_ x: Tensor,
                        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
                        on cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(x, on: cmd)
        let u = upProj(x, on: cmd)
        let inner = Ops.mul(Ops.silu(g, on: cmd), u, on: cmd)
        return downProj(inner, on: cmd)
    }
}

// ─── Integration note for MoE-bearing families ───────────────────────
//
// A hybrid family (NemotronH, GraniteMoeHybrid, Jamba, Qwen3.5 MoE,
// Gemma4 MoE) places an `MoELayer` in its `[any DecoderLayer]` stack
// exactly where a dense MLP layer would otherwise sit. To wire one up:
//
//  1. Build the router with the architecture's gating math:
//       let router = MoERouter(nExperts: cfg.numExperts,
//                               topK: cfg.numExpertsPerToken,
//                               gatingMode: .softmaxThenTopK,   // Qwen
//                               normTopKProb: cfg.normTopkProb)
//     Use `.topKThenSoftmax` for GraniteMoeHybrid-style checkpoints.
//
//  2. Expert weights ship stacked as `[nExperts, outDim, inDim]`
//     tensors. Slice per-expert with `Tensor.slicedRows` and wrap:
//       let gateProj = (0..<nExperts).map { e in
//           AnyLinear(Linear(weight:
//               stackedGate.slicedRows(start: e, count: 1)
//                          .reshaped(to: [moeIntermediate, hidden])))
//       }
//     `up_proj` / `down_proj` follow the same shape. For
//     affine-quantised experts use `QuantizedLinear` per slice (the
//     stacked weight / scales / biases each slice along dim 0).
//
//  3. Optionally build the shared expert (GraniteMoeHybrid). Pass all
//     three shared projections to the `MoELayer` initialiser, or none.
//
//  4. The MoE layer's residual add + pre-norm stay in the host layer
//     (same as the dense MLP): `h + moeLayer.decode(postAttnNorm(h)…)`.
//
//  5. CRITICAL — `MoELayer.decode` commits the command buffer it is
//     given. The host model's decode loop must refresh `workCmd` after
//     the call, e.g.:
//       h = moeLayer.decode(h, position: pos, cache: caches[i],
//                           cmd: workCmd, device: device)
//       workCmd = device.makeCommandBuffer()   // ← fresh buffer
//     This is the same commit-and-replace shape `InspectTap` uses for
//     layer-boundary dumps.
