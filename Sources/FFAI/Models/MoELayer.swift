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

    public init(nExperts: Int, topK: Int, gatingMode: MoEGatingMode,
                normTopKProb: Bool = true) {
        precondition(nExperts > 0, "MoERouter: nExperts must be positive")
        precondition(topK > 0 && topK <= nExperts,
                     "MoERouter: topK (\(topK)) must be in 1...nExperts (\(nExperts))")
        self.nExperts = nExperts
        self.topK = topK
        self.gatingMode = gatingMode
        self.normTopKProb = normTopKProb
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
            // 2. top-K of the softmax probabilities.
            let idx = Self.topKIndices(probs, k: topK)
            var weights = idx.map { probs[$0] }
            // 3. optional re-normalisation of the K picked probs.
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
        // Partial-sort: maintain a k-sized min-heap of (value, index)
        // pairs, ordered by (value asc, index desc) so the smallest
        // value with the largest index is at the top (i.e. easiest to
        // evict). At the end, drain the heap and reverse — gives the
        // top-K ordered by (value desc, index asc), matching the prior
        // full-sort semantics.
        //
        // Complexity: O(n log k) vs the prior `Array.sorted` O(n log n).
        // At Qwen3.6-A3B (nExperts=256, k=8) this is ~280 ops vs ~2048
        // ops per call → ~7× host CPU per route × 40 MoE layers per
        // decode token ≈ 5-7 ms / token saved. Stale comment claimed
        // "nExperts ≤ 128 so a full sort is cheaper" — at 256 it
        // already wasn't, and partial-sort is the right structure
        // regardless.
        precondition(k > 0 && k <= x.count, "topKIndices: k must be in 1...n")
        var heap: [(value: Float, index: Int)] = []
        heap.reserveCapacity(k)
        // Min-heap ordering: smallest value with largest index at root.
        // Returns true when `a` is "smaller" (closer to root / first to
        // evict). For tie on value, the entry with the LARGER index is
        // smaller (we want smaller indices to win ties on the final
        // sort, so the larger-index entry is the easier evict).
        @inline(__always) func less(_ a: (Float, Int), _ b: (Float, Int)) -> Bool {
            if a.0 != b.0 { return a.0 < b.0 }
            return a.1 > b.1
        }
        @inline(__always) func siftUp(_ start: Int) {
            var i = start
            while i > 0 {
                let parent = (i - 1) >> 1
                if less(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    i = parent
                } else { return }
            }
        }
        @inline(__always) func siftDown(_ start: Int) {
            var i = start
            let n = heap.count
            while true {
                let l = 2 * i + 1
                let r = 2 * i + 2
                var smallest = i
                if l < n && less(heap[l], heap[smallest]) { smallest = l }
                if r < n && less(heap[r], heap[smallest]) { smallest = r }
                if smallest == i { return }
                heap.swapAt(i, smallest)
                i = smallest
            }
        }
        for i in 0..<x.count {
            let entry = (x[i], i)
            if heap.count < k {
                heap.append(entry)
                siftUp(heap.count - 1)
            } else if less(heap[0], entry) {
                // Heap root is the easiest evict; the new entry beats
                // it on (value desc, index asc).
                heap[0] = entry
                siftDown(0)
            }
        }
        // Drain — `heap` is min-ordered, but the public contract returns
        // largest-first (`value desc, index asc`). Sort the final k
        // entries by the same (value desc, index asc) criterion.
        heap.sort { a, b in
            if a.value != b.value { return a.value > b.value }
            return a.index < b.index
        }
        return heap.map { $0.index }
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

    /// Stacked-expert weight handles for the batched gather BGEMM fast
    /// path. When set, `decode` dispatches `mt_moe_gather_qmm_mma_int4_bm16`
    /// instead of running `topK` sequential per-expert SwiGLU triplets —
    /// 3 kernel launches per token instead of `3 * topK` (24 → 3 at
    /// `topK=8`). Populated by the host model's MoE builder when (a) the
    /// checkpoint is mlx int4-quantized, (b) the stacked weight tensors
    /// are intact (not sliced away by the per-expert wrapper path), and
    /// (c) the moeIntermediate / hidden shapes satisfy the bm16 kernel's
    /// `N % 32 == 0` and `K % 32 == 0` tile contract. Left nil for
    /// everyone else — the legacy serial-expert loop still runs.
    public struct StackedInt4Experts: Sendable {
        /// `[numExperts, moeIntermediate, hidden/8]` u32 packed.
        public let gateWeight: Tensor
        /// `[numExperts, moeIntermediate, hidden/groupSize]` in `dtype`.
        public let gateScales: Tensor
        public let gateBiases: Tensor
        /// `[numExperts, moeIntermediate, hidden/8]` u32 packed.
        public let upWeight: Tensor
        public let upScales: Tensor
        public let upBiases: Tensor
        /// `[numExperts, hidden, moeIntermediate/8]` u32 packed.
        public let downWeight: Tensor
        public let downScales: Tensor
        public let downBiases: Tensor
        public let numExperts: Int
        public let moeIntermediate: Int
        public let hidden: Int
        public let groupSize: Int
        /// Activation dtype the scales / biases / activations all use.
        public let dtype: DType

        public init(gateWeight: Tensor, gateScales: Tensor, gateBiases: Tensor,
                    upWeight: Tensor, upScales: Tensor, upBiases: Tensor,
                    downWeight: Tensor, downScales: Tensor, downBiases: Tensor,
                    numExperts: Int, moeIntermediate: Int, hidden: Int,
                    groupSize: Int, dtype: DType) {
            self.gateWeight = gateWeight
            self.gateScales = gateScales
            self.gateBiases = gateBiases
            self.upWeight = upWeight
            self.upScales = upScales
            self.upBiases = upBiases
            self.downWeight = downWeight
            self.downScales = downScales
            self.downBiases = downBiases
            self.numExperts = numExperts
            self.moeIntermediate = moeIntermediate
            self.hidden = hidden
            self.groupSize = groupSize
            self.dtype = dtype
        }
    }
    public let stackedInt4Experts: StackedInt4Experts?

    /// - gate: hidden → nExperts router projection.
    /// - gateProj/upProj/downProj: `nExperts`-long arrays of per-expert
    ///   SwiGLU projections, index-aligned with the expert id.
    /// - sharedGate/Up/DownProj: optional shared-expert SwiGLU; pass all
    ///   three or none.
    /// - router: the top-K + gating-math configuration.
    /// - stackedInt4Experts: optional batched-BGEMM fast path. Per-expert
    ///   arrays above remain the source of truth for `parameters()`
    ///   (checkpoint binding) and the fallback decode path.
    public init(gate: AnyLinear,
                gateProj: [AnyLinear], upProj: [AnyLinear], downProj: [AnyLinear],
                sharedGateProj: AnyLinear? = nil,
                sharedUpProj: AnyLinear? = nil,
                sharedDownProj: AnyLinear? = nil,
                router: MoERouter, hidden: Int,
                stackedInt4Experts: StackedInt4Experts? = nil) {
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
        if let s = stackedInt4Experts {
            precondition(s.numExperts == router.nExperts,
                         "MoELayer: stackedInt4Experts.numExperts \(s.numExperts) ≠ router.nExperts \(router.nExperts)")
            precondition(s.hidden == hidden,
                         "MoELayer: stackedInt4Experts.hidden \(s.hidden) ≠ MoELayer.hidden \(hidden)")
            precondition(s.moeIntermediate % 32 == 0 && hidden % 32 == 0,
                         "MoELayer: stackedInt4Experts shape (moeIntermediate=\(s.moeIntermediate), hidden=\(hidden)) violates bm16 N%32 / K%32 tile contract")
        }
        self.stackedInt4Experts = stackedInt4Experts
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
        // Batched gather-BGEMM fast path is opt-in via FFAI_MOE_BGEMM=1.
        // At T=1 decode with topK=8 / m_total=8 it currently regresses
        // vs the sequential per-expert matvec path on M5 Max — the
        // bm16 tile pads to 16 rows but only 8 commit, so the kernel
        // pays full weight-reload bandwidth for 50% useful work. Real
        // win shape is prefill (m_total scales with T, more rows share
        // each expert's weight tile) or NAX (FFAI_MOE_BGEMM_MPP=1).
        // Path is wired + correctness-verified — flip the default once
        // the m_total<16 regression is closed (likely via either a
        // dedicated bm8 kernel emit or by promoting prefill to use
        // this path first).
        let enableBGEMM = ProcessInfo.processInfo.environment["FFAI_MOE_BGEMM"] != nil
        if let stacked = stackedInt4Experts, stacked.dtype == h.dtype, enableBGEMM {
            // Fast path: one batched gather BGEMM per projection. The
            // kernel expects rows of activations sorted by expert id; we
            // sort the topK indices ascending and replicate `h` into the
            // gathered row order. The per-row expert assignment goes in
            // an int32 indices buffer.
            accumulator = batchedSwiGLU(h, stacked: stacked, routing: routing,
                                        on: work, device: device)
        } else {
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
        }

        // ── 5. Optional always-on shared expert ──────────────────────
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            let sharedOut = swiGLU(h, gateProj: sg, upProj: su, downProj: sd, on: work)
            accumulator = accumulator.map { Ops.add($0, sharedOut, on: work) } ?? sharedOut
        }

        // topK ≥ 1 so `accumulator` is always non-nil here.
        let result = accumulator!
        // Commit without wait: `result` is on the in-flight `work`
        // buffer; the caller's residual-add cmd will hazard-track the
        // read against this write. Saves ~0.5-1 ms host-stall per
        // MoE layer per token (Qwen3.6-A3B = 40 layers).
        work.commit()
        return result
    }

    /// Batched gather BGEMM fast path. T=1 decode: `mTotal = topK`. Each
    /// row of the gathered batch holds the same `[hidden]` activation
    /// vector but is paired with a different expert id via `indices`.
    /// The kernel walks rows in tile order — boundary masking lets it
    /// process the trailing partial tile when `mTotal < BM`. One
    /// dispatch per projection replaces `topK` sequential per-expert
    /// SwiGLU triplets:
    ///   24 (topK=8) dispatches → 3 dispatches (gate / up / down) plus
    ///   element-wise SiLU + mul + scaled scatter-sum (~6 small ops).
    /// Outputs are scaled by the router's combine weights on the CPU
    /// (the same `Tensor.filled([hidden])` trick the serial path uses,
    /// but only once per topK slot — still tiny vs the BGEMM cost).
    ///
    /// Tile selection:
    ///   - `FFAI_MOE_BGEMM_BM8=1` + `topK ≤ 8` → bm8 MPP kernel
    ///     (BM=8 fills the tile exactly at Qwen3.6-A3B topK=8 decode;
    ///     bm16 would waste 50 % of the trailing tile rows).
    ///     Requires Apple10+ GPU (M5 Max) + macOS 26.2+.
    ///   - otherwise → bm16 path (Ops.moeGatherDequantGemmInt4),
    ///     which itself respects `FFAI_MOE_BGEMM_MPP=1` for the
    ///     MPP/NAX variant at larger m_total.
    private func batchedSwiGLU(_ h: Tensor,
                               stacked: StackedInt4Experts,
                               routing: MoERouter.Routing,
                               on cmd: MTLCommandBuffer,
                               device: Device) -> Tensor {
        let topK = routing.indices.count
        let moeIntermediate = stacked.moeIntermediate
        let groupSize = stacked.groupSize
        let dtype = stacked.dtype
        let useBm8 = topK <= 8
            && ProcessInfo.processInfo.environment["FFAI_MOE_BGEMM_BM8"] != nil
        // `FFAI_MOE_M1=1` routes the gather through the scalar m1
        // kernel `mt_moe_gather_qmm_int4` — no MPP / cooperative-tensor
        // overhead. At decode T=1 the cooperative-tensor variants
        // (bm8 / bm16 / bm64) lose ~15 % to descriptor setup cost; the
        // m1 path does a one-output-cell-per-TG dot-product with
        // `simd_sum` reduction and is closer in shape to the production
        // per-expert dequant_gemv path.
        let useM1 = !useBm8
            && ProcessInfo.processInfo.environment["FFAI_MOE_M1"] != nil

        // ── Sort topK by expert id ascending (kernel contract) ──
        // Track the original slot so we can apply the right combine
        // weight after the BGEMM. The slot mapping never goes through
        // the GPU, so sorting on the host is cheap.
        let sorted = routing.indices.enumerated().sorted { $0.element < $1.element }
        let sortedExperts = sorted.map { UInt32($0.element) }
        let sortedSlots = sorted.map { $0.offset }

        // ── Materialise the [topK, hidden] activation gather + indices ──
        let xGathered = Tensor.empty(shape: [topK, hidden], dtype: dtype, device: device)
        let inputBytes = hidden * dtype.byteSize
        // h is the same vector for every row at T=1 decode. memcpy from
        // host backing — Tensors here are storage-shared.
        let src = h.buffer.contents().advanced(by: h.offset)
        let dst = xGathered.buffer.contents().advanced(by: xGathered.offset)
        for r in 0..<topK {
            dst.advanced(by: r * inputBytes)
                .copyMemory(from: src, byteCount: inputBytes)
        }
        let indices = Tensor.empty(shape: [topK], dtype: .u32, device: device)
        indices.copyIn(from: sortedExperts)

        // For the m1 path we also need CSR `expert_offsets`. Build on
        // host (cheap: nExperts ≤ 256). `expert_offsets[e]` = first row
        // assigned to expert e (or t_rows if expert e is unselected).
        // The kernel does a linear walk over this on each lane.
        var expertOffsetsT: Tensor? = nil
        if useM1 {
            let nExperts = stacked.numExperts
            var csrHost = [UInt32](repeating: UInt32(topK), count: nExperts + 1)
            var firstRowForExpert = [Int](repeating: topK, count: nExperts)
            for (row, e) in sortedExperts.enumerated() {
                let ei = Int(e)
                if firstRowForExpert[ei] == topK { firstRowForExpert[ei] = row }
            }
            var minOffset = topK
            for e in (0...nExperts).reversed() {
                if e < nExperts && firstRowForExpert[e] < minOffset {
                    minOffset = firstRowForExpert[e]
                }
                csrHost[e] = UInt32(minOffset)
            }
            let offsetsT = Tensor.empty(shape: [nExperts + 1], dtype: .u32, device: device)
            offsetsT.copyIn(from: csrHost)
            expertOffsetsT = offsetsT
        }

        // ── Gate / up projections: [topK, moeIntermediate] ──
        let gateOut = Tensor.empty(shape: [topK, moeIntermediate], dtype: dtype, device: device)
        let upOut = Tensor.empty(shape: [topK, moeIntermediate], dtype: dtype, device: device)

        if let offsetsT = expertOffsetsT {
            let nExperts = stacked.numExperts
            Ops.moeGatherDequantGemmInt4M1(
                xGathered,
                stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
                offsetsT,
                topK, moeIntermediate, hidden,
                nExperts, groupSize, cmd, gateOut)
            Ops.moeGatherDequantGemmInt4M1(
                xGathered,
                stacked.upWeight, stacked.upScales, stacked.upBiases,
                offsetsT,
                topK, moeIntermediate, hidden,
                nExperts, groupSize, cmd, upOut)
        } else {
            let bgemm = useBm8 ? Ops.moeGatherDequantGemmInt4Bm8 : Ops.moeGatherDequantGemmInt4
            bgemm(
                xGathered,
                stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
                indices,
                topK, moeIntermediate, hidden,
                groupSize, cmd, gateOut)
            bgemm(
                xGathered,
                stacked.upWeight, stacked.upScales, stacked.upBiases,
                indices,
                topK, moeIntermediate, hidden,
                groupSize, cmd, upOut)
        }

        // ── SwiGLU activation: silu(gate) * up — fused into one dispatch ──
        let inner = Ops.swiglu(gate: gateOut, up: upOut, on: cmd)

        // ── Down projection: [topK, hidden] ──
        let downOut = Tensor.empty(shape: [topK, hidden], dtype: dtype, device: device)
        if let offsetsT = expertOffsetsT {
            let nExperts = stacked.numExperts
            Ops.moeGatherDequantGemmInt4M1(
                inner,
                stacked.downWeight, stacked.downScales, stacked.downBiases,
                offsetsT,
                topK, hidden, moeIntermediate,
                nExperts, groupSize, cmd, downOut)
        } else {
            let bgemm = useBm8 ? Ops.moeGatherDequantGemmInt4Bm8 : Ops.moeGatherDequantGemmInt4
            bgemm(
                inner,
                stacked.downWeight, stacked.downScales, stacked.downBiases,
                indices,
                topK, hidden, moeIntermediate,
                groupSize, cmd, downOut)
        }

        // ── Weighted scatter-sum back to a single [hidden] vector ──
        // For each routed slot, scale the corresponding row of downOut
        // by `routing.weights[slot]` and accumulate. The Tensor.filled
        // broadcast scalar trick mirrors the serial path's per-slot
        // weight application.
        var acc: Tensor?
        for (sortedIdx, originalSlot) in sortedSlots.enumerated() {
            // Row view into downOut at the sorted position.
            let row = Tensor(
                buffer: downOut.buffer,
                offset: downOut.offset + sortedIdx * hidden * dtype.byteSize,
                shape: [hidden], dtype: dtype)
            let w = Tensor.filled(routing.weights[originalSlot],
                                  shape: [hidden], dtype: dtype, device: device)
            let scaled = Ops.mul(row, w, on: cmd)
            acc = acc.map { Ops.add($0, scaled, on: cmd) } ?? scaled
        }
        return acc!
    }

    /// One SwiGLU FFN: down(silu(gate(x)) * up(x)).
    ///
    /// `silu(gate) * up` runs as a single fused `mt_swiglu` dispatch
    /// instead of two launches (silu → mul). Saves one elementwise
    /// kernel + one full-tensor RMW per call. At Qwen3.6-A3B decode T=1
    /// the per-expert intermediate is [moeIntermediate=768] so the win
    /// is small per dispatch, but the loop runs 8 experts × ≤40 MoE
    /// layers per token, so the per-token saving compounds.
    private func swiGLU(_ x: Tensor,
                        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
                        on cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(x, on: cmd)
        let u = upProj(x, on: cmd)
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
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
