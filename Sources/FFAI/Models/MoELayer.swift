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
import MetalTileSwift

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
        case .softmaxThenTopK where normTopKProb && expertBias == nil:
            // Fast path: `softmaxThenTopK + normTopKProb=true` produces
            // the same weights as `.topKThenSoftmax`. Proof:
            //   softmax(L)[i] = exp(L[i] - M) / Z, M = max(L), Z = Σ
            //   top-K of softmax probs by value desc = top-K of raw
            //     logits (softmax is monotonic on selection)
            //   weights_pre[k] = exp(L[idx_k] - M) / Z
            //   renorm: weights[k] = weights_pre[k] / Σ_k weights_pre
            //     = exp(L[idx_k]) / Σ_k exp(L[idx_k])
            //     = softmax over just the K picked logits
            // Skips the full softmax over `nExperts=256` (~500 ops
            // including 256 exp + sum reduction + 256 div). Saves
            // ~40 K host ops / token on Qwen3.6-A3B (40 MoE layers ×
            // ~1000 ops per route saved). Qwen3 / Qwen3.5 / Qwen3.6 MoE
            // all hit this branch (`norm_topk_prob: true` in every
            // shipped config).
            //
            // REQUIRES `expertBias == nil`: the proof's "top-K of raw
            // logits == top-K of softmax probs" step breaks once an
            // additive per-expert bias is applied AFTER softmax (the
            // bias is not monotonic in the raw logit). LFM2-MoE supplies
            // an `expert_bias`, so it falls through to the general
            // `.softmaxThenTopK` path below, which selects on
            // `softmax(logits) + bias`.
            let idx = Self.topKIndices(logits, k: topK)
            let weights = Self.softmax(idx.map { logits[$0] })
            return Routing(indices: idx, weights: weights)

        case .softmaxThenTopK:
            // `normTopKProb=false` case — semantically distinct
            // (unnormalised softmax probs as weights). Must walk the
            // full softmax over all nExperts.
            let probs = Self.softmax(logits)
            // Optional per-expert additive bias (LFM2-MoE `expert_bias`).
            // With no bias `gated == probs`, so other softmaxThenTopK
            // families are byte-identical.
            let gated: [Float]
            if let bias = expertBias {
                gated = zip(probs, bias).map { $0 + $1 }
            } else {
                gated = probs
            }
            let idx = Self.topKIndices(gated, k: topK)
            var weights = idx.map { gated[$0] }
            // Optional re-normalisation of the K picked weights.
            if normTopKProb {
                let sum = weights.reduce(0, +)
                if sum > 0 { weights = weights.map { $0 / sum } }
            }
            return Routing(indices: idx, weights: weights)

        case .topKThenSoftmax:
            // Top-K of raw logits, softmax over just the K picked —
            // always normalised by construction so `normTopKProb`
            // doesn't apply.
            let idx = Self.topKIndices(logits, k: topK)
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

    /// Env-flag cache — read once at init so the decode loop doesn't
    /// pay 3 × `ProcessInfo.processInfo.environment[...]` dictionary
    /// lookups per MoE layer per token. At Qwen3.6-A3B that's 120
    /// lookups / token saved (40 layers × 3 envs). Each lookup is
    /// ~100 ns on Foundation, so the total is sub-noise (~12 µs / token)
    /// — shipped for cleanliness more than the µbench delta.
    public let enableBGEMM: Bool
    public let useBm8Env: Bool
    public let useM1Env: Bool

    /// Lazy cache of the dequantized gate weight in `[outDim=nExperts,
    /// inDim=hidden]` layout — what `Ops.gemm` expects for its `weight`
    /// argument (`ffai_gemm` computes `out = input @ weight^T`). For
    /// 8-bit gates, `QuantizedLinear.callMany` falls into a T-loop of
    /// per-row `dequantGemv` (T·40 = 20480 launches at T=512 prefill);
    /// the dense path collapses each layer's gate to one `Ops.gemm`
    /// dispatch regardless of T (no T % 64 alignment constraint).
    /// Skipped when:
    ///   • `gate.inner` is not `QuantizedLinear`, or
    ///   • bits != 8, or
    ///   • env `FFAI_MOE_NO_DEQUANT_GATE=1`.
    /// Materialised lazily on first `gateLogitsMany` to avoid eager work
    /// in layers that never see a forward pass.
    private var dequantizedGateCache: Tensor?
    /// Sibling cache in `[hidden, nExperts]` (K, N) layout for the
    /// `mt_steel_gemm_64x64x16_2x2_*` fast path. Engaged when T and
    /// nExperts both divide 64 — covers the production prefill shape
    /// (T=64, 128, 512; nExperts=128).
    private var dequantizedGateCacheKN: Tensor?
    private var dequantizedGateAttempted = false

    /// Cached scratch buffers for the per-expert weighted-sum loop at
    /// decode T=1. Reused across decode calls — avoids 1 host alloc +
    /// 1 host memset per layer per token (~10 µs × 40 layers = 400 µs
    /// per Qwen3.6-A3B decode token).
    /// * `accumulatorScratch`: [hidden] tensor; zeroed at the start of
    ///   each decode call, then accumulated into via `Ops.scalarFMA`.
    /// * `topKScalarsBuf`: small [topK × dtype.byteSize] MTLBuffer
    ///   holding the 8 routing weights packed in the model dtype.
    private var accumulatorScratch: Tensor?
    private var topKScalarsBuf: MTLBuffer?
    /// ITER 32 (post-OOM): per-expert g/u scratches cached at init.
    /// Replaces the per-call `Tensor.empty([moeIntermediate])` in the
    /// swiGLU helper that caused 640 fresh tensor allocations per
    /// Qwen3.6-A3B decode token (8 experts × 40 layers × 2 = 640).
    /// Each [moeIntermediate] bf16 = 2 KiB; 16 KiB per layer total.
    /// Safe across decode tokens because the caller commits + waits
    /// the layer's cmd before the next forward starts.
    /// Indexed by slot (0..topK-1).
    private var expertGScratches: [Tensor] = []
    private var expertUScratches: [Tensor] = []
    /// ITER 36: cache swiglu inner output per expert slot.
    private var expertInnerScratches: [Tensor] = []
    /// ITER 38: cached per-expert down output for the batched 8-expert
    /// down qmm path. Indexed by slot.
    private var expertOutScratches: [Tensor] = []
    /// ITER 56 (Bagel 2): GPU MoE router scratch buffers. `routerIndicesScratch`
    /// is `[topK]` u32; `routerWeightsScratch` is `[topK]` in `h.dtype`. Both
    /// written by `mt_moe_router_topk` once per MoE layer per token, then
    /// consumed by `dequantGemvInt4ExpertIndexed` (per-slot expert id) and
    /// `scalarFMAChain8` (per-slot routing weight) — eliminates the gate→
    /// experts CPU sync that the legacy `cmd.commit + waitUntilCompleted`
    /// path needed.
    private var routerIndicesScratch: Tensor?
    private var routerWeightsScratch: Tensor?
    /// ITER 42: cached output of dense gate gemv (ITER 15).
    private var gateLogitsScratch: Tensor?

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
        let env = ProcessInfo.processInfo.environment
        self.enableBGEMM = env["FFAI_MOE_BGEMM"] != nil
        self.useBm8Env = env["FFAI_MOE_BGEMM_BM8"] != nil
        self.useM1Env = env["FFAI_MOE_M1"] != nil
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
        // ITER 15: if we've already dequantized the gate weight (via a
        // prior prefill call to gateLogitsMany), use the dense fp16
        // gemv path — bypasses the slow 8-bit dequant-gemv at T=1.
        // For pure decode without prefill, this opportunistically
        // dequantizes once per MoE layer (40 layers × hidden=2048 ×
        // nExperts=128 × 2 bytes = ~21 MB total — done once per
        // layer's first decode call).
        let logitsTensor: Tensor
        if !dequantizedGateAttempted {
            dequantizedGateAttempted = true
            if ProcessInfo.processInfo.environment["FFAI_MOE_NO_DEQUANT_GATE"] == nil,
               let q = gate.inner as? QuantizedLinear,
               q.bits == 8 {
                dequantizedGateCache = Self.dequantizeQuantizedLinear(q, device: device)
            }
        }
        if let dense = dequantizedGateCache {
            if gateLogitsScratch == nil
                || gateLogitsScratch!.elementCount != router.nExperts
                || gateLogitsScratch!.dtype != h.dtype {
                gateLogitsScratch = Tensor.empty(shape: [router.nExperts], dtype: h.dtype)
            }
            logitsTensor = Ops.gemv(weight: dense, input: h, on: cmd,
                                     into: gateLogitsScratch)
        } else {
            logitsTensor = gate(h, on: cmd)
        }

        // ── 2a. ITER 56 (Bagel 2): GPU MoE router fast path ──────────
        // When `FFAI_MOE_GPU_ROUTER=1` is set AND the layer has a stacked
        // int4 expert layout AND routing matches `mt_moe_router_topk`'s
        // assumptions (softmax-then-topK + Qwen-MoE-style renorm, topK=8),
        // skip the CPU sync entirely. The router writes topK indices +
        // weights to a GPU buffer; per-expert qmms read their slot's
        // expert id from that buffer via `dequantGemvInt4ExpertIndexed`;
        // the per-slot routing weight feeds `scalarFMAChain8`.
        // Net: ~40 × `waitUntilCompleted` per token → 0.
        // OPT-IN by default (env flag) until A/B-bench confirmed.
        let gpuRouterEnabled = ProcessInfo.processInfo.environment["FFAI_MOE_GPU_ROUTER"] == "1"
        if gpuRouterEnabled,
           let stacked = stackedInt4Experts,
           stacked.dtype == h.dtype,
           router.gatingMode == .softmaxThenTopK,
           router.normTopKProb,
           router.expertBias == nil,
           router.topK == 8
        {
            // Lazy-init per-instance scratches.
            if routerIndicesScratch == nil
                || routerIndicesScratch!.elementCount != router.topK {
                routerIndicesScratch = Tensor.empty(shape: [router.topK], dtype: .u32)
                routerWeightsScratch = Tensor.empty(shape: [router.topK], dtype: h.dtype)
            }
            if accumulatorScratch == nil
                || accumulatorScratch!.dtype != h.dtype
                || accumulatorScratch!.elementCount != hidden {
                accumulatorScratch = Tensor.empty(shape: [hidden], dtype: h.dtype,
                                                   device: device)
            }
            let outDim = stacked.moeIntermediate
            for slot in 0..<router.topK {
                while expertGScratches.count <= slot {
                    expertGScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                    expertUScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                    expertInnerScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                    expertOutScratches.append(Tensor.empty(shape: [hidden], dtype: h.dtype))
                }
            }

            // GPU router: logits → topK indices + weights (still on cmd).
            Ops.moeRouterTopK(
                logits: logitsTensor,
                indicesOut: routerIndicesScratch!,
                weightsOut: routerWeightsScratch!,
                nExperts: router.nExperts, k: router.topK,
                normTopkProb: router.normTopKProb,
                on: cmd)

            // 8 × gate qmm + 8 × up qmm, each reading expert id from the
            // router's GPU-resident indices buffer at `slot·4` offset.
            for slot in 0..<router.topK {
                let expertIdx = Tensor(buffer: routerIndicesScratch!.buffer,
                                       offset: routerIndicesScratch!.offset + slot * 4,
                                       shape: [1], dtype: .u32)
                Ops.dequantGemvInt4ExpertIndexed(
                    weightsStacked: stacked.gateWeight,
                    scalesStacked: stacked.gateScales,
                    biasesStacked: stacked.gateBiases,
                    input: h, expertIndex: expertIdx,
                    groupSize: stacked.groupSize,
                    on: cmd, into: expertGScratches[slot])
                Ops.dequantGemvInt4ExpertIndexed(
                    weightsStacked: stacked.upWeight,
                    scalesStacked: stacked.upScales,
                    biasesStacked: stacked.upBiases,
                    input: h, expertIndex: expertIdx,
                    groupSize: stacked.groupSize,
                    on: cmd, into: expertUScratches[slot])
            }

            // Batched 8-slot SwiGLU on shared encoder (ITER 39 pattern).
            let gs = Array(expertGScratches.prefix(router.topK))
            let us = Array(expertUScratches.prefix(router.topK))
            let inners = Array(expertInnerScratches.prefix(router.topK))
            Ops.swigluMany(gates: gs, ups: us, outs: inners, on: cmd)

            // 8 × down qmm, same indexed-expert pattern as gate/up.
            for slot in 0..<router.topK {
                let expertIdx = Tensor(buffer: routerIndicesScratch!.buffer,
                                       offset: routerIndicesScratch!.offset + slot * 4,
                                       shape: [1], dtype: .u32)
                Ops.dequantGemvInt4ExpertIndexed(
                    weightsStacked: stacked.downWeight,
                    scalesStacked: stacked.downScales,
                    biasesStacked: stacked.downBiases,
                    input: expertInnerScratches[slot], expertIndex: expertIdx,
                    groupSize: stacked.groupSize,
                    on: cmd, into: expertOutScratches[slot])
            }

            // Phase 3 chain8 — fused 8-way weighted accumulator reading
            // per-slot scalar from router's GPU weights buffer. Writes
            // acc directly (no read), so no zero-pass needed.
            var scalars: [Tensor] = []
            scalars.reserveCapacity(router.topK)
            for slot in 0..<router.topK {
                scalars.append(Tensor(buffer: routerWeightsScratch!.buffer,
                                       offset: routerWeightsScratch!.offset + slot * h.dtype.byteSize,
                                       shape: [1], dtype: h.dtype))
            }
            let outs = Array(expertOutScratches.prefix(router.topK))
            let acc = accumulatorScratch!
            Ops.scalarFMAChain8(scalars: scalars, values: outs, out: acc, on: cmd)

            // Commit cmd without wait — preserves the caller contract
            // (caller starts a fresh cmd for residual-add / shared
            // expert). The win is the ABSENT `waitUntilCompleted` that
            // the CPU-sync path needed before this branch.
            cmd.commit()
            return acc
        }

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
        // At T=1 decode with topK=8 / m_total=8 it MEASURABLY REGRESSES
        // vs the sequential per-expert matvec path on M5 Max — the
        // bm16 tile pads to 16 rows but only 8 commit, so the kernel
        // pays full weight-reload bandwidth for 50% useful work. Bench
        // shows ~2.1× slowdown at decode T=1 on Qwen3.6-A3B
        // (Iter 9: 20.34 tps with BGEMM → 43.20 tps without).
        //
        // The env flag stays opt-in for the prefill path (decodeMany)
        // where mTotal scales with T and BGEMM wins, but at T=1 we
        // hard-disable to avoid the regression even if the env is set.
        // Override: `FFAI_MOE_BGEMM_FORCE_T1=1` re-enables for the rare
        // case someone wants to bench the regression directly.
        let forceBGEMMAtT1 = ProcessInfo.processInfo.environment["FFAI_MOE_BGEMM_FORCE_T1"] != nil
        let useBGEMMAtT1 = enableBGEMM && forceBGEMMAtT1
        if let stacked = stackedInt4Experts, stacked.dtype == h.dtype, useBGEMMAtT1 {
            // Fast path: one batched gather BGEMM per projection. The
            // kernel expects rows of activations sorted by expert id; we
            // sort the topK indices ascending and replicate `h` into the
            // gathered row order. The per-row expert assignment goes in
            // an int32 indices buffer.
            accumulator = batchedSwiGLU(h, stacked: stacked, routing: routing,
                                        on: work, device: device)
        } else {
            // Pack the topK routing weights into a single [topK] scalar
            // buffer once, then dispatch one `scalar_fma` per expert
            // instead of the old `Tensor.filled([hidden]) + Ops.mul +
            // Ops.add` chain. Saves a host alloc + 2 dispatches per
            // expert; at Qwen3.6-A3B topK=8 × 40 layers that's 320
            // allocations + 640 dispatches per decode token.
            //
            // First iteration: bootstrap the accumulator by allocating
            // [hidden] zeros so scalar_fma can read it as base. The
            // zero buffer itself is cheap (1 dispatch on a small tensor
            // OR one Tensor.empty + .zero); we hold it until the loop
            // is done. Subsequent iterations alias `accumulator` as
            // both base and out (safe — the kernel reads at idx then
            // writes at idx).
            // Reuse cached scratch slots — see field comment on
            // `accumulatorScratch` + `topKScalarsBuf`.
            let scalarBufBytes = routing.weights.count * h.dtype.byteSize
            if topKScalarsBuf == nil || topKScalarsBuf!.length < scalarBufBytes {
                topKScalarsBuf = device.makeBuffer(length: scalarBufBytes)
            }
            Self.writeTopKScalars(routing.weights, dtype: h.dtype,
                                   into: topKScalarsBuf!)
            let scalarBuf = topKScalarsBuf!

            // Lazy-init accumulator scratch.
            if accumulatorScratch == nil
                || accumulatorScratch!.dtype != h.dtype
                || accumulatorScratch!.elementCount != hidden {
                accumulatorScratch = Tensor.empty(shape: [hidden], dtype: h.dtype,
                                                   device: device)
            }
            let acc = accumulatorScratch!

            // ITER 38: phase-2 batched down qmm. When all 8 expert
            // downProjs are int4 QuantizedLinear with same groupSize,
            // dispatch all 8 down qmms in ONE encoder. Uses cached
            // inner + out scratches (no allocation per call).
            var canBatchDown = true
            var downWeights: [Tensor] = []
            var downScales: [Tensor] = []
            var downBiases: [Tensor] = []
            var downGroupSize: Int = 0
            for expertId in routing.indices {
                if let q = downProj[expertId].inner as? QuantizedLinear, q.bits == 4 {
                    if downGroupSize == 0 { downGroupSize = q.groupSize }
                    if q.groupSize != downGroupSize { canBatchDown = false; break }
                    downWeights.append(q.weight)
                    downScales.append(q.scales)
                    downBiases.append(q.biases)
                } else {
                    canBatchDown = false
                    break
                }
            }

            // ITER 47 (Bagel 2): zeroing is conditional on path. Chain8
            // (fused kernel) WRITES the full sum directly and ignores
            // acc's prior contents; zeroing it would be wasted work.
            // Other paths (Many or legacy loop) accumulate via FMAs
            // that read acc as base.
            let useChain8 = canBatchDown && routing.indices.count == 8
            if !useChain8 {
                acc.zero()
            }

            if canBatchDown {
                // Phase 1a: all 16 gate+up qmms (8 experts × 2 projs)
                // on ONE shared encoder (ITER 46/Bagel2). Replaces 8
                // dequantGemvInt4Two calls (8 encoders) with one
                // dequantGemvInt4Many (1 encoder). Saves 7 encoder
                // begin/end pairs per layer × 40 = 280/decode token.
                // Falls back to per-expert Two if groupSize differs
                // (rare — Qwen3.6 all int4 weights share groupSize=64).
                var gateUpCanBatch = true
                var gateUpGS = 0
                for expertId in routing.indices {
                    guard let qg = gateProj[expertId].inner as? QuantizedLinear,
                          let qu = upProj[expertId].inner as? QuantizedLinear,
                          qg.bits == 4, qu.bits == 4 else {
                        gateUpCanBatch = false; break
                    }
                    if gateUpGS == 0 { gateUpGS = qg.groupSize }
                    if qg.groupSize != gateUpGS || qu.groupSize != gateUpGS {
                        gateUpCanBatch = false; break
                    }
                }
                // Allocate scratches up to slot count (unconditional —
                // both batched + fallback paths read them).
                for (slot, expertId) in routing.indices.enumerated() {
                    if let qg = gateProj[expertId].inner as? QuantizedLinear {
                        let outDim = qg.weight.shape[0]
                        while expertGScratches.count <= slot {
                            expertGScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                            expertUScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                            expertInnerScratches.append(Tensor.empty(shape: [outDim], dtype: h.dtype))
                        }
                    }
                    while expertOutScratches.count <= slot {
                        expertOutScratches.append(Tensor.empty(shape: [hidden], dtype: h.dtype))
                    }
                }
                if gateUpCanBatch {
                    var ws: [Tensor] = [], ss: [Tensor] = [], bs: [Tensor] = []
                    var ins: [Tensor] = [], outs: [Tensor] = []
                    ws.reserveCapacity(routing.indices.count * 2)
                    ss.reserveCapacity(routing.indices.count * 2)
                    bs.reserveCapacity(routing.indices.count * 2)
                    ins.reserveCapacity(routing.indices.count * 2)
                    outs.reserveCapacity(routing.indices.count * 2)
                    for (slot, expertId) in routing.indices.enumerated() {
                        let qg = gateProj[expertId].inner as! QuantizedLinear
                        let qu = upProj[expertId].inner as! QuantizedLinear
                        ws.append(qg.weight); ss.append(qg.scales); bs.append(qg.biases)
                        ins.append(h); outs.append(expertGScratches[slot])
                        ws.append(qu.weight); ss.append(qu.scales); bs.append(qu.biases)
                        ins.append(h); outs.append(expertUScratches[slot])
                    }
                    Ops.dequantGemvInt4Many(
                        weights: ws, scales: ss, biases: bs,
                        inputs: ins, outputs: outs,
                        groupSize: gateUpGS, on: work)
                } else {
                    for (slot, expertId) in routing.indices.enumerated() {
                        let qg = gateProj[expertId].inner as! QuantizedLinear
                        let qu = upProj[expertId].inner as! QuantizedLinear
                        Ops.dequantGemvInt4Two(
                            input: h,
                            w0: qg.weight, s0: qg.scales, b0: qg.biases,
                            out0: expertGScratches[slot],
                            w1: qu.weight, s1: qu.scales, b1: qu.biases,
                            out1: expertUScratches[slot],
                            groupSize: qg.groupSize, on: work)
                    }
                }
                // Phase 1b (ITER 39): batched 8-expert swiglu in ONE
                // encoder using cached scratches. Saves 7 begin/end
                // pairs per layer × 40 = 280/token.
                let gs = Array(expertGScratches.prefix(routing.indices.count))
                let us = Array(expertUScratches.prefix(routing.indices.count))
                let innersOut = Array(expertInnerScratches.prefix(routing.indices.count))
                Ops.swigluMany(gates: gs, ups: us, outs: innersOut, on: work)
                // Phase 2: batched 8-expert down qmm in ONE encoder.
                let inners = Array(expertInnerScratches.prefix(routing.indices.count))
                let outs = Array(expertOutScratches.prefix(routing.indices.count))
                Ops.dequantGemvInt4Many(
                    weights: downWeights, scales: downScales, biases: downBiases,
                    inputs: inners, outputs: outs,
                    groupSize: downGroupSize, on: work)
                // Phase 3: top-K accumulator.
                // ITER 47 (Bagel 2): when topK == 8 (Qwen3.6 default),
                // use the fused mt_scalar_fma_chain8 kernel — one
                // dispatch computes acc = Σ s_k * v_k directly,
                // skipping the acc.zero() step and reading acc only
                // once instead of 8 times.
                // Fallback to scalarFMAMany shared encoder (ITER 45)
                // for non-8 slot counts.
                var scalars: [Tensor] = []
                scalars.reserveCapacity(outs.count)
                for slot in 0..<outs.count {
                    scalars.append(Tensor(buffer: scalarBuf,
                                          offset: slot * h.dtype.byteSize,
                                          shape: [1], dtype: h.dtype))
                }
                if outs.count == 8 {
                    Ops.scalarFMAChain8(scalars: scalars, values: outs, out: acc,
                                        on: work)
                } else {
                    Ops.scalarFMAMany(scalars: scalars, values: outs, acc: acc,
                                      on: work)
                }
            } else {
                // Legacy: per-expert swiGLU chain.
                for (slot, expertId) in routing.indices.enumerated() {
                    let expertOut = swiGLU(h, slot: slot,
                                           gateProj: gateProj[expertId],
                                           upProj: upProj[expertId],
                                           downProj: downProj[expertId],
                                           on: work)
                    let scalarT = Tensor(buffer: scalarBuf,
                                         offset: slot * h.dtype.byteSize,
                                         shape: [1], dtype: h.dtype)
                    Ops.scalarFMA(scalar: scalarT, value: expertOut, base: acc,
                                  into: acc, on: work)
                }
            }
            accumulator = acc
        }

        // ── 5. Optional always-on shared expert ──────────────────────
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            // Shared expert uses slot = topK to avoid collision with
            // per-expert scratches (e.g., slot 8 when topK = 8).
            let sharedOut = swiGLU(h, slot: router.topK,
                                   gateProj: sg, upProj: su, downProj: sd, on: work)
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

    /// T-batched MoE forward. `hFlat` is `[T, hidden]` flat; returns
    /// `[T, hidden]` flat. The mTotal = T·topK rows fan into a single
    /// `Ops.moeGatherDequantGemmInt4` (or bm8 variant) per projection
    /// instead of T·topK sequential per-expert SwiGLU triplets. At
    /// Qwen3.6-A3B T=32, topK=8 → mTotal=256 fills the BM=16 tiles
    /// 16-deep — the regime where cooperative-tensor weight sharing
    /// PAYS (the same kernels regress at T=1 because every row is a
    /// unique expert with no shared weight).
    ///
    /// Requires `stackedInt4Experts` (the BGEMM weights layout). Falls
    /// back to per-token `decode` loop otherwise.
    ///
    /// Architecture:
    ///   1. Gate gemv batched (`gate.callMany`) → `[T, nExperts]`.
    ///   2. `cmd.commit() + wait` — router needs logits on CPU.
    ///   3. Per-token routing on host (T calls, each O(nExperts) + a
    ///      partial sort).
    ///   4. Build plan over T·topK rows: `sourceToken[m]`,
    ///      `expertId[m]`, `weight[m]`. Sort by `expertId` ascending so
    ///      consecutive rows share an expert (the kernel walks rows in
    ///      tile order; sorted = weight tile reuse across rows in a
    ///      tile).
    ///   5. `Ops.gather(h, gatherIdx)` → `xGathered [mTotal, hidden]`
    ///      in one dispatch (vs T·topK host memcpys).
    ///   6. `moeGatherDequantGemmInt4` × 3 (gate / up / down) on the
    ///      gathered batch — one dispatch each.
    ///   7. `Ops.swiglu` element-wise across `[mTotal, moeIntermediate]`.
    ///   8. Weighted scatter-sum: for each `m`, scale `downOut[m]` by
    ///      `weights[m]` and accumulate into `outFlat[sourceToken[m]]`.
    ///      Uses the same `Tensor.filled([hidden])` broadcast pattern as
    ///      the single-token `batchedSwiGLU`, mTotal times. Scatter-sum
    ///      is the dominant residual cost at large `mTotal`; a future
    ///      `mt_moe_scatter_scale_add` kernel collapses these mTotal
    ///      dispatches into one.
    ///   9. Shared expert (if any): one per-row SwiGLU call across T
    ///      tokens (also currently a T-loop because the single-token
    ///      `swiGLU` path is the standing API).
    ///
    /// All work after the gate-readback runs on a fresh `work` cmd that
    /// commits once at the end. The caller's `cmd` is consumed by the
    /// gate gemv and the commit + wait.
    public func decodeMany(_ hFlat: Tensor, t: Int,
                           cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(hFlat.elementCount == t * hidden,
                     "MoELayer.decodeMany: hFlat size \(hFlat.elementCount) ≠ T·hidden = \(t * hidden)")
        precondition(t > 0, "MoELayer.decodeMany: T must be positive")

        let dt = hFlat.dtype

        // ── 1. Gate gemv batched on caller's cmd ─────────────────────────
        let hRows = hFlat.reshaped(to: [t, hidden])
        let gateLogitsAll = gateLogitsMany(hRows, t: t, on: cmd, device: device)
        // gateLogitsAll shape: [T, nExperts]

        // ── 2. Commit + wait so the router can read logits ───────────────
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── 3. Per-token routing on host ─────────────────────────────────
        // `router.route` is a pure function over a per-row logit slice —
        // independent across `r`, so the T calls fan out across CPU cores
        // via `DispatchQueue.concurrentPerform`. At Qwen3.6-A3B T=512 ×
        // 40 layers = 20 480 route() calls per prefill; serial they add
        // up. Pre-allocate the result array sized to `t` so each iteration
        // writes its own slot (race-free).
        let nExperts = router.nExperts
        let topK = router.topK
        let logitsHost = gateLogitsAll.toFloatArray()  // [T·nExperts]
        var routings = [MoERouter.Routing](repeating: MoERouter.Routing(indices: [], weights: []),
                                           count: t)
        routings.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: t) { r in
                let start = r * nExperts
                let rowLogits = Array(logitsHost[start..<(start + nExperts)])
                buf[r] = router.route(logits: rowLogits)
            }
        }

        // ── 4. Build sorted plan over T·topK rows ────────────────────────
        let mTotal = t * topK
        // Tuple: (sortKey expertId, sourceToken, originalSlot, weight)
        var planTuples: [(Int, Int, Int, Float)] = []
        planTuples.reserveCapacity(mTotal)
        for r in 0..<t {
            let routing = routings[r]
            for slot in 0..<topK {
                planTuples.append((routing.indices[slot], r, slot, routing.weights[slot]))
            }
        }
        planTuples.sort { $0.0 < $1.0 }
        var sortedExpertsHost = [UInt32](); sortedExpertsHost.reserveCapacity(mTotal)
        var sourceTokensHost = [UInt32](); sourceTokensHost.reserveCapacity(mTotal)
        var sortedWeightsHost = [Float](); sortedWeightsHost.reserveCapacity(mTotal)
        for tuple in planTuples {
            sortedExpertsHost.append(UInt32(tuple.0))
            sourceTokensHost.append(UInt32(tuple.1))
            sortedWeightsHost.append(tuple.3)
        }
        // ── 4b. Build inv_perm + per-token-order weights for the
        // mt_moe_unpermute fused scatter-sum kernel. `invPermHost[r·k+s]`
        // = sortedIdx where (token r, slot s) landed in `downOut`.
        // `weightsTokenOrderHost[r·k+s]` = routings[r].weights[s]
        // (unsorted, matches the kernel's per-token lookup).
        var invPermHost = [UInt32](repeating: 0, count: mTotal)
        for (sortedIdx, tuple) in planTuples.enumerated() {
            let r = tuple.1
            let slot = tuple.2
            invPermHost[r * topK + slot] = UInt32(sortedIdx)
        }
        var weightsTokenOrderHost = [Float](); weightsTokenOrderHost.reserveCapacity(mTotal)
        for r in 0..<t {
            for slot in 0..<topK {
                weightsTokenOrderHost.append(routings[r].weights[slot])
            }
        }

        // ── 5. Fall back to per-token decode if no stacked-int4 weights ──
        guard let stacked = stackedInt4Experts, stacked.dtype == dt else {
            let outFlat = Tensor.empty(shape: [t * hidden], dtype: dt, device: device)
            let dtBytes = dt.byteSize
            var workCmd = device.makeCommandBuffer()
            for r in 0..<t {
                let hRow = Tensor(buffer: hFlat.buffer,
                                  offset: hFlat.offset + r * hidden * dtBytes,
                                  shape: [hidden], dtype: dt)
                let rowOut = decode(hRow, position: 0,
                                    cache: StatelessLayerCache(),
                                    cmd: workCmd, device: device)
                let outRow = Tensor(buffer: outFlat.buffer,
                                    offset: outFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                let copyCmd = device.makeCommandBuffer()
                Ops.copy(rowOut, into: outRow, on: copyCmd)
                copyCmd.commit()
                workCmd = device.makeCommandBuffer()
            }
            return outFlat
        }

        // ── 6. Work cmd for the rest of the layer ────────────────────────
        let work = device.makeCommandBuffer()
        let moeIntermediate = stacked.moeIntermediate
        let groupSize = stacked.groupSize

        // ── 7. Build gather index + expert ids on GPU ────────────────────
        let gatherIdxBuf = device.makeBuffer(length: mTotal * 4)
        sourceTokensHost.withUnsafeBytes {
            _ = memcpy(gatherIdxBuf.contents(), $0.baseAddress!, mTotal * 4)
        }
        let gatherIdxTensor = Tensor(buffer: gatherIdxBuf, offset: 0,
                                     shape: [mTotal], dtype: .u32)
        let expertIdsBuf = device.makeBuffer(length: mTotal * 4)
        sortedExpertsHost.withUnsafeBytes {
            _ = memcpy(expertIdsBuf.contents(), $0.baseAddress!, mTotal * 4)
        }
        let indices = Tensor(buffer: expertIdsBuf, offset: 0,
                             shape: [mTotal], dtype: .u32)

        // ── 8. Gather activations: [mTotal, hidden] ──────────────────────
        // Source rows are picked from h[sourceToken] — one dispatch.
        let xGathered = Ops.gather(table: hRows, tokenIds: gatherIdxTensor, on: work)
        precondition(xGathered.elementCount == mTotal * hidden,
                     "MoELayer.decodeMany: gather output unexpected size")

        // ── 9. Gate / up BGEMM → [mTotal, moeIntermediate] ───────────────
        let gateOut = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: dt,
                                   device: device)
        let upOut = Tensor.empty(shape: [mTotal, moeIntermediate], dtype: dt,
                                 device: device)
        // Tile selection for the batched-prefill regime:
        //   - mTotal ≥ 64 + `FFAI_MOE_BGEMM_BM64=1` → bm64_mpp (NAX
        //     cooperative-tensor). Wrapper in tree but dispatch-shape
        //     fails the equivalence canary (argmax 220 — the same
        //     dispatchThreads/threadgroups grid bug we saw at m1).
        //     Needs verification against the metaltile-ffai kernel
        //     source. **Default off** until that's resolved.
        //   - 16 ≤ mTotal → bm16 (the default `moeGatherDequant
        //     GemmInt4`). 16-deep tile fill across consecutive
        //     sorted-by-expert rows. THIS is the path that drives the
        //     2.69× T=32 win.
        //   - mTotal ≤ 8 + `FFAI_MOE_BGEMM_BM8=1` → bm8 (decode T=1
        //     fallback).
        // bm64_mpp default-on for mTotal ≥ 1024. Re-bench at fresher
        // calibration (after the `>= 256` initial bump) showed bm16 still
        // wins at T=32 and T=64 — the prior single-bench T=32 data point
        // that showed a 9% bm64 win was high-variance noise.
        //
        // Bench Qwen3.6-A3B M5 Max (forwardManyBench medians, refreshed):
        //   T=8  / mTotal=64:    bm16 59.12 vs bm64 25.63 tps  — bm16 +131%
        //   T=16 / mTotal=128:   bm16 115.90 vs bm64 58.80 tps — bm16  +97%
        //   T=32 / mTotal=256:   bm16 195.67 vs bm64 116.56 tps — bm16 +68%
        //   T=64 / mTotal=512:   bm16 274.38 vs bm64 230.24 tps — bm16 +19%
        //   T=128 / mTotal=1024: bm16 249.53 vs bm64 295.16 tps — bm64 +18%
        //   T=512 / mTotal=4096: bm16 253.99 vs bm64 262.93 tps — bm64  +4%
        //   T=2K / mTotal=16384: bm16 379.10 vs bm64 405.15 tps — bm64  +7%
        //
        // bm64's NAX cooperative-tensor matmul has per-tile setup cost
        // (sub-run boundary detection on heterogeneous expert tiles) that
        // wastes time when only ~10 BM=64 tiles fill the BGEMM. The
        // crossover sits between mTotal=512 (19% bm16 win) and
        // mTotal=1024 (18% bm64 win) — `>= 1024` is the safe pick.
        // Opt out via `FFAI_MOE_BGEMM_NO_BM64=1`.
        let useBm64 = mTotal >= 1024
            && ProcessInfo.processInfo.environment["FFAI_MOE_BGEMM_NO_BM64"] == nil
        let useBm8 = !useBm64 && topK <= 8 && useBm8Env && mTotal <= 8
        let bgemm: (Tensor, Tensor, Tensor, Tensor, Tensor, Int, Int, Int, Int, MTLCommandBuffer, Tensor) -> Void
        if useBm64 {
            bgemm = Ops.moeGatherDequantGemmInt4Bm64Mpp
        } else if useBm8 {
            bgemm = Ops.moeGatherDequantGemmInt4Bm8
        } else {
            bgemm = Ops.moeGatherDequantGemmInt4
        }
        bgemm(xGathered,
              stacked.gateWeight, stacked.gateScales, stacked.gateBiases,
              indices, mTotal, moeIntermediate, hidden, groupSize, work, gateOut)
        bgemm(xGathered,
              stacked.upWeight, stacked.upScales, stacked.upBiases,
              indices, mTotal, moeIntermediate, hidden, groupSize, work, upOut)

        // ── 10. SwiGLU fused: silu(gate) * up ────────────────────────────
        let inner = Ops.swiglu(gate: gateOut, up: upOut, on: work)

        // ── 11. Down BGEMM → [mTotal, hidden] ────────────────────────────
        let downOut = Tensor.empty(shape: [mTotal, hidden], dtype: dt,
                                   device: device)
        bgemm(inner,
              stacked.downWeight, stacked.downScales, stacked.downBiases,
              indices, mTotal, hidden, moeIntermediate, groupSize, work, downOut)

        // ── 12. Weighted scatter-sum back to [T, hidden] via fused
        // `mt_moe_unpermute` kernel — ONE dispatch over T·hidden
        // elements instead of mTotal·2 small `Tensor.filled([hidden])`
        // + mul + add per row. At Qwen3.6-A3B T=32 topK=8 mTotal=256
        // that's 512 dispatches → 1 per gate/up/down stage.
        let dtBytes = dt.byteSize
        let outFlat = Tensor.empty(shape: [t * hidden], dtype: dt, device: device)
        // invPerm + weights buffers (built on host in the plan loop).
        let invPermBuf = device.makeBuffer(length: mTotal * 4)
        invPermHost.withUnsafeBytes {
            _ = memcpy(invPermBuf.contents(), $0.baseAddress!, mTotal * 4)
        }
        let invPermTensor = Tensor(buffer: invPermBuf, offset: 0,
                                   shape: [mTotal], dtype: .u32)
        let weightsTensor = Tensor.empty(shape: [mTotal], dtype: dt, device: device)
        // Host → GPU weights in dtype.
        switch dt {
        case .f32:
            weightsTensor.copyIn(from: weightsTokenOrderHost)
        case .bf16:
            weightsTensor.copyIn(from: weightsTokenOrderHost.map { v -> UInt16 in
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
        case .f16:
            weightsTensor.copyIn(from: weightsTokenOrderHost.map { Float16($0) })
        default:
            fatalError("MoELayer.decodeMany: unsupported dtype \(dt) for unpermute weights")
        }
        Ops.moeUnpermute(
            expertOutputs: downOut, invPerm: invPermTensor,
            topKWeights: weightsTensor, into: outFlat,
            nRows: t, hidden: hidden, k: topK, on: work)
        let _ = dtBytes  // retained for the shared-expert fan-out below

        // ── 13. Optional shared expert — per-row T-loop ─────────────────
        // Shared expert is one always-on per-token SwiGLU. Looping T
        // calls here mirrors today's single-token flow; a batched
        // shared-expert SwiGLU is a small follow-up (gate/up/down all
        // accept callMany inputs).
        if let sg = sharedGateProj, let su = sharedUpProj, let sd = sharedDownProj {
            for r in 0..<t {
                let hRow = Tensor(buffer: hFlat.buffer,
                                  offset: hFlat.offset + r * hidden * dtBytes,
                                  shape: [hidden], dtype: dt)
                let sharedOut = swiGLU(hRow, slot: router.topK,
                                       gateProj: sg, upProj: su, downProj: sd,
                                       on: work)
                let outRow = Tensor(buffer: outFlat.buffer,
                                    offset: outFlat.offset + r * hidden * dtBytes,
                                    shape: [hidden], dtype: dt)
                _ = Ops.add(outRow, sharedOut, on: work, into: outRow)
            }
        }

        // ── 14. Commit without wait — outFlat is in-flight; the caller's
        // next read hazard-tracks against this write. Mirrors decode's
        // commit pattern.
        work.commit()
        return outFlat
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
        let useBm8 = topK <= 8 && useBm8Env
        // `FFAI_MOE_M1=1` routes the gather through the scalar m1
        // kernel `mt_moe_gather_qmm_int4` — no MPP / cooperative-tensor
        // overhead. At decode T=1 the cooperative-tensor variants
        // (bm8 / bm16 / bm64) lose ~15 % to descriptor setup cost; the
        // m1 path does a one-output-cell-per-TG dot-product with
        // `simd_sum` reduction and is closer in shape to the production
        // per-expert dequant_gemv path.
        let useM1 = !useBm8 && useM1Env

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
    private func swiGLU(_ x: Tensor, slot: Int,
                        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
                        on cmd: MTLCommandBuffer) -> Tensor {
        // ITER 25 + ITER 32 (post-OOM): when gate+up are both int4
        // QuantizedLinear with same groupSize, batch them via shared
        // encoder using PRE-CACHED scratches. Allocating fresh Tensors
        // here per call caused OOM under the bench's pipelined
        // cmd buffer fan-out (cmd_buf=64 in flight × 8 experts × 40
        // layers worth of Tensor.empty unable to retire fast enough).
        let g: Tensor
        let u: Tensor
        if let qGate = gateProj.inner as? QuantizedLinear,
           let qUp = upProj.inner as? QuantizedLinear,
           qGate.bits == 4 && qUp.bits == 4 && qGate.groupSize == qUp.groupSize {
            // Ensure scratch pool sized for `slot`.
            let outDim = qGate.weight.shape[0]
            while expertGScratches.count <= slot {
                expertGScratches.append(Tensor.empty(shape: [outDim], dtype: x.dtype))
                expertUScratches.append(Tensor.empty(shape: [outDim], dtype: x.dtype))
                expertInnerScratches.append(Tensor.empty(shape: [outDim], dtype: x.dtype))
            }
            g = expertGScratches[slot]
            u = expertUScratches[slot]
            Ops.dequantGemvInt4Two(
                input: x,
                w0: qGate.weight, s0: qGate.scales, b0: qGate.biases, out0: g,
                w1: qUp.weight,   s1: qUp.scales,   b1: qUp.biases,   out1: u,
                groupSize: qGate.groupSize, on: cmd)
            _ = Ops.swiglu(gate: g, up: u, on: cmd, into: expertInnerScratches[slot])
            return downProj(expertInnerScratches[slot], on: cmd)
        } else {
            g = gateProj(x, on: cmd)
            u = upProj(x, on: cmd)
        }
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        return downProj(inner, on: cmd)
    }

    /// Batched gate logits dispatch with lazy 8-bit dense weight materialise.
    ///
    /// For 4-bit gates, falls through to `gate.callMany` (already routes
    /// to the fast `mt_qmm_mma` batched kernel via `dequantGemmDynamicM`).
    /// For 8-bit gates — common on Qwen3.5/3.6 MoE checkpoints —
    /// `QuantizedLinear.callMany` loops T `dequantGemv` dispatches per
    /// call (T·40 layers = 20 480 dispatches at T=512 prefill). We
    /// dequantise the 8-bit weight to a dense `[nExperts, hidden]` tensor
    /// once on first invocation and dispatch ONE `Ops.gemm` per layer
    /// (which routes to `ffai_gemm_{f32,f16,bf16}` — handles arbitrary
    /// T, no tile-alignment constraint). Memory cost: one extra
    /// `[nExperts, hidden]` tensor per 8-bit gate; at Qwen3.6-A3B
    /// that's 40 × 128 × 2048 × 2 B = ~20 MB.
    private func gateLogitsMany(_ hRows: Tensor, t: Int,
                                on cmd: MTLCommandBuffer,
                                device: Device) -> Tensor {
        if !dequantizedGateAttempted {
            dequantizedGateAttempted = true
            if ProcessInfo.processInfo.environment["FFAI_MOE_NO_DEQUANT_GATE"] == nil,
               let q = gate.inner as? QuantizedLinear,
               q.bits == 8 {
                dequantizedGateCache = Self.dequantizeQuantizedLinear(
                    q, device: device)
                // ALSO build the [K, N] (transposed) layout for the
                // steel-gemm fast path when T % 64 == 0. Steel-gemm uses
                // half-precision MMA accumulators — slightly less precise
                // than `ffai_gemm`'s fp32 accumulator, but ~7% faster at
                // T=512 prefill where the tile size lines up.
                dequantizedGateCacheKN = Self.dequantizeQuantizedLinearKN(
                    q, device: device)
            }
        }
        // Steel-gemm fast path: T-tile and N-tile align with the 64x64x16
        // block. Falls back to `Ops.gemm` for ragged T.
        if let denseKN = dequantizedGateCacheKN,
           t % 64 == 0,
           let nExperts = denseKN.shape.last,
           nExperts % 64 == 0
        {
            let out = Tensor.empty(shape: [t, nExperts], dtype: hRows.dtype, device: device)
            let nTiles = nExperts / 64
            let mTiles = t / 64
            let tgWidth = 1024
            let grid = MTLSize(width: nTiles, height: mTiles, depth: 1)
            let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
            switch hRows.dtype {
            case .bf16:
                MetalTileKernels.mt_steel_gemm_64x64x16_2x2_bf16(
                    a: hRows.buffer, aOffset: hRows.offset,
                    b: denseKN.buffer, bOffset: denseKN.offset,
                    out: out.buffer, outOffset: out.offset,
                    m: UInt32(t), n: UInt32(nExperts), k: UInt32(hidden),
                    gridSize: grid, threadgroupSize: tg, on: cmd)
                return out
            case .f16:
                MetalTileKernels.mt_steel_gemm_64x64x16_2x2_f16(
                    a: hRows.buffer, aOffset: hRows.offset,
                    b: denseKN.buffer, bOffset: denseKN.offset,
                    out: out.buffer, outOffset: out.offset,
                    m: UInt32(t), n: UInt32(nExperts), k: UInt32(hidden),
                    gridSize: grid, threadgroupSize: tg, on: cmd)
                return out
            case .f32:
                MetalTileKernels.mt_steel_gemm_64x64x16_2x2_f32(
                    a: hRows.buffer, aOffset: hRows.offset,
                    b: denseKN.buffer, bOffset: denseKN.offset,
                    out: out.buffer, outOffset: out.offset,
                    m: UInt32(t), n: UInt32(nExperts), k: UInt32(hidden),
                    gridSize: grid, threadgroupSize: tg, on: cmd)
                return out
            default:
                break
            }
        }
        // General path: ffai_gemm handles arbitrary T (no tile alignment).
        if let dense = dequantizedGateCache {
            return Ops.gemm(weight: dense, input: hRows, nRows: t, on: cmd)
        }
        return gate.callMany(hRows, t: t, on: cmd, device: device)
    }

    /// CPU-side dense materialisation of a `QuantizedLinear` into the
    /// `[outDim, inDim]` layout that `Ops.gemm` expects (i.e., the same
    /// shape `QuantizedLinear.weight` already has but with dequantised
    /// values in the activation dtype instead of u32-packed int8s).
    /// Supports 8-bit only — the path that `QuantizedLinear.callMany`
    /// falls into a T-loop for.
    /// Write the routing weights into an existing MTLBuffer — the
    /// buffer is cached on the MoELayer instance via `topKScalarsBuf`
    /// to avoid a fresh alloc per decode token. Caller indexes into
    /// the buffer via `slot * dtype.byteSize`.
    private static func writeTopKScalars(_ weights: [Float], dtype: DType,
                                          into buf: MTLBuffer) {
        let topK = weights.count
        switch dtype {
        case .f32:
            var arr = weights
            memcpy(buf.contents(), &arr, topK * 4)
        case .f16:
            var arr = weights.map { Float16($0) }
            memcpy(buf.contents(), &arr, topK * 2)
        case .bf16:
            var arr = [UInt16]()
            arr.reserveCapacity(topK)
            for v in weights {
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                arr.append(UInt16(rounded >> 16))
            }
            memcpy(buf.contents(), &arr, topK * 2)
        default:
            fatalError("writeTopKScalars: unsupported dtype \(dtype)")
        }
    }

    private static func dequantizeQuantizedLinear(
        _ q: QuantizedLinear, device: Device
    ) -> Tensor? {
        precondition(q.bits == 8, "dequantizeQuantizedLinear: only 8-bit supported")
        let outDim = q.weight.shape[0]                    // nExperts
        let packsPerRow = q.weight.shape[1]
        let inDim = packsPerRow * 4                       // hidden
        let groupsPerRow = inDim / q.groupSize
        precondition(inDim % q.groupSize == 0,
                     "dequantizeQuantizedLinear: inDim \(inDim) not aligned to groupSize \(q.groupSize)")

        let wPacked = q.weight.toArray(as: UInt32.self)
        precondition(wPacked.count == outDim * packsPerRow,
                     "dequantizeQuantizedLinear: weight buffer size mismatch")
        let scalesF = q.scales.toFloatArray()
        let biasesF = q.biases.toFloatArray()

        // Output [outDim, inDim] row-major: dense[r, c] = q_byte * scale + bias.
        var dense = [Float](repeating: 0, count: outDim * inDim)
        for r in 0..<outDim {
            let rwBase = r * packsPerRow
            let rsBase = r * groupsPerRow
            let rdBase = r * inDim
            for c in 0..<inDim {
                let packIdx = rwBase + (c >> 2)
                let shift = UInt32((c & 3) << 3)
                let qByte = Float((wPacked[packIdx] >> shift) & 0xFF)
                let gIdx = rsBase + (c / q.groupSize)
                dense[rdBase + c] = qByte * scalesF[gIdx] + biasesF[gIdx]
            }
        }

        let dtype = q.scales.dtype
        let denseT = Tensor.empty(shape: [outDim, inDim], dtype: dtype, device: device)
        switch dtype {
        case .f32:
            denseT.copyIn(from: dense)
        case .f16:
            denseT.copyIn(from: dense.map { Float16($0) })
        case .bf16:
            let bf16Bits = dense.map { f -> UInt16 in
                let bits = f.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(truncatingIfNeeded: rounded >> 16)
            }
            denseT.copyIn(from: bf16Bits)
        default:
            return nil
        }
        return denseT
    }

    /// `[K=inDim, N=outDim]` (transposed) materialisation of a
    /// `QuantizedLinear` — the `b` operand `mt_steel_gemm_64x64x16_2x2_*`
    /// expects. Same dequant math as the row-major sibling, just
    /// permuted on store.
    private static func dequantizeQuantizedLinearKN(
        _ q: QuantizedLinear, device: Device
    ) -> Tensor? {
        precondition(q.bits == 8, "dequantizeQuantizedLinearKN: only 8-bit supported")
        let outDim = q.weight.shape[0]
        let packsPerRow = q.weight.shape[1]
        let inDim = packsPerRow * 4
        let groupsPerRow = inDim / q.groupSize
        precondition(inDim % q.groupSize == 0,
                     "dequantizeQuantizedLinearKN: inDim not aligned")

        let wPacked = q.weight.toArray(as: UInt32.self)
        let scalesF = q.scales.toFloatArray()
        let biasesF = q.biases.toFloatArray()

        var dense = [Float](repeating: 0, count: inDim * outDim)
        for r in 0..<outDim {
            let rwBase = r * packsPerRow
            let rsBase = r * groupsPerRow
            for c in 0..<inDim {
                let packIdx = rwBase + (c >> 2)
                let shift = UInt32((c & 3) << 3)
                let qByte = Float((wPacked[packIdx] >> shift) & 0xFF)
                let gIdx = rsBase + (c / q.groupSize)
                dense[c * outDim + r] = qByte * scalesF[gIdx] + biasesF[gIdx]
            }
        }

        let dtype = q.scales.dtype
        let denseT = Tensor.empty(shape: [inDim, outDim], dtype: dtype, device: device)
        switch dtype {
        case .f32:
            denseT.copyIn(from: dense)
        case .f16:
            denseT.copyIn(from: dense.map { Float16($0) })
        case .bf16:
            let bf16Bits = dense.map { f -> UInt16 in
                let bits = f.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(truncatingIfNeeded: rounded >> 16)
            }
            denseT.copyIn(from: bf16Bits)
        default:
            return nil
        }
        return denseT
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
