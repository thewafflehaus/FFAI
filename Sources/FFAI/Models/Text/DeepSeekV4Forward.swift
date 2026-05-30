// Copyright 2026 Tom Turney (@TheTom)
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
// DeepSeek V4 single-token decode forward path — full-attention
// sub-block. Lands incrementally: this file currently scaffolds the
// attention path against the existing Ops surface; the FFN sub-block
// (MoE + shared expert + mHC), the CSA / HCA paths, and the
// end-to-end `forward(...)` driver land in follow-ups.

import Foundation
import Metal

// MARK: - Per-call decode state

extension DeepSeekV4Model {
    /// Sliding-window MQA KV cache for one layer. Holds up to
    /// `n_swa=128` 512-d entries; appends grow `swCount` until the
    /// cache wraps. Indexing within the window stays in slot order
    /// (the SDPA kernel walks `[0..n_visible)` directly).
    public final class LayerKVState: @unchecked Sendable {
        public var swCache: Tensor   // [n_swa, head_dim]
        public var swCount: Int
        public let nSWA: Int
        public let headDim: Int

        public init(headDim: Int, nSWA: Int, dtype: DType) {
            self.swCache = Tensor.empty(shape: [nSWA, headDim], dtype: dtype)
            self.swCount = 0
            self.nSWA = nSWA
            self.headDim = headDim
        }
    }

    /// One forward-call decode state.
    public final class DecodeState: @unchecked Sendable {
        public var layerStates: [LayerKVState]
        /// 4-channel mHC residual state, `[n_hc=4, hidden]`.
        public var hcState: Tensor
        public var position: Int

        public init(layerStates: [LayerKVState], hcState: Tensor, position: Int = 0) {
            self.layerStates = layerStates
            self.hcState = hcState
            self.position = position
        }
    }

    public func makeDecodeState() -> DecodeState {
        let cfg = textConfig
        let states = (0..<cfg.nLayers).map { _ in
            LayerKVState(
                headDim: cfg.headDim, nSWA: cfg.slidingWindow, dtype: activationDtype)
        }
        let hc = Tensor.empty(shape: [4, cfg.hidden], dtype: activationDtype)
        return DecodeState(layerStates: states, hcState: hc, position: 0)
    }
}

// MARK: - Errors

enum DeepSeekV4ForwardError: Error, CustomStringConvertible {
    case notImplementedForRegime(Int)
    var description: String {
        switch self {
        case .notImplementedForRegime(let r):
            return "DSv4 forward path not yet implemented for compress_ratio=\(r)"
        }
    }
}

// MARK: - Shape helpers

extension Tensor {
    /// GGUF stores matmul weights as `[n_in_fast, n_out_slow]` in
    /// dimensions order, but `Ops.gemv` expects `[n_out, n_in]`.
    /// Swap the two dim labels (no data movement — same byte layout,
    /// different interpretation).
    public func asGgufMatmulWeight() -> Tensor {
        precondition(shape.count == 2, "asGgufMatmulWeight: rank must be 2")
        return reshaped(to: [shape[1], shape[0]])
    }
}

// MARK: - Ones-tensor cache (for per-head unit-RMS Q-norm)

extension DeepSeekV4Model {
    /// `[head_dim]` ones tensor, cached lazily on first access so
    /// the per-head Q unit-RMS norm has a no-op weight to pass to
    /// `Ops.rmsNormRows`. Backed by the generic `Tensor.filled(...)`
    /// constructor — any model that needs a constant-valued weight
    /// for a no-learnable-weight norm can reuse the same primitive.
    fileprivate func qHeadNormOnes(_ dt: DType) -> Tensor {
        if let cached = qHeadNormOnesCache { return cached }
        let t = Tensor.filled(1.0, shape: [textConfig.headDim], dtype: dt, device: device)
        qHeadNormOnesCache = t
        return t
    }
}

// MARK: - Full-attention sub-block forward
//
// ## Infrastructure gaps blocking the runnable body
//
// 1. **Mixed-dtype rmsNorm**. `Ops.rmsNorm` enforces
//    `x.dtype == weight.dtype` but DSv4 ships norm weights as f32
//    while activations are f16. Either (a) cast f32 norm weights to
//    f16 at load time inside `GGUFTensorBundle.tensor(named:outDtype:)`
//    (currently f32/f16/bf16 sources ignore `outDtype` and pass
//    through with their on-disk dtype), or (b) widen Ops.rmsNorm to
//    accept f32 weight + f16 input.
//
// 2. **No-weight rmsNormRows**. The per-head Q-norm has no learnable
//    weight (just `eps`). `Ops.rmsNormRows` requires a `[rowSize]`
//    weight tensor. Either allocate a ones tensor once, or add a
//    `rmsNormRowsNoWeight` variant.
//
// 3. **Grouped O-LoRA `mul_mat_id`**. `attn_output_a` is a single
//    [4096 × 8192] tensor that must be applied as 8 distinct
//    [4096 × 1024] slices, each driven by a different [4096] slice
//    of the [n_heads × head_dim] attention output. No Ops surface
//    today does this without 8 sequential gemvs against
//    output-axis-strided weight views — and `slicedRows` only
//    slices the leading dim.
//
// 4. **GGUF matmul-weight layout swap**. GGUF dimensions list the
//    fast dim first: `[n_in, n_out]`. Ops.gemv expects `[n_out, n_in]`.
//    The `Tensor.asGgufMatmulWeight()` helper above swaps the dim
//    labels (no data movement). Verified correct for `Ops.gemv` by
//    inspection but not yet unit-tested.
//
// 5. **Sliding-window cache append**. `Ops.copy(_:into:)` writes the
//    [head_dim] kv_norm into `swCache.slicedRows(start: slot, count: 1)`
//    which is shape `[1, head_dim]` — element-count matches but
//    dtype precondition may need the slice to be the same dtype as
//    src (currently fine, both are activation dtype). Untested.
//
// The decode-state types below are correct as-is; the
// `forwardFullAttnSubblock` function body lives in a working
// branch until the 5 gaps are closed.

extension DeepSeekV4Model {

    /// Decode one full-attention layer's attention sub-block.
    /// Reads `state.hcState` (the 4-channel residual), runs the
    /// full-attn block, writes the new 4-channel state back into
    /// `state.hcState`, and returns the un-residualised
    /// `block_out [hidden]` for downstream introspection.
    ///
    /// Wired against the real Ops API (`gemv` with GGUF shape-swap,
    /// `rmsNorm` for the learnable norms, `dsv4MhcSinkhornSplit /
    /// Collapse / Expand` for the mHC dance, `dsv4PartialRope` for
    /// the K/Q tail rotation, `dsv4SdpaDecodeD512Sink` for the
    /// MQA attention with attn_sinks).
    ///
    /// Known-incorrect details flagged with `// FIXME` — these
    /// matter for numerical correctness but not for "does the
    /// dispatch chain compile and run without NaN?":
    /// - Per-head Q-norm (eps-only, no learnable weight) is **skipped**
    ///   — needs a `[head_dim]` ones tensor or a no-weight rms variant.
    /// - Grouped O-LoRA collapses to a single 32768 → 8192 → 4096
    ///   matmul that **does NOT** apply per-group LoRA-A slices.
    ///   Output dims match; values are wrong until the proper
    ///   per-group dispatch lands.
    public func forwardFullAttnSubblock(
        layer: DeepSeekV4Layer, state: DecodeState, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        let headDim = cfg.headDim
        let qLoraRank = cfg.qLoraRank
        let nHeads = cfg.nHeads
        let qkRopeDim = cfg.qkRopeHeadDim
        let nNope = headDim - qkRopeDim

        // ── mHC pre/post/comb split ──
        // mixes = hc_attn_fn @ flatten(H)  →  [24]
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let hcAttnFnW = layer.hcAttnFn.asGgufMatmulWeight()
        let mixes = Ops.gemv(weight: hcAttnFnW, input: flatH, on: cmd)
        let (preAttn, postAttn, combAttn) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcAttnScale, base: layer.hcAttnBase,
            nTokens: 1, eps: cfg.hcEpsilon, sinkhornIters: cfg.hcSinkhornIterations,
            on: cmd)

        // ── mHC collapse: H[4, hidden] → x[hidden] (drop n_tokens=1 dim) ──
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preAttn,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmd)
        let x = xWithTokens.reshaped(to: [hidden])

        // ── attn_norm ──
        let xNorm = Ops.rmsNorm(x, weight: layer.attnNorm, eps: cfg.rmsNormEps, on: cmd)

        // ── Q low-rank chain: x → q_a → q_a_norm → q_b ──
        let qA = Ops.gemv(weight: layer.attnQA.asGgufMatmulWeight(), input: xNorm, on: cmd)
        let qANorm = Ops.rmsNorm(qA, weight: layer.attnQANorm, eps: cfg.rmsNormEps, on: cmd)
        let q = Ops.gemv(weight: layer.attnQB.asGgufMatmulWeight(), input: qANorm, on: cmd)
        // Per-head unit-RMS Q-norm: normalize each [head_dim] row
        // independently with no learnable weight. Pass a ones-tensor
        // of shape [head_dim] cached on the model.
        Ops.rmsNormRows(
            q, weight: qHeadNormOnes(dt), eps: cfg.rmsNormEps,
            nRows: nHeads, rowSize: headDim, on: cmd, into: q)

        // ── Partial RoPE on Q tail ──
        let qRoped = Tensor.empty(shape: q.shape, dtype: dt)
        Ops.copy(q, into: qRoped, on: cmd)
        Ops.dsv4PartialRope(
            qk: qRoped, out: qRoped,
            nHeads: nHeads, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: cfg.ropeTheta, inverse: false, on: cmd)

        // ── KV down-projection + norm + partial RoPE ──
        let kv = Ops.gemv(weight: layer.attnKV.asGgufMatmulWeight(), input: xNorm, on: cmd)
        let kvNorm = Ops.rmsNorm(kv, weight: layer.attnKVANorm, eps: cfg.rmsNormEps, on: cmd)
        Ops.dsv4PartialRope(
            qk: kvNorm, out: kvNorm,
            nHeads: 1, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: cfg.ropeTheta, inverse: false, on: cmd)

        // ── Append to sliding-window cache ──
        let layerState = state.layerStates[layer.layerIndex]
        let slot = layerState.swCount % layerState.nSWA
        Ops.copy(kvNorm, into: layerState.swCache.slicedRows(start: slot, count: 1), on: cmd)
        layerState.swCount += 1
        let nVisible = min(layerState.swCount, layerState.nSWA)

        // ── MQA SDPA with attn_sinks: K == V, n_kv_heads=1 ──
        let kvBuf = layerState.swCache.slicedRows(start: 0, count: nVisible)
        let scale = 1.0 / Float(headDim).squareRoot()
        let attnOut = Ops.dsv4SdpaDecodeD512Sink(
            q: qRoped, k: kvBuf, v: kvBuf, sinkLogit: layer.attnSinks,
            nQHeads: nHeads, nKvHeads: 1, headDim: headDim,
            nKv: nVisible, kvStride: layerState.nSWA,
            scale: scale, outDtype: dt, on: cmd)

        // ── Inverse partial RoPE on attention output ──
        Ops.dsv4PartialRope(
            qk: attnOut, out: attnOut,
            nHeads: nHeads, headDim: headDim, nNope: nNope,
            position: state.position, thetaBase: cfg.ropeTheta, inverse: true, on: cmd)

        // ── Grouped O-LoRA: 8 groups × [4096, 1024] then [8192, 4096] ──
        // Reshape attnOut [n_heads, head_dim] → [n_groups, group_dim]
        // = [8, 4096]. Each group consumes a different LoRA-A slice;
        // since attn_output_a is stored [n_in=4096, n_out=8192] in
        // GGUF (= [8192, 4096] after the as-weight swap), and the
        // 8192-output-dim is the **concatenation of 8 × 1024
        // per-group LoRA-A outputs**, the per-group dispatch is:
        //   oLow[g, :1024] = wA[g, :1024, :] @ attnOut_group[g, :]
        // 8 sequential gemvs, each [1024, 4096], with weight slice
        // taken as a row-range of the swapped weight tensor (axis-0
        // slicing — contiguous and supported by `slicedRows`).
        let oGroups = 8
        let groupDim = (nHeads * headDim) / oGroups  // 4096
        let oLoraRank = cfg.oLoraRank  // 1024
        let attnOutGrouped = attnOut.reshaped(to: [oGroups, groupDim])
        let oLow = Tensor.empty(shape: [oGroups * oLoraRank], dtype: dt)
        let outputAW = layer.attnOutputA.asGgufMatmulWeight()  // [8192, 4096]
        for g in 0..<oGroups {
            let weightSlice = outputAW.slicedRows(start: g * oLoraRank, count: oLoraRank)
            let inputSlice = attnOutGrouped.slicedRows(start: g, count: 1).reshaped(to: [groupDim])
            let outSlice = oLow.slicedRows(start: g * oLoraRank, count: oLoraRank)
            _ = Ops.gemv(
                weight: weightSlice, input: inputSlice, on: cmd, into: outSlice)
        }
        let blockOut = Ops.gemv(
            weight: layer.attnOutputB.asGgufMatmulWeight(), input: oLow, on: cmd)

        // ── mHC expand: write new 4-channel state ──
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postAttn, comb: combAttn,
            residualState: state.hcState,
            hiddenDim: hidden, nHc: 4, nTokens: 1, on: cmd)
        state.hcState = newH
        return blockOut
    }

    /// FFN sub-block — runs the mHC dance + RMS norm + MoE-top-6 +
    /// shared expert + mHC expand. Single token, decode mode.
    ///
    /// Selection path: full sqrtsoftplus router scoring → top-6 via
    /// CPU readback (no GPU `argpartition` Op yet, so this is the
    /// quick-correct path). Expert dispatch is 6 × 3 gemvs against
    /// per-expert slices of the [n_experts, intermediate, hidden]
    /// tensors. Combine = weighted sum of expert outputs by
    /// `score_unbiased * routed_scaling_factor`, plus the
    /// always-on shared expert.
    public func forwardFfnSubblock(
        layer: DeepSeekV4Layer, state: DecodeState, on cmd: MTLCommandBuffer
    ) -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden
        let intermediate = cfg.moeIntermediate
        let nExperts = layer.ffnGateExps.shape.last ?? cfg.nExperts
        let topK = cfg.nExpertsPerToken
        let scaling = cfg.routerScalingFactor

        // ── mHC pre/post/comb split ──
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let hcFnW = layer.hcFfnFn.asGgufMatmulWeight()
        let mixes = Ops.gemv(weight: hcFnW, input: flatH, on: cmd)
        let (preFfn, postFfn, combFfn) = Ops.dsv4MhcSinkhornSplit(
            mixes: mixes, scale: layer.hcFfnScale, base: layer.hcFfnBase,
            nTokens: 1, eps: cfg.hcEpsilon, sinkhornIters: cfg.hcSinkhornIterations,
            on: cmd)

        // ── mHC collapse + ffn_norm ──
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preFfn,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmd)
        let x = xWithTokens.reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: layer.ffnNorm, eps: cfg.rmsNormEps, on: cmd)

        // ── Router scoring: logits = ffn_gate_inp @ xNorm ──
        let routerLogits = Ops.gemv(
            weight: layer.ffnGateInp.asGgufMatmulWeight(), input: xNorm, on: cmd)
        // The sqrtsoftplus router Op takes f32 logits + bias and writes
        // f32 score_unbiased + score_biased. Routerlogits is `dt`
        // (activation dtype). Cast to f32 first.
        let routerLogitsF32 = Tensor.empty(shape: routerLogits.shape, dtype: .f32)
        Ops.castToF32(routerLogits, into: routerLogitsF32, on: cmd)
        let bias: Tensor
        if let b = layer.expProbsBias {
            bias = b
        } else {
            bias = Tensor.filled(0.0, shape: [nExperts], dtype: .f32, device: device)
        }
        let (scoreUnbiased, scoreBiased) = Ops.dsv4MoeRouterSqrtsoftplus(
            logits: routerLogitsF32, bias: bias, on: cmd)

        // CPU-side top-K selection. Sync flush, readback, argpartition.
        // Slow but correct; replace with a GPU top-K when one lands.
        cmd.commit()
        cmd.waitUntilCompleted()
        let biasedHost = scoreBiased.toArray(as: Float.self)
        let unbiasedHost = scoreUnbiased.toArray(as: Float.self)
        var indexed = Array(biasedHost.enumerated())
        indexed.sort { $0.element > $1.element }
        let top = Array(indexed.prefix(topK))
        let topIndices = top.map { $0.offset }
        let topWeights = topIndices.map { unbiasedHost[$0] * scaling }
        // Re-normalize weights so they sum to 1 (a common router
        // convention — keeps activations stable across hash-route +
        // sqrtsoftplus-route boundaries).
        let weightSum = topWeights.reduce(0, +)
        let normWeights: [Float] = weightSum > 0
            ? topWeights.map { $0 / weightSum }
            : Array(repeating: 1.0 / Float(topK), count: topK)

        // ── Expert dispatch ──
        // gate_exps / up_exps:  [hidden, intermediate, n_experts]
        //   → reshape [n_experts, intermediate, hidden] (no data move,
        //     fast/slow swap), slice expert e, get [intermediate, hidden]
        //     = [n_out, n_in] which Ops.gemv accepts directly.
        // down_exps: [intermediate, hidden, n_experts]
        //   → reshape [n_experts, hidden, intermediate], slice e,
        //     [hidden, intermediate] = [n_out, n_in].
        // GPU-side accumulator: moeOut += w_k * expert_out_k for each
        // of topK experts, then + shared-expert output. Uses Ops.add
        // (vector_add) to keep the chain on-GPU — no per-expert
        // CPU sync.
        let gateExps3D = layer.ffnGateExps.reshaped(to: [nExperts, intermediate, hidden])
        let upExps3D = layer.ffnUpExps.reshaped(to: [nExperts, intermediate, hidden])
        let downExps3D = layer.ffnDownExps.reshaped(to: [nExperts, hidden, intermediate])
        let cmd2 = device.makeCommandBuffer()
        let moeAccum = Tensor.filled(0.0, shape: [hidden], dtype: dt, device: device)
        for k in 0..<topK {
            let e = topIndices[k]
            let w = normWeights[k]
            let gateW = gateExps3D.slicedRows(start: e, count: 1)
                .reshaped(to: [intermediate, hidden])
            let upW = upExps3D.slicedRows(start: e, count: 1)
                .reshaped(to: [intermediate, hidden])
            let downW = downExps3D.slicedRows(start: e, count: 1)
                .reshaped(to: [hidden, intermediate])
            let gateOut = Ops.gemv(weight: gateW, input: xNorm, on: cmd2)
            let upOut = Ops.gemv(weight: upW, input: xNorm, on: cmd2)
            let inner = Ops.swiglu(gate: gateOut, up: upOut, on: cmd2)
            let expertOut = Ops.gemv(weight: downW, input: inner, on: cmd2)
            let wT = Tensor.filled(w, shape: [hidden], dtype: dt, device: device)
            let scaled = Ops.mul(expertOut, wT, on: cmd2)
            _ = Ops.add(moeAccum, scaled, on: cmd2, into: moeAccum)
        }
        // Shared expert
        let sGate = Ops.gemv(
            weight: layer.ffnGateShexp.asGgufMatmulWeight(), input: xNorm, on: cmd2)
        let sUp = Ops.gemv(
            weight: layer.ffnUpShexp.asGgufMatmulWeight(), input: xNorm, on: cmd2)
        let sInner = Ops.swiglu(gate: sGate, up: sUp, on: cmd2)
        let shexpOut = Ops.gemv(
            weight: layer.ffnDownShexp.asGgufMatmulWeight(), input: sInner, on: cmd2)
        let blockOut = Ops.add(moeAccum, shexpOut, on: cmd2)

        // mHC expand
        let newH = Ops.dsv4MhcExpand(
            blockOut: blockOut, post: postFfn, comb: combFfn,
            residualState: state.hcState,
            hiddenDim: hidden, nHc: 4, nTokens: 1, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        state.hcState = newH
        return blockOut
    }

    /// Full single-token decode forward through all `nLayers` layers
    /// + output mHC head + output norm + LM head. Returns the logits
    /// vector `[vocab]`.
    ///
    /// **WIP**: CSA / HCA forward paths aren't implemented yet, so
    /// `forwardFullAttnSubblock` is used on ALL layers regardless of
    /// `compress_ratio`. The output is dispatch-correct but the
    /// numerics for CSA/HCA layers are wrong (they should run the
    /// indexer + compressed-cache attention). Quality of the
    /// generated token will be garbage until those paths land.
    public func forwardAllLayers(
        inputTokenId: Int, state: DecodeState
    ) throws -> Tensor {
        let cfg = textConfig
        let dt = activationDtype
        let hidden = cfg.hidden

        // Seed hcState with the input token's embedding broadcast
        // across all 4 mHC channels.
        let embedRow = tokenEmbd.asGgufMatmulWeight()
            .slicedRows(start: inputTokenId, count: 1).reshaped(to: [hidden])
        let cmdSeed = device.makeCommandBuffer()
        for c in 0..<4 {
            let dst = state.hcState.slicedRows(start: c, count: 1).reshaped(to: [hidden])
            Ops.copy(embedRow, into: dst, on: cmdSeed)
        }
        cmdSeed.commit()
        cmdSeed.waitUntilCompleted()

        // Iterate layers wrapped in autoreleasepool so the per-layer
        // tensors' MTLBuffer wrappers are released at end-of-block
        // (Device.makeBuffer is not pooled — every Tensor.empty in
        // the sub-block bodies allocates a fresh shared-storage
        // MTLBuffer, ~100 per layer × 43 = ~4300 transient buffers
        // per token without the pool drain).
        for layerIdx in 0..<cfg.nLayers {
            try autoreleasepool {
                let layer = try self.layer(layerIdx)
                let cmdAttn = device.makeCommandBuffer()
                _ = forwardFullAttnSubblock(layer: layer, state: state, on: cmdAttn)
                cmdAttn.commit()
                cmdAttn.waitUntilCompleted()
                _ = forwardFfnSubblock(
                    layer: layer, state: state, on: device.makeCommandBuffer())
                self.releaseLayer(layerIdx)
                print("forwardAllLayers: layer \(layerIdx) done")
            }
        }

        // Output mHC head: simpler decomposition than per-layer mHC.
        // pre = sigmoid(output_hc_fn^T @ flatten(H) * scale + base) + eps  → [4]
        let flatH = state.hcState.reshaped(to: [4 * hidden])
        let cmdHead = device.makeCommandBuffer()
        let pre4 = Ops.gemv(
            weight: outputHcFn.asGgufMatmulWeight(), input: flatH, on: cmdHead)
        // Sigmoid + scale + base: compute on host (4 elements).
        cmdHead.commit()
        cmdHead.waitUntilCompleted()
        let pre4Host = pre4.toArray(as: Float.self)
        let scaleHost = outputHcScale.toArray(as: Float.self)
        let baseHost = outputHcBase.toArray(as: Float.self)
        let eps = cfg.hcEpsilon
        var preFinal = [Float](repeating: 0, count: 4)
        for c in 0..<4 {
            let z = pre4Host[c] * scaleHost[0] + baseHost[c]
            preFinal[c] = 1.0 / (1.0 + Foundation.exp(-z)) + eps
        }
        let preTensor = Tensor.empty(shape: [4], dtype: .f32)
        preTensor.copyIn(from: preFinal)

        // Collapse H → x using preFinal.
        let cmdCollapse = device.makeCommandBuffer()
        let xWithTokens = Ops.dsv4MhcCollapse(
            state: state.hcState, pre: preTensor,
            hiddenDim: hidden, nHc: 4, nTokens: 1, outDtype: dt, on: cmdCollapse)
        let x = xWithTokens.reshaped(to: [hidden])
        let xNorm = Ops.rmsNorm(x, weight: outputNorm, eps: cfg.rmsNormEps, on: cmdCollapse)
        let logits = Ops.gemv(
            weight: outputHead.asGgufMatmulWeight(), input: xNorm, on: cmdCollapse)
        cmdCollapse.commit()
        cmdCollapse.waitUntilCompleted()
        return logits
    }
}
