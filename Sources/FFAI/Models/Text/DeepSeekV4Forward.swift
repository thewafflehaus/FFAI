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
    fileprivate func asGgufMatmulWeight() -> Tensor {
        precondition(shape.count == 2, "asGgufMatmulWeight: rank must be 2")
        return reshaped(to: [shape[1], shape[0]])
    }
}

// MARK: - Ones-tensor cache (for per-head unit-RMS Q-norm)

extension DeepSeekV4Model {
    /// `[head_dim]` ones tensor, cached lazily on first access so
    /// the per-head Q unit-RMS norm has a no-op weight to pass to
    /// `Ops.rmsNormRows`.
    fileprivate func qHeadNormOnes(_ dt: DType) -> Tensor {
        if let cached = qHeadNormOnesCache { return cached }
        let n = textConfig.headDim
        let buf = device.makeBuffer(length: max(n * dt.byteSize, dt.byteSize))
        switch dt {
        case .f32:
            let p = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<n { p[i] = 1.0 }
        case .f16:
            let p = buf.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<n { p[i] = 1.0 }
        case .bf16:
            let p = buf.contents().assumingMemoryBound(to: UInt16.self)
            let oneBits: UInt16 = 0x3F80  // bf16 1.0
            for i in 0..<n { p[i] = oneBits }
        default:
            fatalError("qHeadNormOnes: unsupported dtype \(dt)")
        }
        let t = Tensor(buffer: buf, offset: 0, shape: [n], dtype: dt)
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
}
