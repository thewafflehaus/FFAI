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
// FishSpeechLayers — transformer building blocks for the FishSpeech dual-AR model.
//
// Contains:
//   - FishSpeechRoPECache:      CPU-side cos/sin tables for RoPE positional encoding.
//   - FishSpeechAttentionLayer: GQA attention with fused QKV and optional QK-norm.
//   - FishSpeechFFN:            SwiGLU feed-forward network.
//   - FishSpeechBlock:          Pre-norm residual transformer block (attention + FFN).
//   - BFloat16 helpers:         bfloat16ToFloat / floatToBfloat16.

import Foundation
import Metal

// ─── RoPE cache ───────────────────────────────────────────────────────────

/// Precomputed cos/sin tables for RoPE. Stored as flat Float32 arrays;
/// CPU fills them at init time, GPU reads them during decode.
///
/// Using CPU-side tables avoids a separate GPU init pass and is negligible
/// overhead (~2 MB for maxSeq=32768, headDim=128).
final class FishSpeechRoPECache: @unchecked Sendable {
    let cosTable: [Float]  // [maxSeq * headDim/2]
    let sinTable: [Float]
    let headDim: Int
    let maxSeq: Int

    init(headDim: Int, ropeBase: Float, maxSeq: Int) {
        self.headDim = headDim
        self.maxSeq = maxSeq
        let half = headDim / 2
        var cos = [Float](repeating: 0, count: maxSeq * half)
        var sin = [Float](repeating: 0, count: maxSeq * half)
        for i in 0 ..< half {
            // θ_i = 1 / (ropeBase ^ (2i / headDim))
            let theta = 1.0 / pow(ropeBase, Float(2 * i) / Float(headDim))
            for pos in 0 ..< maxSeq {
                let angle = Float(pos) * theta
                cos[pos * half + i] = Foundation.cos(angle)
                sin[pos * half + i] = Foundation.sin(angle)
            }
        }
        self.cosTable = cos
        self.sinTable = sin
    }
}

// ─── BFloat16 helpers ─────────────────────────────────────────────────────

@inline(__always) func bfloat16ToFloat(_ bits: UInt16) -> Float {
    var u32: UInt32 = UInt32(bits) << 16
    return withUnsafeBytes(of: &u32) { $0.load(as: Float.self) }
}

@inline(__always) func floatToBfloat16(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    return UInt16(truncatingIfNeeded: bits >> 16)
}

// ─── Attention layer ─────────────────────────────────────────────────────

/// GQA attention with fused QKV projection and optional QK-norm.
/// Backed by FFAI `KVCache` for incremental decode.
final class FishSpeechAttentionLayer: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let hasQKNorm: Bool

    let wqkv: AnyLinear
    let wo: AnyLinear
    /// Per-head layer norms on Q and K if `attentionQKNorm` is true.
    let qNorm: RMSNorm?
    let kNorm: RMSNorm?

    let rope: FishSpeechRoPECache

    init(
        nHeads: Int,
        nKVHeads: Int,
        dim: Int,
        headDim: Int,
        ropeBase: Float,
        maxSeq: Int,
        qkvBias: Bool,
        oBias: Bool,
        qkNorm: Bool,
        normEps: Float,
        wqkv: AnyLinear,
        wo: AnyLinear,
        qNorm: RMSNorm?,
        kNorm: RMSNorm?
    ) {
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
        self.hasQKNorm = qkNorm
        self.wqkv = wqkv
        self.wo = wo
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.rope = FishSpeechRoPECache(headDim: headDim, ropeBase: ropeBase, maxSeq: maxSeq)
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in wqkv.parameters() { out.append(("attention.wqkv.\(k)", v)) }
        for (k, v) in wo.parameters() { out.append(("attention.wo.\(k)", v)) }
        if let q = qNorm {
            for (k, v) in q.parameters() { out.append(("attention.q_norm.\(k)", v)) }
        }
        if let k = kNorm {
            for (kk, v) in k.parameters() { out.append(("attention.k_norm.\(kk)", v)) }
        }
        return out
    }

    /// Single-token decode forward. Manages its own GPU command buffers
    /// because a CPU sync is required after the QKV projection to split
    /// the fused tensor and apply CPU-side RoPE.
    /// `h` is the residual stream [dim]; returns updated [dim] in `h.dtype`.
    func forward(
        _ h: Tensor,
        position: Int,
        cache: any KVCacheProtocol,
        device: Device
    ) -> Tensor {
        let dim = nHeads * headDim
        let kvDim = nKVHeads * headDim

        // ① Fused QKV projection (GPU) → commit → CPU readback for split.
        let qkvCmd = device.makeCommandBuffer()
        let qkv = wqkv(h, on: qkvCmd)
        qkvCmd.commit()
        qkvCmd.waitUntilCompleted()

        // CPU split: Q [nHeads × headDim], K [nKVHeads × headDim], V [nKVHeads × headDim].
        let qkv32 = toFloat32(qkv)
        let qSlice = Array(qkv32.prefix(dim))
        let kSlice = Array(qkv32[dim ..< dim + kvDim])
        let vSlice = Array(qkv32[(dim + kvDim)...])

        var q = Tensor.empty(shape: [nHeads, headDim], dtype: .f32, device: device)
        var kTensor = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32, device: device)
        let v = Tensor.empty(shape: [nKVHeads, headDim], dtype: .f32, device: device)
        q.copyIn(from: qSlice)
        kTensor.copyIn(from: kSlice)
        v.copyIn(from: vSlice)

        // ② Optional per-head QK-norm (one GPU cmd per head, created inside helper).
        if let qn = qNorm {
            q = applyPerHeadNorm(q, norm: qn, nHeads: nHeads, headDim: headDim, device: device)
        }
        if let kn = kNorm {
            kTensor = applyPerHeadNorm(
                kTensor, norm: kn, nHeads: nKVHeads, headDim: headDim, device: device)
        }

        // ③ CPU RoPE.
        q = applyRoPE(q, rope: rope, position: position, nHeads: nHeads, device: device)
        kTensor = applyRoPE(
            kTensor, rope: rope, position: position, nHeads: nKVHeads, device: device)

        // ④ KV-cache append + SDPA (GPU).
        let sdpaCmd = device.makeCommandBuffer()
        cache.appendOnGPU(kFlat: kTensor, vFlat: v, on: sdpaCmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: sdpaCmd)
        let attnOut = Ops.sdpaDecode(
            q: q, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: sdpaCmd
        )
        sdpaCmd.commit()
        sdpaCmd.waitUntilCompleted()

        // ⑤ Output projection.
        let oCmd = device.makeCommandBuffer()
        let flat = attnOut.reshaped(to: [nHeads * headDim])
        let flatTyped = flat.dtype == h.dtype ? flat : castTensor(flat, to: h.dtype, device: device)
        let out = wo(flatTyped, on: oCmd)
        oCmd.commit()
        oCmd.waitUntilCompleted()
        return out
    }

    // ─── Helpers ────────────────────────────────────────────────────

    /// Read a Tensor to a CPU Float32 array (for split / minor ops).
    private func toFloat32(_ t: Tensor) -> [Float] {
        switch t.dtype {
        case .f32: return t.toArray(as: Float.self)
        case .f16:
            let raw = t.toArray(as: Float16.self)
            return raw.map { Float($0) }
        case .bf16:
            let raw = t.toArray(as: UInt16.self)
            return raw.map { bfloat16ToFloat($0) }
        default: fatalError("FishSpeech: unsupported dtype \(t.dtype)")
        }
    }

    /// Apply RoPE on CPU (position lookup). Returns a new f32 Tensor.
    private func applyRoPE(
        _ qk: Tensor,
        rope: FishSpeechRoPECache,
        position: Int,
        nHeads: Int,
        device: Device
    ) -> Tensor {
        var data = qk.toArray(as: Float.self)  // already f32
        let half = headDim / 2
        let base = position * half
        for h in 0 ..< nHeads {
            let hBase = h * headDim
            for i in 0 ..< half {
                let c = rope.cosTable[base + i]
                let s = rope.sinTable[base + i]
                let x0 = data[hBase + i]
                let x1 = data[hBase + i + half]
                data[hBase + i] = x0 * c - x1 * s
                data[hBase + i + half] = x0 * s + x1 * c
            }
        }
        let out = Tensor.empty(shape: [nHeads, headDim], dtype: .f32, device: device)
        out.copyIn(from: data)
        return out
    }

    /// Apply per-head RMSNorm: each head's [headDim] slice is normalised
    /// independently. Creates one GPU command buffer per head to allow
    /// CPU readback of each normalised slice for writeback into the result.
    private func applyPerHeadNorm(
        _ qk: Tensor,
        norm: RMSNorm,
        nHeads: Int,
        headDim: Int,
        device: Device
    ) -> Tensor {
        let result = Tensor.empty(shape: [nHeads, headDim], dtype: qk.dtype, device: device)
        for h in 0 ..< nHeads {
            let slice = qk.slicedRows(start: h, count: 1).reshaped(to: [headDim])
            let headCmd = device.makeCommandBuffer()
            let normSlice = norm(slice, on: headCmd)
            headCmd.commit()
            headCmd.waitUntilCompleted()
            let dst = result.slicedRows(start: h, count: 1).reshaped(to: [headDim])
            dst.copyIn(from: normSlice.toArray(as: Float.self))
        }
        return result
    }

    /// Cast a Tensor to a target DType using CPU copy (used to convert
    /// f32 attention output back to the model's activation dtype).
    private func castTensor(_ t: Tensor, to dtype: DType, device: Device) -> Tensor {
        let out = Tensor.empty(shape: t.shape, dtype: dtype, device: device)
        let f32 = t.toArray(as: Float.self)
        switch dtype {
        case .f32:
            out.copyIn(from: f32)
        case .f16:
            let halves = f32.map { Float16($0) }
            out.copyIn(from: halves)
        case .bf16:
            let u16 = f32.map { floatToBfloat16($0) }
            out.copyIn(from: u16)
        default:
            fatalError("castTensor: unsupported dtype \(dtype)")
        }
        return out
    }
}

// ─── FFN layer ────────────────────────────────────────────────────────────

/// SwiGLU feed-forward: w2(silu(w1(x)) * w3(x)).
final class FishSpeechFFN: Module {
    let w1, w2, w3: AnyLinear

    init(w1: AnyLinear, w2: AnyLinear, w3: AnyLinear) {
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in w1.parameters() { out.append(("feed_forward.w1.\(k)", v)) }
        for (k, v) in w2.parameters() { out.append(("feed_forward.w2.\(k)", v)) }
        for (k, v) in w3.parameters() { out.append(("feed_forward.w3.\(k)", v)) }
        return out
    }

    func forward(_ x: Tensor, cmd: MTLCommandBuffer) -> Tensor {
        let gate = w1(x, on: cmd)
        let up = w3(x, on: cmd)
        let act = Ops.silu(gate, on: cmd)
        let inner = Ops.mul(act, up, on: cmd)
        return w2(inner, on: cmd)
    }
}

// ─── Transformer block ────────────────────────────────────────────────────

final class FishSpeechBlock: Module {
    let attn: FishSpeechAttentionLayer
    let ffn: FishSpeechFFN
    let attnNorm: RMSNorm
    let ffnNorm: RMSNorm

    init(
        attn: FishSpeechAttentionLayer,
        ffn: FishSpeechFFN,
        attnNorm: RMSNorm,
        ffnNorm: RMSNorm
    ) {
        self.attn = attn
        self.ffn = ffn
        self.attnNorm = attnNorm
        self.ffnNorm = ffnNorm
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out += attn.parameters()
        out += ffn.parameters()
        for (k, v) in attnNorm.parameters() { out.append(("attention_norm.\(k)", v)) }
        for (k, v) in ffnNorm.parameters() { out.append(("ffn_norm.\(k)", v)) }
        return out
    }

    /// Pre-norm residual block. Each GPU op gets its own command buffer so
    /// the CPU can read back intermediate tensors when needed (e.g. for the
    /// QKV split inside attention). This is intentionally eager / simple.
    func forward(
        _ h: Tensor,
        position: Int,
        cache: any KVCacheProtocol,
        device: Device
    ) -> Tensor {
        // Attention pre-norm
        let cmdAN = device.makeCommandBuffer()
        let normedAttn = attnNorm(h, on: cmdAN)
        cmdAN.commit()
        cmdAN.waitUntilCompleted()

        // Attention forward (manages its own cmdbufs internally).
        let attnOut = attn.forward(
            normedAttn, position: position,
            cache: cache, device: device)

        // Residual + FFN pre-norm
        let cmdR1 = device.makeCommandBuffer()
        let h2 = Ops.add(h, attnOut, on: cmdR1)
        cmdR1.commit()
        cmdR1.waitUntilCompleted()

        let cmdFN = device.makeCommandBuffer()
        let normedFFN = ffnNorm(h2, on: cmdFN)
        cmdFN.commit()
        cmdFN.waitUntilCompleted()

        // FFN
        let cmdFFN = device.makeCommandBuffer()
        let ffnOut = ffn.forward(normedFFN, cmd: cmdFFN)
        cmdFFN.commit()
        cmdFFN.waitUntilCompleted()

        // Residual
        let cmdR2 = device.makeCommandBuffer()
        let result = Ops.add(h2, ffnOut, on: cmdR2)
        cmdR2.commit()
        cmdR2.waitUntilCompleted()
        return result
    }
}
