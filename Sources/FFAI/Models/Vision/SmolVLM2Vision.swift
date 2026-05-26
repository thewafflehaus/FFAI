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
// SmolVLM2 — vision tower internals.
//
// Config structs, CPU-path vision ops, `SmolVLM2VisionEncoder`,
// `SmolVLM2Connector`, and `SmolVLM2Model` (the `LanguageModel`-conforming
// engine) live here. SmolVLM2 is unusual in that it does not route through
// `VisionModel` — the engine handles vision prefill internally.
//
// The family orchestrator (registry metadata, `SmolVLM2Error`,
// `SmolVLM2Dense.loadModel`) lives in `Models/SmolVLM2.swift`.
//
// Reference: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
// Upstream Python impl: transformers models/idefics3 (SmolVLM2 = Idefics3)
//
// CPU-path vision encoder:
//   The vision encoder runs once per image during prefill, before the KV cache
//   is populated. GPU kernels exist for the Llama-style language backbone but
//   not for LayerNorm, GELU-tanh, or Conv2d. The vision encoder therefore uses
//   CPU-side BF16→F32 computation reading from shared MTLBuffers. This is
//   acceptable because it is a one-time cost per image, not per decode step.
//
// VL forward flow:
//   1. loadVision: extract image patches → vision encoder → connector → [nPatches, textHidden]
//   2. buildPrefillEmbeds: embed all input tokens, splice image features at image-token positions
//   3. prefillWithEmbeds: run language backbone forward pass for each embedding row
//   4. decode: standard LlamaModel token-by-token decode

import Foundation
import Metal

// ─── Config structs ───────────────────────────────────────────────────────────

/// Vision (SigLIP-style ViT) configuration decoded from "vision_config" sub-object.
public struct SmolVLM2VisionConfig: Sendable {
    public let hiddenSize: Int          // 768
    public let intermediateSize: Int    // 3072
    public let numHiddenLayers: Int     // 12
    public let numAttentionHeads: Int   // 12
    public let numChannels: Int         // 3
    public let patchSize: Int           // 16
    public let imageSize: Int           // 512
    public let layerNormEps: Float      // 1e-6
    public let headDim: Int             // hiddenSize / numAttentionHeads

    public init(from raw: [String: Any]) throws {
        guard let hs = raw["hidden_size"] as? Int else {
            throw SmolVLM2Error.missingVisionConfig("hidden_size")
        }
        guard let ps = raw["patch_size"] as? Int else {
            throw SmolVLM2Error.missingVisionConfig("patch_size")
        }
        guard let imgSz = raw["image_size"] as? Int else {
            throw SmolVLM2Error.missingVisionConfig("image_size")
        }
        guard let nLayers = raw["num_hidden_layers"] as? Int else {
            throw SmolVLM2Error.missingVisionConfig("num_hidden_layers")
        }
        guard let nHeads = raw["num_attention_heads"] as? Int else {
            throw SmolVLM2Error.missingVisionConfig("num_attention_heads")
        }
        let intermediate = raw["intermediate_size"] as? Int ?? 3072
        let nCh = raw["num_channels"] as? Int ?? 3
        let eps = (raw["layer_norm_eps"] as? Double).map(Float.init) ?? 1e-6

        self.hiddenSize         = hs
        self.intermediateSize   = intermediate
        self.numHiddenLayers    = nLayers
        self.numAttentionHeads  = nHeads
        self.numChannels        = nCh
        self.patchSize          = ps
        self.imageSize          = imgSz
        self.layerNormEps       = eps
        self.headDim            = hs / nHeads
    }
}

/// Text (Llama-style) configuration decoded from "text_config" sub-object.
public struct SmolVLM2TextConfig: Sendable {
    public let hiddenSize: Int          // 960
    public let intermediateSize: Int    // 2560
    public let numHiddenLayers: Int     // 32
    public let numAttentionHeads: Int   // 15
    public let numKeyValueHeads: Int    // 5
    public let headDim: Int             // 64
    public let vocabSize: Int           // 49280
    public let maxPositionEmbeddings: Int // 8192
    public let rmsNormEps: Float        // 1e-5
    public let ropeTheta: Float         // 100000
    public let tieWordEmbeddings: Bool

    public init(from raw: [String: Any]) throws {
        guard let hs = raw["hidden_size"] as? Int else {
            throw SmolVLM2Error.missingTextConfig("hidden_size")
        }
        guard let vocab = raw["vocab_size"] as? Int else {
            throw SmolVLM2Error.missingTextConfig("vocab_size")
        }
        guard let nLayers = raw["num_hidden_layers"] as? Int else {
            throw SmolVLM2Error.missingTextConfig("num_hidden_layers")
        }
        guard let nHeads = raw["num_attention_heads"] as? Int else {
            throw SmolVLM2Error.missingTextConfig("num_attention_heads")
        }
        let nKV  = raw["num_key_value_heads"] as? Int ?? nHeads
        let hDim = raw["head_dim"] as? Int ?? (hs / nHeads)
        let inter = raw["intermediate_size"] as? Int ?? 2560
        let maxPos = raw["max_position_embeddings"] as? Int ?? 8192
        let eps = (raw["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-5
        let theta = (raw["rope_theta"] as? Double).map(Float.init) ?? 100_000
        let tieEmbed = raw["tie_word_embeddings"] as? Bool ?? false

        self.hiddenSize             = hs
        self.intermediateSize       = inter
        self.numHiddenLayers        = nLayers
        self.numAttentionHeads      = nHeads
        self.numKeyValueHeads       = nKV
        self.headDim                = hDim
        self.vocabSize              = vocab
        self.maxPositionEmbeddings  = maxPos
        self.rmsNormEps             = eps
        self.ropeTheta              = theta
        self.tieWordEmbeddings      = tieEmbed
    }
}

/// Top-level SmolVLM2 model config.
public struct SmolVLM2Config: Sendable {
    public let visionConfig: SmolVLM2VisionConfig
    public let textConfig: SmolVLM2TextConfig
    /// `scale_factor` controls pixel-shuffle downsampling of vision features
    /// before projection into text embedding space.
    public let scaleFactor: Int         // 4
    /// Token id used as placeholder for image patches in the input sequence.
    public let imageTokenId: Int        // 49190
    public let vocabSize: Int           // 49280

    public init(from raw: [String: Any]) throws {
        guard let vcRaw = raw["vision_config"] as? [String: Any] else {
            throw SmolVLM2Error.missingConfig("vision_config")
        }
        guard let tcRaw = raw["text_config"] as? [String: Any] else {
            throw SmolVLM2Error.missingConfig("text_config")
        }
        self.visionConfig  = try SmolVLM2VisionConfig(from: vcRaw)
        self.textConfig    = try SmolVLM2TextConfig(from: tcRaw)
        self.scaleFactor   = raw["scale_factor"] as? Int ?? 4
        self.imageTokenId  = raw["image_token_id"] as? Int ?? 49190
        self.vocabSize     = raw["vocab_size"] as? Int ?? 49280
    }
}

// ─── CPU-path vision ops ──────────────────────────────────────────────────────
//
// The vision encoder runs once per image, not per decode step, so a CPU-side
// BF16→F32 computation path is acceptable here. All tensors are in shared
// MTLBuffers so the CPU can read/write them directly.

/// Helpers for reading BF16 (bfloat16) values from raw memory.
///
/// BF16 is stored as 16-bit values where the bit pattern corresponds to the
/// upper 16 bits of a 32-bit IEEE float (truncated mantissa, same exponent).
private enum BF16 {
    /// Read a BF16 word from a pointer and expand to Float32.
    @inline(__always)
    static func load(_ p: UnsafeRawPointer, at index: Int) -> Float {
        let raw = p.load(fromByteOffset: index * 2, as: UInt16.self)
        let f32bits = UInt32(raw) << 16
        return Float(bitPattern: f32bits)
    }

    /// Write a Float32 value as a BF16 word (round-to-nearest-even truncation).
    @inline(__always)
    static func store(_ value: Float, to p: UnsafeMutableRawPointer, at index: Int) {
        let bits = value.bitPattern
        // Round-to-nearest-even: if the dropped bits > 0x8000, round up.
        // If they equal 0x8000 (half-way), round to even (check low mantissa bit).
        let droppedBits = bits & 0xFFFF
        var upper = UInt16(bits >> 16)
        if droppedBits > 0x8000 || (droppedBits == 0x8000 && (upper & 1) != 0) {
            upper &+= 1
        }
        p.storeBytes(of: upper, toByteOffset: index * 2, as: UInt16.self)
    }
}

/// Helpers for reading F16 (float16) values from raw memory via UInt16 bit manipulation.
private enum F16 {
    @inline(__always)
    static func load(_ p: UnsafeRawPointer, at index: Int) -> Float {
        let raw = p.load(fromByteOffset: index * 2, as: UInt16.self)
        // Decode IEEE 754 float16: sign(1) | exp(5) | mantissa(10)
        let sign: UInt32 = UInt32(raw & 0x8000) << 16
        let exp16: UInt32 = UInt32((raw >> 10) & 0x1F)
        let mant: UInt32 = UInt32(raw & 0x3FF)
        let exp32: UInt32
        if exp16 == 0 {
            // Subnormal
            if mant == 0 { return Float(bitPattern: sign) }
            var m = mant
            var e: UInt32 = 0
            while (m & 0x400) == 0 { m <<= 1; e += 1 }
            exp32 = (127 - 15 - e + 1) << 23
            let bits = sign | exp32 | ((m & 0x3FF) << 13)
            return Float(bitPattern: bits)
        } else if exp16 == 0x1F {
            // Inf / NaN
            exp32 = 0xFF << 23
        } else {
            exp32 = (exp16 + (127 - 15)) << 23
        }
        return Float(bitPattern: sign | exp32 | (mant << 13))
    }
}

/// Read a 1D tensor of any dtype into a Float array (CPU-side).
private func tensorToFloats(_ t: Tensor) -> [Float] {
    let ptr = t.buffer.contents().advanced(by: t.offset)
    let n = t.elementCount
    var out = [Float](repeating: 0, count: n)
    switch t.dtype {
    case .f32:
        let p = ptr.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { out[i] = p[i] }
    case .bf16:
        for i in 0..<n { out[i] = BF16.load(ptr, at: i) }
    case .f16:
        for i in 0..<n { out[i] = F16.load(ptr, at: i) }
    default:
        fatalError("tensorToFloats: unsupported dtype \(t.dtype)")
    }
    return out
}

/// Write Float array back to a Tensor (CPU-side), respecting the target dtype.
private func floatsToTensor(_ values: [Float], shape: [Int], dtype: DType,
                             device: Device = .shared) -> Tensor {
    let n = shape.reduce(1, *)
    precondition(values.count == n, "floatsToTensor: count mismatch")
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    let ptr = t.buffer.contents()
    switch dtype {
    case .f32:
        let p = ptr.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { p[i] = values[i] }
    case .bf16:
        for i in 0..<n { BF16.store(values[i], to: ptr, at: i) }
    case .f16:
        // F16 write: convert Float → half via a round-trip through BF16 is not correct;
        // use truncation. For simplicity we store as BF16 bits using Float16 bitPattern
        // workaround: use bitPattern of the rounded 16-bit representation.
        for i in 0..<n {
            let f = values[i]
            // Swift does not expose Float16 directly before Swift 5.9 on macOS 13+.
            // Encode via UInt16 using the f32-to-f16 algorithm.
            let bits = f.bitPattern
            let sign: UInt16 = UInt16((bits >> 31) & 1) << 15
            let exp32: Int32 = Int32((bits >> 23) & 0xFF) - 127
            let mant: UInt32 = bits & 0x7FFFFF
            let word: UInt16
            if exp32 < -24 {
                word = sign              // underflow → zero
            } else if exp32 < -14 {
                // Subnormal
                let shift = UInt32(-14 - exp32)
                let m = (0x800000 | mant) >> shift
                word = sign | UInt16(m >> 13)
            } else if exp32 > 15 {
                word = sign | 0x7C00   // overflow → inf
            } else {
                let exp16 = UInt16(exp32 + 15) << 10
                word = sign | exp16 | UInt16(mant >> 13)
            }
            ptr.storeBytes(of: word, toByteOffset: i * 2, as: UInt16.self)
        }
    default:
        fatalError("floatsToTensor: unsupported dtype \(dtype)")
    }
    return t
}

/// Standard LayerNorm (subtract mean, divide by std) with per-channel affine.
/// Input: [n], weight: [n], bias: [n] (optional) → output: [n]
/// This is the full sequence variant used in the vision encoder.
private func layerNorm1D(_ x: [Float], weight: [Float], bias: [Float]?, eps: Float) -> [Float] {
    let n = x.count
    let mean = x.reduce(0, +) / Float(n)
    let variance = x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(n)
    let std = (variance + eps).squareRoot()
    var out = [Float](repeating: 0, count: n)
    if let bias = bias {
        for i in 0..<n { out[i] = ((x[i] - mean) / std) * weight[i] + bias[i] }
    } else {
        for i in 0..<n { out[i] = ((x[i] - mean) / std) * weight[i] }
    }
    return out
}

/// Tanh-approximated GELU: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
private func geluTanh(_ x: Float) -> Float {
    let sqrt2overPi: Float = 0.7978845608028654
    let coeff: Float = 0.044715
    let inner = sqrt2overPi * (x + coeff * x * x * x)
    return x * 0.5 * (1 + tanh(inner))
}

// (CPU `matmul` and `addBias` retired during the 2026-05-24 GPU-GEMM
// migration. Per-layer projections dispatch through `smolVLM2GemmBiased`
// → `Ops.gemm + Ops.add`; the connector projection runs as a single
// `Ops.gemm`. Add them back only if a fresh CPU fallback is needed.)

/// Scaled dot-product attention for the vision encoder.
/// q, k, v: [nHeads, seqLen, headDim] — output: [seqLen, nHeads * headDim]
///
/// Now GPU-resident: one `Ops.sdpaBidirectional(headDim: 64)` dispatch.
/// K/V layout `[nHeads, seqLen, headDim]` matches the kernel's
/// `[nKVHeads, kvStride, headDim]` contract (vision MHA: nQHeads ==
/// nKVHeads, kvStride == seqLen, baseKV == 0). Q is transposed once
/// from `[nHeads, seqLen, headDim]` → `[seqLen, nHeads, headDim]` to
/// match the kernel's Q layout. Output `[seqLen, nHeads, headDim]` is
/// reinterpreted flat as `[seqLen, nHeads*headDim]` for the caller.
private func visionSDPA(q: [Float], k: [Float], v: [Float],
                         nHeads: Int, seqLen: Int, headDim: Int) -> [Float] {
    let scale = 1.0 / Float(headDim).squareRoot()
    let device = Device.shared

    // Transpose Q from [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim].
    var qSeqMajor = [Float](repeating: 0, count: seqLen * nHeads * headDim)
    for h in 0..<nHeads {
        for s in 0..<seqLen {
            let src = (h * seqLen + s) * headDim
            let dst = (s * nHeads + h) * headDim
            for d in 0..<headDim { qSeqMajor[dst + d] = q[src + d] }
        }
    }

    let qT = floatsToTensor(qSeqMajor, shape: [seqLen, nHeads, headDim],
                            dtype: .f32, device: device)
    let kT = floatsToTensor(k, shape: [nHeads, seqLen, headDim],
                            dtype: .f32, device: device)
    let vT = floatsToTensor(v, shape: [nHeads, seqLen, headDim],
                            dtype: .f32, device: device)
    let cmd = device.makeCommandBuffer()
    let outT = Ops.sdpaBidirectional(
        q: qT, k: kT, v: vT,
        nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
        baseKV: 0, nQuery: seqLen, kvStride: seqLen,
        scale: scale, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    return outT.toFloatArray()
}

// ─── Vision encoder layers (GPU GEMM-resident projections) ───────────────────

/// Loaded weights for a single SigLIP vision encoder layer.
///
/// Projection weights (Q/K/V/Out, fc1/fc2) and their biases are uploaded
/// to f32 GPU `Tensor`s at load time so each per-layer projection
/// collapses into a single `Ops.gemm + Ops.add` dispatch over the
/// `[seqLen, dim]` patch batch. LayerNorm weights stay CPU-side; the
/// per-row LayerNorm is cheap relative to the matmul it used to gate.
struct SmolVLM2EncoderLayer {
    // Self-attention projections (weight: [dim, dim], bias: [dim])
    // Stored as f32 GPU tensors after the 2026-05-24 GEMM migration.
    let qW: Tensor; let qB: Tensor
    let kW: Tensor; let kB: Tensor
    let vW: Tensor; let vB: Tensor
    let oW: Tensor; let oB: Tensor
    // MLP: fc1 [intermediate, dim], fc2 [dim, intermediate]
    let fc1W: Tensor; let fc1B: Tensor
    let fc2W: Tensor; let fc2B: Tensor
    // Layer norms (still CPU — applied per row before the GEMMs)
    let ln1W: [Float]; let ln1B: [Float]
    let ln2W: [Float]; let ln2B: [Float]

    let dim: Int
    let intermediate: Int
    let nHeads: Int
    let headDim: Int

    /// Load one encoder layer from the safetensors bundle.
    init(index i: Int, weights: SafeTensorsBundle, cfg: SmolVLM2VisionConfig) throws {
        let p = "vision_model.encoder.layers.\(i)"
        let dim = cfg.hiddenSize
        let inter = cfg.intermediateSize
        let nHeads = cfg.numAttentionHeads
        let headDim = cfg.headDim

        self.dim        = dim
        self.intermediate = inter
        self.nHeads     = nHeads
        self.headDim    = headDim

        // Re-host each projection weight + bias as an f32 GPU Tensor.
        func upW(_ key: String, shape: [Int]) throws -> Tensor {
            let floats = tensorToFloats(try weights.tensor(named: key))
            return floatsToTensor(floats, shape: shape, dtype: .f32)
        }
        func upB(_ key: String) throws -> Tensor {
            let floats = tensorToFloats(try weights.tensor(named: key))
            return floatsToTensor(floats, shape: [floats.count], dtype: .f32)
        }
        qW = try upW("\(p).self_attn.q_proj.weight", shape: [dim, dim])
        qB = try upB("\(p).self_attn.q_proj.bias")
        kW = try upW("\(p).self_attn.k_proj.weight", shape: [dim, dim])
        kB = try upB("\(p).self_attn.k_proj.bias")
        vW = try upW("\(p).self_attn.v_proj.weight", shape: [dim, dim])
        vB = try upB("\(p).self_attn.v_proj.bias")
        oW = try upW("\(p).self_attn.out_proj.weight", shape: [dim, dim])
        oB = try upB("\(p).self_attn.out_proj.bias")

        fc1W = try upW("\(p).mlp.fc1.weight", shape: [inter, dim])
        fc1B = try upB("\(p).mlp.fc1.bias")
        fc2W = try upW("\(p).mlp.fc2.weight", shape: [dim, inter])
        fc2B = try upB("\(p).mlp.fc2.bias")

        ln1W = tensorToFloats(try weights.tensor(named: "\(p).layer_norm1.weight"))
        ln1B = tensorToFloats(try weights.tensor(named: "\(p).layer_norm1.bias"))
        ln2W = tensorToFloats(try weights.tensor(named: "\(p).layer_norm2.weight"))
        ln2B = tensorToFloats(try weights.tensor(named: "\(p).layer_norm2.bias"))
    }

    /// Forward pass: input x is [seqLen, dim] flat. Returns [seqLen, dim] flat.
    /// Each projection dispatches a single `Ops.gemm + Ops.add` on a
    /// shared command buffer — replaces the previous per-(row × col) CPU
    /// matmul loop that bottlenecked SmolVLM2 vision prefill.
    func forward(_ x: [Float], seqLen: Int, eps: Float) -> [Float] {
        let device = Device.shared

        // ─── Self-attention ─────────────────────────────────────
        var h = x
        // Layer norm 1: apply per-row (cheap CPU pass).
        var normed1 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let rowSlice = Array(h[start..<start + dim])
            let normRow = layerNorm1D(rowSlice, weight: ln1W, bias: ln1B, eps: eps)
            normed1.replaceSubrange(start..<start + dim, with: normRow)
        }

        // Upload normed input once, dispatch Q/K/V on a shared command buffer.
        let normedT = floatsToTensor(normed1, shape: [seqLen, dim],
                                      dtype: .f32, device: device)
        let cmd = device.makeCommandBuffer()
        let qT = smolVLM2GemmBiased(input: normedT, weight: qW, bias: qB,
                                     nRows: seqLen, outDim: dim,
                                     device: device, on: cmd)
        let kT = smolVLM2GemmBiased(input: normedT, weight: kW, bias: kB,
                                     nRows: seqLen, outDim: dim,
                                     device: device, on: cmd)
        let vT = smolVLM2GemmBiased(input: normedT, weight: vW, bias: vB,
                                     nRows: seqLen, outDim: dim,
                                     device: device, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let q = qT.toFloatArray()
        let k = kT.toFloatArray()
        let v = vT.toFloatArray()

        // Reshape to [nHeads, seqLen, headDim] for SDPA
        // q is [seqLen, nHeads * headDim]; we re-order to [nHeads, seqLen, headDim]
        var qHeads = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        var kHeads = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        var vHeads = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        for s in 0..<seqLen {
            for nh in 0..<nHeads {
                for d in 0..<headDim {
                    let srcIdx = s * dim + nh * headDim + d
                    let dstIdx = nh * seqLen * headDim + s * headDim + d
                    qHeads[dstIdx] = q[srcIdx]
                    kHeads[dstIdx] = k[srcIdx]
                    vHeads[dstIdx] = v[srcIdx]
                }
            }
        }

        let attnOut = visionSDPA(q: qHeads, k: kHeads, v: vHeads,
                                  nHeads: nHeads, seqLen: seqLen, headDim: headDim)
        // attnOut: [seqLen, dim]
        let attnT = floatsToTensor(attnOut, shape: [seqLen, dim],
                                    dtype: .f32, device: device)
        let cmd2 = device.makeCommandBuffer()
        let oTgpu = smolVLM2GemmBiased(input: attnT, weight: oW, bias: oB,
                                        nRows: seqLen, outDim: dim,
                                        device: device, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        let oOut = oTgpu.toFloatArray()

        // Residual
        for i in 0..<h.count { h[i] += oOut[i] }

        // ─── MLP ─────────────────────────────────────────────────
        var normed2 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let rowSlice = Array(h[start..<start + dim])
            let normRow = layerNorm1D(rowSlice, weight: ln2W, bias: ln2B, eps: eps)
            normed2.replaceSubrange(start..<start + dim, with: normRow)
        }

        let normed2T = floatsToTensor(normed2, shape: [seqLen, dim],
                                       dtype: .f32, device: device)
        let cmd3 = device.makeCommandBuffer()
        let fc1Tgpu = smolVLM2GemmBiased(input: normed2T, weight: fc1W, bias: fc1B,
                                          nRows: seqLen, outDim: intermediate,
                                          device: device, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()
        var fc1Out = fc1Tgpu.toFloatArray()
        // GELU tanh approximation (SigLIP / Idefics3 uses gelu_pytorch_tanh)
        for i in 0..<fc1Out.count { fc1Out[i] = geluTanh(fc1Out[i]) }

        let fc1OutT = floatsToTensor(fc1Out, shape: [seqLen, intermediate],
                                      dtype: .f32, device: device)
        let cmd4 = device.makeCommandBuffer()
        let fc2Tgpu = smolVLM2GemmBiased(input: fc1OutT, weight: fc2W, bias: fc2B,
                                          nRows: seqLen, outDim: dim,
                                          device: device, on: cmd4)
        cmd4.commit()
        cmd4.waitUntilCompleted()
        let fc2Out = fc2Tgpu.toFloatArray()

        // Residual
        for i in 0..<h.count { h[i] += fc2Out[i] }
        return h
    }
}

/// `out = input · weightᵀ + bias` (bias broadcast across `nRows`) as one
/// `Ops.gemm` + `Ops.add` on the supplied command buffer. Bias tile is
/// staged CPU-side and uploaded once per call. Caller commits and reads
/// back.
private func smolVLM2GemmBiased(input: Tensor, weight: Tensor, bias: Tensor,
                                 nRows: Int, outDim: Int, device: Device,
                                 on cmd: MTLCommandBuffer) -> Tensor {
    let out = Ops.gemm(weight: weight, input: input, nRows: nRows, on: cmd)
    let biasVals = bias.toFloatArray()
    var tiled = [Float](repeating: 0, count: nRows * outDim)
    for r in 0..<nRows {
        let base = r * outDim
        for c in 0..<outDim { tiled[base + c] = biasVals[c] }
    }
    let tiledT = floatsToTensor(tiled, shape: [nRows, outDim],
                                 dtype: .f32, device: device)
    return Ops.add(out, tiledT, on: cmd)
}

// ─── Vision encoder ───────────────────────────────────────────────────────────

/// Loaded SmolVLM2 vision encoder (SigLIP-style ViT).
/// Weights are kept as Float arrays for CPU-side computation during prefill.
public final class SmolVLM2VisionEncoder: Module {
    let cfg: SmolVLM2VisionConfig
    // Patch embedding: weight [hiddenSize, numChannels, patchSize, patchSize], bias [hiddenSize]
    let patchW: [Float]
    let patchB: [Float]
    // Position embedding: [numPatches, hiddenSize]
    let posEmbed: [Float]
    let numPatches: Int
    // Post layer norm
    let postLnW: [Float]
    let postLnB: [Float]
    // Encoder layers
    let layers: [SmolVLM2EncoderLayer]

    // Keep original tensors for parameters()
    private let patchWTensor: Tensor
    private let patchBTensor: Tensor
    private let posEmbedTensor: Tensor
    private let postLnWTensor: Tensor
    private let postLnBTensor: Tensor

    public init(cfg: SmolVLM2VisionConfig, weights: SafeTensorsBundle) throws {
        self.cfg = cfg

        // Patch embedding weight shape: [hiddenSize, numChannels, patchSize, patchSize]
        let patchWTens = try weights.tensor(named: "vision_model.embeddings.patch_embedding.weight")
        let patchBTens = try weights.tensor(named: "vision_model.embeddings.patch_embedding.bias")
        let posEmbedTens = try weights.tensor(named: "vision_model.embeddings.position_embedding.weight")
        let postLnWTens = try weights.tensor(named: "vision_model.post_layernorm.weight")
        let postLnBTens = try weights.tensor(named: "vision_model.post_layernorm.bias")

        self.patchWTensor     = patchWTens
        self.patchBTensor     = patchBTens
        self.posEmbedTensor   = posEmbedTens
        self.postLnWTensor    = postLnWTens
        self.postLnBTensor    = postLnBTens

        self.patchW    = tensorToFloats(patchWTens)
        self.patchB    = tensorToFloats(patchBTens)
        self.posEmbed  = tensorToFloats(posEmbedTens)

        let n = (cfg.imageSize / cfg.patchSize) * (cfg.imageSize / cfg.patchSize)
        self.numPatches = n

        self.postLnW = tensorToFloats(postLnWTens)
        self.postLnB = tensorToFloats(postLnBTens)

        var layers: [SmolVLM2EncoderLayer] = []
        layers.reserveCapacity(cfg.numHiddenLayers)
        for i in 0..<cfg.numHiddenLayers {
            layers.append(try SmolVLM2EncoderLayer(index: i, weights: weights, cfg: cfg))
        }
        self.layers = layers
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out.append(("vision_model.embeddings.patch_embedding.weight", patchWTensor))
        out.append(("vision_model.embeddings.patch_embedding.bias",   patchBTensor))
        out.append(("vision_model.embeddings.position_embedding.weight", posEmbedTensor))
        out.append(("vision_model.post_layernorm.weight", postLnWTensor))
        out.append(("vision_model.post_layernorm.bias",   postLnBTensor))
        // Layer weights are not exposed here (already loaded into Float arrays)
        return out
    }

    /// Extract patch embeddings from an image tile.
    ///
    /// `pixels` is [height, width, channels] float32 after normalization.
    /// Returns [numPatches, hiddenSize] as Float array.
    ///
    /// Conv2d with stride == kernel (patch extraction) is equivalent to slicing
    /// non-overlapping windows and projecting each through the patch weight matrix.
    func patchEmbeddings(pixels: [Float], height: Int, width: Int) -> [Float] {
        let ps  = cfg.patchSize
        let dim = cfg.hiddenSize
        let nC  = cfg.numChannels
        let nRows = height / ps
        let nCols = width  / ps
        let nPatch = nRows * nCols
        // patchW shape: [dim, nC, ps, ps] — each row is a filter of size nC*ps*ps
        let filterSize = nC * ps * ps

        var out = [Float](repeating: 0, count: nPatch * dim)

        for pr in 0..<nRows {
            for pc in 0..<nCols {
                let pIdx = pr * nCols + pc
                // Extract one patch: [ps, ps, nC]
                var patch = [Float](repeating: 0, count: filterSize)
                for r in 0..<ps {
                    for c in 0..<ps {
                        let pixRow = pr * ps + r
                        let pixCol = pc * ps + c
                        for ch in 0..<nC {
                            // pixels layout: [height, width, nC] row-major
                            patch[r * ps * nC + c * nC + ch] = pixels[pixRow * width * nC + pixCol * nC + ch]
                        }
                    }
                }
                // Rearrange patch to [nC, ps, ps] to match filter layout
                var patchCHW = [Float](repeating: 0, count: filterSize)
                for ch in 0..<nC {
                    for r in 0..<ps {
                        for c in 0..<ps {
                            patchCHW[ch * ps * ps + r * ps + c] = patch[r * ps * nC + c * nC + ch]
                        }
                    }
                }
                // Dot patchCHW [filterSize] with each filter row [filterSize] → [dim]
                for d in 0..<dim {
                    var dot: Float = 0
                    let fRow = d * filterSize
                    for j in 0..<filterSize { dot += patchW[fRow + j] * patchCHW[j] }
                    out[pIdx * dim + d] = dot + patchB[d]
                }
            }
        }
        return out
    }

    /// Run the full vision encoder on a single image tile (already normalized).
    ///
    /// `pixels` is [height, width, channels] Float32 after normalize.
    /// Returns [numPatches, hiddenSize] Float — the pooler output after post-LayerNorm.
    func encode(pixels: [Float], height: Int, width: Int) -> [Float] {
        let dim = cfg.hiddenSize
        let nPatch = (height / cfg.patchSize) * (width / cfg.patchSize)

        // Patch embeddings + position embeddings
        var x = patchEmbeddings(pixels: pixels, height: height, width: width)
        // x: [nPatch, dim]
        for i in 0..<nPatch {
            for d in 0..<dim {
                x[i * dim + d] += posEmbed[i * dim + d]
            }
        }

        // Transformer layers
        for layer in layers {
            x = layer.forward(x, seqLen: nPatch, eps: cfg.layerNormEps)
        }

        // Post layer norm (applied per patch)
        var postNormed = [Float](repeating: 0, count: nPatch * dim)
        for row in 0..<nPatch {
            let start = row * dim
            let rowSlice = Array(x[start..<start + dim])
            let normRow = layerNorm1D(rowSlice, weight: postLnW, bias: postLnB,
                                      eps: cfg.layerNormEps)
            postNormed.replaceSubrange(start..<start + dim, with: normRow)
        }
        return postNormed
    }
}

// ─── Connector (pixel-shuffle + MLP projection) ───────────────────────────────

/// SmolVLM2 connector: pixel-shuffle (scale_factor=4) then a linear projection
/// from (visionHidden * scaleFactor²) → textHidden.
public final class SmolVLM2Connector: Module {
    let scaleFactor: Int
    /// f32 GPU tensor view of the projector weight `[textHidden, visionHidden·sf²]`.
    /// The CPU `[Float]` copy was retired during the GPU-GEMM migration;
    /// `projWTensor` keeps the original checkpoint reference for `parameters()`.
    private let projWGpu: Tensor
    let projWTensor: Tensor

    public init(cfg: SmolVLM2Config, weights: SafeTensorsBundle) throws {
        self.scaleFactor = cfg.scaleFactor
        let projTensor = try weights.tensor(named: "connector.modality_projection.proj.weight")
        self.projWTensor = projTensor
        let floats = tensorToFloats(projTensor)
        self.projWGpu = floatsToTensor(floats, shape: projTensor.shape, dtype: .f32)
    }

    public func parameters() -> [(String, Tensor)] {
        [("connector.modality_projection.proj.weight", projWTensor)]
    }

    /// Pixel-shuffle then project.
    ///
    /// `visionOut` is [nPatches, visionHidden] float (e.g., [1024, 768]).
    /// After pixel-shuffle with scaleFactor=4: [nPatches/sf², visionHidden*sf²]
    /// After projection: [nPatches/sf², textHidden]
    func forward(visionOut: [Float], nPatches: Int, visionHidden: Int, textHidden: Int) -> [Float] {
        let sf = scaleFactor
        let sf2 = sf * sf

        // Pixel-shuffle: interpret [nPatches, visionHidden] as [side, side, visionHidden]
        // and reorganize to [side/sf, side/sf, visionHidden * sf²]
        let side = Int(Double(nPatches).squareRoot())
        precondition(side * side == nPatches, "SmolVLM2 connector: nPatches must be a perfect square")
        precondition(side % sf == 0, "SmolVLM2 connector: side (\(side)) must be divisible by scale_factor (\(sf))")

        let newSide = side / sf
        let newHidden = visionHidden * sf2
        let newNPatches = newSide * newSide

        // Pixel shuffle rearrangement matching Python:
        //   reshaped = x.reshape(B, side, side, embed_dim)
        //   reshaped = reshaped.reshape(B, side, side/sf, embed_dim*sf)
        //   reshaped = reshaped.transpose(0, 2, 1, 3)
        //   reshaped = reshaped.reshape(B, side/sf, side/sf, embed_dim*sf²)
        //   reshaped = reshaped.transpose(0, 2, 1, 3)
        //   reshaped = reshaped.reshape(B, seq/sf², embed_dim*sf²)
        // (B=1 throughout, so we work without the batch dim)

        // Step 1: [side, side, visionHidden]
        // Step 2: [side, side/sf, visionHidden*sf]
        var step2 = [Float](repeating: 0, count: side * newSide * visionHidden * sf)
        for r in 0..<side {
            for c2 in 0..<newSide {
                for e in 0..<visionHidden {
                    for s in 0..<sf {
                        let srcRow = r
                        let srcCol = c2 * sf + s
                        let srcIdx = srcRow * side * visionHidden + srcCol * visionHidden + e
                        let dstIdx = r * newSide * visionHidden * sf + c2 * visionHidden * sf + e * sf + s
                        step2[dstIdx] = visionOut[srcIdx]
                    }
                }
            }
        }

        // Step 3: transpose(0, 2, 1, 3) → [side/sf, side, visionHidden*sf]
        var step3 = [Float](repeating: 0, count: newSide * side * visionHidden * sf)
        for c2 in 0..<newSide {
            for r in 0..<side {
                for d in 0..<(visionHidden * sf) {
                    let srcIdx = r * newSide * visionHidden * sf + c2 * visionHidden * sf + d
                    let dstIdx = c2 * side * visionHidden * sf + r * visionHidden * sf + d
                    step3[dstIdx] = step2[srcIdx]
                }
            }
        }

        // Step 4: reshape to [side/sf, side/sf, visionHidden*sf²]
        // = further reshape of side dim: [side/sf, side/sf, sf, visionHidden*sf] then last two merged
        var step4 = [Float](repeating: 0, count: newSide * newSide * newHidden)
        for r2 in 0..<newSide {
            for c2 in 0..<newSide {
                for s in 0..<sf {
                    for e in 0..<(visionHidden * sf) {
                        let srcRow = r2
                        let srcInnerRow = c2 * sf + s
                        let srcIdx = srcRow * side * visionHidden * sf + srcInnerRow * visionHidden * sf + e
                        let dstIdx = r2 * newSide * newHidden + c2 * newHidden + s * visionHidden * sf + e
                        step4[dstIdx] = step3[srcIdx]
                    }
                }
            }
        }

        // Step 5: transpose(0, 2, 1, 3) on [side/sf, side/sf, visionHidden*sf²]
        // interpreted as [newSide, newSide, newHidden] → already flattened, no extra transpose needed
        // (the reshape in step 4 already has the right layout; second transpose is identity for 2D patch grid)

        // Step 6: reshape to [newNPatches, newHidden] (already that shape from step 4)
        // = step4 as-is

        // Linear projection: [newNPatches, newHidden] × projWGpu^T → [newNPatches, textHidden]
        // One GPU GEMM dispatch instead of the nested-loop CPU matmul.
        let device = Device.shared
        let inputT = floatsToTensor(step4, shape: [newNPatches, newHidden],
                                     dtype: .f32, device: device)
        let cmd = device.makeCommandBuffer()
        let outT = Ops.gemm(weight: projWGpu, input: inputT,
                            nRows: newNPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        _ = textHidden  // textHidden is encoded in projWGpu.shape[0]
        return outT.toFloatArray()
    }
}

// ─── SmolVLM2Model (implements LanguageModel) ─────────────────────────────────
//
// Wraps LlamaModel for the text backbone. Adds vision prefill capability:
// `forwardVisionPrefill` encodes an image tile and splices the visual features
// into the residual stream before the normal KV-cache decode loop runs.

public final class SmolVLM2Model: LanguageModel {
    public let llamaModel: LlamaModel
    public let visionEncoder: SmolVLM2VisionEncoder
    public let connector: SmolVLM2Connector
    public let cfg: SmolVLM2Config
    public let device: Device

    // LanguageModel conformance — delegate to the Llama backbone
    public var hidden:    Int { llamaModel.hidden }
    public var nLayers:   Int { llamaModel.nLayers }
    public var nHeads:    Int { llamaModel.nHeads }
    public var nKVHeads:  Int { llamaModel.nKVHeads }
    public var headDim:   Int { llamaModel.headDim }
    public var vocab:     Int { llamaModel.vocab }
    public var maxSeq:    Int { llamaModel.maxSeq }
    public var dtype:     DType { llamaModel.dtype }

    public init(llamaModel: LlamaModel,
                visionEncoder: SmolVLM2VisionEncoder,
                connector: SmolVLM2Connector,
                cfg: SmolVLM2Config,
                device: Device) {
        self.llamaModel    = llamaModel
        self.visionEncoder = visionEncoder
        self.connector     = connector
        self.cfg           = cfg
        self.device        = device
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out += visionEncoder.parameters()
        out += connector.parameters()
        out += llamaModel.parameters()
        return out
    }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        llamaModel.makeLayerCaches(maxSeq: maxSeq, device: device)
    }

    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        llamaModel.forward(tokenId: tokenId, position: position,
                           caches: caches, on: cmd, device: device)
    }

    public func forwardSample(tokenId: Int, position: Int,
                              caches: [any LayerCacheProtocol], device: Device) -> Int {
        llamaModel.forwardSample(tokenId: tokenId, position: position,
                                  caches: caches, device: device)
    }

    public func forwardSampleCategorical(
        tokenId: Int, position: Int, caches: [any LayerCacheProtocol],
        temperature: Float, uniformDraw: Float, device: Device
    ) -> Int {
        llamaModel.forwardSampleCategorical(
            tokenId: tokenId, position: position, caches: caches,
            temperature: temperature, uniformDraw: uniformDraw, device: device
        )
    }

    // ─── Vision prefill ──────────────────────────────────────────────────────

    /// Encode a single image tile and return visual feature embeddings.
    ///
    /// `pixels` is a [height, width, channels] Float32 array after normalization
    /// (mean/std subtraction). Returns [numImageTokens, textHidden] Float32
    /// embeddings that should be spliced into the text sequence at image-token
    /// positions before the language model decode loop.
    ///
    /// The number of image tokens equals nPatches / scaleFactor² where nPatches
    /// is (imageSize / patchSize)².
    public func encodeImage(pixels: [Float],
                             height: Int, width: Int) -> [Float] {
        let vc = cfg.visionConfig
        let nPatches = (height / vc.patchSize) * (width / vc.patchSize)
        let sf2 = cfg.scaleFactor * cfg.scaleFactor
        let nImageTokens = nPatches / sf2

        // Run vision encoder
        let visionFeatures = visionEncoder.encode(pixels: pixels,
                                                   height: height, width: width)
        // Run connector: pixel-shuffle + projection
        let imageEmbeds = connector.forward(
            visionOut: visionFeatures,
            nPatches: nPatches,
            visionHidden: vc.hiddenSize,
            textHidden: cfg.textConfig.hiddenSize
        )
        precondition(imageEmbeds.count == nImageTokens * cfg.textConfig.hiddenSize,
                     "SmolVLM2: image embeds shape mismatch")
        return imageEmbeds
    }

    /// Number of text-stream tokens one video frame contributes to the
    /// prompt. Equals `nPatches / scaleFactor²` where
    /// `nPatches = (imageSize / patchSize)²`.
    ///
    /// SmolVLM2 does not use a separate video token id — each frame is
    /// encoded as an independent image using the same `<image>` placeholder
    /// (cfg.imageTokenId). The caller should place
    /// `frameCount × imageTokensPerFrame` consecutive image-token
    /// placeholders in the prompt before calling `encodeVideoFrames`.
    public var imageTokensPerFrame: Int {
        let vc = cfg.visionConfig
        let nPatches = (vc.imageSize / vc.patchSize) * (vc.imageSize / vc.patchSize)
        return nPatches / (cfg.scaleFactor * cfg.scaleFactor)
    }

    /// Encode a sequence of video frames and return the concatenated
    /// visual-feature embeddings ready for `prefillWithImage`.
    ///
    /// Each `pixels` element is a `[height * width * 3]` Float32 array
    /// (HWC, values in [-1, 1] after SmolVLM2 normalization). Returns a
    /// flat `[frameCount × imageTokensPerFrame × textHidden]` Float32
    /// array suitable for direct use with `prefillWithImage`.
    ///
    /// SmolVLM2 encodes each frame independently through the same SigLIP
    /// ViT + pixel-shuffle connector as a single image (unlike Qwen
    /// 2/2.5/3 VL which folds frames into a temporal-patch axis). The
    /// per-frame `[imageTokensPerFrame × textHidden]` embedding slices
    /// are concatenated in display order.
    public func encodeVideoFrames(
        frames: [[Float]],
        height: Int, width: Int
    ) -> [Float] {
        precondition(!frames.isEmpty,
                     "SmolVLM2Model.encodeVideoFrames: expected at least one frame")
        // Encode each frame independently — reuse the single-image path.
        var allEmbeds: [Float] = []
        allEmbeds.reserveCapacity(frames.count * imageTokensPerFrame * cfg.textConfig.hiddenSize)
        for pixels in frames {
            let frameEmbeds = encodeImage(pixels: pixels, height: height, width: width)
            allEmbeds.append(contentsOf: frameEmbeds)
        }
        return allEmbeds
    }

    /// Prefill the KV cache with a mixed sequence of text tokens and image embeddings.
    ///
    /// `tokenIds` is the full input token sequence including image-token placeholders
    /// (cfg.imageTokenId). `imageEmbeds` is [nImageTokens, textHidden] float, returned
    /// by `encodeImage`. The image embeddings replace image placeholder tokens in order.
    ///
    /// Returns the next-token logits for the last position (ready for decode).
    ///
    /// This prefill runs one forward pass per token (the same decode path) with
    /// image embeddings substituted in place of embedding-table lookups at image
    /// placeholder positions. This is correct for the single-image case and avoids
    /// needing a separate batched prefill path.
    public func prefillWithImage(
        tokenIds: [Int],
        imageEmbeds: [Float],
        caches: [any LayerCacheProtocol],
        device: Device
    ) -> Tensor {
        let textHidden = cfg.textConfig.hiddenSize
        let imageTokenId = cfg.imageTokenId
        let cmd = device.makeCommandBuffer()
        let _ = cmd  // use the cmd buffer for GPU operations below

        var imageIdx = 0
        var lastLogits: Tensor = Tensor.empty(shape: [vocab], dtype: dtype, device: device)
        let seqLen = tokenIds.count

        for (pos, tokenId) in tokenIds.enumerated() {
            if tokenId == imageTokenId && imageIdx < (imageEmbeds.count / textHidden) {
                // Substitute image embedding for this position
                let embedStart = imageIdx * textHidden
                let embedSlice = Array(imageEmbeds[embedStart..<embedStart + textHidden])
                imageIdx += 1

                // Write the float embedding into a GPU tensor then run layers
                let h = floatsToTensor(embedSlice, shape: [textHidden], dtype: dtype, device: device)
                // Run the Llama layer stack on this embedding directly
                lastLogits = forwardFromEmbedding(h, position: pos, caches: caches, device: device)
            } else {
                // Normal token: use embedding table
                lastLogits = llamaModel.forward(
                    tokenId: tokenId, position: pos, caches: caches, device: device
                )
            }

            if pos < seqLen - 1 {
                try? Task.checkCancellation()
            }
        }

        return lastLogits
    }

    /// Run the Llama layer stack on a pre-computed embedding vector.
    /// `embedding` is [hidden] in the model's activation dtype.
    ///
    /// Used during VL prefill to process image-feature tokens that bypass the
    /// normal embedding table lookup.
    public func forwardFromEmbedding(_ embedding: Tensor, position: Int,
                                      caches: [any LayerCacheProtocol],
                                      device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        var h = embedding.reshaped(to: [llamaModel.hidden])

        // Cast to activation dtype if needed
        // (the floatsToTensor call above writes in self.dtype, so this is usually a no-op)

        for (i, layer) in llamaModel.layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed  = llamaModel.finalNorm(h, on: cmd)
        let logits  = llamaModel.lmHead(normed, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    /// Multi-token forward — delegates to LlamaModel's optimised
    /// chunked path. SmolVLM2's text backbone is a LlamaModel, so the
    /// text-only AR prefill picks up the full TTFT win
    /// (batched Ops.gemm projections + one Ops.sdpaMulti per layer)
    /// for free. The vision-prefill image-substitution path stays as
    /// its own per-token routine.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        llamaModel.forwardMulti(tokenIds: tokenIds, startingAt: position,
                                caches: caches, on: cmd, device: device)
    }
}
