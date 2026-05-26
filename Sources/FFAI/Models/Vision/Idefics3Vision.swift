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
// Idefics3 vision internals — config structs, CPU vision ops, encoder,
// connector, key remapper, and Idefics3Model.
//
// The family orchestrator (`enum Idefics3`, `enum Idefics3Error`,
// `struct Idefics3Dense`) lives in `Models/Idefics3.swift`. This file
// contains the implementation types:
//   • Idefics3VisionConfig / Idefics3TextConfig / Idefics3Config — decoded
//     from the checkpoint's config.json sub-objects.
//   • CPU BF16/F16/F32 helpers — loadBF16 / storeBF16 / loadF16 /
//     idefics3TensorToFloats / idefics3FloatsToTensor.
//   • CPU vision primitives — idefics3LayerNorm1D / idefics3GeluTanh /
//     idefics3VisionSDPA.
//   • Idefics3EncoderLayer — one SigLIP-style ViT block (GPU GEMM projections
//     + CPU LayerNorm).
//   • idefics3GemmBiased — helper: Ops.gemm + tiled bias add.
//   • Idefics3VisionEncoder — full SigLIP-style ViT (patch embed + blocks).
//   • Idefics3Connector — pixel-shuffle + linear projection (GPU GEMM).
//   • loadIdefics3Linear / loadIdefics3Embedding — layer-load helpers for
//     the Idefics3RemappedBundle wrapper.
//   • Idefics3RemappedBundle — transparent HF→FFAI key remapper.
//   • Idefics3Model — the assembled VLM: SigLIP encoder + connector + Llama
//     backbone, with encodeImage / prefillWithImage / forwardFromEmbedding.

import Foundation
import Metal

// ─── Config structs ───────────────────────────────────────────────────────────

/// Vision (SigLIP-style ViT) configuration decoded from "vision_config" sub-object.
public struct Idefics3VisionConfig: Sendable {
    public let hiddenSize: Int          // 1152 for Idefics3-8B
    public let intermediateSize: Int    // 4304
    public let numHiddenLayers: Int     // 27
    public let numAttentionHeads: Int   // 16
    public let numChannels: Int         // 3
    public let patchSize: Int           // 14
    public let imageSize: Int           // 364
    public let layerNormEps: Float      // 1e-6
    public let headDim: Int             // hiddenSize / numAttentionHeads

    public init(from raw: [String: Any]) throws {
        guard let hs = raw["hidden_size"] as? Int else {
            throw Idefics3Error.missingVisionConfig("hidden_size")
        }
        guard let ps = raw["patch_size"] as? Int else {
            throw Idefics3Error.missingVisionConfig("patch_size")
        }
        guard let imgSz = raw["image_size"] as? Int else {
            throw Idefics3Error.missingVisionConfig("image_size")
        }
        guard let nLayers = raw["num_hidden_layers"] as? Int else {
            throw Idefics3Error.missingVisionConfig("num_hidden_layers")
        }
        guard let nHeads = raw["num_attention_heads"] as? Int else {
            throw Idefics3Error.missingVisionConfig("num_attention_heads")
        }
        let intermediate = raw["intermediate_size"] as? Int ?? (hs * 4)
        let nCh  = raw["num_channels"] as? Int ?? 3
        let eps  = (raw["layer_norm_eps"] as? Double).map(Float.init) ?? 1e-6

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
public struct Idefics3TextConfig: Sendable {
    public let hiddenSize: Int          // 4096 for Idefics3-8B
    public let intermediateSize: Int    // 14336
    public let numHiddenLayers: Int     // 32
    public let numAttentionHeads: Int   // 32
    public let numKeyValueHeads: Int    // 8
    public let headDim: Int             // 128
    public let vocabSize: Int           // 128259
    public let maxPositionEmbeddings: Int // 8192
    public let rmsNormEps: Float        // 1e-5
    public let ropeTheta: Float         // 500_000
    public let tieWordEmbeddings: Bool

    public init(from raw: [String: Any]) throws {
        guard let hs = raw["hidden_size"] as? Int else {
            throw Idefics3Error.missingTextConfig("hidden_size")
        }
        guard let vocab = raw["vocab_size"] as? Int else {
            throw Idefics3Error.missingTextConfig("vocab_size")
        }
        guard let nLayers = raw["num_hidden_layers"] as? Int else {
            throw Idefics3Error.missingTextConfig("num_hidden_layers")
        }
        guard let nHeads = raw["num_attention_heads"] as? Int else {
            throw Idefics3Error.missingTextConfig("num_attention_heads")
        }
        let nKV    = raw["num_key_value_heads"] as? Int ?? nHeads
        let hDim   = raw["head_dim"] as? Int ?? (hs / nHeads)
        let inter  = raw["intermediate_size"] as? Int ?? (hs * 4)
        let maxPos = raw["max_position_embeddings"] as? Int ?? 8192
        let eps    = (raw["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-5
        let theta  = (raw["rope_theta"] as? Double).map(Float.init) ?? 500_000
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

/// Top-level Idefics3 model config.
public struct Idefics3Config: Sendable {
    public let visionConfig: Idefics3VisionConfig
    public let textConfig: Idefics3TextConfig
    /// `scale_factor` controls pixel-shuffle downsampling of vision features
    /// before projection into text embedding space. Default is 2 for Idefics3
    /// (SmolVLM2 uses 4).
    public let scaleFactor: Int          // 2
    /// Token id used as placeholder for image patches in the input sequence.
    public let imageTokenId: Int         // 49153
    public let vocabSize: Int            // 128259

    public init(from raw: [String: Any]) throws {
        guard let vcRaw = raw["vision_config"] as? [String: Any] else {
            throw Idefics3Error.missingConfig("vision_config")
        }
        guard let tcRaw = raw["text_config"] as? [String: Any] else {
            throw Idefics3Error.missingConfig("text_config")
        }
        self.visionConfig = try Idefics3VisionConfig(from: vcRaw)
        self.textConfig   = try Idefics3TextConfig(from: tcRaw)
        self.scaleFactor  = raw["scale_factor"] as? Int ?? 2
        self.imageTokenId = raw["image_token_id"] as? Int
            ?? raw["image_token_index"] as? Int
            ?? 49153
        self.vocabSize    = raw["vocab_size"] as? Int ?? 128259
    }
}

// ─── CPU-path BF16 / F16 / F32 helpers ───────────────────────────────────────
//
// Shared with SmolVLM2; duplicated here to avoid cross-file private dependencies.
// These run once per image during prefill — not per decode step.

/// Read a BF16 word from a pointer and expand to Float32.
@inline(__always)
private func loadBF16(_ p: UnsafeRawPointer, at index: Int) -> Float {
    let raw = p.load(fromByteOffset: index * 2, as: UInt16.self)
    return Float(bitPattern: UInt32(raw) << 16)
}

/// Write a Float32 as BF16 (round-to-nearest-even).
@inline(__always)
private func storeBF16(_ value: Float, to p: UnsafeMutableRawPointer, at index: Int) {
    let bits = value.bitPattern
    let droppedBits = bits & 0xFFFF
    var upper = UInt16(bits >> 16)
    if droppedBits > 0x8000 || (droppedBits == 0x8000 && (upper & 1) != 0) {
        upper &+= 1
    }
    p.storeBytes(of: upper, toByteOffset: index * 2, as: UInt16.self)
}

/// Read an F16 word from a pointer and expand to Float32.
@inline(__always)
private func loadF16(_ p: UnsafeRawPointer, at index: Int) -> Float {
    let raw = p.load(fromByteOffset: index * 2, as: UInt16.self)
    let sign:  UInt32 = UInt32(raw & 0x8000) << 16
    let exp16: UInt32 = UInt32((raw >> 10) & 0x1F)
    let mant:  UInt32 = UInt32(raw & 0x3FF)
    if exp16 == 0 {
        if mant == 0 { return Float(bitPattern: sign) }
        var m = mant; var e: UInt32 = 0
        while (m & 0x400) == 0 { m <<= 1; e += 1 }
        return Float(bitPattern: sign | ((127 - 15 - e + 1) << 23) | ((m & 0x3FF) << 13))
    }
    let exp32: UInt32 = exp16 == 0x1F ? 0xFF << 23 : (exp16 + (127 - 15)) << 23
    return Float(bitPattern: sign | exp32 | (mant << 13))
}

/// Read all elements of a 1-D (or flat multi-D) Tensor into a Float array.
private func idefics3TensorToFloats(_ t: Tensor) -> [Float] {
    let ptr = t.buffer.contents().advanced(by: t.offset)
    let n   = t.elementCount
    var out = [Float](repeating: 0, count: n)
    switch t.dtype {
    case .f32:
        let p = ptr.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { out[i] = p[i] }
    case .bf16:
        for i in 0..<n { out[i] = loadBF16(ptr, at: i) }
    case .f16:
        for i in 0..<n { out[i] = loadF16(ptr, at: i) }
    default:
        fatalError("idefics3TensorToFloats: unsupported dtype \(t.dtype)")
    }
    return out
}

/// Write a Float array into a new Tensor.
private func idefics3FloatsToTensor(_ values: [Float], shape: [Int], dtype: DType,
                                    device: Device = .shared) -> Tensor {
    let n = shape.reduce(1, *)
    precondition(values.count == n, "idefics3FloatsToTensor: count mismatch")
    let t   = Tensor.empty(shape: shape, dtype: dtype, device: device)
    let ptr = t.buffer.contents()
    switch dtype {
    case .f32:
        let p = ptr.bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { p[i] = values[i] }
    case .bf16:
        for i in 0..<n { storeBF16(values[i], to: ptr, at: i) }
    case .f16:
        for i in 0..<n {
            let f   = values[i]
            let bits = f.bitPattern
            let sign: UInt16  = UInt16((bits >> 31) & 1) << 15
            let exp32: Int32  = Int32((bits >> 23) & 0xFF) - 127
            let mant: UInt32  = bits & 0x7FFFFF
            let word: UInt16
            if exp32 < -24 {
                word = sign
            } else if exp32 < -14 {
                let shift = UInt32(-14 - exp32)
                let m = (0x800000 | mant) >> shift
                word = sign | UInt16(m >> 13)
            } else if exp32 > 15 {
                word = sign | 0x7C00
            } else {
                word = sign | (UInt16(exp32 + 15) << 10) | UInt16(mant >> 13)
            }
            ptr.storeBytes(of: word, toByteOffset: i * 2, as: UInt16.self)
        }
    default:
        fatalError("idefics3FloatsToTensor: unsupported dtype \(dtype)")
    }
    return t
}

// ─── CPU vision primitives ────────────────────────────────────────────────────

/// Standard LayerNorm with per-channel affine. Input: [n], weight/bias: [n].
private func idefics3LayerNorm1D(_ x: [Float], weight: [Float], bias: [Float],
                                  eps: Float) -> [Float] {
    let n    = x.count
    let mean = x.reduce(0, +) / Float(n)
    let variance = x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(n)
    let std  = (variance + eps).squareRoot()
    var out  = [Float](repeating: 0, count: n)
    for i in 0..<n { out[i] = ((x[i] - mean) / std) * weight[i] + bias[i] }
    return out
}

/// Tanh-approximated GELU: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715·x³))).
/// Idefics3 uses `gelu_pytorch_tanh` activation in its vision MLP.
@inline(__always)
private func idefics3GeluTanh(_ x: Float) -> Float {
    let sqrt2OverPi: Float = 0.7978845608028654
    let coeff: Float = 0.044715
    let inner = sqrt2OverPi * (x + coeff * x * x * x)
    return x * 0.5 * (1 + tanh(inner))
}

// (CPU `idefics3Matmul` and `idefics3AddBias` were retired during the
// 2026-05-24 GPU-GEMM migration. Every per-layer projection now dispatches
// through `idefics3GemmBiased` → `Ops.gemm + Ops.add`, and the connector
// projection runs as a single `Ops.gemm`. Reintroduce them only if a new
// CPU-side fallback is genuinely needed.)

/// Scaled dot-product attention for the vision encoder.
/// q, k, v: [nHeads, seqLen, headDim] — output: [seqLen, nHeads * headDim]
///
/// Now GPU-resident: dispatches one `Ops.sdpaBidirectional(headDim: 72)`
/// kernel. The input layout `[nHeads, seqLen, headDim]` matches the
/// kernel's `[nKVHeads, kvStride, headDim]` K/V contract exactly
/// (vision-tower MHA: nQHeads == nKVHeads, kvStride == seqLen,
/// baseKV == 0). Output `[seqLen, nHeads, headDim]` is flattened to
/// `[seqLen, nHeads * headDim]` for the caller.
private func idefics3VisionSDPA(q: [Float], k: [Float], v: [Float],
                                 nHeads: Int, seqLen: Int, headDim: Int) -> [Float] {
    let scale = 1.0 / Float(headDim).squareRoot()
    let device = Device.shared

    // Wrap Q in [seqLen, nHeads, headDim] layout (kernel's Q contract).
    // The CPU buffer was [nHeads, seqLen, headDim], so transpose here.
    var qSeqMajor = [Float](repeating: 0, count: seqLen * nHeads * headDim)
    for h in 0..<nHeads {
        for s in 0..<seqLen {
            let src = (h * seqLen + s) * headDim
            let dst = (s * nHeads + h) * headDim
            for d in 0..<headDim { qSeqMajor[dst + d] = q[src + d] }
        }
    }

    let qT = idefics3FloatsToTensor(qSeqMajor, shape: [seqLen, nHeads, headDim],
                                    dtype: .f32, device: device)
    let kT = idefics3FloatsToTensor(k, shape: [nHeads, seqLen, headDim],
                                    dtype: .f32, device: device)
    let vT = idefics3FloatsToTensor(v, shape: [nHeads, seqLen, headDim],
                                    dtype: .f32, device: device)
    let cmd = device.makeCommandBuffer()
    let outT = Ops.sdpaBidirectional(
        q: qT, k: kT, v: vT,
        nQHeads: nHeads, nKVHeads: nHeads, headDim: headDim,
        baseKV: 0, nQuery: seqLen, kvStride: seqLen,
        scale: scale, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    // Output is [seqLen, nHeads, headDim] = [seqLen, nHeads*headDim] flat.
    return outT.toFloatArray()
}

// ─── Vision encoder layers (GPU GEMM-resident projections) ───────────────────

/// Loaded weights for a single SigLIP-style encoder block.
///
/// Projection weights (Q/K/V/Out, fc1/fc2) and their biases live as f32
/// GPU `Tensor`s so each per-layer projection collapses into a single
/// `Ops.gemm` dispatch over the full `[seqLen, dim]` patch batch. Layer-
/// norm weights stay on the CPU for now — the per-row LayerNorm is
/// cheap relative to the matmul bandwidth it used to bottleneck.
struct Idefics3EncoderLayer {
    // Self-attention: weight [dim, dim] f32 GPU, bias [dim] f32 GPU
    let qW: Tensor; let qB: Tensor
    let kW: Tensor; let kB: Tensor
    let vW: Tensor; let vB: Tensor
    // Output projection is named "out_proj" in Idefics3 (not "o_proj")
    let oW: Tensor; let oB: Tensor
    // MLP: fc1 [intermediate, dim] f32 GPU, fc2 [dim, intermediate] f32 GPU
    let fc1W: Tensor; let fc1B: Tensor
    let fc2W: Tensor; let fc2B: Tensor
    // LayerNorms (still CPU — applied per row before the GEMMs)
    let ln1W: [Float]; let ln1B: [Float]
    let ln2W: [Float]; let ln2B: [Float]

    let dim: Int
    let intermediate: Int
    let nHeads: Int
    let headDim: Int

    /// Load one encoder block from the (potentially remapped) safetensors bundle.
    /// Weight paths follow the FFAI flat layout (post-remapping):
    ///   vision_model.encoder.layers.<i>.self_attn.{q,k,v}_proj.{weight,bias}
    ///   vision_model.encoder.layers.<i>.self_attn.out_proj.{weight,bias}
    ///   vision_model.encoder.layers.<i>.mlp.{fc1,fc2}.{weight,bias}
    ///   vision_model.encoder.layers.<i>.layer_norm{1,2}.{weight,bias}
    init(index i: Int, weights: Idefics3RemappedBundle, cfg: Idefics3VisionConfig) throws {
        let p   = "vision_model.encoder.layers.\(i)"
        let dim = cfg.hiddenSize

        self.dim         = dim
        self.intermediate = cfg.intermediateSize
        self.nHeads      = cfg.numAttentionHeads
        self.headDim     = cfg.headDim

        // Re-host each projection weight + bias as an f32 GPU Tensor so
        // `Ops.gemm` / `Ops.add` can consume them directly.
        func upW(_ key: String, shape: [Int]) throws -> Tensor {
            let floats = idefics3TensorToFloats(try weights.tensor(named: key))
            return idefics3FloatsToTensor(floats, shape: shape, dtype: .f32)
        }
        func upB(_ key: String) throws -> Tensor {
            let floats = idefics3TensorToFloats(try weights.tensor(named: key))
            return idefics3FloatsToTensor(floats, shape: [floats.count], dtype: .f32)
        }
        qW = try upW("\(p).self_attn.q_proj.weight", shape: [dim, dim])
        qB = try upB("\(p).self_attn.q_proj.bias")
        kW = try upW("\(p).self_attn.k_proj.weight", shape: [dim, dim])
        kB = try upB("\(p).self_attn.k_proj.bias")
        vW = try upW("\(p).self_attn.v_proj.weight", shape: [dim, dim])
        vB = try upB("\(p).self_attn.v_proj.bias")
        oW = try upW("\(p).self_attn.out_proj.weight", shape: [dim, dim])
        oB = try upB("\(p).self_attn.out_proj.bias")

        fc1W = try upW("\(p).mlp.fc1.weight", shape: [intermediate, dim])
        fc1B = try upB("\(p).mlp.fc1.bias")
        fc2W = try upW("\(p).mlp.fc2.weight", shape: [dim, intermediate])
        fc2B = try upB("\(p).mlp.fc2.bias")

        ln1W = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm1.weight"))
        ln1B = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm1.bias"))
        ln2W = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm2.weight"))
        ln2B = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm2.bias"))
    }

    /// Forward pass: x is [seqLen, dim] flat. Returns [seqLen, dim] flat.
    /// Each projection now dispatches one `Ops.gemm + Ops.add` pair on a
    /// single command buffer instead of the per-(row × col) CPU matmul
    /// loop that previously pinned the CPU during Idefics3 vision prefill.
    func forward(_ x: [Float], seqLen: Int, eps: Float) -> [Float] {
        let device = Device.shared

        // ── Self-attention ──────────────────────────────────────────────────
        // Per-row LayerNorm runs CPU (cheap O(seqLen·dim)) before we
        // upload `normed1` to the GPU.
        var h = x
        var normed1 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let slice = Array(h[start..<start + dim])
            let n = idefics3LayerNorm1D(slice, weight: ln1W, bias: ln1B, eps: eps)
            normed1.replaceSubrange(start..<start + dim, with: n)
        }

        // Upload normed input once, then dispatch Q/K/V/O over the same
        // command buffer. All three projections share the input tensor.
        let normedT = idefics3FloatsToTensor(normed1, shape: [seqLen, dim],
                                              dtype: .f32, device: device)
        let cmd = device.makeCommandBuffer()
        let qT  = idefics3GemmBiased(input: normedT, weight: qW, bias: qB,
                                      nRows: seqLen, outDim: dim, device: device, on: cmd)
        let kT  = idefics3GemmBiased(input: normedT, weight: kW, bias: kB,
                                      nRows: seqLen, outDim: dim, device: device, on: cmd)
        let vT  = idefics3GemmBiased(input: normedT, weight: vW, bias: vB,
                                      nRows: seqLen, outDim: dim, device: device, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let q = qT.toFloatArray()
        let k = kT.toFloatArray()
        let v = vT.toFloatArray()

        // Reshape [seqLen, nHeads * headDim] → [nHeads, seqLen, headDim] for SDPA
        var qH = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        var kH = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        var vH = [Float](repeating: 0, count: nHeads * seqLen * headDim)
        for s in 0..<seqLen {
            for nh in 0..<nHeads {
                for d in 0..<headDim {
                    let src = s * dim + nh * headDim + d
                    let dst = nh * seqLen * headDim + s * headDim + d
                    qH[dst] = q[src]; kH[dst] = k[src]; vH[dst] = v[src]
                }
            }
        }

        // GPU scaled dot-product attention (already migrated).
        let attnOut = idefics3VisionSDPA(q: qH, k: kH, v: vH,
                                         nHeads: nHeads, seqLen: seqLen, headDim: headDim)

        // Output projection + residual. Same GPU pattern as Q/K/V.
        let attnT = idefics3FloatsToTensor(attnOut, shape: [seqLen, dim],
                                            dtype: .f32, device: device)
        let cmd2 = device.makeCommandBuffer()
        let oTGpu = idefics3GemmBiased(input: attnT, weight: oW, bias: oB,
                                        nRows: seqLen, outDim: dim, device: device, on: cmd2)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        let oOut = oTGpu.toFloatArray()
        for i in 0..<h.count { h[i] += oOut[i] }

        // ── MLP ─────────────────────────────────────────────────────────────
        var normed2 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let slice = Array(h[start..<start + dim])
            let n = idefics3LayerNorm1D(slice, weight: ln2W, bias: ln2B, eps: eps)
            normed2.replaceSubrange(start..<start + dim, with: n)
        }

        let normed2T = idefics3FloatsToTensor(normed2, shape: [seqLen, dim],
                                               dtype: .f32, device: device)
        let cmd3 = device.makeCommandBuffer()
        let fc1Tgpu = idefics3GemmBiased(input: normed2T, weight: fc1W, bias: fc1B,
                                          nRows: seqLen, outDim: intermediate,
                                          device: device, on: cmd3)
        cmd3.commit()
        cmd3.waitUntilCompleted()
        var fc1Out = fc1Tgpu.toFloatArray()
        // GELU tanh approximation (gelu_pytorch_tanh)
        for i in 0..<fc1Out.count { fc1Out[i] = idefics3GeluTanh(fc1Out[i]) }

        let fc1OutT = idefics3FloatsToTensor(fc1Out, shape: [seqLen, intermediate],
                                              dtype: .f32, device: device)
        let cmd4 = device.makeCommandBuffer()
        let fc2Tgpu = idefics3GemmBiased(input: fc1OutT, weight: fc2W, bias: fc2B,
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

/// Run `out = input · weightᵀ + bias` (broadcast bias across `nRows`)
/// as one `Ops.gemm` + bias-tile + `Ops.add` on the supplied command
/// buffer. The bias tile is built CPU-side once and uploaded to GPU.
/// Returns the result tensor; caller commits and reads back.
private func idefics3GemmBiased(input: Tensor, weight: Tensor, bias: Tensor,
                                 nRows: Int, outDim: Int, device: Device,
                                 on cmd: MTLCommandBuffer) -> Tensor {
    let out = Ops.gemm(weight: weight, input: input, nRows: nRows, on: cmd)
    let biasVals = bias.toFloatArray()
    var tiled = [Float](repeating: 0, count: nRows * outDim)
    for r in 0..<nRows {
        let base = r * outDim
        for c in 0..<outDim { tiled[base + c] = biasVals[c] }
    }
    let tiledT = idefics3FloatsToTensor(tiled, shape: [nRows, outDim],
                                         dtype: .f32, device: device)
    return Ops.add(out, tiledT, on: cmd)
}

// ─── Vision encoder ───────────────────────────────────────────────────────────

/// Loaded Idefics3 vision encoder (SigLIP-style ViT).
/// Weights are kept as Float arrays for CPU-side computation during prefill.
public final class Idefics3VisionEncoder: Module {
    let cfg: Idefics3VisionConfig
    // Patch embedding: conv2d weight [hiddenSize, numChannels, patchSize, patchSize]
    let patchW: [Float]
    let patchB: [Float]
    // Position embedding: [numPatches, hiddenSize]
    let posEmbed: [Float]
    let numPatches: Int
    // Post layer norm
    let postLnW: [Float]
    let postLnB: [Float]
    // Encoder layers
    let layers: [Idefics3EncoderLayer]

    // Keep raw Tensors for parameters()
    private let patchWTensor:   Tensor
    private let patchBTensor:   Tensor
    private let posEmbedTensor: Tensor
    private let postLnWTensor:  Tensor
    private let postLnBTensor:  Tensor

    init(cfg: Idefics3VisionConfig, weights: Idefics3RemappedBundle) throws {
        self.cfg = cfg

        let patchWTens    = try weights.tensor(named: "vision_model.embeddings.patch_embedding.weight")
        let patchBTens    = try weights.tensor(named: "vision_model.embeddings.patch_embedding.bias")
        let posEmbedTens  = try weights.tensor(named: "vision_model.embeddings.position_embedding.weight")
        let postLnWTens   = try weights.tensor(named: "vision_model.post_layernorm.weight")
        let postLnBTens   = try weights.tensor(named: "vision_model.post_layernorm.bias")

        self.patchWTensor   = patchWTens
        self.patchBTensor   = patchBTens
        self.posEmbedTensor = posEmbedTens
        self.postLnWTensor  = postLnWTens
        self.postLnBTensor  = postLnBTens

        self.patchW   = idefics3TensorToFloats(patchWTens)
        self.patchB   = idefics3TensorToFloats(patchBTens)
        self.posEmbed = idefics3TensorToFloats(posEmbedTens)

        let n = (cfg.imageSize / cfg.patchSize) * (cfg.imageSize / cfg.patchSize)
        self.numPatches = n

        self.postLnW = idefics3TensorToFloats(postLnWTens)
        self.postLnB = idefics3TensorToFloats(postLnBTens)

        var layers: [Idefics3EncoderLayer] = []
        layers.reserveCapacity(cfg.numHiddenLayers)
        for i in 0..<cfg.numHiddenLayers {
            layers.append(try Idefics3EncoderLayer(index: i, weights: weights, cfg: cfg))
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
        return out
    }

    /// Extract patch embeddings from a single normalized image tile.
    ///
    /// `pixels` is [height, width, channels] float32 after normalization.
    /// Returns [numPatches, hiddenSize] as a Float array.
    ///
    /// Conv2d with stride == kernel (patch extraction) is equivalent to slicing
    /// non-overlapping windows and projecting each through the patch weight matrix.
    /// patchW layout: [hiddenSize, numChannels, patchSize, patchSize] — OIHW.
    func patchEmbeddings(pixels: [Float], height: Int, width: Int) -> [Float] {
        let ps     = cfg.patchSize
        let dim    = cfg.hiddenSize
        let nC     = cfg.numChannels
        let nRows  = height / ps
        let nCols  = width  / ps
        let nPatch = nRows * nCols
        let filterSize = nC * ps * ps

        var out = [Float](repeating: 0, count: nPatch * dim)

        for pr in 0..<nRows {
            for pc in 0..<nCols {
                let pIdx = pr * nCols + pc
                // Extract one patch [ps, ps, nC] from the HWC image
                var patchCHW = [Float](repeating: 0, count: filterSize)
                for ch in 0..<nC {
                    for r in 0..<ps {
                        for c in 0..<ps {
                            let pixRow = pr * ps + r
                            let pixCol = pc * ps + c
                            // pixels: [height, width, nC] row-major
                            let srcIdx = pixRow * width * nC + pixCol * nC + ch
                            patchCHW[ch * ps * ps + r * ps + c] = pixels[srcIdx]
                        }
                    }
                }
                // patchW[d, *] is filter d of size filterSize — dot with patchCHW
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

    /// Run the full vision encoder on a single normalized image tile.
    ///
    /// `pixels` is [height, width, channels] Float32 after mean/std normalization.
    /// Returns [numPatches, hiddenSize] Float — the pooler output after post-LayerNorm.
    func encode(pixels: [Float], height: Int, width: Int) -> [Float] {
        let dim    = cfg.hiddenSize
        let nPatch = (height / cfg.patchSize) * (width / cfg.patchSize)

        // Patch embeddings + position embeddings
        var x = patchEmbeddings(pixels: pixels, height: height, width: width)
        for i in 0..<nPatch {
            for d in 0..<dim { x[i * dim + d] += posEmbed[i * dim + d] }
        }

        // Transformer layers
        for layer in layers {
            x = layer.forward(x, seqLen: nPatch, eps: cfg.layerNormEps)
        }

        // Post layer norm (applied per patch)
        var postNormed = [Float](repeating: 0, count: nPatch * dim)
        for row in 0..<nPatch {
            let start = row * dim
            let slice = Array(x[start..<start + dim])
            let n = idefics3LayerNorm1D(slice, weight: postLnW, bias: postLnB,
                                        eps: cfg.layerNormEps)
            postNormed.replaceSubrange(start..<start + dim, with: n)
        }
        return postNormed
    }
}

// ─── Connector (pixel-shuffle + MLP projection) ───────────────────────────────

/// Idefics3 connector: pixel-shuffle (scale_factor=2) then a linear projection
/// from (visionHidden * scaleFactor²) → textHidden.
///
/// The pixel-shuffle algorithm mirrors the Python reference exactly:
///   1. Reshape [nPatches, visionHidden] → [side, side, visionHidden]
///   2. Reshape → [side, side/sf, visionHidden*sf]
///   3. Transpose(0,2,1,3) → [side/sf, side, visionHidden*sf]
///   4. Reshape → [side/sf, side/sf, visionHidden*sf²]
///   5. Transpose(0,2,1,3) — no-op for the 2D patch grid layout
///   6. Reshape → [nPatches/sf², visionHidden*sf²]
public final class Idefics3Connector: Module {
    let scaleFactor: Int
    /// f32 GPU `Tensor` view of the projector weight `[textHidden, visionHidden·sf²]`.
    /// The CPU `[[Float]]` copy was retired during the GPU-GEMM migration;
    /// the original checkpoint tensor is exposed via `projWTensor` for
    /// `parameters()`.
    private let projWGpu: Tensor
    let projWTensor: Tensor

    init(cfg: Idefics3Config, weights: Idefics3RemappedBundle) throws {
        self.scaleFactor = cfg.scaleFactor
        let projTensor = try weights.tensor(named: "connector.modality_projection.proj.weight")
        self.projWTensor = projTensor
        // Re-host the projector weight as f32 on the GPU; it is matmul'd
        // against the pixel-shuffle output every vision prefill, so we
        // pay the conversion once.
        let floats = idefics3TensorToFloats(projTensor)
        self.projWGpu = idefics3FloatsToTensor(floats, shape: projTensor.shape,
                                                dtype: .f32)
    }

    public func parameters() -> [(String, Tensor)] {
        [("connector.modality_projection.proj.weight", projWTensor)]
    }

    /// Pixel-shuffle then project.
    ///
    /// `visionOut` is [nPatches, visionHidden].
    /// Returns [nPatches/sf², textHidden].
    func forward(visionOut: [Float], nPatches: Int, visionHidden: Int, textHidden: Int) -> [Float] {
        let sf  = scaleFactor
        let sf2 = sf * sf
        let side = Int(Double(nPatches).squareRoot())
        precondition(side * side == nPatches,
                     "Idefics3 connector: nPatches must be a perfect square, got \(nPatches)")
        precondition(side % sf == 0,
                     "Idefics3 connector: side (\(side)) must be divisible by scale_factor (\(sf))")

        let newSide    = side / sf
        let newHidden  = visionHidden * sf2
        let newNPatches = newSide * newSide

        // Step 2: [side, side, visionHidden] → [side, side/sf, visionHidden*sf]
        var step2 = [Float](repeating: 0, count: side * newSide * visionHidden * sf)
        for r in 0..<side {
            for c2 in 0..<newSide {
                for e in 0..<visionHidden {
                    for s in 0..<sf {
                        let srcIdx = r * side * visionHidden + (c2 * sf + s) * visionHidden + e
                        let dstIdx = r * newSide * visionHidden * sf + c2 * visionHidden * sf + e * sf + s
                        step2[dstIdx] = visionOut[srcIdx]
                    }
                }
            }
        }

        // Step 3: transpose(0,2,1,3) → [side/sf, side, visionHidden*sf]
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

        // Step 4: reshape → [side/sf, side/sf, visionHidden*sf²]
        var step4 = [Float](repeating: 0, count: newSide * newSide * newHidden)
        for r2 in 0..<newSide {
            for c2 in 0..<newSide {
                for s in 0..<sf {
                    for e in 0..<(visionHidden * sf) {
                        let srcIdx = r2 * side * visionHidden * sf + (c2 * sf + s) * visionHidden * sf + e
                        let dstIdx = r2 * newSide * newHidden + c2 * newHidden + s * visionHidden * sf + e
                        step4[dstIdx] = step3[srcIdx]
                    }
                }
            }
        }

        // Linear projection: [newNPatches, newHidden] × projWGpu^T → [newNPatches, textHidden]
        // One GPU GEMM dispatch instead of the nested-loop CPU matmul that
        // previously bottlenecked the connector at 4608 → textHidden.
        let device = Device.shared
        let inputT = idefics3FloatsToTensor(step4, shape: [newNPatches, newHidden],
                                             dtype: .f32, device: device)
        let cmd = device.makeCommandBuffer()
        let outT = Ops.gemm(weight: projWGpu, input: inputT,
                            nRows: newNPatches, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return outT.toFloatArray()
    }
}

// ─── Remapped-bundle load helpers ────────────────────────────────────────────
// Mirrors loadLinear / loadEmbedding from Layers.swift but accepts the
// Idefics3RemappedBundle wrapper instead of SafeTensorsBundle directly.

func loadIdefics3Linear(
    base: String, in bundle: Idefics3RemappedBundle,
    quantization: ModelConfig.QuantizationConfig?
) throws -> AnyLinear {
    if let q = quantization, [3, 4, 5, 6, 8].contains(q.bits), bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyLinear(QuantizedLinear(
            weight: t.weight, scales: t.scales, biases: t.biases,
            bits: q.bits, groupSize: q.groupSize
        ))
    }
    return AnyLinear(Linear(weight: try bundle.tensor(named: "\(base).weight")))
}

func loadIdefics3Embedding(
    base: String, in bundle: Idefics3RemappedBundle,
    hidden: Int, quantization: ModelConfig.QuantizationConfig?
) throws -> AnyEmbedding {
    if let q = quantization, [3, 4, 5, 6, 8].contains(q.bits), bundle.isQuantized(base) {
        let t = try bundle.quantizedTriplet(base)
        return AnyEmbedding(QuantizedEmbedding(
            weight: t.weight, scales: t.scales, biases: t.biases,
            hidden: hidden, bits: q.bits, groupSize: q.groupSize
        ))
    }
    return AnyEmbedding(Embedding(weight: try bundle.tensor(named: "\(base).weight")))
}

// ─── Key remapper ─────────────────────────────────────────────────────────────

/// A lightweight view over a `SafeTensorsBundle` that transparently remaps
/// weight keys from HF's original Idefics3 layout to the flat layout FFAI
/// expects.
///
/// HF layout:
///   model.text_model.<suffix>   → language_model.<suffix>
///   model.vision_model.<suffix> → vision_model.<suffix>
///   model.connector.<suffix>    → connector.<suffix>
///   lm_head.<suffix>            → language_model.lm_head.<suffix>
///
/// mlx-community conversions are already flat; the remapping is a no-op for
/// those keys (the original key is tried first, then the HF form).
final class Idefics3RemappedBundle {
    private let inner: SafeTensorsBundle

    init(_ bundle: SafeTensorsBundle) { self.inner = bundle }

    /// Resolve the canonical storage key for a FFAI-style flat key.
    ///
    /// Try the flat key directly first (mlx-community format), then the HF
    /// form. This is correct because mlx-community keys are already in the
    /// flat FFAI form, so the first lookup always wins for those checkpoints.
    private func storageKey(for key: String) -> String {
        if inner.has(key) { return key }
        // Try reverse-mapping: flat → HF
        if key.hasPrefix("language_model.lm_head.") {
            let hfKey = String(key.dropFirst("language_model.".count))
            if inner.has(hfKey) { return hfKey }
        }
        if key.hasPrefix("language_model.") {
            let hfKey = "model.text_model." + key.dropFirst("language_model.".count)
            if inner.has(hfKey) { return hfKey }
        }
        if key.hasPrefix("vision_model.") {
            let hfKey = "model.vision_model." + key.dropFirst("vision_model.".count)
            if inner.has(hfKey) { return hfKey }
        }
        if key.hasPrefix("connector.") {
            let hfKey = "model.connector." + key.dropFirst("connector.".count)
            if inner.has(hfKey) { return hfKey }
        }
        return key  // fall back to the original key; tensor(named:) will throw if absent
    }

    func tensor(named key: String) throws -> Tensor {
        try inner.tensor(named: storageKey(for: key))
    }

    func isQuantized(_ base: String) -> Bool {
        inner.isQuantized(storageKey(for: base))
    }

    func has(_ key: String) -> Bool {
        inner.has(storageKey(for: key))
    }

    func quantizedTriplet(_ base: String) throws -> SafeTensorsBundle.QuantizedTriplet {
        try inner.quantizedTriplet(storageKey(for: base))
    }
}

// ─── Idefics3Model (implements LanguageModel) ─────────────────────────────────

/// The assembled Idefics3 VLM: SigLIP vision encoder + pixel-shuffle
/// connector + Llama-style text backbone. Implements the `LanguageModel`
/// protocol by delegating to the internal `LlamaModel`.
///
/// Vision prefill (`prefillWithImage`) drives the KV cache one token at a
/// time, substituting image embeddings for image-placeholder tokens.
public final class Idefics3Model: LanguageModel {
    public let llamaModel:     LlamaModel
    public let visionEncoder:  Idefics3VisionEncoder
    public let connector:      Idefics3Connector
    public let cfg:            Idefics3Config
    public let device:         Device

    // LanguageModel conformance — delegate to the Llama backbone
    public var hidden:   Int   { llamaModel.hidden }
    public var nLayers:  Int   { llamaModel.nLayers }
    public var nHeads:   Int   { llamaModel.nHeads }
    public var nKVHeads: Int   { llamaModel.nKVHeads }
    public var headDim:  Int   { llamaModel.headDim }
    public var vocab:    Int   { llamaModel.vocab }
    public var maxSeq:   Int   { llamaModel.maxSeq }
    public var dtype:    DType { llamaModel.dtype }

    public init(llamaModel:    LlamaModel,
                visionEncoder: Idefics3VisionEncoder,
                connector:     Idefics3Connector,
                cfg:           Idefics3Config,
                device:        Device) {
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
                        device: Device) -> Tensor {
        llamaModel.forward(tokenId: tokenId, position: position,
                           caches: caches, device: device)
    }

    /// Command-buffer-aware forward — required by `LanguageModel` for
    /// command-buffer chaining across decode steps. Delegates to
    /// LlamaModel's `forward(...:on:device:)`.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        llamaModel.forward(tokenId: tokenId, position: position,
                           caches: caches, on: cmd, device: device)
    }

    /// Multi-token forward — delegates to LlamaModel's optimised
    /// chunked path (Ops.gemm batched projections + one
    /// Ops.sdpaMulti(causal: true) per layer). Idefics3 gets the full
    /// TTFT win for the text-only AR prefill path for free,
    /// since its text backbone IS a LlamaModel. Image-prefill
    /// (vision-substitution) still routes through `prefillWithImage`
    /// which is its own block path.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        llamaModel.forwardMulti(tokenIds: tokenIds, startingAt: position,
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
    public func encodeImage(pixels: [Float], height: Int, width: Int) -> [Float] {
        let vc          = cfg.visionConfig
        let nPatches    = (height / vc.patchSize) * (width / vc.patchSize)
        let sf2         = cfg.scaleFactor * cfg.scaleFactor
        let nImageTokens = nPatches / sf2

        let visionFeatures = visionEncoder.encode(pixels: pixels, height: height, width: width)
        let imageEmbeds    = connector.forward(
            visionOut: visionFeatures,
            nPatches: nPatches,
            visionHidden: vc.hiddenSize,
            textHidden: cfg.textConfig.hiddenSize
        )
        precondition(imageEmbeds.count == nImageTokens * cfg.textConfig.hiddenSize,
                     "Idefics3: image embeds count mismatch")
        return imageEmbeds
    }

    /// Prefill the KV cache with a mixed sequence of text tokens and image embeddings.
    ///
    /// `tokenIds` is the full input token sequence including image-token placeholders
    /// (`cfg.imageTokenId`). `imageEmbeds` is [nImageTokens, textHidden] float,
    /// returned by `encodeImage`. The image embeddings replace image placeholder
    /// tokens in order.
    ///
    /// Returns the next-token logits for the last position (ready for decode).
    public func prefillWithImage(
        tokenIds: [Int],
        imageEmbeds: [Float],
        caches: [any LayerCacheProtocol],
        device: Device
    ) -> Tensor {
        let textHidden   = cfg.textConfig.hiddenSize
        let imageTokenId = cfg.imageTokenId
        let totalImageTokens = imageEmbeds.count / textHidden

        var imageIdx = 0
        var lastLogits = Tensor.empty(shape: [vocab], dtype: dtype, device: device)
        let seqLen = tokenIds.count

        for (pos, tokenId) in tokenIds.enumerated() {
            if tokenId == imageTokenId && imageIdx < totalImageTokens {
                // Substitute the image embedding for this position
                let embedStart = imageIdx * textHidden
                let embedSlice = Array(imageEmbeds[embedStart..<embedStart + textHidden])
                imageIdx += 1

                let h = idefics3FloatsToTensor(embedSlice, shape: [textHidden],
                                               dtype: dtype, device: device)
                lastLogits = forwardFromEmbedding(h, position: pos, caches: caches, device: device)
            } else {
                // Normal text token
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

        for (i, layer) in llamaModel.layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed = llamaModel.finalNorm(h, on: cmd)
        let logits = llamaModel.lmHead(normed, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }
}
