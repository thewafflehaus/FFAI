// Idefics3 family — HuggingFace Idefics3 / SmolVLM (the predecessor to SmolVLM2).
//
// Architecture: SigLIP-style ViT vision encoder + pixel-shuffle connector +
// Llama-style language backbone. Config type is "idefics3" with architecture
// string "Idefics3ForConditionalGeneration".
//
// Reference:
//   mlx-swift-lm: Libraries/MLXVLM/Models/Idefics3.swift
//   mlx-vlm: mlx_vlm/models/idefics3.py
//   HuggingFace: HuggingFaceM4/Idefics3-8B-Llama3
//
// Design mirrors SmolVLM2.swift exactly — Idefics3 is the ancestor and
// SmolVLM2 is a renamed descendant with a larger scale_factor (4 vs 2).
//
// CPU-path vision encoder:
//   The vision encoder runs once per image during prefill, before the KV cache
//   is populated. GPU kernels exist for the Llama-style language backbone but
//   not for LayerNorm, GELU-tanh, or Conv2d. The vision encoder therefore uses
//   CPU-side BF16→F32 computation reading from shared MTLBuffers. This is
//   acceptable because it is a one-time cost per image, not per decode step.
//
// Key differences from SmolVLM2:
//   - Default scale_factor = 2 (SmolVLM2 uses 4)
//   - Default image_token_id = 49153 (SmolVLM2 uses 49190)
//   - Weight prefix remapping: HF stores text weights under "model.text_model.*"
//     and vision weights under "model.vision_model.*"; we remap on load.
//   - The Idefics3-8B checkpoint uses Llama3-style rope_theta = 500_000.
//
// VL forward flow:
//   1. encodeImage: extract image patches → vision encoder → connector → [nImageTokens, textHidden]
//   2. prefillWithImage: embed all input tokens, splice image features at image-token positions
//   3. decode: standard LlamaModel token-by-token decode

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────────────

public enum Idefics3 {
    public static let modelTypes: Set<String>    = ["idefics3"]
    public static let architectures: Set<String> = ["Idefics3ForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> Idefics3Dense.Type {
        return Idefics3Dense.self
    }
}

public enum Idefics3Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingVisionConfig(String)
    case missingTextConfig(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):       return "Idefics3: missing top-level config field: \(f)"
        case .missingVisionConfig(let f): return "Idefics3: missing vision_config.\(f)"
        case .missingTextConfig(let f):   return "Idefics3: missing text_config.\(f)"
        }
    }
}

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

/// Matrix multiply: C[m,n] = A[m,k] × B^T[n,k].  B is stored row-major as [n,k].
private func idefics3Matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
    var out = [Float](repeating: 0, count: m * n)
    for i in 0..<m {
        for j in 0..<n {
            var sum: Float = 0
            let aRow = i * k; let bRow = j * k
            for l in 0..<k { sum += a[aRow + l] * b[bRow + l] }
            out[i * n + j] = sum
        }
    }
    return out
}

/// Bias-add: out[i] += bias[i % biasLen].
private func idefics3AddBias(_ x: inout [Float], bias: [Float]) {
    let n = bias.count
    for i in 0..<x.count { x[i] += bias[i % n] }
}

/// Scaled dot-product attention for the vision encoder.
/// q, k, v: [nHeads, seqLen, headDim] — output: [seqLen, nHeads * headDim]
///
/// The outer (head, query-row) index space is embarrassingly parallel:
/// each (h, i) pair writes a disjoint [headDim]-wide slice of `out`.
private func idefics3VisionSDPA(q: [Float], k: [Float], v: [Float],
                                 nHeads: Int, seqLen: Int, headDim: Int) -> [Float] {
    let scale  = 1.0 / Float(headDim).squareRoot()
    let stride = nHeads * headDim
    var out    = [Float](repeating: 0, count: seqLen * stride)

    // Each (h, i) pair writes a disjoint [headDim]-wide output slice — no
    // two iterations touch the same element, so the writes are race-free.
    // The unsafe pointer send is safe because the slice invariant holds.
    q.withUnsafeBufferPointer { qBuf in
    k.withUnsafeBufferPointer { kBuf in
    v.withUnsafeBufferPointer { vBuf in
    out.withUnsafeMutableBufferPointer { outBuf in
        let qPtr = qBuf.baseAddress!
        let kPtr = kBuf.baseAddress!
        let vPtr = vBuf.baseAddress!
        let outPtr = outBuf.baseAddress!
        let nH = nHeads; let sL = seqLen; let hD = headDim
        let st = stride; let sc = scale
        DispatchQueue.concurrentPerform(iterations: nH * sL) { work in
            let h = work / sL
            let i = work % sL
            let hOff = h * hD
            var scores = [Float](repeating: 0, count: sL)
            var maxS: Float = -.greatestFiniteMagnitude
            for j in 0..<sL {
                var dot: Float = 0
                let qBase = h * sL * hD + i * hD
                let kBase = h * sL * hD + j * hD
                for d in 0..<hD { dot += qPtr[qBase + d] * kPtr[kBase + d] }
                let s = dot * sc; scores[j] = s
                if s > maxS { maxS = s }
            }
            var sumExp: Float = 0
            for j in 0..<sL {
                let e = exp(scores[j] - maxS); scores[j] = e; sumExp += e
            }
            let inv = sumExp > 0 ? 1.0 / sumExp : 0
            let oBase = i * st + hOff
            for j in 0..<sL {
                let w = scores[j] * inv
                let vBase = h * sL * hD + j * hD
                for d in 0..<hD { outPtr[oBase + d] += w * vPtr[vBase + d] }
            }
        }
    }}}}
    return out
}

// ─── Vision encoder layers (CPU-side) ────────────────────────────────────────

/// Loaded weights for a single SigLIP-style encoder block.
/// All weights are pre-converted to Float arrays for CPU-side computation.
struct Idefics3EncoderLayer {
    // Self-attention: weight [dim, dim], bias [dim]
    let qW: [Float]; let qB: [Float]
    let kW: [Float]; let kB: [Float]
    let vW: [Float]; let vB: [Float]
    // Output projection is named "out_proj" in Idefics3 (not "o_proj")
    let oW: [Float]; let oB: [Float]
    // MLP: fc1 [intermediate, dim], fc2 [dim, intermediate]
    let fc1W: [Float]; let fc1B: [Float]
    let fc2W: [Float]; let fc2B: [Float]
    // LayerNorms
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

        qW   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.q_proj.weight"))
        qB   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.q_proj.bias"))
        kW   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.k_proj.weight"))
        kB   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.k_proj.bias"))
        vW   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.v_proj.weight"))
        vB   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.v_proj.bias"))
        oW   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.out_proj.weight"))
        oB   = idefics3TensorToFloats(try weights.tensor(named: "\(p).self_attn.out_proj.bias"))

        fc1W = idefics3TensorToFloats(try weights.tensor(named: "\(p).mlp.fc1.weight"))
        fc1B = idefics3TensorToFloats(try weights.tensor(named: "\(p).mlp.fc1.bias"))
        fc2W = idefics3TensorToFloats(try weights.tensor(named: "\(p).mlp.fc2.weight"))
        fc2B = idefics3TensorToFloats(try weights.tensor(named: "\(p).mlp.fc2.bias"))

        ln1W = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm1.weight"))
        ln1B = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm1.bias"))
        ln2W = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm2.weight"))
        ln2B = idefics3TensorToFloats(try weights.tensor(named: "\(p).layer_norm2.bias"))
    }

    /// Forward pass: x is [seqLen, dim] flat. Returns [seqLen, dim] flat.
    func forward(_ x: [Float], seqLen: Int, eps: Float) -> [Float] {
        var h = x

        // ── Self-attention ──────────────────────────────────────────────────
        // LayerNorm per row
        var normed1 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let slice = Array(h[start..<start + dim])
            let n = idefics3LayerNorm1D(slice, weight: ln1W, bias: ln1B, eps: eps)
            normed1.replaceSubrange(start..<start + dim, with: n)
        }

        // Project Q/K/V: [seqLen, dim] × weight^T [dim, dim] → [seqLen, dim]
        var q = idefics3Matmul(normed1, qW, m: seqLen, k: dim, n: dim)
        idefics3AddBias(&q, bias: qB)
        var k = idefics3Matmul(normed1, kW, m: seqLen, k: dim, n: dim)
        idefics3AddBias(&k, bias: kB)
        var v = idefics3Matmul(normed1, vW, m: seqLen, k: dim, n: dim)
        idefics3AddBias(&v, bias: vB)

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

        // Parallel scaled dot-product attention
        let attnOut = idefics3VisionSDPA(q: qH, k: kH, v: vH,
                                         nHeads: nHeads, seqLen: seqLen, headDim: headDim)

        // Output projection + residual
        var oOut = idefics3Matmul(attnOut, oW, m: seqLen, k: dim, n: dim)
        idefics3AddBias(&oOut, bias: oB)
        for i in 0..<h.count { h[i] += oOut[i] }

        // ── MLP ─────────────────────────────────────────────────────────────
        var normed2 = [Float](repeating: 0, count: seqLen * dim)
        for row in 0..<seqLen {
            let start = row * dim
            let slice = Array(h[start..<start + dim])
            let n = idefics3LayerNorm1D(slice, weight: ln2W, bias: ln2B, eps: eps)
            normed2.replaceSubrange(start..<start + dim, with: n)
        }

        var fc1Out = idefics3Matmul(normed2, fc1W, m: seqLen, k: dim, n: intermediate)
        idefics3AddBias(&fc1Out, bias: fc1B)
        // GELU tanh approximation (gelu_pytorch_tanh)
        for i in 0..<fc1Out.count { fc1Out[i] = idefics3GeluTanh(fc1Out[i]) }

        var fc2Out = idefics3Matmul(fc1Out, fc2W, m: seqLen, k: intermediate, n: dim)
        idefics3AddBias(&fc2Out, bias: fc2B)

        // Residual
        for i in 0..<h.count { h[i] += fc2Out[i] }
        return h
    }
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
    let projW: [Float]     // [textHidden, visionHidden * sf²]
    let projWTensor: Tensor

    init(cfg: Idefics3Config, weights: Idefics3RemappedBundle) throws {
        self.scaleFactor = cfg.scaleFactor
        let projTensor = try weights.tensor(named: "connector.modality_projection.proj.weight")
        self.projWTensor = projTensor
        self.projW = idefics3TensorToFloats(projTensor)
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

        // Linear projection: [newNPatches, newHidden] × projW^T [textHidden, newHidden]
        return idefics3Matmul(step4, projW, m: newNPatches, k: newHidden, n: textHidden)
    }
}

// ─── Remapped-bundle load helpers ────────────────────────────────────────────
// Mirrors loadLinear / loadEmbedding from Layers.swift but accepts the
// Idefics3RemappedBundle wrapper instead of SafeTensorsBundle directly.

private func loadIdefics3Linear(
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

private func loadIdefics3Embedding(
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

// ─── Dense variant ───────────────────────────────────────────────────────────

public struct Idefics3Dense {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut, .visionIn]
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Idefics3Model {
        guard let vcRaw = config.nested("vision_config") else {
            throw Idefics3Error.missingConfig("vision_config")
        }
        guard let tcRaw = config.nested("text_config") else {
            throw Idefics3Error.missingConfig("text_config")
        }
        let vc       = try Idefics3VisionConfig(from: vcRaw)
        let tc       = try Idefics3TextConfig(from: tcRaw)
        let idefCfg  = try Idefics3Config(from: config.raw)

        // ─── Remap weight prefixes ────────────────────────────────────────────
        // HF Idefics3 stores weights as:
        //   model.text_model.*       → language_model.*
        //   model.vision_model.*     → vision_model.*
        //   model.connector.*        → connector.*
        //   lm_head.*                → language_model.lm_head.*  (when not tied)
        //
        // mlx-community conversions may already use the remapped form; the
        // `Idefics3RemappedBundle` wrapper transparently tries the flat key first,
        // then the HF form, making it idempotent for both checkpoint styles.
        let remapped = Idefics3RemappedBundle(weights)

        // ─── Vision encoder & connector ──────────────────────────────────────
        let visionEncoder = try Idefics3VisionEncoder(cfg: vc, weights: remapped)
        let connector     = try Idefics3Connector(cfg: idefCfg, weights: remapped)

        // ─── Language backbone (Llama-style) ─────────────────────────────────
        let quant = config.quantization

        let embedTokens = try loadIdefics3Embedding(
            base: "language_model.embed_tokens", in: remapped,
            hidden: tc.hiddenSize, quantization: quant
        )

        var llamaLayers: [LlamaLayer] = []
        llamaLayers.reserveCapacity(tc.numHiddenLayers)
        for i in 0..<tc.numHiddenLayers {
            let p = "language_model.layers.\(i)"
            let qProj    = try loadIdefics3Linear(base: "\(p).self_attn.q_proj", in: remapped, quantization: quant)
            let kProj    = try loadIdefics3Linear(base: "\(p).self_attn.k_proj", in: remapped, quantization: quant)
            let vProj    = try loadIdefics3Linear(base: "\(p).self_attn.v_proj", in: remapped, quantization: quant)
            let oProj    = try loadIdefics3Linear(base: "\(p).self_attn.o_proj", in: remapped, quantization: quant)
            let gateProj = try loadIdefics3Linear(base: "\(p).mlp.gate_proj",    in: remapped, quantization: quant)
            let upProj   = try loadIdefics3Linear(base: "\(p).mlp.up_proj",      in: remapped, quantization: quant)
            let downProj = try loadIdefics3Linear(base: "\(p).mlp.down_proj",    in: remapped, quantization: quant)
            let inputNorm = RMSNorm(
                weight: try remapped.tensor(named: "\(p).input_layernorm.weight"),
                eps: tc.rmsNormEps)
            let postAttnNorm = RMSNorm(
                weight: try remapped.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: tc.rmsNormEps)
            llamaLayers.append(LlamaLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: tc.hiddenSize,
                nHeads: tc.numAttentionHeads, nKVHeads: tc.numKeyValueHeads,
                headDim: tc.headDim, intermediate: tc.intermediateSize,
                ropeTheta: tc.ropeTheta,
                ropeScaling: .none
            ))
        }

        let finalNorm = RMSNorm(
            weight: try remapped.tensor(named: "language_model.norm.weight"),
            eps: tc.rmsNormEps)

        // LM head — Idefics3-8B has tieWordEmbeddings == false and ships lm_head.weight.
        let lmHead: AnyLinear
        if !tc.tieWordEmbeddings, remapped.has("language_model.lm_head.weight") {
            lmHead = try loadIdefics3Linear(base: "language_model.lm_head", in: remapped, quantization: quant)
        } else if let q = quant, remapped.isQuantized("language_model.embed_tokens") {
            let t = try remapped.quantizedTriplet("language_model.embed_tokens")
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize
            ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation dtype
        let activationDtype: DType
        if remapped.isQuantized("language_model.embed_tokens"),
           let scales = try? remapped.tensor(named: "language_model.embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        let llamaModel = LlamaModel(
            embedTokens: embedTokens, layers: llamaLayers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: tc.hiddenSize, nLayers: tc.numHiddenLayers,
            nHeads: tc.numAttentionHeads, nKVHeads: tc.numKeyValueHeads,
            headDim: tc.headDim, vocab: tc.vocabSize,
            maxSeq: tc.maxPositionEmbeddings, ropeTheta: tc.ropeTheta,
            dtype: activationDtype,
            kvCacheKind: options.kvCache
        )

        return Idefics3Model(
            llamaModel: llamaModel,
            visionEncoder: visionEncoder,
            connector: connector,
            cfg: idefCfg,
            device: device
        )
    }
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
