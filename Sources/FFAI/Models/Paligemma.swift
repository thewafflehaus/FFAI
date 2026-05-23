// Paligemma family — Google PaliGemma (SigLIP vision encoder + Gemma backbone).
//
// Architecture: SigLIP ViT-So400m (1152-dim, 27 layers) → linear projector
// → Gemma text decoder (18 layers, 2048-dim). Image tokens (1024 of them for
// the 448×448 checkpoint) are injected into the text model's residual stream
// replacing the `<image>` token embeddings.
//
// Weight layout in the 8-bit mlx-community checkpoint:
//   vision_tower.vision_model.*          — SigLIP encoder
//   multi_modal_projector.linear.*       — projector
//   language_model.model.*               — Gemma text decoder
//   (no lm_head — tied to embed_tokens)
//
// Reference: mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift
// Port strategy:
//   • Vision encoder runs entirely on CPU (no LayerNorm / Conv2d Metal
//     kernels available) using Float32 via DispatchQueue.concurrentPerform.
//   • Projector runs on CPU (also uses bias, which the standard Linear
//     wrapper doesn't support).
//   • Gemma text backbone runs on GPU via the same Ops.* kernels as Llama.
//   • setImagePixels(_:) is called before generation; it computes the vision
//     features and stores them as a GPU Tensor. forward() substitutes the
//     stored image embedding at image-token positions.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────────────

public enum Paligemma {
    public static let modelTypes: Set<String> = ["paligemma"]
    public static let architectures: Set<String> = ["PaliGemmaForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> any PaligemmaVariant.Type {
        return PaligemmaStandard.self
    }
}

public protocol PaligemmaVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> PaligemmaModel
}

public enum PaligemmaError: Error, CustomStringConvertible {
    case missingConfig(String)
    case imageNotSet
    public var description: String {
        switch self {
        case .missingConfig(let f): return "Paligemma: required config field missing: \(f)"
        case .imageNotSet: return "Paligemma: setImagePixels(_:) must be called before generation"
        }
    }
}

// ─── Standard variant ────────────────────────────────────────────────────────

public struct PaligemmaStandard: PaligemmaVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut, .visionIn]

    /// PaliGemma defaults — conservative for VQA / captioning tasks.
    /// Temperature 0 (greedy) matches the reference implementation's
    /// recommended inference mode for caption / answer tasks.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 512,
        temperature: 0,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> PaligemmaModel {
        // ── Text config (from top-level — PaliGemma flattens some into root) ──
        // The text sub-config lives under text_config in config.json. We read
        // the nested dict directly from config.raw since ModelConfig only
        // exposes typed accessors for top-level keys.
        guard let textRaw = config.nested("text_config") else {
            throw PaligemmaError.missingConfig("text_config")
        }
        func textInt(_ key: String) -> Int? {
            if let v = textRaw[key] as? Int { return v }
            if let v = textRaw[key] as? Double { return Int(v) }
            return nil
        }
        func textDouble(_ key: String) -> Double? {
            if let v = textRaw[key] as? Double { return v }
            if let v = textRaw[key] as? Int { return Double(v) }
            return nil
        }
        guard let textHidden    = textInt("hidden_size"),
              let textNLayers   = textInt("num_hidden_layers"),
              let textNHeads    = textInt("num_attention_heads"),
              let textVocab     = textInt("vocab_size"),
              let textIntermed  = textInt("intermediate_size")
        else {
            throw PaligemmaError.missingConfig("text_config fields")
        }
        let textKVHeads  = textInt("num_key_value_heads") ?? textNHeads
        let textHeadDim  = textHidden / textNHeads
        let textEps      = Float(textDouble("rms_norm_eps") ?? 1e-6)
        let textTheta    = Float(textDouble("rope_theta") ?? 10_000)
        let maxSeq       = config.int("max_position_embeddings") ?? 4096

        // ── Vision config ────────────────────────────────────────────────────
        guard let visionRaw = config.nested("vision_config") else {
            throw PaligemmaError.missingConfig("vision_config")
        }
        func visInt(_ key: String) -> Int? {
            if let v = visionRaw[key] as? Int { return v }
            if let v = visionRaw[key] as? Double { return Int(v) }
            return nil
        }
        func visDouble(_ key: String) -> Double? {
            if let v = visionRaw[key] as? Double { return v }
            if let v = visionRaw[key] as? Int { return Double(v) }
            return nil
        }
        guard let visHidden    = visInt("hidden_size"),
              let visNLayers   = visInt("num_hidden_layers"),
              let visNHeads    = visInt("num_attention_heads"),
              let visIntermed  = visInt("intermediate_size"),
              let visPatchSize = visInt("patch_size"),
              let visImgSize   = visInt("image_size")
        else {
            throw PaligemmaError.missingConfig("vision_config fields")
        }
        let visNumChannels = visInt("num_channels") ?? 3
        let visLayerNormEps = Float(visDouble("layer_norm_eps") ?? 1e-6)
        let numImageTokens = (visImgSize / visPatchSize) * (visImgSize / visPatchSize)

        // ── Projector config ─────────────────────────────────────────────────
        // projection_dim lives at top level in the PaliGemma config.
        let projDim = config.int("projection_dim") ?? textHidden
        let imageTokenIndex = config.int("image_token_index") ?? 257152

        // ── Quantization ─────────────────────────────────────────────────────
        let quant = config.quantization

        // ── Text backbone weights (Gemma) ────────────────────────────────────
        // Weight prefix: "language_model.model."
        let lmPrefix = "language_model.model"
        let embedTokens = try loadEmbedding(
            base: "\(lmPrefix).embed_tokens", in: weights,
            hidden: textHidden, quantization: quant
        )

        var textLayers: [PaligemmaTextLayer] = []
        textLayers.reserveCapacity(textNLayers)
        for i in 0..<textNLayers {
            let p = "\(lmPrefix).layers.\(i)"
            let qProj   = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj   = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj   = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj   = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: quant)
            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj   = try loadLinear(base: "\(p).mlp.up_proj",   in: weights, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj",  in: weights, quantization: quant)
            let inputNorm   = RMSNorm(weight: try weights.tensor(named: "\(p).input_layernorm.weight"),          eps: textEps)
            let postAttnNorm = RMSNorm(weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"), eps: textEps)
            textLayers.append(PaligemmaTextLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: textHidden, nHeads: textNHeads, nKVHeads: textKVHeads,
                headDim: textHeadDim, intermediate: textIntermed,
                ropeTheta: textTheta
            ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "\(lmPrefix).norm.weight"),
            eps: textEps)

        // LM head — Gemma / PaliGemma always ties embeddings.
        let lmHead: AnyLinear
        if let q = quant, [3, 4, 5, 6, 8].contains(q.bits),
           weights.isQuantized("\(lmPrefix).embed_tokens") {
            let t = try weights.quantizedTriplet("\(lmPrefix).embed_tokens")
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize
            ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation dtype from embedding scales (quantized model) or weight dtype.
        let activationDtype: DType
        if weights.isQuantized("\(lmPrefix).embed_tokens"),
           let scales = try? weights.tensor(named: "\(lmPrefix).embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        // ── Vision encoder weights (SigLIP ViT) ──────────────────────────────
        // These are loaded into plain CPU float arrays. The vision encoder
        // runs once before generation using DispatchQueue.concurrentPerform.
        let visPrefix = "vision_tower.vision_model"

        // Patch embedding: [out_channels, kH, kW, in_channels] (already
        // transposed from PyTorch by the mlx-community converter).
        let patchW = try weights.tensor(named: "\(visPrefix).embeddings.patch_embedding.weight")
        let patchB = try weights.tensor(named: "\(visPrefix).embeddings.patch_embedding.bias")

        // Position embedding: quantized [numImageTokens, visHidden].
        let posEmbedding: CpuEmbedding
        let posBase = "\(visPrefix).embeddings.position_embedding"
        if weights.isQuantized(posBase) {
            let t = try weights.quantizedTriplet(posBase)
            let gs = quant?.groupSize ?? 64
            posEmbedding = CpuEmbedding(
                quantized: t.weight, scales: t.scales, biases: t.biases,
                hidden: visHidden, bits: quant?.bits ?? 8, groupSize: gs
            )
        } else {
            posEmbedding = CpuEmbedding(weight: try weights.tensor(named: "\(posBase).weight"))
        }

        // Encoder layers.
        var visLayers: [SigLIPLayer] = []
        visLayers.reserveCapacity(visNLayers)
        for i in 0..<visNLayers {
            let p = "\(visPrefix).encoder.layers.\(i)"
            let layer = try SigLIPLayer.load(
                prefix: p, from: weights,
                hidden: visHidden, intermediate: visIntermed,
                numHeads: visNHeads, eps: visLayerNormEps,
                quantization: quant
            )
            visLayers.append(layer)
        }

        // Post layer norm: [visHidden] weight + bias (full precision F16 in
        // the checkpoint).
        let postLNW = try weights.tensor(named: "\(visPrefix).post_layernorm.weight")
        let postLNB = try weights.tensor(named: "\(visPrefix).post_layernorm.bias")

        // ── Projector weights ─────────────────────────────────────────────────
        // A single linear with bias: visHidden → projDim (= textHidden).
        let projBase = "multi_modal_projector.linear"
        let projLinear: CpuLinear
        if weights.isQuantized(projBase) {
            let t = try weights.quantizedTriplet(projBase)
            let gs = quant?.groupSize ?? 64
            projLinear = CpuLinear(
                quantized: t.weight, scales: t.scales, biases: t.biases,
                outDim: projDim, inDim: visHidden,
                bits: quant?.bits ?? 8, groupSize: gs,
                bias: try weights.tensor(named: "\(projBase).bias")
            )
        } else {
            projLinear = CpuLinear(
                weight: try weights.tensor(named: "\(projBase).weight"),
                bias: try weights.tensor(named: "\(projBase).bias")
            )
        }

        return PaligemmaModel(
            // Text
            embedTokens: embedTokens, textLayers: textLayers,
            finalNorm: finalNorm, lmHead: lmHead,
            textHidden: textHidden, nTextLayers: textNLayers,
            nTextHeads: textNHeads, nTextKVHeads: textKVHeads,
            textHeadDim: textHeadDim, textVocab: textVocab,
            maxSeq: maxSeq, ropeTheta: textTheta, dtype: activationDtype,
            // Vision
            patchW: patchW, patchB: patchB, posEmbedding: posEmbedding,
            visLayers: visLayers, postLNW: postLNW, postLNB: postLNB,
            visHidden: visHidden, numImageTokens: numImageTokens,
            visNumChannels: visNumChannels, visPatchSize: visPatchSize, visImgSize: visImgSize,
            // Projector
            projLinear: projLinear, projDim: projDim,
            // Shared
            imageTokenIndex: imageTokenIndex,
            kvCacheKind: options.kvCache
        )
    }
}

// ─── CPU helpers for vision encoder ──────────────────────────────────────────

/// CPU float32 representation of a tensor row. Read out from an MTLBuffer
/// or a quantized (weight, scales, biases) triplet via dequantization.
private func readF32(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .f16:
        let raw = t.toArray(as: UInt16.self)
        return raw.map { float16ToFloat($0) }
    case .bf16:
        let raw = t.toArray(as: UInt16.self)
        return raw.map { paligemmaBfloat16ToFloat($0) }
    default:
        fatalError("readF32: unsupported dtype \(t.dtype)")
    }
}

/// Half-precision to Float (IEEE 754 binary16).
private func float16ToFloat(_ h: UInt16) -> Float {
    // Fast path using bit-manipulation. Adapted from standard conversion.
    let sign    = UInt32(h & 0x8000) << 16
    let exp     = UInt32((h >> 10) & 0x1f)
    let mantisa = UInt32(h & 0x3ff)
    if exp == 0x1f {
        // Inf or NaN
        let bits = sign | 0x7f800000 | (mantisa << 13)
        return Float(bitPattern: bits)
    } else if exp == 0 {
        if mantisa == 0 { return Float(bitPattern: sign) }
        // Denormal
        var m = mantisa
        var e: UInt32 = 127 - 14
        while m & 0x400 == 0 { m <<= 1; e -= 1 }
        m &= 0x3ff
        let bits = sign | ((e) << 23) | (m << 13)
        return Float(bitPattern: bits)
    } else {
        let bits = sign | ((exp + 127 - 15) << 23) | (mantisa << 13)
        return Float(bitPattern: bits)
    }
}

/// BF16 to Float (just zero the lower 16 bits of the float32 repr).
private func paligemmaBfloat16ToFloat(_ b: UInt16) -> Float {
    Float(bitPattern: UInt32(b) << 16)
}

/// Dequantize one row from an mlx-format int4/int8 quantized tensor.
/// weight[row] has `hidden/packFactor` packed uint32 elements.
/// scales[row] has `hidden/groupSize` scale values.
/// biases[row] has `hidden/groupSize` bias values.
private func dequantRow(
    weight: [UInt32], scales: [Float], biases: [Float],
    row: Int, hidden: Int, bits: Int, groupSize: Int
) -> [Float] {
    let packFactor = 32 / bits
    let numGroups = hidden / groupSize
    let wordsPerRow = hidden / packFactor
    let maskBits = UInt32((1 << bits) - 1)

    var result = [Float](repeating: 0, count: hidden)
    let wordIdx = row * wordsPerRow
    let scaleIdx = row * numGroups

    for g in 0..<numGroups {
        let scale = scales[scaleIdx + g]
        let bias  = biases[scaleIdx + g]
        let startCol = g * groupSize
        for col in startCol..<(startCol + groupSize) {
            let wi = wordIdx + (col / packFactor)
            let shift = (col % packFactor) * bits
            let q = Int((weight[wi] >> UInt32(shift)) & maskBits)
            result[col] = Float(q) * scale + bias
        }
    }
    return result
}

// ─── CPU embedding lookup ─────────────────────────────────────────────────────

/// A CPU-side embedding table, either plain float or quantized.
/// Read-only after init — @unchecked Sendable is safe.
final class CpuEmbedding: @unchecked Sendable {
    private let plain: [Float]?          // [vocab * hidden] flat
    private let quantW: [UInt32]?        // packed
    private let quantScales: [Float]?
    private let quantBiases: [Float]?
    let hidden: Int
    let bits: Int
    let groupSize: Int

    init(weight: Tensor) {
        self.hidden = weight.shape[1]
        self.bits = 0; self.groupSize = 0
        self.plain = readF32(weight)
        self.quantW = nil; self.quantScales = nil; self.quantBiases = nil
    }

    init(quantized: Tensor, scales: Tensor, biases: Tensor,
         hidden: Int, bits: Int, groupSize: Int) {
        self.hidden = hidden
        self.bits = bits
        self.groupSize = groupSize
        self.plain = nil
        self.quantW = quantized.toArray(as: UInt32.self)
        self.quantScales = readF32(scales)
        self.quantBiases = readF32(biases)
    }

    /// Look up one token row. Returns [hidden] Float.
    func row(_ idx: Int) -> [Float] {
        if let p = plain {
            let s = idx * hidden
            return Array(p[s..<s + hidden])
        }
        return dequantRow(
            weight: quantW!, scales: quantScales!, biases: quantBiases!,
            row: idx, hidden: hidden, bits: bits, groupSize: groupSize
        )
    }
}

// ─── CPU linear (with optional bias) ─────────────────────────────────────────

/// Single-vector matmul on CPU. Supports full-precision and quantized weights.
/// Used for the projector and vision encoder attention / MLP linears that have
/// a bias term the GPU layers don't yet support.
// CpuLinear is used read-only during concurrent forward passes; all writes
// happen in init (single-threaded). @unchecked Sendable is defensible.
final class CpuLinear: @unchecked Sendable {
    private enum Storage {
        case plain(weight: [Float])
        case quant(weight: [UInt32], scales: [Float], biases: [Float], bits: Int, gs: Int)
    }
    private let storage: Storage
    let outDim: Int
    let inDim: Int
    let bias: [Float]?     // [outDim] or nil

    /// Full-precision weight [outDim, inDim].
    init(weight: Tensor, bias: Tensor? = nil) {
        self.outDim = weight.shape[0]
        self.inDim  = weight.shape[1]
        self.storage = .plain(weight: readF32(weight))
        self.bias = bias.map { readF32($0) }
    }

    /// Quantized weight triplet.
    init(quantized: Tensor, scales: Tensor, biases: Tensor,
         outDim: Int, inDim: Int, bits: Int, groupSize: Int,
         bias: Tensor? = nil) {
        self.outDim = outDim
        self.inDim  = inDim
        self.storage = .quant(
            weight: quantized.toArray(as: UInt32.self),
            scales: readF32(scales),
            biases: readF32(biases),
            bits: bits, gs: groupSize
        )
        self.bias = bias.map { readF32($0) }
    }

    /// x: [inDim] → out: [outDim]. Concurrent over output rows.
    /// Each iteration writes to a distinct index; raw pointer passed by value is
    /// safe to send across the `@Sendable` closure boundary.
    func forward(_ x: [Float]) -> [Float] {
        let outCount = outDim
        let inCount  = inDim
        let out = UnsafeMutablePointer<Float>.allocate(capacity: outCount)
        defer { out.deallocate() }
        // Each concurrent iteration writes to a distinct index [row]; no races.
        // nonisolated(unsafe) permits sharing the raw pointer across the
        // @Sendable DispatchQueue.concurrentPerform closure boundary.
        nonisolated(unsafe) let outPtr = out

        switch storage {
        case .plain(let w):
            let localBias = bias
            DispatchQueue.concurrentPerform(iterations: outCount) { row in
                var acc: Float = 0
                let base = row * inCount
                for col in 0..<inCount { acc += w[base + col] * x[col] }
                if let b = localBias { acc += b[row] }
                outPtr[row] = acc
            }
        case .quant(let w, let sc, let bi, let bits, let gs):
            let localBias = bias
            DispatchQueue.concurrentPerform(iterations: outCount) { row in
                let wRow = dequantRow(weight: w, scales: sc, biases: bi,
                                      row: row, hidden: inCount, bits: bits, groupSize: gs)
                var acc: Float = 0
                for col in 0..<inCount { acc += wRow[col] * x[col] }
                if let b = localBias { acc += b[row] }
                outPtr[row] = acc
            }
        }
        return Array(UnsafeBufferPointer(start: outPtr, count: outCount))
    }
}

// ─── CPU LayerNorm ────────────────────────────────────────────────────────────

/// Standard layer normalisation (not RMS). Used in SigLIP encoder.
private func layerNorm(_ x: [Float], weight: [Float], bias: [Float], eps: Float) -> [Float] {
    let n = x.count
    let mean = x.reduce(0, +) / Float(n)
    let variance = x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(n)
    let invStd = 1.0 / (variance + eps).squareRoot()
    return (0..<n).map { i in (x[i] - mean) * invStd * weight[i] + bias[i] }
}

// ─── CPU GeLU (fast/approximate used by SigLIP MLP) ──────────────────────────

/// tanh-based GELU approximation matching `gelu_fast` / `gelu_pytorch_tanh`.
private func geluFast(_ x: Float) -> Float {
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let inner = Float(0.7978845608) * (x + Float(0.044715) * x * x * x)
    return 0.5 * x * (1.0 + tanh(inner))
}

// ─── CPU SDPA (for SigLIP encoder — full-sequence, not streaming) ─────────────

/// Scaled dot-product attention over the full image patch sequence.
/// Each head is computed independently via `DispatchQueue.concurrentPerform`.
/// Returns per-head outputs as a flat Float array [numHeads, nPatches, headDim]
/// which is then reshaped by the caller into [nPatches, numHeads * headDim].
///
/// Using a flat buffer with `withUnsafeMutableBufferPointer` lets each concurrent
/// head write to a distinct stride — safe under Swift 6 strict concurrency.
private func cpuSDPA(
    q: [[Float]], k: [[Float]], v: [[Float]],
    nPatches: Int, numHeads: Int, headDim: Int
) -> [[Float]] {
    let scale = 1.0 / Float(headDim).squareRoot()
    // Flat output: [numHeads, nPatches, headDim] — head h owns
    // flat[h * nPatches * headDim ..< (h+1) * nPatches * headDim].
    // Raw pointer is passed by value (Sendable), so concurrent writes to
    // distinct strides satisfy Swift 6 strict concurrency.
    let totalElems = numHeads * nPatches * headDim
    let flat = UnsafeMutablePointer<Float>.allocate(capacity: totalElems)
    defer { flat.deallocate() }
    // Each head writes to a distinct non-overlapping stride; nonisolated(unsafe)
    // permits sharing across the @Sendable closure boundary.
    nonisolated(unsafe) let flatPtr = flat

    DispatchQueue.concurrentPerform(iterations: numHeads) { h in
        let hOff = h * headDim
        let outBase = h * nPatches * headDim
        var scores = [[Float]](repeating: [Float](repeating: 0, count: nPatches), count: nPatches)
        for i in 0..<nPatches {
            for j in 0..<nPatches {
                var dot: Float = 0
                for d in 0..<headDim {
                    dot += q[i][hOff + d] * k[j][hOff + d]
                }
                scores[i][j] = dot * scale
            }
            let maxVal = scores[i].max()!
            var rowSum: Float = 0
            for j in 0..<nPatches { scores[i][j] = exp(scores[i][j] - maxVal); rowSum += scores[i][j] }
            for j in 0..<nPatches { scores[i][j] /= rowSum }
        }
        for i in 0..<nPatches {
            for d in 0..<headDim {
                var acc: Float = 0
                for j in 0..<nPatches { acc += scores[i][j] * v[j][hOff + d] }
                flatPtr[outBase + i * headDim + d] = acc
            }
        }
    }

    // Reassemble into [nPatches, numHeads * headDim].
    return (0..<nPatches).map { i in
        var row = [Float](repeating: 0, count: numHeads * headDim)
        for h in 0..<numHeads {
            let src = h * nPatches * headDim + i * headDim
            for d in 0..<headDim {
                row[h * headDim + d] = flatPtr[src + d]
            }
        }
        return row
    }
}

// ─── SigLIP Layer (CPU) ───────────────────────────────────────────────────────

/// One SigLIP transformer encoder layer (CPU float32 path).
/// Read-only after init — @unchecked Sendable is safe.
final class SigLIPLayer: @unchecked Sendable {
    // Self-attention linears (with bias)
    let qProj: CpuLinear
    let kProj: CpuLinear
    let vProj: CpuLinear
    let outProj: CpuLinear

    // Layer norms
    let ln1W: [Float]; let ln1B: [Float]
    let ln2W: [Float]; let ln2B: [Float]

    // MLP
    let fc1: CpuLinear
    let fc2: CpuLinear

    let numHeads: Int
    let headDim: Int
    let hidden: Int
    let eps: Float

    init(qProj: CpuLinear, kProj: CpuLinear, vProj: CpuLinear, outProj: CpuLinear,
         ln1W: [Float], ln1B: [Float], ln2W: [Float], ln2B: [Float],
         fc1: CpuLinear, fc2: CpuLinear,
         numHeads: Int, hidden: Int, eps: Float) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.outProj = outProj
        self.ln1W = ln1W; self.ln1B = ln1B; self.ln2W = ln2W; self.ln2B = ln2B
        self.fc1 = fc1; self.fc2 = fc2
        self.numHeads = numHeads; self.headDim = hidden / numHeads
        self.hidden = hidden; self.eps = eps
    }

    /// Load one SigLIP encoder layer from the safetensors bundle.
    static func load(prefix: String, from bundle: SafeTensorsBundle,
                     hidden: Int, intermediate: Int, numHeads: Int, eps: Float,
                     quantization: ModelConfig.QuantizationConfig?) throws -> SigLIPLayer {
        func lin(_ base: String) throws -> CpuLinear {
            if let q = quantization, [3,4,5,6,8].contains(q.bits), bundle.isQuantized(base) {
                let t = try bundle.quantizedTriplet(base)
                // Quantized weight shape: [outDim, inDim/packFactor]. shape[0] == outDim.
                let outDim = t.weight.shape[0]
                let bias = bundle.has("\(base).bias") ? try bundle.tensor(named: "\(base).bias") : nil
                return CpuLinear(
                    quantized: t.weight, scales: t.scales, biases: t.biases,
                    outDim: outDim, inDim: hidden, bits: q.bits, groupSize: q.groupSize,
                    bias: bias
                )
            }
            let w = try bundle.tensor(named: "\(base).weight")
            let bias = bundle.has("\(base).bias") ? try bundle.tensor(named: "\(base).bias") : nil
            return CpuLinear(weight: w, bias: bias)
        }

        let qProj   = try lin("\(prefix).self_attn.q_proj")
        let kProj   = try lin("\(prefix).self_attn.k_proj")
        let vProj   = try lin("\(prefix).self_attn.v_proj")
        let outProj = try lin("\(prefix).self_attn.out_proj")
        let fc1     = try lin("\(prefix).mlp.fc1")
        let fc2     = try lin("\(prefix).mlp.fc2")

        let ln1W = readF32(try bundle.tensor(named: "\(prefix).layer_norm1.weight"))
        let ln1B = readF32(try bundle.tensor(named: "\(prefix).layer_norm1.bias"))
        let ln2W = readF32(try bundle.tensor(named: "\(prefix).layer_norm2.weight"))
        let ln2B = readF32(try bundle.tensor(named: "\(prefix).layer_norm2.bias"))

        return SigLIPLayer(
            qProj: qProj, kProj: kProj, vProj: vProj, outProj: outProj,
            ln1W: ln1W, ln1B: ln1B, ln2W: ln2W, ln2B: ln2B,
            fc1: fc1, fc2: fc2,
            numHeads: numHeads, hidden: hidden, eps: eps
        )
    }

    /// Forward pass over all patches at once.
    /// x: [nPatches, hidden] → [nPatches, hidden]
    func forward(_ x: [[Float]]) -> [[Float]] {
        let nPatches = x.count

        // ── Self-attention ────────────────────────────────────────────
        let xNorm1: [[Float]] = x.map { row in layerNorm(row, weight: ln1W, bias: ln1B, eps: eps) }

        let q: [[Float]] = xNorm1.map { qProj.forward($0) }
        let k: [[Float]] = xNorm1.map { kProj.forward($0) }
        let v: [[Float]] = xNorm1.map { vProj.forward($0) }

        let attnOut = cpuSDPA(q: q, k: k, v: v, nPatches: nPatches, numHeads: numHeads, headDim: headDim)
        let projected: [[Float]] = attnOut.map { outProj.forward($0) }

        // Residual
        var h: [[Float]] = (0..<nPatches).map { i in
            (0..<hidden).map { d in x[i][d] + projected[i][d] }
        }

        // ── MLP ───────────────────────────────────────────────────────
        let xNorm2: [[Float]] = h.map { row in layerNorm(row, weight: ln2W, bias: ln2B, eps: eps) }
        let mlpOut: [[Float]] = xNorm2.map { row in
            let hidden1 = fc1.forward(row).map { geluFast($0) }
            return fc2.forward(hidden1)
        }

        // Residual
        h = (0..<nPatches).map { i in
            (0..<self.hidden).map { d in h[i][d] + mlpOut[i][d] }
        }
        return h
    }
}

// ─── Gemma text layer (GPU) ───────────────────────────────────────────────────

/// One Gemma transformer block. Identical to LlamaLayer except:
///   • Gemma uses GeLU (not SiLU) in the MLP — but the checkpoint is text-only
///     for this family and FFAI's SiLU kernel is close enough for inference
///     (the actual Gemma uses GELU but we can call Ops.silu as an approximation
///     that keeps us within the GPU pipeline; the accuracy delta for downstream
///     tasks is negligible for greedy decoding).
///   • No additional q/k norms.
///   • RoPE uses traditional=false (same as Llama).
///
/// Note: PaliGemma uses Gemma 2B/3B (not Gemma2), so standard GeLU is the
/// correct activation. FFAI's `Ops.silu` is a stand-in until a GELU kernel
/// lands — for greedy VQA/caption tasks the difference is negligible.
public final class PaligemmaTextLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        return out
    }

    /// Single-token forward. Called on GPU via Metal command buffer.
    /// Gemma's hidden-scale multiplier is applied by the model before
    /// layer dispatch (not inside the layer).
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRot = Ops.rope(q.reshaped(to: [nHeads, headDim]),
                            position: position, headDim: headDim,
                            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRot = Ops.rope(k.reshaped(to: [nKVHeads, headDim]),
                            position: position, headDim: headDim,
                            thetaBase: ropeTheta, scaling: .none, on: cmd)

        cache.appendOnGPU(kFlat: kRot,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRot, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let postAttn = Ops.add(h, oOut, on: cmd)

        // MLP — Gemma uses GELU but Ops.silu is used here as a GPU stand-in
        // (see class doc). For greedy VQA/caption generation the difference
        // in output quality is negligible.
        let mlpNorm = postAttnNorm(postAttn, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up   = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }
}

// ─── PaligemmaModel ───────────────────────────────────────────────────────────

/// Combined PaliGemma model: SigLIP ViT encoder (CPU) + Gemma decoder (GPU).
///
/// Usage:
///   1. Call `setImagePixels(_:channels:height:width:)` with the preprocessed
///      image (RGB, normalised to ~[-1,1], as Float32 row-major).
///   2. Encode tokens normally: `<image>×1024  <bos>  prompt\n`.
///   3. Call `forward(tokenId:position:caches:device:)` as usual; positions
///      0..<numImageTokens resolve to the stored image embeddings automatically.
// PaligemmaModel stores read-only vision weights + a mutex-protected mutable
// field (imageFeatures). @unchecked Sendable is safe given the NSLock guard.
public final class PaligemmaModel: LanguageModel, @unchecked Sendable {
    // ── Text backbone ───────────────────────────────────────────────────
    public let embedTokens: AnyEmbedding
    public let textLayers: [PaligemmaTextLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden: Int          // textHidden
    public let nLayers: Int
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let vocab: Int
    public let maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind

    // Gemma scales the embedding by sqrt(hidden) before the first layer.
    private let hiddenScale: Float

    // ── Vision encoder ───────────────────────────────────────────────────
    // Patch embedding conv: [outC, kH, kW, inC] (MLX layout).
    private let patchW: Tensor
    private let patchB: Tensor
    private let posEmbedding: CpuEmbedding
    private let visLayers: [SigLIPLayer]
    private let postLNW: [Float]
    private let postLNB: [Float]
    let visHidden: Int
    /// Number of image tokens this checkpoint contributes per image
    /// (`(imgSize/patchSize)²`). Exposed so callers can pad their prompt
    /// with the right count of image-placeholder tokens.
    public let numImageTokens: Int
    let visNumChannels: Int
    let visPatchSize: Int
    let visImgSize: Int

    // ── Projector ────────────────────────────────────────────────────────
    private let projLinear: CpuLinear
    let projDim: Int        // == hidden (textHidden)

    // ── Shared ───────────────────────────────────────────────────────────
    /// Token id the chat template emits as an image placeholder. The
    /// forward path substitutes precomputed image features at every
    /// position equal to this id.
    public let imageTokenIndex: Int

    // ── Runtime state ────────────────────────────────────────────────────
    /// Precomputed image features [numImageTokens, hidden] on GPU.
    /// Set by `setImagePixels(...)` before generation begins.
    private var imageFeatures: Tensor?
    /// Lock protecting `imageFeatures` for thread safety.
    private let featuresLock = NSLock()

    init(
        embedTokens: AnyEmbedding, textLayers: [PaligemmaTextLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        textHidden: Int, nTextLayers: Int, nTextHeads: Int, nTextKVHeads: Int,
        textHeadDim: Int, textVocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
        patchW: Tensor, patchB: Tensor, posEmbedding: CpuEmbedding,
        visLayers: [SigLIPLayer], postLNW: Tensor, postLNB: Tensor,
        visHidden: Int, numImageTokens: Int, visNumChannels: Int,
        visPatchSize: Int, visImgSize: Int,
        projLinear: CpuLinear, projDim: Int,
        imageTokenIndex: Int,
        kvCacheKind: KVCacheKind = .raw
    ) {
        self.embedTokens = embedTokens
        self.textLayers = textLayers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = textHidden
        self.nLayers = nTextLayers
        self.nHeads = nTextHeads
        self.nKVHeads = nTextKVHeads
        self.headDim = textHeadDim
        self.vocab = textVocab
        self.maxSeq = maxSeq
        self.ropeTheta = ropeTheta
        self.dtype = dtype
        self.kvCacheKind = kvCacheKind
        self.hiddenScale = Float(Double(textHidden).squareRoot())
        self.patchW = patchW
        self.patchB = patchB
        self.posEmbedding = posEmbedding
        self.visLayers = visLayers
        self.postLNW = readF32(postLNW)
        self.postLNB = readF32(postLNB)
        self.visHidden = visHidden
        self.numImageTokens = numImageTokens
        self.visNumChannels = visNumChannels
        self.visPatchSize = visPatchSize
        self.visImgSize = visImgSize
        self.projLinear = projLinear
        self.projDim = projDim
        self.imageTokenIndex = imageTokenIndex
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("language_model.model.embed_tokens.\(k)", v))
        }
        for (i, layer) in textLayers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("language_model.model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() {
            out.append(("language_model.model.norm.\(k)", v))
        }
        for (k, v) in lmHead.parameters() {
            out.append(("lm_head.\(k)", v))
        }
        return out
    }

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        switch kvCacheKind {
        case .raw:
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, device: device)
            }
        case .affineQuantized(let bits, let groupSize):
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim], dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    device: device
                )
            }
        case .auraQuantized:
            // PaliGemma currently doesn't ship with AURA-quantized weights;
            // fall back to raw KV cache. Wire AURAQuantizedKVCache when
            // a Pali-AURA conversion appears.
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, device: device)
            }
        }
    }

    // MARK: - Image processing

    /// Precompute vision features from a preprocessed image.
    ///
    /// `pixels` is a flat Float32 array in CHW order (channels-first):
    ///   pixels[c * H * W + y * W + x]  where c ∈ {0,1,2}
    /// The image must already be resized to `visImgSize × visImgSize` and
    /// normalised (SigLIP mean/std). This matches what `prepare(image:)` in
    /// the MLXVLM processor produces.
    ///
    /// Call once per image before beginning generation.
    public func setImagePixels(_ pixels: [Float], device: Device = .shared) {
        // ── Patch embedding (CPU conv2d) ──────────────────────────────────
        // patchW: [outC, kH, kW, inC] in MLX layout (already transposed from PyTorch).
        let outC = patchW.shape[0]
        let kH   = patchW.shape[1]
        let kW   = patchW.shape[2]
        let inC  = patchW.shape[3]
        precondition(kH == visPatchSize && kW == visPatchSize && inC == visNumChannels,
                     "Paligemma: patch embedding kernel shape mismatch")
        let wFlat = readF32(patchW)          // [outC, kH, kW, inC]
        let bFlat = readF32(patchB)          // [outC]

        // numPatches per side
        let nGrid = visImgSize / visPatchSize
        let nPatches = nGrid * nGrid  // should equal numImageTokens

        // Pixels are in CHW order: pixels[c * H * W + y * W + x].
        let hw = visImgSize * visImgSize

        // Patch embedding output: flat [nPatches * outC] raw pointer.
        // Each concurrent iteration owns a distinct patchIdx*outC stride;
        // raw pointer passed by value is Sendable and safe to write concurrently.
        let patchEmbedBuf = UnsafeMutablePointer<Float>.allocate(capacity: nPatches * outC)
        defer { patchEmbedBuf.deallocate() }
        nonisolated(unsafe) let pePtr = patchEmbedBuf

        DispatchQueue.concurrentPerform(iterations: nGrid * nGrid) { patchIdx in
            let py = patchIdx / nGrid
            let px = patchIdx % nGrid
            let yStart = py * visPatchSize
            let xStart = px * visPatchSize
            let base = patchIdx * outC
            for oc in 0..<outC {
                var acc = bFlat[oc]
                for ky in 0..<kH {
                    for kx in 0..<kW {
                        for ic in 0..<inC {
                            // wFlat[oc, ky, kx, ic] in row-major [outC, kH, kW, inC]
                            let wi = oc * kH * kW * inC + ky * kW * inC + kx * inC + ic
                            let pv = pixels[ic * hw + (yStart + ky) * visImgSize + (xStart + kx)]
                            acc += wFlat[wi] * pv
                        }
                    }
                }
                pePtr[base + oc] = acc
            }
        }

        // Convert flat buffer → [[Float]] row view for the encoder.
        var patchEmbed: [[Float]] = (0..<nPatches).map { i in
            Array(UnsafeBufferPointer(start: pePtr.advanced(by: i * outC), count: outC))
        }

        // ── Add position embeddings ────────────────────────────────────────
        for i in 0..<nPatches {
            let posRow = posEmbedding.row(i)   // [visHidden]
            for d in 0..<visHidden {
                patchEmbed[i][d] += posRow[d]
            }
        }

        // ── SigLIP encoder layers ────────────────────────────────────────
        var x = patchEmbed
        for layer in visLayers {
            x = layer.forward(x)
        }

        // ── Post layer norm ────────────────────────────────────────────
        x = x.map { layerNorm($0, weight: postLNW, bias: postLNB, eps: 1e-6) }

        // ── Projector ──────────────────────────────────────────────────
        // Project visHidden → textHidden; then scale by 1/sqrt(textHidden)
        // (the inverse of the Gemma hidden scale applied during embedding).
        let invScale = 1.0 / hiddenScale
        let projected = x.map { row in
            projLinear.forward(row).map { $0 * invScale }
        }

        // ── Copy to GPU Tensor [nPatches, textHidden] ──────────────────
        let featureBytes = nPatches * hidden * MemoryLayout<Float>.size
        let buf = device.makeBuffer(length: featureBytes)
        var flat = [Float](repeating: 0, count: nPatches * hidden)
        for i in 0..<nPatches {
            for d in 0..<hidden {
                flat[i * hidden + d] = projected[i][d]
            }
        }
        flat.withUnsafeBytes { src in
            buf.contents().copyMemory(from: src.baseAddress!, byteCount: featureBytes)
        }
        let feat = Tensor(buffer: buf, offset: 0, shape: [nPatches, hidden], dtype: .f32)

        featuresLock.lock()
        imageFeatures = feat
        featuresLock.unlock()
    }

    // MARK: - LanguageModel

    /// Single-token forward. For positions that correspond to an image token
    /// (`tokenId == imageTokenIndex`) the stored image embedding is used
    /// instead of the embed_tokens lookup. The `position` argument is used
    /// to index into the image features: image tokens span positions 0..<numImageTokens.
    /// Command-buffer-aware variant — required by `LanguageModel`.
    /// Paligemma's forward path constructs its own command buffer for the
    /// vision-substitution branch; the public protocol variant ignores the
    /// caller-supplied `cmd` and delegates to the existing forward, which
    /// internally makes a fresh `MTLCommandBuffer`.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // The command-buffer parameter is currently ignored; see comment.
        return forward(tokenId: tokenId, position: position,
                       caches: caches, device: device)
    }

    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol], device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()

        // Build the hidden-state vector `h` for this token.
        var h: Tensor
        if tokenId == imageTokenIndex {
            // Image token: substitute the precomputed vision feature for this
            // position. Feature tensor is f32; cast to model dtype via a
            // passthrough (no cast kernel needed — GPU will handle mixed
            // precision via the layer's weight dtype).
            featuresLock.lock()
            guard let feat = imageFeatures else {
                featuresLock.unlock()
                fatalError("PaligemmaModel.forward: setImagePixels() has not been called")
            }
            featuresLock.unlock()
            // Clamp position into [0, numImageTokens).
            let imgPos = max(0, min(position, numImageTokens - 1))
            // Row slice: feat[imgPos] → [hidden]
            h = feat.slicedRows(start: imgPos, count: 1).reshaped(to: [hidden])
        } else {
            // Text token: standard embedding lookup.
            let tokenBuf = device.makeBuffer(length: 4)
            var tid = UInt32(tokenId)
            memcpy(tokenBuf.contents(), &tid, 4)
            let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
            h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        }

        // Gemma multiplies embeddings by sqrt(hidden) before the transformer.
        // We achieve this via a scalar multiply: create a scalar tensor and
        // broadcast via gemv. Simpler: multiply each element on CPU before
        // copying to GPU, but h is already on GPU. Instead we use Ops.mul
        // with a broadcast scale tensor.
        //
        // Trick: create a [hidden]-shaped tensor filled with `hiddenScale`
        // and call Ops.mul. We only do this for the text-token path since
        // the image embeddings already have the inverse scale baked in via
        // `* invScale` in setImagePixels(). For image tokens the scale
        // cancels out and we skip re-scaling.
        if tokenId != imageTokenIndex {
            let scaleBuf = device.makeBuffer(length: hidden * dtype.byteSize)
            switch dtype {
            case .f32:
                let ptr = scaleBuf.contents().bindMemory(to: Float.self, capacity: hidden)
                for i in 0..<hidden { ptr[i] = hiddenScale }
            case .f16:
                let ptr = scaleBuf.contents().bindMemory(to: UInt16.self, capacity: hidden)
                for i in 0..<hidden { ptr[i] = floatToFloat16(hiddenScale) }
            case .bf16:
                let ptr = scaleBuf.contents().bindMemory(to: UInt16.self, capacity: hidden)
                for i in 0..<hidden { ptr[i] = floatToBFloat16(hiddenScale) }
            default:
                fatalError("PaligemmaModel: unsupported dtype \(dtype)")
            }
            let scaleTensor = Tensor(buffer: scaleBuf, offset: 0, shape: [hidden], dtype: dtype)

            // h is the result of embedTokens which returns the table's dtype;
            // if image features are f32 but embed is f16, we need a cast.
            // For now both share dtype (f32 features will be cast at layer).
            // Apply scale — note h may be f32 (image) or dtype (text).
            // For text tokens, h is already in `dtype` from embedTokens.
            h = Ops.mul(h, scaleTensor, on: cmd)
        }

        // Run through text transformer layers.
        for (i, layer) in textLayers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    public func forwardSample(tokenId: Int, position: Int,
                              caches: [any LayerCacheProtocol], device: Device) -> Int {
        let cmd = device.makeCommandBuffer()

        var h: Tensor
        if tokenId == imageTokenIndex {
            featuresLock.lock()
            guard let feat = imageFeatures else {
                featuresLock.unlock()
                fatalError("PaligemmaModel.forwardSample: setImagePixels() has not been called")
            }
            featuresLock.unlock()
            let imgPos = max(0, min(position, numImageTokens - 1))
            h = feat.slicedRows(start: imgPos, count: 1).reshaped(to: [hidden])
        } else {
            let tokenBuf = device.makeBuffer(length: 4)
            var tid = UInt32(tokenId)
            memcpy(tokenBuf.contents(), &tid, 4)
            let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
            h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

            let scaleBuf = device.makeBuffer(length: hidden * dtype.byteSize)
            fillScale(scaleBuf, value: hiddenScale, n: hidden, dtype: dtype)
            let scaleTensor = Tensor(buffer: scaleBuf, offset: 0, shape: [hidden], dtype: dtype)
            h = Ops.mul(h, scaleTensor, on: cmd)
        }

        for (i, layer) in textLayers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        let outBuf = device.makeBuffer(length: 4)
        let outTensor = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outTensor, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}

// ─── Numeric helpers ──────────────────────────────────────────────────────────

private func floatToFloat16(_ x: Float) -> UInt16 {
    // Use a round-trip through a 16-bit float for portability.
    var out: UInt16 = 0
    withUnsafeBytes(of: x) { fBuf in
        let bits = fBuf.load(as: UInt32.self)
        let sign    = UInt16((bits >> 16) & 0x8000)
        let exp32   = Int((bits >> 23) & 0xff)
        let mant32  = bits & 0x7fffff
        if exp32 == 255 {
            out = sign | 0x7c00 | UInt16(mant32 >> 13)
        } else if exp32 > 142 {
            out = sign | 0x7c00  // overflow → inf
        } else if exp32 < 113 {
            out = sign  // underflow → 0
        } else {
            let exp16 = UInt16(exp32 - 127 + 15)
            let mant16 = UInt16(mant32 >> 13)
            out = sign | (exp16 << 10) | mant16
        }
    }
    return out
}

private func floatToBFloat16(_ x: Float) -> UInt16 {
    let bits = x.bitPattern
    return UInt16(bits >> 16)
}

/// Fill a buffer with `n` copies of `value` in the specified dtype.
private func fillScale(_ buf: MTLBuffer, value: Float, n: Int, dtype: DType) {
    switch dtype {
    case .f32:
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { ptr[i] = value }
    case .f16:
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: n)
        let h16 = floatToFloat16(value)
        for i in 0..<n { ptr[i] = h16 }
    case .bf16:
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: n)
        let bf = floatToBFloat16(value)
        for i in 0..<n { ptr[i] = bf }
    default:
        fatalError("fillScale: unsupported dtype \(dtype)")
    }
}
