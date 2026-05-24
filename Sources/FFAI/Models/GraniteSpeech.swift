// GraniteSpeech — IBM Granite-4.0 Speech model family.
//
// Architecture:
//   1. CTC Conformer audio encoder  (encoder.*)
//   2. QFormer / BLIP-2 projector   (projector.*)
//   3. Granite LM backbone          (language_model.*)  — Llama-like with
//      Granite-specific multipliers (embedding, residual, logits_scaling).
//
// Reference: mlx-audio-swift/Sources/MLXAudioSTT/Models/GraniteSpeech/
//
// Implementation strategy:
//   • Encoder + projector run ONCE per audio clip on the CPU fast path
//     (read weights to Float arrays, compute in Swift via DispatchQueue.concurrentPerform
//     for the multi-head / multi-layer loops). No GPU involvement here —
//     encoder/projector are not in the hot decode loop.
//   • LM backbone follows the same GPU token-by-token decode pattern as
//     LlamaModel (single MTLCommandBuffer per token, KV cache, argmax/sample).
//
// Weight key layout (from mlx-community/granite-4.0-1b-speech-5bit):
//   encoder.*
//   projector.*
//   language_model.model.*
//   language_model.lm_head.*

import Foundation
import Metal
import Tokenizers

// MARK: - Config

/// Top-level config decoded from config.json. Mirrors GraniteSpeechModelConfig
/// in the mlx-audio-swift reference.
public struct GraniteSpeechConfig: Sendable {
    public struct EncoderConfig: Sendable {
        public let inputDim: Int       // mel features (160 = 2×80)
        public let numLayers: Int      // conformer blocks
        public let hiddenDim: Int
        public let feedforwardMult: Int
        public let numHeads: Int
        public let dimHead: Int
        public let outputDim: Int      // CTC vocab size / projection dim
        public let contextSize: Int    // block size for relative pos attention
        public let maxPosEmb: Int
        public let convKernelSize: Int
        public let convExpansionFactor: Int
    }

    public struct ProjectorConfig: Sendable {
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let intermediateSize: Int
        public let layerNormEps: Float
        public let encoderHiddenSize: Int  // matches encoder.hiddenDim
    }

    public struct TextConfig: Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let attentionBias: Bool
        public let mlpBias: Bool
        public let attentionMultiplier: Float
        public let embeddingMultiplier: Float
        public let residualMultiplier: Float
        public let logitsScaling: Float
        public let tieWordEmbeddings: Bool
    }

    public let encoderConfig: EncoderConfig
    public let projectorConfig: ProjectorConfig
    public let textConfig: TextConfig
    public let audioTokenIndex: Int   // token id replaced by audio embeddings
    public let downsampleRate: Int    // QFormer output compression
    public let windowSize: Int        // QFormer chunk size (frames)
    public let quantization: ModelConfig.QuantizationConfig?

    // Derived: number of QFormer output tokens per window.
    public var numQueriesPerWindow: Int { windowSize / downsampleRate }

    public static func load(from config: ModelConfig) throws -> GraniteSpeechConfig {
        let enc = config.nested("encoder_config") ?? [:]
        let proj = config.nested("projector_config") ?? [:]
        let txt  = config.nested("text_config") ?? [:]

        func intE(_ k: String, _ d: Int) -> Int {
            (enc[k] as? Int) ?? d
        }
        func intP(_ k: String, _ d: Int) -> Int {
            (proj[k] as? Int) ?? d
        }
        func intT(_ k: String, _ d: Int) -> Int {
            (txt[k] as? Int) ?? d
        }
        func floatP(_ k: String, _ d: Float) -> Float {
            Float((proj[k] as? Double) ?? Double(d))
        }
        func floatT(_ k: String, _ d: Float) -> Float {
            // Config values come in as Double (JSON numbers)
            if let v = txt[k] as? Double { return Float(v) }
            if let v = txt[k] as? Int { return Float(v) }
            return d
        }
        func boolT(_ k: String, _ d: Bool) -> Bool {
            (txt[k] as? Bool) ?? d
        }

        let encoderCfg = EncoderConfig(
            inputDim:           intE("input_dim",          160),
            numLayers:          intE("num_layers",          16),
            hiddenDim:          intE("hidden_dim",         1024),
            feedforwardMult:    intE("feedforward_mult",      4),
            numHeads:           intE("num_heads",             8),
            dimHead:            intE("dim_head",           128),
            outputDim:          intE("output_dim",         348),
            contextSize:        intE("context_size",       200),
            maxPosEmb:          intE("max_pos_emb",        512),
            convKernelSize:     intE("conv_kernel_size",    15),
            convExpansionFactor:intE("conv_expansion_factor", 2)
        )

        let projCfg = ProjectorConfig(
            hiddenSize:         intP("hidden_size",       1024),
            numHiddenLayers:    intP("num_hidden_layers",    2),
            numAttentionHeads:  intP("num_attention_heads", 16),
            intermediateSize:   intP("intermediate_size", 4096),
            layerNormEps:       floatP("layer_norm_eps",  1e-12),
            encoderHiddenSize:  intP("encoder_hidden_size", 1024)
        )

        let textCfg = TextConfig(
            vocabSize:            intT("vocab_size",           100353),
            hiddenSize:           intT("hidden_size",            2048),
            intermediateSize:     intT("intermediate_size",     4096),
            numHiddenLayers:      intT("num_hidden_layers",       40),
            numAttentionHeads:    intT("num_attention_heads",     16),
            numKeyValueHeads:     intT("num_key_value_heads",      4),
            maxPositionEmbeddings:intT("max_position_embeddings", 4096),
            rmsNormEps:           floatT("rms_norm_eps",         1e-5),
            ropeTheta:            floatT("rope_theta",         10000),
            attentionBias:        boolT("attention_bias",      false),
            mlpBias:              boolT("mlp_bias",             false),
            attentionMultiplier:  floatT("attention_multiplier", 0.0078125),
            embeddingMultiplier:  floatT("embedding_multiplier",    12.0),
            residualMultiplier:   floatT("residual_multiplier",     0.22),
            logitsScaling:        floatT("logits_scaling",           8.0),
            tieWordEmbeddings:    boolT("tie_word_embeddings",    false)
        )

        return GraniteSpeechConfig(
            encoderConfig:  encoderCfg,
            projectorConfig: projCfg,
            textConfig: textCfg,
            audioTokenIndex: config.int("audio_token_index") ?? 100352,
            downsampleRate: config.int("downsample_rate") ?? 5,
            windowSize:     config.int("window_size")      ?? 15,
            quantization:   config.quantization
        )
    }
}

// MARK: - CPU float helpers

/// Tiny weight store for CPU-side computation. Holds a decoded float32 array
/// of a weight tensor that normally lives in safetensors (which may be bf16).
/// We decode to f32 once at load time for all encoder/projector weights.
private func loadF32(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .f16:
        let halfs = t.toArray(as: UInt16.self)
        return halfs.map { float16ToFloat32($0) }
    case .bf16:
        let bhalfs = t.toArray(as: UInt16.self)
        return bhalfs.map { bfloat16ToFloat32($0) }
    default:
        fatalError("GraniteSpeech: unsupported encoder weight dtype \(t.dtype)")
    }
}

/// Load a bias tensor (may be absent for certain layers). Returns zeros if missing.
private func loadBiasF32(_ bundle: SafeTensorsBundle, _ key: String, size: Int) -> [Float] {
    guard let t = try? bundle.tensor(named: key) else { return [Float](repeating: 0, count: size) }
    return loadF32(t)
}

private func float16ToFloat32(_ bits: UInt16) -> Float {
    // IEEE 754 half → float conversion.
    let sign: UInt32 = UInt32(bits >> 15) << 31
    let exp  = (bits >> 10) & 0x1F
    let mant = bits & 0x3FF
    if exp == 0 {
        // subnormal
        if mant == 0 { return Float(bitPattern: sign) }
        var e: UInt32 = 0
        var m = UInt32(mant)
        while m & 0x400 == 0 { m <<= 1; e += 1 }
        let fullExp = (127 - 15 - e + 1) << 23
        return Float(bitPattern: sign | fullExp | ((m & 0x3FF) << 13))
    }
    if exp == 0x1F {
        // inf / nan
        return Float(bitPattern: sign | 0x7F800000 | (UInt32(mant) << 13))
    }
    let fullExp = (UInt32(exp) + (127 - 15)) << 23
    return Float(bitPattern: sign | fullExp | (UInt32(mant) << 13))
}

private func bfloat16ToFloat32(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

/// Dense matrix-vector multiply: y = W·x + b.
/// W shape: [outDim, inDim] (row-major), x shape: [inDim], b shape: [outDim].
private func matVec(_ w: [Float], _ x: [Float], _ b: [Float]? = nil,
                    outDim: Int, inDim: Int) -> [Float] {
    precondition(w.count == outDim * inDim)
    precondition(x.count == inDim)
    var y = [Float](repeating: 0, count: outDim)
    // Parallelise over rows so multi-layer conformer attention
    // uses all CPU cores when processing the encoder.
    DispatchQueue.concurrentPerform(iterations: outDim) { i in
        var acc: Float = 0
        let base = i * inDim
        for j in 0..<inDim { acc += w[base + j] * x[j] }
        y[i] = acc
    }
    if let bias = b {
        for i in 0..<outDim { y[i] += bias[i] }
    }
    return y
}

/// Batch matrix-vector: each row of `x` (shape [n, inDim]) projected by W.
/// Output shape [n, outDim]. Parallelised over rows × output channels.
private func batchMatVec(_ w: [Float], _ xRows: [[Float]], _ b: [Float]? = nil,
                         outDim: Int, inDim: Int) -> [[Float]] {
    let n = xRows.count
    var out = [[Float]](repeating: [Float](repeating: 0, count: outDim), count: n)
    DispatchQueue.concurrentPerform(iterations: n) { row in
        let x = xRows[row]
        var y = [Float](repeating: 0, count: outDim)
        for i in 0..<outDim {
            var acc: Float = 0
            let base = i * inDim
            for j in 0..<inDim { acc += w[base + j] * x[j] }
            y[i] = acc
        }
        if let bias = b {
            for i in 0..<outDim { y[i] += bias[i] }
        }
        out[row] = y
    }
    return out
}

/// Softmax over a 1D array.
private func softmax(_ x: [Float]) -> [Float] {
    let maxVal = x.max() ?? 0
    var exps = x.map { expf($0 - maxVal) }
    let sum = exps.reduce(0, +)
    for i in exps.indices { exps[i] /= sum }
    return exps
}

/// SiLU activation: x * sigmoid(x).
private func silu(_ x: inout [Float]) {
    for i in x.indices { x[i] = x[i] / (1 + expf(-x[i])) }
}

/// GELU activation (approximate: tanh variant).
private func gelu(_ x: inout [Float]) {
    let c: Float = 0.7978845608028654  // sqrt(2/pi)
    for i in x.indices {
        let xi = x[i]
        let inner = c * (xi + 0.044715 * xi * xi * xi)
        x[i] = 0.5 * xi * (1 + tanhf(inner))
    }
}

/// Sigmoid element-wise.
private func sigmoid(_ x: [Float]) -> [Float] { x.map { 1 / (1 + expf(-$0)) } }

/// Layer norm: normalise each vector in-place given weight (scale) and bias.
private func layerNorm(_ x: inout [Float], weight: [Float], bias: [Float], eps: Float) {
    let n = x.count
    let mean = x.reduce(0, +) / Float(n)
    let variance = x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(n)
    let std = sqrtf(variance + eps)
    for i in 0..<n {
        x[i] = (x[i] - mean) / std * weight[i] + bias[i]
    }
}

/// RMS norm: in-place normalise then scale.
private func rmsNorm(_ x: inout [Float], weight: [Float], eps: Float) {
    let n = x.count
    let meanSq = x.map { $0 * $0 }.reduce(0, +) / Float(n)
    let scale = 1.0 / sqrtf(meanSq + eps)
    for i in 0..<n { x[i] = x[i] * scale * weight[i] }
}

// MARK: - Encoder weight store

/// All weights for the CTC Conformer encoder, loaded to CPU float32.
final class GraniteSpeechEncoderWeights {
    let cfg: GraniteSpeechConfig.EncoderConfig

    // input_linear
    let inputW: [Float]
    let inputB: [Float]

    // per-layer weights
    struct LayerWeights {
        // ff1
        let ff1PreNormW: [Float]; let ff1PreNormB: [Float]
        let ff1UpW: [Float];     let ff1UpB: [Float]
        let ff1DownW: [Float];   let ff1DownB: [Float]

        // attn
        let attnPreNormW: [Float]; let attnPreNormB: [Float]
        let attnRelPosW: [Float]   // [2*maxPosEmb+1, dimHead]
        let attnToQW: [Float]      // [numHeads*dimHead, hiddenDim]
        let attnToKVW: [Float]     // [2*numHeads*dimHead, hiddenDim]
        let attnToOutW: [Float];   let attnToOutB: [Float]

        // conv
        let convNormW: [Float]; let convNormB: [Float]
        let convUpW: [Float];   let convUpB: [Float]
        let convDepthW: [Float]  // [innerDim, 1, kernelSize] — already (chanOut,chanIn,k) order
        let convBnW: [Float]; let convBnB: [Float]
        let convBnMean: [Float]; let convBnVar: [Float]
        let convDownW: [Float];  let convDownB: [Float]

        // ff2
        let ff2PreNormW: [Float]; let ff2PreNormB: [Float]
        let ff2UpW: [Float];     let ff2UpB: [Float]
        let ff2DownW: [Float];   let ff2DownB: [Float]

        // post_norm
        let postNormW: [Float]; let postNormB: [Float]
    }

    let layers: [LayerWeights]

    // out linear + out_mid linear
    let outW: [Float];    let outB: [Float]
    let outMidW: [Float]; let outMidB: [Float]

    init(_ cfg: GraniteSpeechConfig.EncoderConfig, bundle: SafeTensorsBundle) throws {
        self.cfg = cfg
        let p = "encoder"

        inputW = loadF32(try bundle.tensor(named: "\(p).input_linear.weight"))
        inputB = loadBiasF32(bundle, "\(p).input_linear.bias", size: cfg.hiddenDim)

        var ls: [LayerWeights] = []
        let innerConvDim = cfg.hiddenDim * cfg.convExpansionFactor
        for i in 0..<cfg.numLayers {
            let lp = "\(p).layers.\(i)"

            let ff1PreNormW = loadF32(try bundle.tensor(named: "\(lp).ff1.pre_norm.weight"))
            let ff1PreNormB = loadBiasF32(bundle, "\(lp).ff1.pre_norm.bias", size: cfg.hiddenDim)
            let ff1UpW   = loadF32(try bundle.tensor(named: "\(lp).ff1.up_proj.weight"))
            let ff1UpB   = loadBiasF32(bundle, "\(lp).ff1.up_proj.bias", size: cfg.hiddenDim * cfg.feedforwardMult)
            let ff1DownW = loadF32(try bundle.tensor(named: "\(lp).ff1.down_proj.weight"))
            let ff1DownB = loadBiasF32(bundle, "\(lp).ff1.down_proj.bias", size: cfg.hiddenDim)

            let attnPreNormW = loadF32(try bundle.tensor(named: "\(lp).attn.pre_norm.weight"))
            let attnPreNormB = loadBiasF32(bundle, "\(lp).attn.pre_norm.bias", size: cfg.hiddenDim)
            let attnRelPosW  = loadF32(try bundle.tensor(named: "\(lp).attn.rel_pos_emb.weight"))
            let attnToQW     = loadF32(try bundle.tensor(named: "\(lp).attn.to_q.weight"))
            let attnToKVW    = loadF32(try bundle.tensor(named: "\(lp).attn.to_kv.weight"))
            let attnToOutW   = loadF32(try bundle.tensor(named: "\(lp).attn.to_out.weight"))
            let attnToOutB   = loadBiasF32(bundle, "\(lp).attn.to_out.bias", size: cfg.hiddenDim)

            let convNormW = loadF32(try bundle.tensor(named: "\(lp).conv.norm.weight"))
            let convNormB = loadBiasF32(bundle, "\(lp).conv.norm.bias", size: cfg.hiddenDim)
            let convUpW   = loadF32(try bundle.tensor(named: "\(lp).conv.up_conv.weight"))
            let convUpB   = loadBiasF32(bundle, "\(lp).conv.up_conv.bias", size: innerConvDim * 2)

            // depthwise conv weight: [innerDim, 1, kernelSize] in PyTorch order
            // MLX has transposed it to [kernelSize, 1, innerDim] (Conv1d format).
            // We store as-loaded and handle the order in convForward.
            let convDepthW = loadF32(try bundle.tensor(named: "\(lp).conv.depth_conv.conv.weight"))

            let convBnW    = loadF32(try bundle.tensor(named: "\(lp).conv.batch_norm.weight"))
            let convBnB    = loadBiasF32(bundle, "\(lp).conv.batch_norm.bias", size: innerConvDim)
            let convBnMean = loadF32(try bundle.tensor(named: "\(lp).conv.batch_norm.running_mean"))
            let convBnVar  = loadF32(try bundle.tensor(named: "\(lp).conv.batch_norm.running_var"))
            let convDownW  = loadF32(try bundle.tensor(named: "\(lp).conv.down_conv.weight"))
            let convDownB  = loadBiasF32(bundle, "\(lp).conv.down_conv.bias", size: cfg.hiddenDim)

            let ff2PreNormW = loadF32(try bundle.tensor(named: "\(lp).ff2.pre_norm.weight"))
            let ff2PreNormB = loadBiasF32(bundle, "\(lp).ff2.pre_norm.bias", size: cfg.hiddenDim)
            let ff2UpW   = loadF32(try bundle.tensor(named: "\(lp).ff2.up_proj.weight"))
            let ff2UpB   = loadBiasF32(bundle, "\(lp).ff2.up_proj.bias", size: cfg.hiddenDim * cfg.feedforwardMult)
            let ff2DownW = loadF32(try bundle.tensor(named: "\(lp).ff2.down_proj.weight"))
            let ff2DownB = loadBiasF32(bundle, "\(lp).ff2.down_proj.bias", size: cfg.hiddenDim)

            let postNormW = loadF32(try bundle.tensor(named: "\(lp).post_norm.weight"))
            let postNormB = loadBiasF32(bundle, "\(lp).post_norm.bias", size: cfg.hiddenDim)

            ls.append(LayerWeights(
                ff1PreNormW: ff1PreNormW, ff1PreNormB: ff1PreNormB,
                ff1UpW: ff1UpW, ff1UpB: ff1UpB,
                ff1DownW: ff1DownW, ff1DownB: ff1DownB,
                attnPreNormW: attnPreNormW, attnPreNormB: attnPreNormB,
                attnRelPosW: attnRelPosW,
                attnToQW: attnToQW, attnToKVW: attnToKVW,
                attnToOutW: attnToOutW, attnToOutB: attnToOutB,
                convNormW: convNormW, convNormB: convNormB,
                convUpW: convUpW, convUpB: convUpB,
                convDepthW: convDepthW,
                convBnW: convBnW, convBnB: convBnB,
                convBnMean: convBnMean, convBnVar: convBnVar,
                convDownW: convDownW, convDownB: convDownB,
                ff2PreNormW: ff2PreNormW, ff2PreNormB: ff2PreNormB,
                ff2UpW: ff2UpW, ff2UpB: ff2UpB,
                ff2DownW: ff2DownW, ff2DownB: ff2DownB,
                postNormW: postNormW, postNormB: postNormB
            ))
        }
        self.layers = ls

        outW    = loadF32(try bundle.tensor(named: "\(p).out.weight"))
        outB    = loadBiasF32(bundle, "\(p).out.bias", size: cfg.outputDim)
        outMidW = loadF32(try bundle.tensor(named: "\(p).out_mid.weight"))
        outMidB = loadBiasF32(bundle, "\(p).out_mid.bias", size: cfg.hiddenDim)
    }
}

// MARK: - Projector weight store

final class GraniteSpeechProjectorWeights {
    let cfg: GraniteSpeechConfig
    // query: [1, numQueries, hiddenSize]
    let query: [Float]      // shape: [numQueries * hiddenSize]
    // qformer.layernorm
    let qfLnW: [Float]; let qfLnB: [Float]
    // per-layer
    struct QFLayerWeights {
        // self-attention
        let saQW: [Float]; let saQB: [Float]
        let saKW: [Float]; let saKB: [Float]
        let saVW: [Float]; let saVB: [Float]
        let saOutDenseW: [Float]; let saOutDenseB: [Float]
        let saOutLnW: [Float];   let saOutLnB: [Float]
        // cross-attention
        let caQW: [Float]; let caQB: [Float]
        let caKW: [Float]; let caKB: [Float]
        let caVW: [Float]; let caVB: [Float]
        let caOutDenseW: [Float]; let caOutDenseB: [Float]
        let caOutLnW: [Float];   let caOutLnB: [Float]
        // intermediate + output
        let intDenseW: [Float]; let intDenseB: [Float]
        let outDenseW: [Float]; let outDenseB: [Float]
        let outLnW: [Float];   let outLnB: [Float]
    }
    let layers: [QFLayerWeights]
    // linear: projector.linear
    let linearW: [Float]; let linearB: [Float]

    init(_ cfg: GraniteSpeechConfig, bundle: SafeTensorsBundle) throws {
        self.cfg = cfg
        let p = "projector"
        let pcfg = cfg.projectorConfig
        let hs = pcfg.hiddenSize
        // ehs: encoder hidden size — used as kvDim in qformerMHA at forward time,
        // not needed at load time (cross-attn biases are sized [hs], the output dim).
        _ = pcfg.encoderHiddenSize

        let numQ = cfg.numQueriesPerWindow
        query = loadF32(try bundle.tensor(named: "\(p).query"))
        qfLnW = loadF32(try bundle.tensor(named: "\(p).qformer.layernorm.weight"))
        qfLnB = loadBiasF32(bundle, "\(p).qformer.layernorm.bias", size: hs)

        var ls: [QFLayerWeights] = []
        for i in 0..<pcfg.numHiddenLayers {
            let lp = "\(p).qformer.encoder.layer.\(i)"

            // self-attention (projection: key/value against self → kvDim=hs)
            let saQW = loadF32(try bundle.tensor(named: "\(lp).attention.attention.query.weight"))
            let saQB = loadBiasF32(bundle, "\(lp).attention.attention.query.bias", size: hs)
            let saKW = loadF32(try bundle.tensor(named: "\(lp).attention.attention.key.weight"))
            let saKB = loadBiasF32(bundle, "\(lp).attention.attention.key.bias", size: hs)
            let saVW = loadF32(try bundle.tensor(named: "\(lp).attention.attention.value.weight"))
            let saVB = loadBiasF32(bundle, "\(lp).attention.attention.value.bias", size: hs)
            let saOutDenseW = loadF32(try bundle.tensor(named: "\(lp).attention.output.dense.weight"))
            let saOutDenseB = loadBiasF32(bundle, "\(lp).attention.output.dense.bias", size: hs)
            let saOutLnW = loadF32(try bundle.tensor(named: "\(lp).attention.output.LayerNorm.weight"))
            let saOutLnB = loadBiasF32(bundle, "\(lp).attention.output.LayerNorm.bias", size: hs)

            // cross-attention (key/value against encoder hidden → kvDim=ehs)
            let caQW = loadF32(try bundle.tensor(named: "\(lp).crossattention.attention.query.weight"))
            let caQB = loadBiasF32(bundle, "\(lp).crossattention.attention.query.bias", size: hs)
            let caKW = loadF32(try bundle.tensor(named: "\(lp).crossattention.attention.key.weight"))
            let caKB = loadBiasF32(bundle, "\(lp).crossattention.attention.key.bias", size: hs)
            let caVW = loadF32(try bundle.tensor(named: "\(lp).crossattention.attention.value.weight"))
            let caVB = loadBiasF32(bundle, "\(lp).crossattention.attention.value.bias", size: hs)
            let caOutDenseW = loadF32(try bundle.tensor(named: "\(lp).crossattention.output.dense.weight"))
            let caOutDenseB = loadBiasF32(bundle, "\(lp).crossattention.output.dense.bias", size: hs)
            let caOutLnW = loadF32(try bundle.tensor(named: "\(lp).crossattention.output.LayerNorm.weight"))
            let caOutLnB = loadBiasF32(bundle, "\(lp).crossattention.output.LayerNorm.bias", size: hs)

            let intDenseW = loadF32(try bundle.tensor(named: "\(lp).intermediate_query.dense.weight"))
            let intDenseB = loadBiasF32(bundle, "\(lp).intermediate_query.dense.bias", size: pcfg.intermediateSize)
            let outDenseW = loadF32(try bundle.tensor(named: "\(lp).output_query.dense.weight"))
            let outDenseB = loadBiasF32(bundle, "\(lp).output_query.dense.bias", size: hs)
            let outLnW = loadF32(try bundle.tensor(named: "\(lp).output_query.LayerNorm.weight"))
            let outLnB = loadBiasF32(bundle, "\(lp).output_query.LayerNorm.bias", size: hs)

            _ = numQ  // suppress "unused" warning — used as numQ queries per window at runtime
            ls.append(QFLayerWeights(
                saQW: saQW, saQB: saQB, saKW: saKW, saKB: saKB, saVW: saVW, saVB: saVB,
                saOutDenseW: saOutDenseW, saOutDenseB: saOutDenseB,
                saOutLnW: saOutLnW, saOutLnB: saOutLnB,
                caQW: caQW, caQB: caQB, caKW: caKW, caKB: caKB, caVW: caVW, caVB: caVB,
                caOutDenseW: caOutDenseW, caOutDenseB: caOutDenseB,
                caOutLnW: caOutLnW, caOutLnB: caOutLnB,
                intDenseW: intDenseW, intDenseB: intDenseB,
                outDenseW: outDenseW, outDenseB: outDenseB,
                outLnW: outLnW, outLnB: outLnB
            ))
        }
        self.layers = ls

        linearW = loadF32(try bundle.tensor(named: "\(p).linear.weight"))
        linearB = loadBiasF32(bundle, "\(p).linear.bias", size: cfg.textConfig.hiddenSize)
    }
}

// MARK: - CPU Encoder forward

/// Compute mel spectrogram from a 16 kHz mono waveform.
/// Returns a 2D array of shape [T, 160] (pairs of 80-dim mel frames).
private func extractMelFeatures(_ waveform: [Float]) -> [[Float]] {
    let nFft     = 512
    let winLen   = 400
    let hopLen   = 160
    let nMels    = 80
    let sampleRate = 16000

    // Hanning window (periodic, padded to nFft)
    var win = [Float](repeating: 0, count: winLen)
    for n in 0..<winLen {
        win[n] = 0.5 * (1 - cos(2.0 * Float.pi * Float(n) / Float(winLen)))
    }
    // Pad: centre the winLen window in nFft
    let padLeft  = (nFft - winLen) / 2
    var winPadded = [Float](repeating: 0, count: nFft)
    for i in 0..<winLen { winPadded[padLeft + i] = win[i] }

    // Reflect-pad audio so first and last windows are centred
    let halfWin = nFft / 2
    var audio = [Float](repeating: 0, count: waveform.count + 2 * halfWin)
    for i in 0..<halfWin { audio[halfWin - 1 - i] = waveform[min(i, waveform.count - 1)] }
    for i in 0..<waveform.count { audio[halfWin + i] = waveform[i] }
    for i in 0..<halfWin {
        let src = waveform.count - 1 - i
        audio[halfWin + waveform.count + i] = waveform[max(0, src)]
    }

    let nFrames = (audio.count - nFft) / hopLen + 1
    let nFreqs  = nFft / 2 + 1

    // STFT: magnitude squared spectrum [nFrames, nFreqs]
    var powerSpec = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nFrames)
    // Process frames — parallelise across frames for speed
    DispatchQueue.concurrentPerform(iterations: nFrames) { frame in
        let offset = frame * hopLen
        // windowed frame (imaginary part unused — pure real input)
        var re = [Float](repeating: 0, count: nFft)
        for k in 0..<nFft { re[k] = audio[offset + k] * winPadded[k] }
        // DFT of this frame (only positive frequencies)
        for freq in 0..<nFreqs {
            var rSum: Float = 0, iSum: Float = 0
            let angle = -2.0 * Float.pi * Float(freq) / Float(nFft)
            for t in 0..<nFft {
                let theta = angle * Float(t)
                rSum += re[t] * cos(theta)
                iSum += re[t] * sin(theta)
            }
            powerSpec[frame][freq] = rSum * rSum + iSum * iSum
        }
    }

    // Mel filter bank [nFreqs, nMels] (HTK triangular, no norm)
    let fMin: Float = 0
    let fMax: Float = Float(sampleRate) / 2.0
    func hzToMel(_ hz: Float) -> Float { 2595 * log10f(1 + hz / 700) }
    func melToHz(_ mel: Float) -> Float { 700 * (pow(10, mel / 2595) - 1) }
    let melMin = hzToMel(fMin)
    let melMax = hzToMel(fMax)
    // nMels+2 evenly spaced mel points
    var melPts = [Float](repeating: 0, count: nMels + 2)
    for i in 0..<(nMels + 2) {
        melPts[i] = melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
    }
    // Convert to bin indices
    let binPts = melPts.map { Int(($0 / (Float(sampleRate) / Float(nFft))).rounded()) }
    // Build filter bank
    var melFB = [[Float]](repeating: [Float](repeating: 0, count: nMels), count: nFreqs)
    for m in 0..<nMels {
        let fLeft  = binPts[m]
        let fCenter = binPts[m + 1]
        let fRight = binPts[m + 2]
        for k in fLeft..<fCenter where fCenter > fLeft {
            melFB[k][m] = Float(k - fLeft) / Float(fCenter - fLeft)
        }
        for k in fCenter..<fRight where fRight > fCenter {
            melFB[k][m] = Float(fRight - k) / Float(fRight - fCenter)
        }
    }

    // Mel spectrogram [nFrames, nMels]
    var melSpec = [[Float]](repeating: [Float](repeating: 0, count: nMels), count: nFrames)
    DispatchQueue.concurrentPerform(iterations: nFrames) { frame in
        for m in 0..<nMels {
            var sum: Float = 0
            for k in 0..<nFreqs { sum += powerSpec[frame][k] * melFB[k][m] }
            melSpec[frame][m] = sum
        }
    }

    // Log-mel: log10(max(1e-10, x)), normalise
    var logMel = melSpec.map { row in row.map { max(1e-10, $0) }.map { log10f($0) } }
    let maxVal = logMel.flatMap { $0 }.max() ?? 0
    for i in logMel.indices {
        for j in logMel[i].indices {
            logMel[i][j] = (max(logMel[i][j], maxVal - 8.0) / 4.0) + 1.0
        }
    }

    // Drop trailing frame if total is odd, then pair adjacent frames: [T, 160]
    var nF = logMel.count
    if nF % 2 == 1 { logMel.removeLast(); nF -= 1 }
    var paired = [[Float]](repeating: [Float](repeating: 0, count: 2 * nMels), count: nF / 2)
    for i in 0..<(nF / 2) {
        paired[i] = logMel[2 * i] + logMel[2 * i + 1]
    }
    return paired
}

/// Apply a 1-D depthwise convolution with pre-computed causal padding.
/// Input: [T, chan], weight: [outChan, 1, kernelSize] (PyTorch layout after MLX transpose).
/// The MLX port transposes the weight to [kernelSize, 1, chan] (Conv1d layout).
/// We handle both layouts — detect by comparing weight count vs shapes.
private func depthwiseConv1d(input: [[Float]], weight: [Float],
                              kernelSize: Int, chan: Int,
                              paddingLeft: Int, paddingRight: Int) -> [[Float]] {
    let padded = (0..<paddingLeft).map { _ in [Float](repeating: 0, count: chan) }
        + input
        + (0..<paddingRight).map { _ in [Float](repeating: 0, count: chan) }
    let outT = padded.count - kernelSize + 1

    // Weight layout from MLX (after sanitise transposition):
    // [kernelSize, 1, chan] — i.e., weight[k * chan + c] = filter coeff for channel c, offset k
    var out = [[Float]](repeating: [Float](repeating: 0, count: chan), count: outT)
    DispatchQueue.concurrentPerform(iterations: outT) { t in
        for c in 0..<chan {
            var acc: Float = 0
            for k in 0..<kernelSize {
                acc += padded[t + k][c] * weight[k * chan + c]
            }
            out[t][c] = acc
        }
    }
    return out
}

/// Apply 1×1 convolution: [T, inChan] → [T, outChan] (equivalent to row-wise linear).
private func conv1x1(input: [[Float]], weight: [Float], bias: [Float],
                     outChan: Int, inChan: Int) -> [[Float]] {
    return batchMatVec(weight, input, bias, outDim: outChan, inDim: inChan)
}

/// Batch norm (inference-only): (x - mean) / sqrt(var + eps) * weight + bias.
private func batchNorm1d(_ x: [[Float]], weight: [Float], bias: [Float],
                          mean: [Float], variance: [Float], eps: Float = 1e-5) -> [[Float]] {
    let chan = weight.count
    return x.map { frame in
        var y = frame
        for c in 0..<chan {
            y[c] = (y[c] - mean[c]) / sqrtf(variance[c] + eps) * weight[c] + bias[c]
        }
        return y
    }
}

/// Conformer feed-forward block: preNorm → up → silu → down, residual scaled by 0.5.
private func conformerFF(
    _ x: [[Float]], w: GraniteSpeechEncoderWeights.LayerWeights,
    isFF1: Bool, cfg: GraniteSpeechConfig.EncoderConfig
) -> [[Float]] {
    let ffDim = cfg.hiddenDim * cfg.feedforwardMult
    let (preNormW, preNormB, upW, upB, downW, downB): ([Float],[Float],[Float],[Float],[Float],[Float])
    if isFF1 {
        (preNormW, preNormB) = (w.ff1PreNormW, w.ff1PreNormB)
        (upW, upB) = (w.ff1UpW, w.ff1UpB)
        (downW, downB) = (w.ff1DownW, w.ff1DownB)
    } else {
        (preNormW, preNormB) = (w.ff2PreNormW, w.ff2PreNormB)
        (upW, upB) = (w.ff2UpW, w.ff2UpB)
        (downW, downB) = (w.ff2DownW, w.ff2DownB)
    }
    let T = x.count
    var normed = x
    for t in 0..<T { layerNorm(&normed[t], weight: preNormW, bias: preNormB, eps: 1e-5) }
    var up = batchMatVec(upW, normed, upB, outDim: ffDim, inDim: cfg.hiddenDim)
    for t in 0..<T { silu(&up[t]) }
    var down = batchMatVec(downW, up, downB, outDim: cfg.hiddenDim, inDim: ffDim)
    // Residual: 0.5 * ff(x) + x
    for t in 0..<T { for d in 0..<cfg.hiddenDim { down[t][d] = 0.5 * down[t][d] + x[t][d] } }
    return down
}

/// Block-wise relative-position conformer attention.
/// x: [T, hiddenDim]; returns [T, hiddenDim].
private func conformerAttn(
    _ x: [[Float]], w: GraniteSpeechEncoderWeights.LayerWeights,
    attnDists: [[Int32]], cfg: GraniteSpeechConfig.EncoderConfig
) -> [[Float]] {
    let T = x.count
    let C = cfg.contextSize
    let H = cfg.numHeads
    let Dh = cfg.dimHead
    let innerDim = H * Dh
    let scale = pow(Float(Dh), -0.5)

    var normed = x
    for t in 0..<T { layerNorm(&normed[t], weight: w.attnPreNormW, bias: w.attnPreNormB, eps: 1e-5) }

    // Pad to multiple of contextSize
    let numBlocks = (T + C - 1) / C
    let remainder = T % C
    var padded = normed
    if remainder > 0 {
        let padLen = C - remainder
        padded += [[Float]](repeating: [Float](repeating: 0, count: cfg.hiddenDim), count: padLen)
    }
    // Q: [numBlocks*C, innerDim]
    let qFlat = batchMatVec(w.attnToQW, padded, nil, outDim: innerDim, inDim: cfg.hiddenDim)
    // KV: [Tp, 2*innerDim]
    let kvFlat = batchMatVec(w.attnToKVW, padded, nil, outDim: innerDim * 2, inDim: cfg.hiddenDim)

    // Output accumulator [T, innerDim]
    var outRows = [[Float]](repeating: [Float](repeating: 0, count: innerDim), count: T)

    // Process each block independently
    for b in 0..<numBlocks {
        let start = b * C
        let isLast = b == numBlocks - 1
        let validLen = isLast && remainder > 0 ? remainder : C

        // Extract Q,K,V for this block: [C, H, Dh]
        // Reshape: q[t][h*Dh + d] → q[h][t][d]
        var qBlock = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C), count: H)
        var kBlock = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C), count: H)
        var vBlock = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C), count: H)

        for t in 0..<C {
            let row = start + t
            for h in 0..<H {
                for d in 0..<Dh {
                    qBlock[h][t][d] = qFlat[row][h * Dh + d]
                    kBlock[h][t][d] = kvFlat[row][h * Dh + d]
                    vBlock[h][t][d] = kvFlat[row][innerDim + h * Dh + d]
                }
            }
        }

        // Compute attention per head — parallelise over heads.
        //
        // TODO(perf): Migrate to a GPU kernel. The Conformer encoder
        // self-attention has Dh=128 (no `Ops.sdpaBidirectional` variant
        // ships beyond d=96) AND adds a per-pair relative-position bias
        // computed against `relPosEmb` — neither fits the current
        // bidirectional SDPA contract. Until a `sdpaBidirectionalRelPos`
        // / d128 kernel lands, this stays on the CPU concurrent loop. The
        // QFormer MHA below (16 × 64) already runs on GPU.
        var headOuts = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C), count: H)
        DispatchQueue.concurrentPerform(iterations: H) { h in
            // Dot-product scores [C, C]: q[h][i] · k[h][j] * scale
            var scores = [[Float]](repeating: [Float](repeating: 0, count: C), count: C)
            for i in 0..<C {
                for j in 0..<C {
                    var dot: Float = 0
                    for d in 0..<Dh { dot += qBlock[h][i][d] * kBlock[h][j][d] }
                    scores[i][j] = dot * scale
                }
            }
            // Add relative position bias: posAttn[i][j] = q[i] · relEmb[dist(i,j)] * scale
            for i in 0..<C {
                for j in 0..<C {
                    let distIdx = Int(attnDists[i][j])  // already clamped+shifted
                    var posDot: Float = 0
                    // relPosEmb: [2*maxPosEmb+1, dimHead] — look up row distIdx
                    let relBase = distIdx * Dh
                    for d in 0..<Dh { posDot += qBlock[h][i][d] * w.attnRelPosW[relBase + d] }
                    scores[i][j] += posDot * scale
                }
            }
            // Mask padding tokens in the last block
            if isLast && remainder > 0 {
                for i in 0..<C {
                    for j in 0..<C {
                        let rowValid = i < validLen
                        let colValid = j < validLen
                        if !rowValid || !colValid { scores[i][j] = -1e9 }
                    }
                }
            }
            // Softmax per row then attend
            for i in 0..<C {
                let w_i = softmax(scores[i])
                for d in 0..<Dh {
                    var acc: Float = 0
                    for j in 0..<C { acc += w_i[j] * vBlock[h][j][d] }
                    headOuts[h][i][d] = acc
                }
            }
        }

        // Concatenate heads → [validLen, innerDim], apply to_out
        for t in 0..<validLen {
            let globalT = start + t
            guard globalT < T else { break }
            var concat = [Float](repeating: 0, count: innerDim)
            for h in 0..<H {
                for d in 0..<Dh { concat[h * Dh + d] = headOuts[h][t][d] }
            }
            let out = matVec(w.attnToOutW, concat, w.attnToOutB, outDim: cfg.hiddenDim, inDim: innerDim)
            for d in 0..<cfg.hiddenDim { outRows[globalT][d] += out[d] }
        }
    }

    // Residual (output of attn + original x)
    for t in 0..<T { for d in 0..<cfg.hiddenDim { outRows[t][d] += x[t][d] } }
    return outRows
}

/// Conformer conv module.
private func conformerConv(
    _ x: [[Float]], w: GraniteSpeechEncoderWeights.LayerWeights,
    cfg: GraniteSpeechConfig.EncoderConfig
) -> [[Float]] {
    let T = x.count
    let innerDim = cfg.hiddenDim * cfg.convExpansionFactor

    // Layer norm
    var normed = x
    for t in 0..<T { layerNorm(&normed[t], weight: w.convNormW, bias: w.convNormB, eps: 1e-5) }

    // up_conv: [T, hiddenDim] → [T, innerDim*2]  (1×1 conv = linear per frame)
    let up = conv1x1(input: normed, weight: w.convUpW, bias: w.convUpB,
                     outChan: innerDim * 2, inChan: cfg.hiddenDim)

    // GLU: split [innerDim*2] → gate + signal, gate through sigmoid
    var gated = [[Float]](repeating: [Float](repeating: 0, count: innerDim), count: T)
    for t in 0..<T {
        let sig = Array(up[t][0..<innerDim])
        let gate = Array(up[t][innerDim..<2 * innerDim])
        let gateAct = sigmoid(gate)
        for d in 0..<innerDim { gated[t][d] = sig[d] * gateAct[d] }
    }

    // Depthwise conv1d
    let kernelSize = cfg.convKernelSize
    let pad = kernelSize / 2
    let padOffset = (kernelSize + 1) % 2
    let paddingLeft = pad
    let paddingRight = pad - padOffset

    var depthOut = depthwiseConv1d(
        input: gated, weight: w.convDepthW,
        kernelSize: kernelSize, chan: innerDim,
        paddingLeft: paddingLeft, paddingRight: paddingRight
    )

    // BatchNorm + SiLU
    depthOut = batchNorm1d(depthOut, weight: w.convBnW, bias: w.convBnB,
                            mean: w.convBnMean, variance: w.convBnVar)
    for t in 0..<T { silu(&depthOut[t]) }

    // down_conv: [T, innerDim] → [T, hiddenDim]
    let out = conv1x1(input: depthOut, weight: w.convDownW, bias: w.convDownB,
                      outChan: cfg.hiddenDim, inChan: innerDim)

    // Residual
    var res = out
    for t in 0..<T { for d in 0..<cfg.hiddenDim { res[t][d] += x[t][d] } }
    return res
}

/// Build the relative-position distance table once.
private func buildAttnDists(contextSize: Int, maxPosEmb: Int) -> [[Int32]] {
    var dists = [[Int32]](repeating: [Int32](repeating: 0, count: contextSize), count: contextSize)
    for i in 0..<contextSize {
        for j in 0..<contextSize {
            let raw = i - j
            let clamped = max(-contextSize, min(contextSize, raw))
            dists[i][j] = Int32(clamped + maxPosEmb)
        }
    }
    return dists
}

/// Run the full CTC Conformer encoder.
/// Input: [[Float]] shape [T, inputDim=160].
/// Output: [[Float]] shape [T, hiddenDim].
func runEncoder(_ input: [[Float]], weights: GraniteSpeechEncoderWeights) -> [[Float]] {
    let cfg = weights.cfg
    let T = input.count

    // input_linear
    var h = batchMatVec(weights.inputW, input, weights.inputB,
                        outDim: cfg.hiddenDim, inDim: cfg.inputDim)

    let attnDists = buildAttnDists(contextSize: cfg.contextSize, maxPosEmb: cfg.maxPosEmb)
    let halfLayers = cfg.numLayers / 2

    for (idx, layer) in weights.layers.enumerated() {
        // Conformer block: ff1 → attn → conv → ff2 → postNorm
        h = conformerFF(h, w: layer, isFF1: true, cfg: cfg)
        h = conformerAttn(h, w: layer, attnDists: attnDists, cfg: cfg)
        h = conformerConv(h, w: layer, cfg: cfg)
        h = conformerFF(h, w: layer, isFF1: false, cfg: cfg)
        for t in 0..<T { layerNorm(&h[t], weight: layer.postNormW, bias: layer.postNormB, eps: 1e-5) }

        // Mid-point softmax residual connection (after layer numLayers/2)
        if idx + 1 == halfLayers {
            // xMid = out(h): [T, outputDim]
            let xMid = batchMatVec(weights.outW, h, weights.outB,
                                   outDim: cfg.outputDim, inDim: cfg.hiddenDim)
            var softMid = xMid
            for t in 0..<T { softMid[t] = softmax(xMid[t]) }
            // outMid(softmax(xMid)): [T, hiddenDim]
            let midProj = batchMatVec(weights.outMidW, softMid, weights.outMidB,
                                      outDim: cfg.hiddenDim, inDim: cfg.outputDim)
            for t in 0..<T { for d in 0..<cfg.hiddenDim { h[t][d] += midProj[t][d] } }
        }
    }
    return h
}

// MARK: - CPU QFormer forward

/// Multi-head attention (self or cross).
/// queries: [L, hs], kvInput: [M, kvDim] → output: [L, hs]
///
/// Hot path. The projector runs a self-attention and a cross-attention
/// per QFormer layer × per audio window, which on long clips (10 s+)
/// becomes the dominant CPU cost — long enough to time the integration
/// test out at 900 s. Migrate the softmax(QKᵀ)·V kernel onto the GPU via
/// `Ops.sdpaBidirectional` (headDim=64 ships a tuned variant). The Q/K/V
/// projections are still CPU `batchMatVec` because their dimensions are
/// modest and we'd otherwise need to copy them up just to copy them back
/// down for the next layer's residual.
private func qformerMHA(
    queries: [[Float]], kvInput: [[Float]],
    qW: [Float], qB: [Float],
    kW: [Float], kB: [Float],
    vW: [Float], vB: [Float],
    numHeads: Int, hs: Int, kvDim: Int
) -> [[Float]] {
    let L = queries.count
    let M = kvInput.count
    let headDim = hs / numHeads
    precondition(hs == numHeads * headDim,
                 "qformerMHA: hs (\(hs)) must equal numHeads*headDim")
    let scale = pow(Float(headDim), -0.5)

    let q = batchMatVec(qW, queries, qB, outDim: hs, inDim: hs)
    let k = batchMatVec(kW, kvInput, kB, outDim: hs, inDim: kvDim)
    let v = batchMatVec(vW, kvInput, vB, outDim: hs, inDim: kvDim)

    // ── GPU path: dispatch Ops.sdpaBidirectional when the head_dim has
    // a kernel shipped. Falls back to the CPU per-head loop below for
    // unusual configurations (e.g. headDim outside {32,64,72,80,96}).
    if OpsValidation.sdpaBidirectionalSupportedHeadDims.contains(headDim) {
        let device = Device.shared
        // Pack Q into [L, numHeads, headDim] — row-major, same memory
        // order as `q[i][h*Dh+d]`. K/V need transpose to
        // [numHeads, M, headDim] for the kernel's
        // `[nKVHeads, kvStride, headDim]` contract.
        var qFlat = [Float](repeating: 0, count: L * hs)
        for i in 0..<L {
            let base = i * hs
            for d in 0..<hs { qFlat[base + d] = q[i][d] }
        }
        var kFlat = [Float](repeating: 0, count: M * hs)
        var vFlat = [Float](repeating: 0, count: M * hs)
        for j in 0..<M {
            for h in 0..<numHeads {
                let srcOff = h * headDim
                let dst = (h * M + j) * headDim
                for d in 0..<headDim {
                    kFlat[dst + d] = k[j][srcOff + d]
                    vFlat[dst + d] = v[j][srcOff + d]
                }
            }
        }
        let qT = Tensor.empty(shape: [L, numHeads, headDim],
                              dtype: .f32, device: device)
        qT.copyIn(from: qFlat)
        let kT = Tensor.empty(shape: [numHeads, M, headDim],
                              dtype: .f32, device: device)
        kT.copyIn(from: kFlat)
        let vT = Tensor.empty(shape: [numHeads, M, headDim],
                              dtype: .f32, device: device)
        vT.copyIn(from: vFlat)
        let cmd = device.makeCommandBuffer()
        let outT = Ops.sdpaBidirectional(
            q: qT, k: kT, v: vT,
            nQHeads: numHeads, nKVHeads: numHeads, headDim: headDim,
            baseKV: 0, nQuery: L, kvStride: M,
            scale: scale, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        let outFlat = outT.toFloatArray()  // [L, numHeads, headDim]
        // Reassemble back into [L, hs] — the row-major layout matches.
        return (0..<L).map { i in
            let src = i * hs
            return Array(outFlat[src..<src + hs])
        }
    }

    // ── CPU fallback (rare head_dim, kept for safety) ───────────────
    var out = [[Float]](repeating: [Float](repeating: 0, count: hs), count: L)
    DispatchQueue.concurrentPerform(iterations: numHeads) { h in
        let base = h * headDim
        var scores = [[Float]](repeating: [Float](repeating: 0, count: M), count: L)
        for i in 0..<L {
            for j in 0..<M {
                var dot: Float = 0
                for d in 0..<headDim { dot += q[i][base + d] * k[j][base + d] }
                scores[i][j] = dot * scale
            }
        }
        for i in 0..<L {
            let attnW = softmax(scores[i])
            for d in 0..<headDim {
                var acc: Float = 0
                for j in 0..<M { acc += attnW[j] * v[j][base + d] }
                out[i][base + d] = acc
            }
        }
    }
    return out
}

/// One QFormer layer: self-attn → cross-attn → FFN.
private func qformerLayer(
    hidden: [[Float]], encoderStates: [[Float]],
    lw: GraniteSpeechProjectorWeights.QFLayerWeights,
    cfg: GraniteSpeechConfig.ProjectorConfig
) -> [[Float]] {
    let hs  = cfg.hiddenSize
    let ehs = cfg.encoderHiddenSize
    let H   = cfg.numAttentionHeads
    let eps = cfg.layerNormEps
    let L   = hidden.count

    // Self-attention
    var saOut = qformerMHA(
        queries: hidden, kvInput: hidden,
        qW: lw.saQW, qB: lw.saQB, kW: lw.saKW, kB: lw.saKB, vW: lw.saVW, vB: lw.saVB,
        numHeads: H, hs: hs, kvDim: hs
    )
    saOut = batchMatVec(lw.saOutDenseW, saOut, lw.saOutDenseB, outDim: hs, inDim: hs)
    // residual + LayerNorm
    var afterSA = [[Float]](repeating: [Float](repeating: 0, count: hs), count: L)
    for i in 0..<L {
        var v = saOut[i]
        for d in 0..<hs { v[d] += hidden[i][d] }
        layerNorm(&v, weight: lw.saOutLnW, bias: lw.saOutLnB, eps: eps)
        afterSA[i] = v
    }

    // Cross-attention (query=afterSA, key/value=encoderStates)
    var caOut = qformerMHA(
        queries: afterSA, kvInput: encoderStates,
        qW: lw.caQW, qB: lw.caQB, kW: lw.caKW, kB: lw.caKB, vW: lw.caVW, vB: lw.caVB,
        numHeads: H, hs: hs, kvDim: ehs
    )
    caOut = batchMatVec(lw.caOutDenseW, caOut, lw.caOutDenseB, outDim: hs, inDim: hs)
    var afterCA = [[Float]](repeating: [Float](repeating: 0, count: hs), count: L)
    for i in 0..<L {
        var v = caOut[i]
        for d in 0..<hs { v[d] += afterSA[i][d] }
        layerNorm(&v, weight: lw.caOutLnW, bias: lw.caOutLnB, eps: eps)
        afterCA[i] = v
    }

    // Intermediate + output FFN
    var ffn = batchMatVec(lw.intDenseW, afterCA, lw.intDenseB,
                          outDim: cfg.intermediateSize, inDim: hs)
    for i in ffn.indices { gelu(&ffn[i]) }
    let ffnOut = batchMatVec(lw.outDenseW, ffn, lw.outDenseB, outDim: hs, inDim: cfg.intermediateSize)
    var result = [[Float]](repeating: [Float](repeating: 0, count: hs), count: L)
    for i in 0..<L {
        var v = ffnOut[i]
        for d in 0..<hs { v[d] += afterCA[i][d] }
        layerNorm(&v, weight: lw.outLnW, bias: lw.outLnB, eps: eps)
        result[i] = v
    }
    return result
}

/// Run the encoder projector (QFormer) to produce audio token embeddings
/// in the LM's hidden space.
/// encoderOut: [[Float]] shape [T, hiddenDim]
/// Returns [[Float]] shape [numAudioTokens, lmHiddenSize]
func runProjector(_ encoderOut: [[Float]], weights: GraniteSpeechProjectorWeights) -> [[Float]] {
    let cfg = weights.cfg
    let T   = encoderOut.count
    let ws  = cfg.windowSize
    let nQ  = cfg.numQueriesPerWindow
    let hs  = cfg.projectorConfig.hiddenSize
    let eps = cfg.projectorConfig.layerNormEps

    // Number of blocks (pad encoder output to multiple of windowSize)
    let nBlocks = (T + ws - 1) / ws
    let padded   = encoderOut + [[Float]](
        repeating: [Float](repeating: 0, count: cfg.projectorConfig.encoderHiddenSize),
        count: nBlocks * ws - T
    )

    // query: [nQ * hs] — expand to [nQ, hs] per block
    let baseQuery: [[Float]] = (0..<nQ).map { qi in
        Array(weights.query[(qi * hs)..<((qi + 1) * hs)])
    }

    var allQueryOutputs = [[Float]]()
    allQueryOutputs.reserveCapacity(nBlocks * nQ)

    for b in 0..<nBlocks {
        let window = Array(padded[(b * ws)..<((b + 1) * ws)])

        // Layer norm on initial query
        var qNormed = baseQuery
        for i in qNormed.indices { layerNorm(&qNormed[i], weight: weights.qfLnW, bias: weights.qfLnB, eps: eps) }

        // QFormer layers
        var h = qNormed
        for lw in weights.layers {
            h = qformerLayer(hidden: h, encoderStates: window, lw: lw, cfg: cfg.projectorConfig)
        }
        allQueryOutputs.append(contentsOf: h)
    }

    // Project to LM hidden size
    let lmHidden = cfg.textConfig.hiddenSize
    return batchMatVec(weights.linearW, allQueryOutputs, weights.linearB,
                       outDim: lmHidden, inDim: hs)
}

// MARK: - Granite LM backbone (GPU decode)

/// One Granite decoder layer (Llama-like + Granite multipliers).
public final class GraniteSpeechLMLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm

    let hidden, nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let ropeScaling: Ops.RoPEScaling
    // Granite-specific multipliers applied to each sub-residual.
    let attentionMultiplier: Float
    let residualMultiplier: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        ropeTheta: Float, ropeScaling: Ops.RoPEScaling,
        attentionMultiplier: Float, residualMultiplier: Float
    ) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.ropeTheta = ropeTheta; self.ropeScaling = ropeScaling
        self.attentionMultiplier = attentionMultiplier
        self.residualMultiplier = residualMultiplier
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

    /// Single-token decode step. Returns updated residual [hidden].
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRotated = Ops.rope(q.reshaped(to: [nHeads, headDim]), position: position,
                                headDim: headDim, thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)
        let kRotated = Ops.rope(k.reshaped(to: [nKVHeads, headDim]), position: position,
                                headDim: headDim, thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)

        cache.appendOnGPU(kFlat: kRotated, vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)

        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: attentionMultiplier, on: cmd
        )
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        // Residual: h + oOut * residualMultiplier
        let scaledO = scalarMul(oOut, scalar: residualMultiplier, device: device, on: cmd)
        let postAttn = Ops.add(h, scaledO, on: cmd)

        // MLP — SwiGLU
        let mlpNorm = postAttnNorm(postAttn, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up   = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        let scaledMlp = scalarMul(mlpOut, scalar: residualMultiplier, device: device, on: cmd)
        return Ops.add(postAttn, scaledMlp, on: cmd)
    }
}

/// Scale a tensor by a scalar (element-wise multiply with a broadcast value).
/// Used for Granite's residual and embedding multipliers.
private func scalarMul(_ x: Tensor, scalar: Float, device: Device, on cmd: MTLCommandBuffer) -> Tensor {
    // Create a scalar broadcast tensor filled with `scalar`, then use Ops.mul.
    // A clean route: fill a same-shaped tensor with the constant and multiply.
    let out = Tensor.empty(shape: x.shape, dtype: x.dtype)
    let n = x.elementCount
    // Write scalar into a [n]-element tensor filled with the constant.
    let scalarBuf = device.makeBuffer(length: n * x.dtype.byteSize)
    let scalarTensor = Tensor(buffer: scalarBuf, offset: 0, shape: x.shape, dtype: x.dtype)
    switch x.dtype {
    case .f32:
        let ptr = scalarBuf.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { ptr[i] = scalar }
    case .f16:
        // Convert f32 scalar to f16
        let bits = float32ToFloat16(scalar)
        let ptr = scalarBuf.contents().bindMemory(to: UInt16.self, capacity: n)
        for i in 0..<n { ptr[i] = bits }
    case .bf16:
        let bits = float32ToBFloat16(scalar)
        let ptr = scalarBuf.contents().bindMemory(to: UInt16.self, capacity: n)
        for i in 0..<n { ptr[i] = bits }
    default:
        fatalError("scalarMul: unsupported dtype \(x.dtype)")
    }
    return Ops.mul(x, scalarTensor, on: cmd, into: out)
}

private func float32ToFloat16(_ v: Float) -> UInt16 {
    // Simple f32 → f16 (may clamp, no denormal handling for constants)
    let bits = v.bitPattern
    let sign = UInt16((bits >> 31) & 1) << 15
    let exp  = Int((bits >> 23) & 0xFF) - 127 + 15
    let mant = bits & 0x7FFFFF
    if exp <= 0 { return sign }
    if exp >= 31 { return sign | 0x7C00 }
    return sign | (UInt16(exp) << 10) | UInt16(mant >> 13)
}

private func float32ToBFloat16(_ v: Float) -> UInt16 {
    UInt16(v.bitPattern >> 16)
}

// MARK: - Output

/// Statistics and decoded text from one transcription call.
public struct TranscriptionResult: Sendable {
    /// Decoded transcript (trimmed).
    public let text: String
    /// Tokens generated by the LM backbone (not counting the audio prompt).
    public let generatedTokens: Int
    /// Wall-clock seconds for the full transcription (encode + project + decode).
    public let totalTimeS: Double

    public init(text: String, generatedTokens: Int, totalTimeS: Double) {
        self.text = text
        self.generatedTokens = generatedTokens
        self.totalTimeS = totalTimeS
    }
}

// MARK: - GraniteSpeechModel

/// The loaded GraniteSpeech model: encoder + projector (CPU) + LM (GPU).
/// Note: does NOT conform to the `AudioModel` protocol declared in
/// `AudioGenerationModel.swift` — that one is for TTS (`synthesize(text:...)`).
/// GraniteSpeech follows the standalone STT pattern: `load(...)` + `transcribe(...)`.
public final class GraniteSpeechModel: Module {
    public let config: GraniteSpeechConfig

    // CPU-side encoder & projector weights
    private let encoderWeights: GraniteSpeechEncoderWeights
    private let projectorWeights: GraniteSpeechProjectorWeights

    // GPU-side LM backbone
    public let embedTokens: AnyEmbedding
    public let lmLayers: [GraniteSpeechLMLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    // Tokenizer for prompt building and decoding
    public let tokenizer: any Tokenizer

    private let device: Device
    private let embeddingMultiplier: Float
    private let logitsScaling: Float
    private let audioTokenId: Int
    private let eosTokenId: Int
    private let bosTokenId: Int

    init(config: GraniteSpeechConfig,
         encoderWeights: GraniteSpeechEncoderWeights,
         projectorWeights: GraniteSpeechProjectorWeights,
         embedTokens: AnyEmbedding,
         lmLayers: [GraniteSpeechLMLayer],
         finalNorm: RMSNorm,
         lmHead: AnyLinear,
         tokenizer: any Tokenizer,
         device: Device) {
        self.config = config
        self.encoderWeights = encoderWeights
        self.projectorWeights = projectorWeights
        self.embedTokens = embedTokens
        self.lmLayers = lmLayers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.tokenizer = tokenizer
        self.device = device
        self.embeddingMultiplier = config.textConfig.embeddingMultiplier
        self.logitsScaling = config.textConfig.logitsScaling
        self.audioTokenId = config.audioTokenIndex
        self.eosTokenId = tokenizer.eosTokenId ?? 100257
        self.bosTokenId = config.textConfig.vocabSize - 1  // Granite uses last vocab slot as BOS
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        // Expose GPU-resident LM parameters only (encoder/projector live on CPU)
        for (k, v) in embedTokens.parameters() {
            out.append(("language_model.model.embed_tokens.\(k)", v))
        }
        for (i, layer) in lmLayers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("language_model.model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("language_model.model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("language_model.lm_head.\(k)", v)) }
        return out
    }

    // MARK: - AudioModel conformance

    /// Transcribe a 16 kHz mono waveform.
    public func transcribe(
        _ waveform: [Float],
        maxNewTokens: Int = 512,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        let start = Date()

        // 1. Extract mel features and encode
        let melFrames = extractMelFeatures(waveform)
        let encoderOut = runEncoder(melFrames, weights: encoderWeights)

        // 2. Project to LM embedding space
        let audioEmbeddings = runProjector(encoderOut, weights: projectorWeights)
        let numAudioTokens = audioEmbeddings.count

        // 3. Build prompt: <|audio|>×N + transcription instruction, apply chat template
        let userPrompt = "can you transcribe the speech into a written format?"
        let audioPlaceholder = String(repeating: "<|audio|>", count: numAudioTokens)
        let content = "\(audioPlaceholder)\(userPrompt)"
        let messages: [Message] = [["role": "user", "content": content]]
        let promptIds: [Int]
        if let ids = try? tokenizer.applyChatTemplate(messages: messages) {
            promptIds = ids
        } else {
            promptIds = tokenizer.encode(text: "USER: \(content)\nASSISTANT:")
        }

        // 4. Build combined embeddings tensor (audio tokens replaced by projected embeddings)
        // First pass: embed all tokens (audio positions will be replaced)
        let promptLen = promptIds.count
        let lmHidden = config.textConfig.hiddenSize

        // Assemble the full embedding sequence on CPU then copy to GPU
        var allEmbeddings = [[Float]](repeating: [Float](repeating: 0, count: lmHidden), count: promptLen)
        var audioIdx = 0
        for (pos, tokenId) in promptIds.enumerated() {
            if tokenId == audioTokenId, audioIdx < numAudioTokens {
                allEmbeddings[pos] = audioEmbeddings[audioIdx]
                audioIdx += 1
            } else {
                // GPU embedding lookup (synchronous for single token)
                let emb = gpuEmbedLookup(tokenId: tokenId)
                allEmbeddings[pos] = emb
            }
        }
        // Apply embedding multiplier
        let em = embeddingMultiplier
        for i in 0..<promptLen {
            for d in 0..<lmHidden { allEmbeddings[i][d] *= em }
        }

        // 5. Run prefill (process all prompt tokens sequentially, feeding embeddings)
        let dtype = embedTokens.weight.dtype
        let caches = makeLMCaches()
        var lastLogits = Tensor.empty(shape: [config.textConfig.vocabSize], dtype: dtype)

        for pos in 0..<promptLen {
            let embVec = allEmbeddings[pos]
            let embBuf = device.makeBuffer(length: lmHidden * dtype.byteSize)
            let embTensor = Tensor(buffer: embBuf, offset: 0, shape: [lmHidden], dtype: dtype)
            copyF32ToTensor(embVec, embTensor)

            let logits = lmForwardEmbed(embTensor, position: pos, caches: caches)
            if pos == promptLen - 1 { lastLogits = logits }
        }

        // 6. Decode
        var generatedTokens: [Int] = []
        var lastTok: Int = -1

        if temperature == 0 {
            lastTok = Sampling.argmax(lastLogits)
        } else {
            var rng = SystemRandomNumberGenerator()
            lastTok = Sampling.sample(lastLogits,
                                      parameters: GenerationParameters(temperature: temperature),
                                      rng: &rng)
        }

        for _ in 0..<maxNewTokens {
            if lastTok == eosTokenId { break }
            generatedTokens.append(lastTok)

            let nextLogits = lmForwardToken(tokenId: lastTok, position: promptLen + generatedTokens.count - 1, caches: caches)
            if temperature == 0 {
                lastTok = Sampling.argmax(nextLogits)
            } else {
                var rng = SystemRandomNumberGenerator()
                lastTok = Sampling.sample(nextLogits,
                                          parameters: GenerationParameters(temperature: temperature),
                                          rng: &rng)
            }
        }

        let text = tokenizer.decode(tokens: generatedTokens)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let elapsed = Date().timeIntervalSince(start)
        return TranscriptionResult(text: text, generatedTokens: generatedTokens.count, totalTimeS: elapsed)
    }

    // MARK: - Internal helpers

    /// Synchronous single-token GPU embedding lookup (CPU↔GPU roundtrip).
    private func gpuEmbedLookup(tokenId: Int) -> [Float] {
        let cmd = device.makeCommandBuffer()
        let tidBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tidBuf.contents(), &tid, 4)
        let tidTensor = Tensor(buffer: tidBuf, offset: 0, shape: [1], dtype: .u32)
        let emb = embedTokens(tidTensor, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return emb.toArray(as: Float.self)
    }

    /// Copy CPU [Float] into a Tensor (handles f32/f16/bf16 output dtype).
    private func copyF32ToTensor(_ src: [Float], _ dst: Tensor) {
        let dstPtr = dst.buffer.contents().advanced(by: dst.offset)
        switch dst.dtype {
        case .f32:
            src.withUnsafeBytes { memcpy(dstPtr, $0.baseAddress!, src.count * 4) }
        case .f16:
            let bits = src.map { float32ToFloat16($0) }
            bits.withUnsafeBytes { memcpy(dstPtr, $0.baseAddress!, src.count * 2) }
        case .bf16:
            let bits = src.map { float32ToBFloat16($0) }
            bits.withUnsafeBytes { memcpy(dstPtr, $0.baseAddress!, src.count * 2) }
        default:
            fatalError("copyF32ToTensor: unsupported dtype \(dst.dtype)")
        }
    }

    /// LM forward from a pre-computed embedding vector. Returns logits [vocab].
    private func lmForwardEmbed(_ emb: Tensor, position: Int,
                                 caches: [any LayerCacheProtocol]) -> Tensor {
        let cmd = device.makeCommandBuffer()
        var h = emb
        for (i, layer) in lmLayers.enumerated() {
            h = layer.forward(h, position: position, cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }
        let normed = finalNorm(h, on: cmd)
        var logits = lmHead(normed, on: cmd)
        // Divide by logits_scaling
        let invScale = 1.0 / logitsScaling
        logits = scalarMul(logits, scalar: invScale, device: device, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return logits
    }

    /// LM forward for a single token id (embedding lookup + layers). Returns logits [vocab].
    private func lmForwardToken(tokenId: Int, position: Int,
                                 caches: [any LayerCacheProtocol]) -> Tensor {
        let cmd = device.makeCommandBuffer()

        // Embedding lookup
        let tidBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tidBuf.contents(), &tid, 4)
        let tidTensor = Tensor(buffer: tidBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tidTensor, on: cmd).reshaped(to: [config.textConfig.hiddenSize])

        // Apply embedding multiplier
        h = scalarMul(h, scalar: embeddingMultiplier, device: device, on: cmd)

        for (i, layer) in lmLayers.enumerated() {
            h = layer.forward(h, position: position, cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }
        let normed = finalNorm(h, on: cmd)
        var logits = lmHead(normed, on: cmd)
        logits = scalarMul(logits, scalar: 1.0 / logitsScaling, device: device, on: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
        return logits
    }

    /// Create fresh KV caches for one inference session.
    private func makeLMCaches() -> [any LayerCacheProtocol] {
        let maxSeq = config.textConfig.maxPositionEmbeddings
        let dtype = embedTokens.weight.dtype
        let nKVHeads = config.textConfig.numKeyValueHeads
        let headDim = config.textConfig.hiddenSize / config.textConfig.numAttentionHeads
        return (0..<config.textConfig.numHiddenLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                    dtype: dtype, device: device)
        }
    }
}

// MARK: - Family entry point

public enum GraniteSpeech {
    public static let modelTypes: Set<String>   = ["granite_speech"]
    public static let architectures: Set<String> = ["GraniteSpeechForConditionalGeneration"]

    public static let defaultTranscriptionParameters = TranscriptionParameters(
        maxNewTokens: 512,
        temperature: 0.0
    )

    /// Whether `config` describes a GraniteSpeech checkpoint.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let a = config.architecture, architectures.contains(a) { return true }
        return false
    }

    /// Load from a resolved local model directory.
    public static func load(
        directory: URL,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> GraniteSpeechModel {
        let rawConfig = try ModelConfig.load(from: directory)
        let cfg = try GraniteSpeechConfig.load(from: rawConfig)

        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let quant  = cfg.quantization

        // Load tokenizer
        let tokenizer = try await TokenizerLoader().load(from: directory)

        // CPU encoder weights
        let encW = try GraniteSpeechEncoderWeights(cfg.encoderConfig, bundle: bundle)

        // CPU projector weights
        let projW = try GraniteSpeechProjectorWeights(cfg, bundle: bundle)

        // GPU LM backbone
        let textCfg = cfg.textConfig
        let nLayers = textCfg.numHiddenLayers
        let hidden  = textCfg.hiddenSize
        let nHeads  = textCfg.numAttentionHeads
        let nKV     = textCfg.numKeyValueHeads
        let headDim = hidden / nHeads
        let theta   = textCfg.ropeTheta
        let attMul  = textCfg.attentionMultiplier
        let resMul  = textCfg.residualMultiplier
        let prefix  = "language_model"

        let embedTokens = try loadEmbedding(
            base: "\(prefix).model.embed_tokens", in: bundle, hidden: hidden, quantization: quant
        )

        var lmLayers: [GraniteSpeechLMLayer] = []
        lmLayers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let lp = "\(prefix).model.layers.\(i)"
            let qProj = try loadLinear(base: "\(lp).self_attn.q_proj", in: bundle, quantization: quant)
            let kProj = try loadLinear(base: "\(lp).self_attn.k_proj", in: bundle, quantization: quant)
            let vProj = try loadLinear(base: "\(lp).self_attn.v_proj", in: bundle, quantization: quant)
            let oProj = try loadLinear(base: "\(lp).self_attn.o_proj", in: bundle, quantization: quant)
            let gProj = try loadLinear(base: "\(lp).mlp.gate_proj",    in: bundle, quantization: quant)
            let uProj = try loadLinear(base: "\(lp).mlp.up_proj",      in: bundle, quantization: quant)
            let dProj = try loadLinear(base: "\(lp).mlp.down_proj",    in: bundle, quantization: quant)
            let inNorm   = RMSNorm(weight: try bundle.tensor(named: "\(lp).input_layernorm.weight"),
                                   eps: textCfg.rmsNormEps)
            let postNorm = RMSNorm(weight: try bundle.tensor(named: "\(lp).post_attention_layernorm.weight"),
                                   eps: textCfg.rmsNormEps)
            lmLayers.append(GraniteSpeechLMLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gProj, upProj: uProj, downProj: dProj,
                inputNorm: inNorm, postAttnNorm: postNorm,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKV, headDim: headDim,
                ropeTheta: theta, ropeScaling: .none,
                attentionMultiplier: attMul, residualMultiplier: resMul
            ))
        }

        let finalNorm = RMSNorm(weight: try bundle.tensor(named: "\(prefix).model.norm.weight"),
                                eps: textCfg.rmsNormEps)

        // LM head — GraniteSpeech always has an explicit lm_head (tieWordEmbeddings=false)
        let lmHead: AnyLinear
        if !textCfg.tieWordEmbeddings, bundle.has("\(prefix).lm_head.weight") {
            lmHead = try loadLinear(base: "\(prefix).lm_head", in: bundle, quantization: quant)
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        return GraniteSpeechModel(
            config: cfg,
            encoderWeights: encW,
            projectorWeights: projW,
            embedTokens: embedTokens,
            lmLayers: lmLayers,
            finalNorm: finalNorm,
            lmHead: lmHead,
            tokenizer: tokenizer,
            device: device
        )
    }
}

// MARK: - Transcription parameters

/// Generation configuration for transcription (mirrors GenerationParameters
/// but for the STT surface).
public struct TranscriptionParameters: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float

    public init(maxNewTokens: Int = 512, temperature: Float = 0.0) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
    }
}
