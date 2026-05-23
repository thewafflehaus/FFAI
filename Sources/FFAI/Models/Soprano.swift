// Soprano — a Qwen3-based LLM text-to-speech family from the mlx-community.
//
// Soprano turns text into speech via an autoregressive language model that
// emits hidden states, which a Vocos-like ConvNeXt + ISTFT decoder converts
// into a waveform:
//
//   text ──text-cleaning──▶ [STOP][TEXT]<text>[START] prompt
//        ──BPE tokenizer──▶ token ids
//        ──Soprano LLM (Qwen3-style: GQA, q_norm, k_norm, SwiGLU)──▶ hidden states
//        ──Vocos decoder (ConvNeXt backbone + ISTFT head)──▶ 32 kHz waveform
//
// Two checkpoint variants are supported:
//
//   Soprano-80M   — model_type "soprano", architecture SopranoForCausalLM.
//                   Weights: language_model.* (LLM) + decoder.* (Vocos).
//                   Config includes all decoder hyper-parameters inline.
//
//   Soprano-1.1   — model_type "soprano", architecture Qwen3ForCausalLM.
//                   Weights: model.* (LLM) + lm_head.* only — no decoder.
//                   synthesize throws SopranoError.decoderNotAvailable.
//
// Soprano weight key layout (Soprano-80M → normalised internal keys):
//   language_model.embed_tokens.*   → model.embed_tokens.*
//   language_model.layers.i.*       → model.layers.i.*
//   language_model.norm.*           → model.norm.*
//   language_model.lm_head.*        → lm_head.*
//   decoder.decoder.embed.*         → loaded directly by SopranoVocosDecoder
//   decoder.decoder.convnext.i.*    → loaded directly by SopranoVocosDecoder
//   decoder.decoder.norm.*          → loaded directly by SopranoVocosDecoder
//   decoder.decoder.final_layer_norm.* → loaded directly by SopranoVocosDecoder
//   decoder.head.out.*              → loaded directly by SopranoVocosDecoder
//
// Reference implementation:
//   ~/Development/personal/ai/mlx-audio-swift/Sources/MLXAudioTTS/Models/Soprano/
// Cached checkpoints:
//   ~/.cache/huggingface/hub/models--mlx-community--Soprano-80M-bf16
//   ~/.cache/huggingface/hub/models--mlx-community--Soprano-1.1-80M-bf16

import Foundation
import Metal
import Tokenizers

// ─── Errors ──────────────────────────────────────────────────────────────────

public enum SopranoError: Error, CustomStringConvertible {
    /// A required field is absent from config.json.
    case missingConfig(String)
    /// The Vocos decoder weights are not present in this checkpoint.
    /// Soprano-1.1 checkpoints ship only LLM weights; the decoder will
    /// be added in a future checkpoint revision.
    case decoderNotAvailable
    /// Tokenizer was not loaded before calling synthesize.
    case tokenizerNotLoaded
    /// The generation loop produced no usable output.
    case generationFailed(String)

    public var description: String {
        switch self {
        case .missingConfig(let field):
            return "Soprano: required config field missing: \(field)"
        case .decoderNotAvailable:
            return "Soprano: decoder weights are absent in this checkpoint. "
                + "Soprano-1.1 ships LLM-only weights. Use Soprano-80M for "
                + "end-to-end synthesis."
        case .tokenizerNotLoaded:
            return "Soprano: tokenizer missing — call load(directory:) first"
        case .generationFailed(let m):
            return "Soprano: generation failed — \(m)"
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Soprano hyper-parameters decoded from config.json.
///
/// Soprano-80M includes all decoder fields; Soprano-1.1 omits them
/// (decoder-related optionals are nil) and synthesize reports
/// `SopranoError.decoderNotAvailable`.
public struct SopranoConfig: Sendable {
    // ── Transformer (Qwen3-compatible) ──
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    // ── Token IDs ──
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    /// [STOP] token is always ID 3 in the Soprano BPE vocabulary.
    public static let stopTokenId = 3

    // ── Audio / decoder (nil → Soprano-1.1 — no decoder in checkpoint) ──
    public let sampleRate: Int?
    public let decoderNumLayers: Int?
    public let decoderDim: Int?
    public let decoderIntermediateDim: Int?
    public let hopLength: Int?
    public let nFft: Int?
    public let upscale: Int?
    public let inputKernel: Int?
    public let dwKernel: Int?
    public let tokenSize: Int?

    /// Whether this config describes a checkpoint with a Vocos decoder.
    /// True for Soprano-80M; false for Soprano-1.1.
    public var hasDecoderConfig: Bool {
        decoderDim != nil && hopLength != nil && nFft != nil
    }

    public static func from(_ config: ModelConfig) -> SopranoConfig? {
        func i(_ k: String) -> Int? { config.int(k) }
        func f(_ k: String) -> Float? {
            guard let v = config.float(k) else { return nil }
            return Float(v)
        }
        guard let hidden       = i("hidden_size"),
              let nLayers      = i("num_hidden_layers"),
              let nHeads       = i("num_attention_heads"),
              let headDim      = i("head_dim"),
              let vocab        = i("vocab_size"),
              let intermediate = i("intermediate_size")
        else { return nil }

        return SopranoConfig(
            hiddenSize: hidden,
            numHiddenLayers: nLayers,
            numAttentionHeads: nHeads,
            numKeyValueHeads: i("num_key_value_heads") ?? nHeads,
            headDim: headDim,
            intermediateSize: intermediate,
            vocabSize: vocab,
            maxPositionEmbeddings: i("max_position_embeddings") ?? 512,
            rmsNormEps: f("rms_norm_eps") ?? 1e-6,
            ropeTheta: f("rope_theta") ?? 10_000,
            tieWordEmbeddings: config.bool("tie_word_embeddings") ?? false,
            bosTokenId: i("bos_token_id") ?? 1,
            eosTokenId: i("eos_token_id") ?? 2,
            padTokenId: i("pad_token_id") ?? 0,
            sampleRate: i("sample_rate"),
            decoderNumLayers: i("decoder_num_layers"),
            decoderDim: i("decoder_dim"),
            decoderIntermediateDim: i("decoder_intermediate_dim"),
            hopLength: i("hop_length"),
            nFft: i("n_fft"),
            upscale: i("upscale"),
            inputKernel: i("input_kernel"),
            dwKernel: i("dw_kernel"),
            tokenSize: i("token_size")
        )
    }
}

// ─── Transformer layer ────────────────────────────────────────────────────────

/// One Soprano transformer block — identical to Qwen3Layer:
/// GQA attention with per-head q_norm/k_norm, standard RoPE, and SwiGLU MLP.
final class SopranoLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm, kNorm: RMSNorm,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int, intermediate: Int,
        ropeTheta: Float
    ) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / sqrtf(Float(headDim))
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in qNorm.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        for (k, v) in kNorm.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        return out
    }

    /// Single-token forward. Returns the updated hidden state `[hidden]`.
    func forward(
        _ h: Tensor, position: Int,
        cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let xNorm = inputNorm(h, on: cmd)

        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head q_norm / k_norm — the Qwen3/Soprano structural marker.
        let qNormed = Ops.rmsNormRows(q, weight: qNorm.weight, eps: qNorm.eps,
                                      nRows: nHeads, rowSize: headDim, on: cmd)
        let kNormed = Ops.rmsNormRows(k, weight: kNorm.weight, eps: kNorm.eps,
                                      nRows: nKVHeads, rowSize: headDim, on: cmd)

        // RoPE — standard, no scaling.
        let qRot = Ops.rope(qNormed.reshaped(to: [nHeads, headDim]),
                            position: position, headDim: headDim,
                            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRot = Ops.rope(kNormed.reshaped(to: [nKVHeads, headDim]),
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

        // SwiGLU MLP
        let mlpNorm = postAttnNorm(postAttn, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }
}

// ─── Soprano LLM backbone ─────────────────────────────────────────────────────

/// The Soprano transformer — a Qwen3-compatible dense LLM backbone.
/// Produces (logits, hidden_state) per token for the Vocos decoder.
final class SopranoLLM: Module {
    let embedTokens: AnyEmbedding
    let layers: [SopranoLayer]
    let finalNorm: RMSNorm
    let lmHead: AnyLinear
    let hidden, nLayers, nHeads, nKVHeads, headDim, maxSeq: Int
    let ropeTheta: Float
    let dtype: DType

    init(
        embedTokens: AnyEmbedding, layers: [SopranoLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int,
        headDim: Int, maxSeq: Int, ropeTheta: Float, dtype: DType
    ) {
        self.embedTokens = embedTokens; self.layers = layers
        self.finalNorm = finalNorm; self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers; self.nHeads = nHeads
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.maxSeq = maxSeq
        self.ropeTheta = ropeTheta; self.dtype = dtype
    }

    func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    func makeLayerCaches(device: Device) -> [KVCache] {
        (0..<nLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: maxSeq,
                    dtype: dtype, eviction: .unbounded, device: device)
        }
    }

    /// Single-token forward. Returns logits `[vocab]` + pre-head hidden `[hidden]`.
    func forward(
        tokenId: Int, position: Int,
        caches: [KVCache],
        cmd: MTLCommandBuffer, device: Device
    ) -> (logits: Tensor, hidden: Tensor) {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0,
                                 shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i], cmd: cmd, device: device)
        }
        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)
        return (logits: logits, hidden: normed)
    }
}

// ─── Soprano Vocos decoder ────────────────────────────────────────────────────

/// Soprano's audio decoder — a Vocos-style ConvNeXt backbone plus ISTFT head.
///
/// Converts a sequence of LLM hidden states into audio:
///   [T, hidden] hidden states
///   ──linear-interpolate (upscale×, align-corners)──▶ [T', hidden]
///   ──ConvNeXt backbone (embed → layernorm → 8× conv-neXt → final layernorm)──▶ [T', decoderDim]
///   ──ISTFT head (linear → split mag/phase → iSTFT overlap-add)──▶ waveform
///
/// CPU-native (same rationale as the existing Vocos port):
/// the decoder runs once per utterance, not in the hot token loop.
/// Overlap-add is parallelised with DispatchQueue.concurrentPerform
/// when numFrames > 64 to avoid serialising on long utterances.
///
/// Weight key layout in the Soprano-80M bundle:
///   decoder.decoder.embed.weight/.bias
///   decoder.decoder.norm.weight/.bias
///   decoder.decoder.convnext.{i}.dwconv.weight/.bias
///   decoder.decoder.convnext.{i}.norm.weight/.bias
///   decoder.decoder.convnext.{i}.pwconv1.weight/.bias
///   decoder.decoder.convnext.{i}.pwconv2.weight/.bias
///   decoder.decoder.convnext.{i}.gamma
///   decoder.decoder.final_layer_norm.weight/.bias
///   decoder.head.out.weight/.bias
public final class SopranoDecoder {
    public let inputChannels: Int     // == LLM hidden size
    public let decoderDim: Int
    public let upscale: Int
    public let nFFT: Int
    public let hopLength: Int

    // ConvNeXt backbone weights
    private let embedW: [Float], embedShape: [Int]
    private let embedB: [Float]?
    private let normW: [Float], normB: [Float]?
    private let blocks: [SopranoConvNeXtBlock]
    private let finalNormW: [Float], finalNormB: [Float]?

    // ISTFT head weights
    private let headOutW: [Float]    // [nFFT+2, decoderDim]
    private let headOutB: [Float]?

    /// Magnitude clamp — iSTFT stability guard (matches MLX reference).
    private static let magClip: Float = 1e2

    fileprivate init(
        inputChannels: Int, decoderDim: Int, upscale: Int, nFFT: Int, hopLength: Int,
        embedW: [Float], embedShape: [Int], embedB: [Float]?,
        normW: [Float], normB: [Float]?,
        blocks: [SopranoConvNeXtBlock],
        finalNormW: [Float], finalNormB: [Float]?,
        headOutW: [Float], headOutB: [Float]?
    ) {
        self.inputChannels = inputChannels; self.decoderDim = decoderDim
        self.upscale = upscale; self.nFFT = nFFT; self.hopLength = hopLength
        self.embedW = embedW; self.embedShape = embedShape; self.embedB = embedB
        self.normW = normW; self.normB = normB
        self.blocks = blocks
        self.finalNormW = finalNormW; self.finalNormB = finalNormB
        self.headOutW = headOutW; self.headOutB = headOutB
    }

    /// Load from a SafeTensorsBundle using Soprano-80M key layout.
    static func load(
        from bundle: SafeTensorsBundle,
        inputChannels: Int, decoderDim: Int, intermediateDim: Int,
        numLayers: Int, inputKernel: Int, dwKernel: Int,
        upscale: Int, nFFT: Int, hopLength: Int
    ) throws -> SopranoDecoder {
        func floats(_ key: String) throws -> [Float] {
            AudioMath.floats(try bundle.tensor(named: key))
        }
        func shape(_ key: String) throws -> [Int] {
            try bundle.tensor(named: key).shape
        }
        func has(_ key: String) -> Bool { bundle.has(key) }

        // Conv weight in MLX NLC layout [Cout, K, Cin] → PyTorch [Cout, Cin, K].
        func convW(_ key: String) throws -> (data: [Float], shape: [Int]) {
            let raw = try floats(key)
            let s = try shape(key)
            let (cOut, k, cIn) = (s[0], s[1], s[2])
            var out = [Float](repeating: 0, count: raw.count)
            for o in 0..<cOut {
                for kk in 0..<k {
                    for ic in 0..<cIn {
                        out[(o * cIn + ic) * k + kk] = raw[(o * k + kk) * cIn + ic]
                    }
                }
            }
            return (out, [cOut, cIn, k])
        }

        let bb = "decoder.decoder"

        let (ew, es) = try convW("\(bb).embed.weight")
        let eb = has("\(bb).embed.bias") ? try floats("\(bb).embed.bias") : nil
        let nw = try floats("\(bb).norm.weight")
        let nb = has("\(bb).norm.bias") ? try floats("\(bb).norm.bias") : nil

        var blockList: [SopranoConvNeXtBlock] = []
        for i in 0..<numLayers {
            let p = "\(bb).convnext.\(i)"
            let (dw, ds) = try convW("\(p).dwconv.weight")
            let dwb = has("\(p).dwconv.bias") ? try floats("\(p).dwconv.bias") : nil
            let cnW = try floats("\(p).norm.weight")
            let cnB = has("\(p).norm.bias") ? try floats("\(p).norm.bias") : nil
            let pw1w = try floats("\(p).pwconv1.weight")
            let pw1b = has("\(p).pwconv1.bias") ? try floats("\(p).pwconv1.bias") : nil
            let pw2w = try floats("\(p).pwconv2.weight")
            let pw2b = has("\(p).pwconv2.bias") ? try floats("\(p).pwconv2.bias") : nil
            let gamma = has("\(p).gamma") ? try floats("\(p).gamma") : nil
            blockList.append(SopranoConvNeXtBlock(
                dwWeight: dw, dwShape: ds, dwBias: dwb,
                normW: cnW, normB: cnB,
                pw1W: pw1w, pw1B: pw1b,
                pw2W: pw2w, pw2B: pw2b,
                gamma: gamma, dim: decoderDim, interDim: intermediateDim))
        }

        let fnw = try floats("\(bb).final_layer_norm.weight")
        let fnb = has("\(bb).final_layer_norm.bias") ? try floats("\(bb).final_layer_norm.bias") : nil

        let hw = try floats("decoder.head.out.weight")
        let hb = has("decoder.head.out.bias") ? try floats("decoder.head.out.bias") : nil

        return SopranoDecoder(
            inputChannels: inputChannels, decoderDim: decoderDim,
            upscale: upscale, nFFT: nFFT, hopLength: hopLength,
            embedW: ew, embedShape: es, embedB: eb,
            normW: nw, normB: nb, blocks: blockList,
            finalNormW: fnw, finalNormB: fnb,
            headOutW: hw, headOutB: hb)
    }

    /// Decode hidden states into a `[L]` waveform Tensor.
    ///
    /// - Parameters:
    ///   - hiddenStates: Flat `[tokenCount × inputChannels]` row-major array.
    ///   - tokenCount: Number of hidden-state rows.
    ///   - device: Metal device for the ISTFT overlap-add kernel.
    public func decode(hiddenStates: [Float], tokenCount: Int,
                       device: Device = .shared) -> Tensor {
        let hidden = inputChannels
        let t = tokenCount

        // Transpose [T, hidden] row-major → NCL [1, hidden, T]
        var ncl = [Float](repeating: 0, count: hidden * t)
        for tok in 0..<t {
            for ch in 0..<hidden {
                ncl[ch * t + tok] = hiddenStates[tok * hidden + ch]
            }
        }
        var curShape = [1, hidden, t]

        // Linear interpolation: align-corners, upscale× expansion.
        let targetSize = upscale * (t - 1) + 1
        (ncl, curShape) = sopranoInterpolate1d(ncl, shape: curShape, size: targetSize)

        // Input conv (backbone embed), "same" padding (K/2).
        let k0 = embedShape[2]
        let (embedded, embeddedShape) = AudioMath.conv1d(
            x: ncl, xShape: curShape, weight: embedW, wShape: embedShape,
            bias: embedB, stride: 1, padding: k0 / 2, dilation: 1, groups: 1)
        curShape = embeddedShape

        // Initial LayerNorm over channels (NCL → rows → NCL).
        var rows = nclToRows(embedded, c: decoderDim, t: curShape[2])
        rows = AudioMath.layerNorm(rows, rows: curShape[2], dim: decoderDim,
                                   weight: normW, bias: normB)
        var h = rowsToNcl(rows, c: decoderDim, t: curShape[2])

        // ConvNeXt blocks — parallelise CPU attention per block.
        for block in blocks { (h, curShape) = block(h, shape: curShape) }

        // Final LayerNorm
        rows = nclToRows(h, c: decoderDim, t: curShape[2])
        rows = AudioMath.layerNorm(rows, rows: curShape[2], dim: decoderDim,
                                   weight: finalNormW, bias: finalNormB)
        h = rowsToNcl(rows, c: decoderDim, t: curShape[2])

        // ISTFT head: project to STFT coefficients, split mag+phase, iSTFT.
        return sopranoISTFT(features: h, shape: curShape, device: device)
    }

    /// Project backbone output to complex STFT and reconstruct audio.
    private func sopranoISTFT(
        features: [Float], shape: [Int], device: Device
    ) -> Tensor {
        let t = shape[2]
        let nFreq = nFFT / 2 + 1

        // Project [T, decoderDim] → [T, nFFT+2] via linear.
        let rows = nclToRows(features, c: decoderDim, t: t)
        let coeffs = AudioMath.linear(rows, rows: t, inDim: decoderDim,
                                      weight: headOutW, outDim: nFFT + 2,
                                      bias: headOutB)

        // Split into magnitude (exp, clipped) and phase → complex STFT [T, nFreq].
        var specRe = [Float](repeating: 0, count: t * nFreq)
        var specIm = [Float](repeating: 0, count: t * nFreq)
        for frame in 0..<t {
            let base = frame * (nFFT + 2)
            for f in 0..<nFreq {
                let mag = min(expf(coeffs[base + f]), Self.magClip)
                let phase = coeffs[base + nFreq + f]
                specRe[frame * nFreq + f] = mag * cosf(phase)
                specIm[frame * nFreq + f] = mag * sinf(phase)
            }
        }

        // GPU iSTFT overlap-add via `Ops.vocoderISTFT`.
        let reT  = Tensor.empty(shape: [t, nFreq], dtype: .f32, device: device)
        reT.copyIn(from: specRe)
        let imT  = Tensor.empty(shape: [t, nFreq], dtype: .f32, device: device)
        imT.copyIn(from: specIm)
        let win  = AudioPreprocessing.hannWindow(nFFT)
        let winT = Tensor.empty(shape: [nFFT], dtype: .f32, device: device)
        winT.copyIn(from: win)

        let cmd = device.makeCommandBuffer()
        let waveform = Ops.vocoderISTFT(
            specRe: reT, specIm: imT, window: winT,
            nFrames: t, nFFT: nFFT, hopLength: hopLength, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        return waveform
    }
}

// ─── ConvNeXt block ───────────────────────────────────────────────────────────

/// One Soprano ConvNeXt block: depthwise conv → LayerNorm → pwconv1 → GELU →
/// pwconv2 → layer-scale → residual add.
///
/// Mirrors VocosConvNeXtBlock from `VocosBackbone.swift` but operates on
/// Soprano-specific weight key prefixes and is self-contained.
private struct SopranoConvNeXtBlock {
    let dwWeight: [Float]    // [dim, 1, K]  (depthwise)
    let dwShape: [Int]
    let dwBias: [Float]?
    let normW: [Float], normB: [Float]?
    let pw1W: [Float], pw1B: [Float]?
    let pw2W: [Float], pw2B: [Float]?
    let gamma: [Float]?
    let dim: Int
    let interDim: Int

    /// Apply the block to an NCL feature map `[1, dim, T]`. Returns the same shape.
    func callAsFunction(_ x: [Float], shape: [Int]) -> (data: [Float], shape: [Int]) {
        let t = shape[2]
        let k = dwShape[2]

        // Depthwise conv — groups == dim.
        let (h, _) = AudioMath.conv1d(
            x: x, xShape: shape, weight: dwWeight, wShape: dwShape,
            bias: dwBias, stride: 1, padding: k / 2, dilation: 1, groups: dim)

        // LayerNorm over channels.
        var rows = nclToRows(h, c: dim, t: t)
        rows = AudioMath.layerNorm(rows, rows: t, dim: dim, weight: normW, bias: normB)

        // Pointwise linear 1 (dim → interDim) + GELU.
        var ff = AudioMath.linear(rows, rows: t, inDim: dim,
                                  weight: pw1W, outDim: interDim, bias: pw1B)
        ff = AudioMath.gelu(ff)

        // Pointwise linear 2 (interDim → dim).
        var out = AudioMath.linear(ff, rows: t, inDim: interDim,
                                   weight: pw2W, outDim: dim, bias: pw2B)

        // Per-channel layer scale.
        if let g = gamma {
            for pos in 0..<t {
                for ch in 0..<dim { out[pos * dim + ch] *= g[ch] }
            }
        }

        // Residual add (out is [T, dim]; x is NCL → transpose back first).
        let outNcl = rowsToNcl(out, c: dim, t: t)
        precondition(outNcl.count == x.count,
                     "SopranoConvNeXtBlock: residual length mismatch")
        var sum = x
        for i in 0..<sum.count { sum[i] += outNcl[i] }
        return (sum, shape)
    }
}

// ─── NCL layout helpers ───────────────────────────────────────────────────────

/// Transpose NCL [1, C, T] tensor to row-major [T, C].
private func nclToRows(_ x: [Float], c: Int, t: Int) -> [Float] {
    var out = [Float](repeating: 0, count: t * c)
    for ch in 0..<c {
        for pos in 0..<t { out[pos * c + ch] = x[ch * t + pos] }
    }
    return out
}

/// Transpose row-major [T, C] back to NCL [1, C, T].
private func rowsToNcl(_ x: [Float], c: Int, t: Int) -> [Float] {
    var out = [Float](repeating: 0, count: c * t)
    for pos in 0..<t {
        for ch in 0..<c { out[ch * t + pos] = x[pos * c + ch] }
    }
    return out
}

/// 1-D linear interpolation in NCL layout [1, C, L] → [1, C, size].
/// align_corners=True: input[0] maps to output[0], input[L-1] to output[size-1].
private func sopranoInterpolate1d(
    _ x: [Float], shape: [Int], size: Int
) -> ([Float], [Int]) {
    let (c, inWidth) = (shape[1], shape[2])
    guard size > 0 && inWidth > 0 else { return (x, shape) }
    if size == inWidth { return (x, shape) }
    if inWidth == 1 {
        var out = [Float](repeating: 0, count: c * size)
        for ch in 0..<c {
            let v = x[ch * inWidth]
            for i in 0..<size { out[ch * size + i] = v }
        }
        return (out, [1, c, size])
    }

    let scale = Float(inWidth - 1) / Float(size - 1)
    var out = [Float](repeating: 0, count: c * size)

    // DispatchQueue.concurrentPerform over channels for parallelism on long
    // utterances (each channel is independent).
    DispatchQueue.concurrentPerform(iterations: c) { ch in
        let base = ch * inWidth
        let outBase = ch * size
        for i in 0..<size {
            let p = Float(i) * scale
            let lo = min(Int(p), inWidth - 1)
            let hi = min(lo + 1, inWidth - 1)
            let frac = p - Float(lo)
            out[outBase + i] = x[base + lo] * (1 - frac) + x[base + hi] * frac
        }
    }
    return (out, [1, c, size])
}

// ─── Text utilities ───────────────────────────────────────────────────────────
// Ported from the MLX reference (Soprano/TextUtils.swift). Converts numbers,
// expands abbreviations, handles special characters, and lowercases.

private let sopranoOnes = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
]
private let sopranoTens = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety"
]
private let sopranoOrdinals: [Int: String] = [
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
    90: "ninetieth"
]

private func sopranoNumToWords(_ n: Int) -> String {
    if n < 0 { return "minus " + sopranoNumToWords(-n) }
    if n == 0 { return "zero" }
    if n < 20 { return sopranoOnes[n] }
    if n < 100 {
        let o = n % 10 == 0 ? "" : " " + sopranoOnes[n % 10]
        return sopranoTens[n / 10] + o
    }
    if n < 1_000 {
        let r = n % 100 == 0 ? "" : " " + sopranoNumToWords(n % 100)
        return sopranoOnes[n / 100] + " hundred" + r
    }
    if n < 1_000_000 {
        let r = n % 1000 == 0 ? "" : " " + sopranoNumToWords(n % 1000)
        return sopranoNumToWords(n / 1000) + " thousand" + r
    }
    if n < 1_000_000_000 {
        let r = n % 1_000_000 == 0 ? "" : " " + sopranoNumToWords(n % 1_000_000)
        return sopranoNumToWords(n / 1_000_000) + " million" + r
    }
    let r = n % 1_000_000_000 == 0 ? "" : " " + sopranoNumToWords(n % 1_000_000_000)
    return sopranoNumToWords(n / 1_000_000_000) + " billion" + r
}

private func sopranoOrdinalToWords(_ n: Int) -> String {
    if let o = sopranoOrdinals[n] { return o }
    if n < 100 {
        let t = n / 10; let o = n % 10
        if o == 0 { return sopranoTens[t] + "ieth" }
        return sopranoTens[t] + " " + (sopranoOrdinals[o] ?? sopranoOnes[o] + "th")
    }
    let base = sopranoNumToWords(n)
    return base.hasSuffix("y") ? String(base.dropLast()) + "ieth" : base + "th"
}

private let sopranoAbbreviations: [(String, String)] = [
    ("\\bmrs\\.", "misuss"), ("\\bms\\.", "miss"), ("\\bmr\\.", "mister"),
    ("\\bdr\\.", "doctor"), ("\\bst\\.", "saint"), ("\\bco\\.", "company"),
    ("\\bjr\\.", "junior"), ("\\bmaj\\.", "major"), ("\\bgen\\.", "general"),
    ("\\bdrs\\.", "doctors"), ("\\brev\\.", "reverend"), ("\\blt\\.", "lieutenant"),
    ("\\bhon\\.", "honorable"), ("\\bsgt\\.", "sergeant"), ("\\bcapt\\.", "captain"),
    ("\\besq\\.", "esquire"), ("\\bltd\\.", "limited"), ("\\bcol\\.", "colonel"),
    ("\\bft\\.", "fort")
]

private let sopranoCasedAbbreviations: [(String, String)] = [
    ("\\bTTS\\b", "text to speech"), ("\\bHz\\b", "hertz"),
    ("\\bkHz\\b", "kilohertz"), ("\\bKBs\\b", "kilobytes"),
    ("\\bKB\\b", "kilobyte"), ("\\bMBs\\b", "megabytes"),
    ("\\bMB\\b", "megabyte"), ("\\bGBs\\b", "gigabytes"),
    ("\\bGB\\b", "gigabyte"), ("\\bTBs\\b", "terabytes"),
    ("\\bTB\\b", "terabyte"), ("\\bAPIs\\b", "a p i's"),
    ("\\bAPI\\b", "a p i"), ("\\bCLIs\\b", "c l i's"),
    ("\\bCLI\\b", "c l i"), ("\\bCPUs\\b", "c p u's"),
    ("\\bCPU\\b", "c p u"), ("\\bGPUs\\b", "g p u's"),
    ("\\bGPU\\b", "g p u"), ("\\bAve\\b", "avenue"),
    ("\\betc\\b", "etcetera")
]

private let sopranoSpecialChars: [(String, String)] = [
    ("@", " at "), ("&", " and "), ("%", " percent "), (":", "."), (";", ","),
    ("\\+", " plus "), ("\\\\", " backslash "), ("~", " about "),
    ("<", " less than "), (">", " greater than "), ("=", " equals "),
    ("/", " slash "), ("_", " ")
]

/// Clean and normalise text for Soprano TTS — converts numbers, expands
/// abbreviations and special characters, lowercases, and collapses whitespace.
func cleanTextForSoprano(_ text: String) -> String {
    var result = text
    // Unicode → ASCII
    result = result.precomposedStringWithCanonicalMapping
        .unicodeScalars.filter { $0.isASCII }.map { String($0) }.joined()
    // Numbers
    result = sopranoNormalizeNumbers(result)
    // Abbreviations (case-insensitive)
    for (pat, rep) in sopranoAbbreviations {
        if let re = try? NSRegularExpression(pattern: pat, options: .caseInsensitive) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: rep)
        }
    }
    // Case-sensitive abbreviations
    for (pat, rep) in sopranoCasedAbbreviations {
        if let re = try? NSRegularExpression(pattern: pat) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: rep)
        }
    }
    // Special characters
    for (pat, rep) in sopranoSpecialChars {
        if let re = try? NSRegularExpression(pattern: pat) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: rep)
        }
    }
    result = result.lowercased()
    // Remove non-allowed characters
    if let re = try? NSRegularExpression(
        pattern: "[^A-Za-z !\\$%&'\\*\\+,\\-./0123456789<>\\?_]") {
        let r = NSRange(result.startIndex..., in: result)
        result = re.stringByReplacingMatches(in: result, range: r, withTemplate: "")
    }
    if let re = try? NSRegularExpression(pattern: "[<>/_+]") {
        let r = NSRange(result.startIndex..., in: result)
        result = re.stringByReplacingMatches(in: result, range: r, withTemplate: "")
    }
    // Collapse whitespace
    if let re = try? NSRegularExpression(pattern: "\\s+") {
        let r = NSRange(result.startIndex..., in: result)
        result = re.stringByReplacingMatches(in: result, range: r, withTemplate: " ")
    }
    if let re = try? NSRegularExpression(pattern: " ([.?!,])") {
        let r = NSRange(result.startIndex..., in: result)
        result = re.stringByReplacingMatches(in: result, range: r, withTemplate: "$1")
    }
    result = result.trimmingCharacters(in: .whitespaces)
    // Dedup punctuation
    for (pat, rep): (String, String) in [
        ("\\.{3,}", "..."), (",+", ","),
        ("[.,]*\\.[.,]*", "."), ("[.,!]*![.,!]*", "!"),
        ("[.,!?]*\\?[.,!?]*", "?")
    ] {
        if let re = try? NSRegularExpression(pattern: pat) {
            let r = NSRange(result.startIndex..., in: result)
            result = re.stringByReplacingMatches(in: result, range: r, withTemplate: rep)
        }
    }
    return result
}

private func sopranoNormalizeNumbers(_ text: String) -> String {
    var result = text

    // Applies regex substitution in reverse-match order to preserve offsets.
    func replace(_ pattern: String, opts: NSRegularExpression.Options = [],
                 transform: (String) -> String) {
        guard let re = try? NSRegularExpression(pattern: pattern, options: opts) else { return }
        let ns = result as NSString
        let rng = NSRange(location: 0, length: ns.length)
        for m in re.matches(in: result, range: rng).reversed() {
            guard let r = Range(m.range, in: result) else { continue }
            result.replaceSubrange(r, with: transform(String(result[r])))
        }
    }

    replace("#\\d") { m in "number \(String(m.dropFirst()))" }
    replace("\\d[KMBTkmbt]") { m in
        let map = ["K":"thousand","M":"million","B":"billion","T":"trillion",
                   "k":"thousand","m":"million","b":"billion","t":"trillion"]
        return "\(String(m.dropLast())) \(map[String(m.last!)] ?? "")"
    }
    replace("(\\d[\\d,]+\\d)") { m in m.replacingOccurrences(of: ",", with: "") }
    replace("\\$([\\d.,]*\\d+)") { m in
        let cleaned = String(m.dropFirst()).replacingOccurrences(of: ",", with: "")
        let parts = cleaned.split(separator: ".", maxSplits: 1)
        let d = parts.first.flatMap { Int($0) } ?? 0
        let c = parts.count > 1 ? (Int(parts[1]) ?? 0) : 0
        if d > 0 && c > 0 {
            return "\(sopranoNumToWords(d)) dollar\(d==1 ? "" : "s"), "
                 + "\(sopranoNumToWords(c)) cent\(c==1 ? "" : "s")"
        }
        if d > 0 { return "\(sopranoNumToWords(d)) dollar\(d==1 ? "" : "s")" }
        if c > 0 { return "\(sopranoNumToWords(c)) cent\(c==1 ? "" : "s")" }
        return "zero dollars"
    }
    replace("\\d+(st|nd|rd|th)") { m in
        let n = m.replacingOccurrences(of: "st|nd|rd|th", with: "",
                                        options: .regularExpression)
        return (Int(n).map { sopranoOrdinalToWords($0) }) ?? m
    }
    replace("\\d+") { m in
        guard let n = Int(m) else { return m }
        if n > 1000 && n < 3000 {
            if n == 2000 { return "two thousand" }
            if n > 2000 && n < 2010 { return "two thousand " + sopranoNumToWords(n % 100) }
            if n % 100 == 0 { return sopranoNumToWords(n / 100) + " hundred" }
            let f = n / 100; let s = n % 100
            return sopranoNumToWords(f) + (s < 10 ? " oh " : " ") + sopranoNumToWords(s)
        }
        return sopranoNumToWords(n)
    }
    return result
}

// ─── Soprano model ────────────────────────────────────────────────────────────

/// A loaded Soprano TTS model. Owns the LLM transformer backbone, the Vocos
/// audio decoder (nil for Soprano-1.1), and the BPE tokenizer.
public final class SopranoModel: @unchecked Sendable {
    /// Decoded configuration.
    public let config: SopranoConfig
    let llm: SopranoLLM
    /// Vocos decoder — nil for Soprano-1.1 (LLM-only checkpoint).
    let decoder: SopranoDecoder?
    /// Loaded BPE tokenizer. Populated by `load(directory:)`.
    var tokenizer: (any Tokenizer)?

    /// Output waveform sample rate (32 000 Hz for all Soprano variants).
    public var sampleRate: Int { config.sampleRate ?? 32_000 }

    init(config: SopranoConfig, llm: SopranoLLM, decoder: SopranoDecoder?) {
        self.config = config
        self.llm = llm
        self.decoder = decoder
    }

    // ── Synthesis ─────────────────────────────────────────────────────

    /// Synthesise speech for `text`. Returns a flat `[Float]` waveform at
    /// `sampleRate` Hz.
    ///
    /// Throws `SopranoError.decoderNotAvailable` for Soprano-1.1 checkpoints
    /// (which ship only the LLM).
    ///
    /// - Parameters:
    ///   - text: Input text to synthesise.
    ///   - parameters: Generation hyper-parameters.
    ///   - device: Metal device (defaults to `.shared`).
    public func synthesize(
        text: String,
        parameters: AudioGenerationParameters = AudioGenerationParameters(),
        device: Device = .shared
    ) throws -> [Float] {
        guard let dec = decoder else { throw SopranoError.decoderNotAvailable }
        guard let tok = tokenizer else { throw SopranoError.tokenizerNotLoaded }

        // 1. Text preprocessing
        let cleaned = cleanTextForSoprano(text.trimmingCharacters(in: .whitespaces))
        let prompt = "[STOP][TEXT]\(cleaned)[START]"
        let tokenIds = tokenize(prompt, tokenizer: tok)
        guard !tokenIds.isEmpty else {
            throw SopranoError.generationFailed("text produced no tokens")
        }

        // 2. Autoregressive LLM → collect hidden-state rows
        let (hiddenRows, tokenCount) = generateHiddenStates(
            promptIds: tokenIds,
            parameters: parameters,
            device: device)

        guard tokenCount > 0 else {
            throw SopranoError.generationFailed("LLM produced no tokens before stop")
        }

        // 3. Vocos decode → waveform
        let waveformTensor = dec.decode(
            hiddenStates: hiddenRows, tokenCount: tokenCount, device: device)

        // 4. Trim Vocos centre-padding (nFFT/2 from each end).
        let raw = waveformTensor.toArray(as: Float.self)
        let trim = dec.nFFT / 2
        let lo = min(trim, raw.count)
        let hi = max(raw.count - trim, lo)
        return Array(raw[lo..<hi])
    }

    // ── Tokenizer ─────────────────────────────────────────────────────

    /// Space token ID in the Soprano BPE vocabulary.
    private static let spaceTokenId = 8004

    /// Tokenise a Soprano prompt string, correctly handling:
    ///   • Special tokens ([STOP], [TEXT], [START]) — encoded as-is.
    ///   • Whitespace runs — substituted with the space-token ID (8004)
    ///     to work around a swift-transformers BPE splitter quirk.
    private func tokenize(_ text: String, tokenizer: any Tokenizer) -> [Int] {
        guard let specialRe = try? NSRegularExpression(
                  pattern: #"\[(STOP|TEXT|START)\]"#),
              let preTokenRe = try? NSRegularExpression(
                  pattern: #"\s+|\w+|[^\w\s]+"#)
        else { return tokenizer.encode(text: text, addSpecialTokens: false) }

        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let specialMatches = specialRe.matches(in: text, range: fullRange)

        var segments: [(text: String, isSpecial: Bool)] = []
        var lastEnd = 0
        for match in specialMatches {
            if match.range.location > lastEnd {
                let r = NSRange(location: lastEnd,
                                length: match.range.location - lastEnd)
                segments.append((nsText.substring(with: r), false))
            }
            segments.append((nsText.substring(with: match.range), true))
            lastEnd = match.range.location + match.range.length
        }
        if lastEnd < nsText.length {
            segments.append((nsText.substring(from: lastEnd), false))
        }

        var allTokens: [Int] = []
        for seg in segments where !seg.text.isEmpty {
            if seg.isSpecial {
                allTokens += tokenizer.encode(text: seg.text,
                                               addSpecialTokens: false)
            } else {
                let segNS = seg.text as NSString
                let r = NSRange(location: 0, length: segNS.length)
                let chunks = preTokenRe.matches(in: seg.text, range: r).map {
                    segNS.substring(with: $0.range)
                }
                for chunk in chunks {
                    if chunk.allSatisfy({ $0.isWhitespace }) {
                        for _ in chunk { allTokens.append(Self.spaceTokenId) }
                    } else {
                        allTokens += tokenizer.encode(
                            text: chunk, addSpecialTokens: false)
                    }
                }
            }
        }
        return allTokens
    }

    // ── Autoregressive generation ─────────────────────────────────────

    /// Run the LLM autoregression loop. Returns flat `[tokenCount × hidden]`
    /// hidden-state rows and the decoded token count.
    ///
    /// Strategy (matches the FFAI generate loop):
    ///   1. Prefill — process all prompt tokens sequentially (seeds KV cache).
    ///   2. Decode — sample tokens until [STOP] (ID 3) or maxTokens.
    ///   3. Collect the pre-head hidden state for every decode step.
    private func generateHiddenStates(
        promptIds: [Int],
        parameters: AudioGenerationParameters,
        device: Device
    ) -> (hiddenRows: [Float], tokenCount: Int) {
        let maxTokens = parameters.maxTokens
        let temperature = parameters.temperature
        let topP = parameters.topP
        let stopId = SopranoConfig.stopTokenId
        let hiddenDim = llm.hidden

        let caches = llm.makeLayerCaches(device: device)
        var position = 0
        var rng = SystemRandomNumberGenerator()

        // --- Prefill pass ---
        // Process all prompt tokens; only the last token's logits are needed.
        var lastLogits: Tensor?
        for tokenId in promptIds {
            let cmd = device.makeCommandBuffer()
            let (logits, _) = llm.forward(tokenId: tokenId, position: position,
                                           caches: caches, cmd: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()
            lastLogits = logits
            position += 1
        }

        // Sample the first decode token from the last prefill logits.
        var currentToken: Int = {
            guard let lg = lastLogits else { return stopId }
            let logitsF = AudioMath.floats(lg)
            return sopranoSample(logits: logitsF, temperature: temperature,
                                 topP: topP, rng: &rng)
        }()

        // --- Decode loop ---
        var hiddenRows: [Float] = []
        hiddenRows.reserveCapacity(maxTokens * hiddenDim)
        var tokenCount = 0

        for _ in 0..<maxTokens {
            if currentToken == stopId { break }

            let cmd = device.makeCommandBuffer()
            let (logits, hiddenTensor) = llm.forward(
                tokenId: currentToken, position: position,
                caches: caches, cmd: cmd, device: device)
            cmd.commit()
            cmd.waitUntilCompleted()

            hiddenRows.append(contentsOf: hiddenTensor.toArray(as: Float.self))
            tokenCount += 1
            position += 1

            let logitsF = AudioMath.floats(logits)
            currentToken = sopranoSample(logits: logitsF, temperature: temperature,
                                         topP: topP, rng: &rng)
        }

        return (hiddenRows, tokenCount)
    }
}

/// Top-p nucleus sampler for the Soprano decode loop.
private func sopranoSample(
    logits: [Float], temperature: Float, topP: Float,
    rng: inout some RandomNumberGenerator
) -> Int {
    guard temperature > 0 else {
        // Greedy: argmax.
        return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
    }
    // Temperature scaling + softmax.
    let maxVal = logits.max() ?? 0
    var probs = logits.map { expf(($0 - maxVal) / temperature) }
    let sum = probs.reduce(0, +)
    guard sum > 0 else { return 0 }
    for i in probs.indices { probs[i] /= sum }

    // Top-p filter (nucleus).
    if topP < 1.0 {
        let sorted = probs.enumerated().sorted { $0.element > $1.element }
        var cumulative: Float = 0
        var keep = Set<Int>()
        for (idx, p) in sorted {
            cumulative += p
            keep.insert(idx)
            if cumulative >= topP { break }
        }
        for i in probs.indices where !keep.contains(i) { probs[i] = 0 }
        let newSum = probs.reduce(0, +)
        if newSum > 0 { for i in probs.indices { probs[i] /= newSum } }
    }

    // Categorical sample.
    let draw = Float.random(in: 0..<1, using: &rng)
    var cumulative: Float = 0
    for (i, p) in probs.enumerated() {
        cumulative += p
        if draw < cumulative { return i }
    }
    return probs.count - 1
}

// ─── Loading ──────────────────────────────────────────────────────────────────

extension SopranoModel {

    /// `model_type` values this family handles.
    public static let modelTypes: Set<String> = ["soprano"]
    /// Architecture strings this family handles (Soprano-80M).
    public static let architectures: Set<String> = ["SopranoForCausalLM"]

    /// Whether a decoded `config.json` describes a Soprano checkpoint.
    ///
    /// Detection rules (Soprano-1.1 uses Qwen3ForCausalLM architecture, so
    /// architecture alone is not sufficient — model_type is the primary key):
    ///   1. model_type == "soprano"  ← canonical, covers both variants.
    ///   2. architecture == "SopranoForCausalLM"  ← structural fallback.
    public static func handles(_ config: ModelConfig) -> Bool {
        if let mt = config.modelType, modelTypes.contains(mt) { return true }
        if let arch = config.architecture, architectures.contains(arch) { return true }
        return false
    }

    /// Load a Soprano checkpoint from a resolved snapshot `directory`.
    ///
    /// Soprano-80M: loads both the LLM (language_model.*) and the Vocos
    /// decoder (decoder.*). Soprano-1.1: loads only the LLM (model.*).
    public static func load(
        directory: URL,
        device: Device = .shared
    ) async throws -> SopranoModel {
        let modelConfig = try ModelConfig.load(from: directory)
        guard let config = SopranoConfig.from(modelConfig) else {
            throw SopranoError.missingConfig("hidden_size / num_hidden_layers")
        }

        let bundle = try SafeTensorsBundle(directory: directory, device: device)
        let quant = modelConfig.quantization

        // Detect which weight-key layout is present.
        // Soprano-80M uses "language_model.*"; Soprano-1.1 uses "model.*".
        let is80M = bundle.has("language_model.embed_tokens.weight")
        let llmPrefix = is80M ? "language_model" : "model"

        let hidden      = config.hiddenSize
        let nLayers     = config.numHiddenLayers
        let nHeads      = config.numAttentionHeads
        let nKVHeads    = config.numKeyValueHeads
        let headDim     = config.headDim
        let intermediate = config.intermediateSize
        let eps         = config.rmsNormEps

        // Embedding
        let embedTokens = try loadEmbedding(
            base: "\(llmPrefix).embed_tokens", in: bundle,
            hidden: hidden, quantization: quant)

        // Transformer layers
        var layers: [SopranoLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "\(llmPrefix).layers.\(i)"
            let qProj  = try loadLinear(base: "\(p).self_attn.q_proj",  in: bundle, quantization: quant)
            let kProj  = try loadLinear(base: "\(p).self_attn.k_proj",  in: bundle, quantization: quant)
            let vProj  = try loadLinear(base: "\(p).self_attn.v_proj",  in: bundle, quantization: quant)
            let oProj  = try loadLinear(base: "\(p).self_attn.o_proj",  in: bundle, quantization: quant)
            let qNorm  = RMSNorm(weight: try bundle.tensor(named: "\(p).self_attn.q_norm.weight"), eps: eps)
            let kNorm  = RMSNorm(weight: try bundle.tensor(named: "\(p).self_attn.k_norm.weight"), eps: eps)
            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: bundle, quantization: quant)
            let upProj   = try loadLinear(base: "\(p).mlp.up_proj",   in: bundle, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: bundle, quantization: quant)
            let inputNorm    = RMSNorm(
                weight: try bundle.tensor(named: "\(p).input_layernorm.weight"), eps: eps)
            let postAttnNorm = RMSNorm(
                weight: try bundle.tensor(named: "\(p).post_attention_layernorm.weight"), eps: eps)
            layers.append(SopranoLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                qNorm: qNorm, kNorm: kNorm,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, intermediate: intermediate,
                ropeTheta: config.ropeTheta))
        }

        // Final norm
        let finalNorm = RMSNorm(
            weight: try bundle.tensor(named: "\(llmPrefix).norm.weight"),
            eps: eps)

        // LM head — Soprano-80M stores under language_model.lm_head or bare lm_head.
        let lmHead: AnyLinear
        let headBase80M = "language_model.lm_head"
        if is80M && bundle.has("\(headBase80M).weight") {
            lmHead = try loadLinear(base: headBase80M, in: bundle, quantization: quant)
        } else if bundle.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: bundle, quantization: quant)
        } else {
            // Tied embeddings — share the embed table as the LM head.
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype = embedTokens.weight.dtype

        let llm = SopranoLLM(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads, nKVHeads: nKVHeads,
            headDim: headDim, maxSeq: config.maxPositionEmbeddings,
            ropeTheta: config.ropeTheta, dtype: activationDtype)

        // Vocos decoder — only present in Soprano-80M.
        // Apply the per-version defaults (Python's __post_init__ logic):
        //   Soprano-80M uses decoder_dim=512, input_kernel=3 (even though
        //   the config may not explicitly list input_kernel).
        let dec: SopranoDecoder?
        if is80M && config.hasDecoderConfig {
            let decoderDim       = config.decoderDim ?? 512
            let inputKernel      = config.inputKernel ?? 3
            let dwKernel         = config.dwKernel ?? 3
            let intermediateDim  = config.decoderIntermediateDim ?? (decoderDim * 3)
            let numDecoderLayers = config.decoderNumLayers ?? 8
            let hopLength        = config.hopLength ?? 512
            let nFft             = config.nFft ?? 2048
            let upscale          = config.upscale ?? 4

            dec = try SopranoDecoder.load(
                from: bundle,
                inputChannels: hidden,
                decoderDim: decoderDim,
                intermediateDim: intermediateDim,
                numLayers: numDecoderLayers,
                inputKernel: inputKernel,
                dwKernel: dwKernel,
                upscale: upscale,
                nFFT: nFft,
                hopLength: hopLength)
        } else {
            dec = nil
        }

        let model = SopranoModel(config: config, llm: llm, decoder: dec)

        // Load BPE tokenizer from the checkpoint directory.
        let loader = TokenizerLoader()
        model.tokenizer = try await loader.load(from: directory)

        return model
    }
}
