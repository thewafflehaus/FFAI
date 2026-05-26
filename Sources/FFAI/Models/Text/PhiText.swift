// Phi text — concrete variants + the dense decoder support for the
// Phi-3 / Phi-3.5 family. The family enum (`enum Phi`), variant
// protocol (`PhiVariant`), and error type (`PhiError`) live in
// `Models/Phi.swift` (the family root / main interface).
//
// This file holds:
//   • `Phi3Dense` — `PhiVariant` conformance + the per-variant
//     `loadModel` entry. Slices the fused `qkv_proj` and
//     `gate_up_proj` weights into row-views that drop straight into
//     `LlamaLayer`, then returns a `LlamaModel` engine.

import Foundation

public struct Phi3Dense: PhiVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Phi-3's reference generation defaults. Slightly cooler than
    /// Llama (temperature 0.0 by default in HF examples), but we
    /// mirror Llama's family default so the user-facing surface is
    /// consistent across families.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let vocab = config.vocabSize,
              let intermediate = config.intermediateSize,
              let eps = config.rmsNormEps
        else {
            throw PhiError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // Phi-3 puts head_dim in the config explicitly on later
        // revisions, but the canonical Phi-3-mini config does not —
        // it implies head_dim = hidden / nHeads.
        let headDim = config.headDim ?? (hidden / nHeads)
        let theta = Float(config.ropeTheta ?? 10_000)
        let maxSeq = config.int("max_position_embeddings") ?? 4096
        let tieEmbed = config.tieWordEmbeddings

        // SuScaledRoPE / longrope is queued as a follow-up. For now
        // refuse those variants explicitly so the user gets a clear
        // error instead of a silently-wrong RoPE.
        if let rs = config.nested("rope_scaling"),
           let t = rs["type"] as? String ?? rs["rope_type"] as? String,
           t != "default" && t != "linear"
        {
            throw PhiError.unsupportedRopeScaling(t)
        }
        // Linear rope_scaling is just a frequency divisor — propagate
        // via the Llama machinery's ropeScaling.scaleFactor.
        let ropeScaling: Ops.RoPEScaling
        if let rs = config.nested("rope_scaling"),
           let factor = rs["factor"] as? Double
        {
            // Linear scaling reduces frequencies uniformly; Llama's
            // Ops.RoPEScaling expresses this as `scaleFactor`.
            ropeScaling = Ops.RoPEScaling(
                scaleFactor: Float(factor),
                lowFreqFactor: 1.0,
                highFreqFactor: 1.0,
                originalMaxPosition: Float(maxSeq)
            )
        } else {
            ropeScaling = .none
        }

        let quant = config.quantization
        // Embedding (Phi-3 fp16/bf16 keep this dense; some 4-bit
        // packs do quantize it — same path as Llama).
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Layers
        var layers: [LlamaLayer] = []
        layers.reserveCapacity(nLayers)

        let qSize = nHeads * headDim
        let kvSize = nKVHeads * headDim

        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            // Fused qkv split. Phi-3 quantized fused projections need
            // group-aligned slicing (see header comment) — bail with a
            // descriptive error rather than silently miscomputing.
            if quant != nil, weights.isQuantized("\(p).self_attn.qkv_proj") {
                throw PhiError.quantizedFusedNotSupported
            }
            let qkvFused = try weights.tensor(named: "\(p).self_attn.qkv_proj.weight")
            // Sanity: [(Q + 2*KV) * head_dim, hidden]
            precondition(
                qkvFused.shape == [qSize + 2 * kvSize, hidden],
                "Phi3 qkv_proj shape mismatch: got \(qkvFused.shape), expected [\(qSize + 2 * kvSize), \(hidden)]"
            )
            let qWeight = qkvFused.slicedRows(start: 0, count: qSize)
            let kWeight = qkvFused.slicedRows(start: qSize, count: kvSize)
            let vWeight = qkvFused.slicedRows(start: qSize + kvSize, count: kvSize)
            let qProj = AnyLinear(Linear(weight: qWeight))
            let kProj = AnyLinear(Linear(weight: kWeight))
            let vProj = AnyLinear(Linear(weight: vWeight))

            let oProj = try loadLinear(base: "\(p).self_attn.o_proj",
                                       in: weights, quantization: quant)

            // Fused gate_up split.
            if quant != nil, weights.isQuantized("\(p).mlp.gate_up_proj") {
                throw PhiError.quantizedFusedNotSupported
            }
            let gateUpFused = try weights.tensor(named: "\(p).mlp.gate_up_proj.weight")
            precondition(
                gateUpFused.shape == [2 * intermediate, hidden],
                "Phi3 gate_up_proj shape mismatch: got \(gateUpFused.shape), expected [\(2 * intermediate), \(hidden)]"
            )
            let gateWeight = gateUpFused.slicedRows(start: 0, count: intermediate)
            let upWeight = gateUpFused.slicedRows(start: intermediate, count: intermediate)
            let gateProj = AnyLinear(Linear(weight: gateWeight))
            let upProj = AnyLinear(Linear(weight: upWeight))

            let downProj = try loadLinear(base: "\(p).mlp.down_proj",
                                          in: weights, quantization: quant)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))

            layers.append(LlamaLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, intermediate: intermediate,
                ropeTheta: theta, ropeScaling: ropeScaling
            ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize
            ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
           let scales = try? weights.tensor(named: "model.embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return LlamaModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction
        )
    }
}
