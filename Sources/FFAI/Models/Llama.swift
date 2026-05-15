// Llama family — Llama 3.x architecture. Phase 2 ships the dense
// variant only (1B / 3B / 8B / 70B; 405B with quant). The protocol +
// per-variant struct pattern is established here even with a single
// variant so the family scales when 4 / 4.x / future variants land.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum Llama {
    public static let modelTypes: Set<String> = ["llama"]
    public static let architectures: Set<String> = ["LlamaForCausalLM"]

    /// Pick the variant struct for a config. Phase 2 only knows about
    /// LlamaDense; future variants (e.g. LlamaMoE if Llama 4 ships one)
    /// would dispatch here.
    public static func variant(for config: ModelConfig) throws -> any LlamaVariant.Type {
        // Only one variant for now.
        return LlamaDense.self
    }
}

public protocol LlamaVariant {
    static var availableCapabilities: Set<Capability> { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel
}

// ─── LlamaDense — standard transformer ───────────────────────────────

public struct LlamaDense: LlamaVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let headDim = config.headDim,
              let vocab = config.vocabSize,
              let intermediate = config.intermediateSize,
              let eps = config.rmsNormEps
        else {
            throw LlamaError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let theta = Float(config.ropeTheta ?? 500_000)
        let maxSeq = config.int("max_position_embeddings") ?? 8192
        let tieEmbed = config.tieWordEmbeddings

        // Llama 3 RoPE scaling (rope_type: "llama3"). For other rope_types
        // (yarn, dynamic) we'd dispatch differently — for Phase 2 we only
        // handle the explicit llama3 case + plain rope.
        var ropeScaling = Ops.RoPEScaling.none
        if let rs = config.nested("rope_scaling"),
           (rs["rope_type"] as? String) == "llama3"
        {
            ropeScaling = Ops.RoPEScaling(
                scaleFactor: Float((rs["factor"] as? Double) ?? (rs["factor"] as? Int).map(Double.init) ?? 1),
                lowFreqFactor: Float((rs["low_freq_factor"] as? Double) ?? 1),
                highFreqFactor: Float((rs["high_freq_factor"] as? Double) ?? 4),
                originalMaxPosition: Float((rs["original_max_position_embeddings"] as? Int) ?? 8192)
            )
        }

        let quant = config.quantization

        // Embedding — quantized if the bundle has matching scales/biases
        // (mlx-community 4-bit checkpoints typically quantize this).
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Layers
        var layers: [LlamaLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: weights, quantization: quant)

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

        // Final norm
        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        // LM head. Three cases:
        //   1. !tieEmbed and the checkpoint has lm_head: load it (quant
        //      if applicable).
        //   2. tieEmbed AND embedding is quantized: reuse the embedding's
        //      QuantizedLinear-shaped triplet for the lm_head gemv.
        //   3. tieEmbed AND embedding is full precision: tie weights.
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, [3, 4, 5, 6, 8].contains(q.bits),
                  weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize
            ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation/inference dtype: prefer the scales dtype for
        // quantized models (the f16/bf16 the model actually computes
        // in), fall back to the embedding weight dtype otherwise.
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
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype
        )
    }
}

public enum LlamaError: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig: return "Llama: required config field missing"
        }
    }
}

// ─── Layer (attention + MLP) ─────────────────────────────────────────

public final class LlamaLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let ropeScaling: Ops.RoPEScaling
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float,
         ropeScaling: Ops.RoPEScaling) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
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

    /// Single-token forward pass. `h` is the residual stream [hidden].
    /// `position` is the absolute sequence index of this token.
    /// Returns the updated residual stream.
    ///
    /// All work is queued on `cmd` — no commit/wait inside. Caller is
    /// responsible for committing once at end-of-token and waiting.
    func forward(_ h: Tensor, position: Int, cache: KVCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // Attention
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)   // [n_heads * head_dim]
        let k = kProj(xNorm, on: cmd)   // [n_kv_heads * head_dim]
        let v = vProj(xNorm, on: cmd)   // [n_kv_heads * head_dim]

        // RoPE on q and k
        let qRotated = Ops.rope(q.reshaped(to: [nHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)
        let kRotated = Ops.rope(k.reshaped(to: [nKVHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)

        // GPU KV cache update — no CPU sync. Bumps cache.length so
        // SDPA below sees this token in its attended-positions count.
        cache.appendOnGPU(kFlat: kRotated,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)

        // SDPA — cache is [nKVHeads, maxSeq, headDim]; kernel takes
        // n_kv = filled length (post-append), kv_stride = maxSeq.
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cache.kBuffer, v: cache.vBuffer,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let postAttn = Ops.add(h, oOut, on: cmd)

        // MLP — SwiGLU
        let mlpNorm = postAttnNorm(postAttn, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }
}

// ─── Whole model ─────────────────────────────────────────────────────

public final class LlamaModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [LlamaLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType

    init(embedTokens: AnyEmbedding, layers: [LlamaLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.ropeTheta = ropeTheta
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("model.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// Make a fresh KV cache for one inference session.
    public func makeKVCache(maxSeq: Int?, device: Device) -> [KVCache] {
        let cap = maxSeq ?? self.maxSeq
        return (0..<nLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap, dtype: dtype, device: device)
        }
    }

    /// Single-token forward. `tokenId` is the input token; `position` is
    /// its absolute sequence index. Returns logits [vocab].
    ///
    /// All N layers + embedding + final norm + lm head are queued on
    /// ONE MTLCommandBuffer with a single commit + wait at the end.
    public func forward(tokenId: Int, position: Int,
                        caches: [KVCache], device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()

        // Embedding
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        // Layers — all queued on the same command buffer, no syncs.
        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position, cache: caches[i],
                              cmd: cmd, device: device)
        }

        // Final norm + lm head
        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        // ONE sync per token.
        cmd.commit()
        cmd.waitUntilCompleted()

        return logits
    }

    /// Forward + GPU argmax fused into one command buffer. Only 4 bytes
    /// (the chosen token id) cross CPU↔GPU per token instead of vocab_size
    /// floats.
    public func forwardSample(tokenId: Int, position: Int,
                              caches: [KVCache], device: Device) -> Int {
        let cmd = device.makeCommandBuffer()

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position, cache: caches[i],
                              cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        // GPU argmax — 4-byte output buffer reused across calls would be
        // ideal; for Phase 4 simplicity allocate fresh per token.
        let outBuf = device.makeBuffer(length: 4)
        let outTensor = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outTensor, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        return Int(result)
    }
}
