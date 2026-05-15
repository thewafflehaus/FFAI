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

        // Embedding
        let embedWeight = try weights.tensor(named: "model.embed_tokens.weight")
        let embedTokens = Embedding(weight: embedWeight)

        // Layers
        var layers: [LlamaLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            let qProj = Linear(weight: try weights.tensor(named: "\(p).self_attn.q_proj.weight"))
            let kProj = Linear(weight: try weights.tensor(named: "\(p).self_attn.k_proj.weight"))
            let vProj = Linear(weight: try weights.tensor(named: "\(p).self_attn.v_proj.weight"))
            let oProj = Linear(weight: try weights.tensor(named: "\(p).self_attn.o_proj.weight"))

            let gateProj = Linear(weight: try weights.tensor(named: "\(p).mlp.gate_proj.weight"))
            let upProj = Linear(weight: try weights.tensor(named: "\(p).mlp.up_proj.weight"))
            let downProj = Linear(weight: try weights.tensor(named: "\(p).mlp.down_proj.weight"))

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

        // LM head
        let lmHead: Linear
        if tieEmbed {
            lmHead = Linear(weight: embedWeight)
        } else if let w = try? weights.tensor(named: "lm_head.weight") {
            lmHead = Linear(weight: w)
        } else {
            // Some checkpoints omit lm_head when tying; fall back to embed.
            lmHead = Linear(weight: embedWeight)
        }

        return LlamaModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: embedWeight.dtype
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
    let qProj, kProj, vProj, oProj: Linear
    let gateProj, upProj, downProj: Linear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let ropeScaling: Ops.RoPEScaling
    let scale: Float

    init(qProj: Linear, kProj: Linear, vProj: Linear, oProj: Linear,
         gateProj: Linear, upProj: Linear, downProj: Linear,
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

        // Append to KV cache. Need to sync the command buffer first so
        // rotated K/V are CPU-visible before memcpy.
        cmd.commit()
        cmd.waitUntilCompleted()

        cache.append(kFlat: kRotated, vFlat: v.reshaped(to: [nKVHeads, headDim]))

        // New command buffer for SDPA + MLP
        let cmd2 = device.makeCommandBuffer()

        // SDPA — cache is [nKVHeads, maxSeq, headDim]; kernel takes
        // n_kv = filled length, kv_stride = maxSeq physical stride.
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cache.kBuffer, v: cache.vBuffer,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd2)
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd2)
        let postAttn = Ops.add(h, oOut, on: cmd2)

        // MLP
        let mlpNorm = postAttnNorm(postAttn, on: cmd2)
        let gate = gateProj(mlpNorm, on: cmd2)
        let up = upProj(mlpNorm, on: cmd2)
        let siluGate = Ops.silu(gate, on: cmd2)
        let mlpInner = Ops.mul(siluGate, up, on: cmd2)
        let mlpOut = downProj(mlpInner, on: cmd2)
        let result = Ops.add(postAttn, mlpOut, on: cmd2)

        cmd2.commit()
        cmd2.waitUntilCompleted()

        return result
    }
}

// ─── Whole model ─────────────────────────────────────────────────────

public final class LlamaModel: Module {
    public let embedTokens: Embedding
    public let layers: [LlamaLayer]
    public let finalNorm: RMSNorm
    public let lmHead: Linear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType

    init(embedTokens: Embedding, layers: [LlamaLayer],
         finalNorm: RMSNorm, lmHead: Linear,
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
    public func makeKVCache(maxSeq: Int? = nil, device: Device = .shared) -> [KVCache] {
        let cap = maxSeq ?? self.maxSeq
        return (0..<nLayers).map { _ in
            KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap, dtype: dtype, device: device)
        }
    }

    /// Single-token forward. `tokenId` is the input token; `position` is
    /// its absolute sequence index. Returns logits [vocab].
    public func forward(tokenId: Int, position: Int,
                        caches: [KVCache], device: Device = .shared) -> Tensor {
        // Embedding
        let cmd0 = device.makeCommandBuffer()
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd0).reshaped(to: [hidden])
        cmd0.commit()
        cmd0.waitUntilCompleted()

        // Layers
        for (i, layer) in layers.enumerated() {
            let cmdL = device.makeCommandBuffer()
            h = layer.forward(h, position: position, cache: caches[i],
                              cmd: cmdL, device: device)
        }

        // Final norm + lm head
        let cmdF = device.makeCommandBuffer()
        let normed = finalNorm(h, on: cmdF)
        let logits = lmHead(normed, on: cmdF)
        cmdF.commit()
        cmdF.waitUntilCompleted()

        return logits
    }
}
