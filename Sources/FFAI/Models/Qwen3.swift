// Qwen3 family — Qwen3 dense (Phase 2.5). Future variants land here:
//
//   Qwen3.5 hybrid (GDN + attention)   — Phase 4 alongside the GDN kernels
//   Qwen3.5 MoE                        — Phase 4
//   Qwen3.5-VL (vision)                — Phase 6
//   Qwen3.5-Omni (vision + audio)      — Phase 7+
//
// The protocol + per-variant struct convention is established now even
// with a single dense variant, so adding 3.5 hybrid/MoE later is a
// new struct + a new entry in `Qwen3.variant(for:)` rather than a
// switch-statement grow-out.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen3 {
    public static let modelTypes: Set<String> = ["qwen3"]
    public static let architectures: Set<String> = ["Qwen3ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any Qwen3Variant.Type {
        return Qwen3Dense.self
    }
}

public protocol Qwen3Variant {
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant. The user can override any
    /// field; absent overrides fall back to the values declared here.
    /// See planning/roadmap.md for which fields are honored today vs
    /// staged for Phase 5+ (sampling kernels).
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen3Model
}

public enum Qwen3Error: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig: return "Qwen3: required config field missing"
        }
    }
}

// ─── Qwen3Dense — standard dense transformer with q_norm / k_norm ─────

public struct Qwen3Dense: Qwen3Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Qwen 3 dense defaults. Tracks mlx-swift-lm's Qwen 3 family
    /// values (temp 0.6, top-p 0.95, top-k 20, min-p 0.0,
    /// rep-penalty 1.0) and a 1024-token prefill chunk for dense
    /// attention. Qwen 3.5 hybrid / MoE will declare their own when
    /// those variants land in Phase 5.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 0.95,
        topK: 20,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen3Model {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let headDim = config.headDim,
              let vocab = config.vocabSize,
              let intermediate = config.intermediateSize,
              let eps = config.rmsNormEps
        else {
            throw Qwen3Error.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let theta = Float(config.ropeTheta ?? 1_000_000)
        let maxSeq = config.int("max_position_embeddings") ?? 32_768
        let tieEmbed = config.tieWordEmbeddings

        // Optional linear RoPE scaling (Qwen3 dense uses rope_scaling
        // {type: "linear", factor: F} which uniformly divides every
        // inv_freq by F. We model this with the Llama-3 scaling shape
        // by setting scaleFactor = F and pinning everything else so the
        // wavelength branches always go down the "low_freq → divide"
        // path, equivalent to a uniform scale.)
        var ropeScaling = Ops.RoPEScaling.none
        if let rs = config.nested("rope_scaling") {
            if let typeStr = (rs["type"] as? String) ?? (rs["rope_type"] as? String),
               typeStr == "linear",
               let factor = rs["factor"] as? Double
            {
                ropeScaling = Ops.RoPEScaling(
                    scaleFactor: Float(factor),
                    lowFreqFactor: 1, highFreqFactor: 1,
                    originalMaxPosition: 1  // forces low_freq_wavelen = 1, so all freqs are "low"
                )
            } else if (rs["rope_type"] as? String) == "llama3" {
                // Some Qwen3 variants ship Llama-3 style scaling; reuse
                // the same parameters.
                ropeScaling = Ops.RoPEScaling(
                    scaleFactor: Float((rs["factor"] as? Double) ?? 1),
                    lowFreqFactor: Float((rs["low_freq_factor"] as? Double) ?? 1),
                    highFreqFactor: Float((rs["high_freq_factor"] as? Double) ?? 4),
                    originalMaxPosition: Float((rs["original_max_position_embeddings"] as? Int) ?? 8192)
                )
            }
        }

        let quant = config.quantization

        // Embedding — quantized if the bundle has matching scales/biases.
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Layers
        var layers: [Qwen3Layer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            // Per-head Q/K RMSNorm — the structural delta vs Llama.
            let qNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                eps: Float(eps))
            let kNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                eps: Float(eps))

            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: weights, quantization: quant)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))

            layers.append(Qwen3Layer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                qNorm: qNorm, kNorm: kNorm,
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

        // LM head. Tied / quantized variants — same as Llama.
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

        // Activation/inference dtype: prefer scales for quantized models.
        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
           let scales = try? weights.tensor(named: "model.embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return Qwen3Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache
        )
    }
}

// ─── Qwen3Layer ──────────────────────────────────────────────────────

public final class Qwen3Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let ropeScaling: Ops.RoPEScaling
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         qNorm: RMSNorm, kNorm: RMSNorm,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float,
         ropeScaling: Ops.RoPEScaling) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
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
        for (k, v) in qNorm.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        for (k, v) in kNorm.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        return out
    }

    /// Single-token forward pass. Same shape as LlamaLayer.forward but
    /// applies q_norm / k_norm to each head's [head_dim] vector before
    /// RoPE. All work queued on `cmd`, no commit/wait inside.
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // Attention
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head q_norm / k_norm
        let qNormed = applyPerHeadRMSNorm(q, weight: qNorm.weight, eps: qNorm.eps,
                                          nHeads: nHeads, headDim: headDim,
                                          on: cmd, device: device)
        let kNormed = applyPerHeadRMSNorm(k, weight: kNorm.weight, eps: kNorm.eps,
                                          nHeads: nKVHeads, headDim: headDim,
                                          on: cmd, device: device)

        // RoPE
        let qRotated = Ops.rope(qNormed.reshaped(to: [nHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)
        let kRotated = Ops.rope(kNormed.reshaped(to: [nKVHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)

        // GPU KV cache update — no CPU sync.
        cache.appendOnGPU(kFlat: kRotated,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
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

    /// Apply RMSNorm independently to each head's [head_dim] slice of a
    /// flat [nHeads * headDim] tensor via a single multi-row dispatch
    /// (Ops.rmsNormRows). Phase 4 collapse from one-launch-per-head.
    private func applyPerHeadRMSNorm(
        _ x: Tensor, weight: Tensor, eps: Float,
        nHeads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        Ops.rmsNormRows(x, weight: weight, eps: eps,
                        nRows: nHeads, rowSize: headDim, on: cmd)
    }
}

// ─── Qwen3Model ──────────────────────────────────────────────────────

public final class Qwen3Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Qwen3Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind

    init(embedTokens: AnyEmbedding, layers: [Qwen3Layer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
         kvCacheKind: KVCacheKind = .raw) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers; self.nHeads = nHeads
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.vocab = vocab
        self.maxSeq = maxSeq; self.ropeTheta = ropeTheta; self.dtype = dtype
        self.kvCacheKind = kvCacheKind
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

    public func makeKVCache(maxSeq: Int?, device: Device) -> [any KVCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        switch kvCacheKind {
        case .raw:
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, device: device)
            }
        case .affineQuantized(let bits, let groupSize):
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    device: device
                )
            }
        }
    }

    public func forward(tokenId: Int, position: Int,
                        caches: [any KVCacheProtocol], device: Device) -> Tensor {
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

        cmd.commit()
        cmd.waitUntilCompleted()
        return logits
    }

    public func forwardSample(tokenId: Int, position: Int,
                              caches: [any KVCacheProtocol], device: Device) -> Int {
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

        let outBuf = device.makeBuffer(length: 4)
        let outTensor = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.argmax(logits, into: outTensor, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()

        let result = outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        return Int(result)
    }

    /// Forward + GPU softmax-categorical-sample fused into one command
    /// buffer. Same structure as `forwardSample` but uses
    /// `softmax_categorical_sample` instead of argmax. Overrides the
    /// LanguageModel default impl that runs forward + sample on two
    /// cmdbufs — fusing them is the perf win for the `gpu-categorical`
    /// path Generate selects when T > 0 with no filters.
    public func forwardSampleCategorical(
        tokenId: Int, position: Int, caches: [any KVCacheProtocol],
        temperature: Float, uniformDraw: Float,
        device: Device
    ) -> Int {
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

        let tBuf = device.makeBuffer(length: 4)
        var tVal = temperature
        memcpy(tBuf.contents(), &tVal, 4)
        let temperatureT = Tensor(buffer: tBuf, offset: 0, shape: [1], dtype: .f32)

        let uBuf = device.makeBuffer(length: 4)
        var uVal = uniformDraw
        memcpy(uBuf.contents(), &uVal, 4)
        let uniformT = Tensor(buffer: uBuf, offset: 0, shape: [1], dtype: .f32)

        let outBuf = device.makeBuffer(length: 4)
        let outTensor = Tensor(buffer: outBuf, offset: 0, shape: [1], dtype: .u32)
        Ops.softmaxCategoricalSample(logits, into: outTensor,
                                     temperature: temperatureT,
                                     uniform: uniformT, on: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        return Int(outBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)
    }
}
