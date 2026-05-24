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
    ) throws -> LlamaModel
}

// ─── LlamaDense — standard transformer ───────────────────────────────

public struct LlamaDense: LlamaVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Llama 3.x dense defaults. Tracks mlx-swift-lm's
    /// `GenerationParameters` baseline (temp 0.6, top-p 1.0) and
    /// mlx-swift-lm's per-family `defaultPrefillStepSize` (1024 for
    /// dense attention models).
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
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction,
            auraDecodePath: options.auraDecodePath
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
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
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

        // AURA cache stores K and V in Π-rotated space; apply Π to Q
        // before SDPA and Π^T to the SDPA output before oProj so the
        // scores cancel and the residual stream stays in the original
        // activation space. Raw / affine caches skip both rotations.
        // See AURAQuantizedKVCache header for the math.
        let qForSdpa: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            qForSdpa = Ops.auraRotatePerHead(
                qRotated.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtype,
                nHeads: nHeads, headDim: headDim, on: cmd
            ).reshaped(to: [nHeads, headDim])
        } else {
            qForSdpa = qRotated
        }

        // SDPA — cache is [nKVHeads, maxSeq, headDim]; kernel takes
        // n_kv = filled length (post-append), kv_stride = maxSeq. For
        // affine-quantized caches, prepareForAttention queues a
        // bulk-dequant pass into the shared working buffer first.
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)

        let attnReadyForOProj: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            attnReadyForOProj = Ops.auraRotatePerHead(
                attnOut.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtypeT,
                nHeads: nHeads, headDim: headDim, on: cmd)
        } else {
            attnReadyForOProj = attnOut.reshaped(to: [nHeads * headDim])
        }
        let oOut = oProj(attnReadyForOProj, on: cmd)

        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Validator gate handles Llama 3 70B (8192).
        // Returns (postAttn residual stream, normed input for MLP).
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, oOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postAttn = fused.residual
            mlpNorm = fused.normed
        } else {
            postAttn = Ops.add(h, oOut, on: cmd)
            mlpNorm = postAttnNorm(postAttn, on: cmd)
        }

        // MLP — SwiGLU
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }

    /// Chunked forward — process `nRows` tokens at once. `h` is the
    /// `[nRows, hidden]` residual stream; `position` is the absolute
    /// sequence index of the FIRST row (subsequent rows take
    /// `position + 1`, `+2`, …). Returns the `[nRows, hidden]`
    /// updated residual stream.
    ///
    /// Hot path for Phase 6.6 chunked prefill. The collapse vs the
    /// per-token loop:
    /// - inputNorm / postAttnNorm: `Ops.rmsNormRows` instead of `nRows`
    ///   single-row RMSNorms.
    /// - q/k/v/o/gate/up/down projections: batched `AnyLinear.batched`
    ///   instead of `nRows` `gemv`/`dequantGemv` calls (full-precision
    ///   path collapses to one `Ops.gemm` per projection).
    /// - Attention: one `Ops.sdpaMulti(causal: true)` over the full
    ///   chunk instead of `nRows` separate `Ops.sdpaDecode` calls —
    ///   the biggest single dispatch saving.
    /// - RoPE + cache append: still loop `nRows` times (single-position
    ///   kernels) but all queued on the same `cmd`; the GPU pipelines
    ///   them, the CPU-side cost is just enqueueing.
    ///
    /// **Cache compatibility.** AURA caches need Q π-rotation +
    /// post-SDPA π^T-unrotation; this fast path doesn't apply them.
    /// `LlamaModel.forwardMulti` checks the cache type and falls back
    /// to the per-token loop when any layer's cache is `AURAQuantizedKVCache`.
    func forwardMulti(_ h: Tensor, startingAt position: Int,
                      cache: any KVCacheProtocol,
                      cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden],
                     "LlamaLayer.forwardMulti: h shape \(h.shape) ≠ [nRows, hidden]")

        // ── Attention ───────────────────────────────────────────────────
        let xNorm = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd
        ).reshaped(to: [nRows, hidden])

        let q = qProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nHeads*headDim]
        let k = kProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nKVHeads*headDim]
        let v = vProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nKVHeads*headDim]

        // RoPE — single-position kernel, looped per row on the same cmd.
        // Allocates rotated buffers up front, RoPE writes in place.
        let qRot = Tensor.empty(shape: [nRows, nHeads * headDim],
                                dtype: q.dtype, device: device)
        let kRot = Tensor.empty(shape: [nRows, nKVHeads * headDim],
                                dtype: k.dtype, device: device)
        for i in 0..<nRows {
            let qRow = q.slicedRows(start: i, count: 1).reshaped(to: [nHeads * headDim])
            let qOut = qRot.slicedRows(start: i, count: 1).reshaped(to: [nHeads * headDim])
            _ = Ops.rope(qRow, position: position + i, headDim: headDim,
                         thetaBase: ropeTheta, scaling: ropeScaling,
                         on: cmd, into: qOut)
            let kRow = k.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads * headDim])
            let kOut = kRot.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads * headDim])
            _ = Ops.rope(kRow, position: position + i, headDim: headDim,
                         thetaBase: ropeTheta, scaling: ropeScaling,
                         on: cmd, into: kOut)
            // Append this token's rotated K + raw V to the cache.
            cache.appendOnGPU(
                kFlat: kOut.reshaped(to: [nKVHeads, headDim]),
                vFlat: v.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads, headDim]),
                on: cmd
            )
        }

        // SDPA — ONE dispatch over the chunk. Causal masking means
        // query row r attends [0, position + r + 1).
        let qForSdpa = qRot.reshaped(to: [nRows, nHeads, headDim])
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaMulti(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            baseKV: position, nQuery: nRows, kvStride: cache.maxSeq,
            causal: true, scale: scale, on: cmd
        )

        let attnFlat = attnOut.reshaped(to: [nRows, nHeads * headDim])
        let oOut = oProj.callMany(attnFlat, t: nRows, on: cmd, device: device)

        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Validator gate handles Llama 3 70B (8192).
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h.reshaped(to: [nRows * hidden]),
                oOut.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: cmd)
            postAttn = fused.residual.reshaped(to: [nRows, hidden])
            mlpNorm = fused.normed.reshaped(to: [nRows, hidden])
        } else {
            postAttn = Ops.add(
                h.reshaped(to: [nRows * hidden]),
                oOut.reshaped(to: [nRows * hidden]),
                on: cmd
            ).reshaped(to: [nRows, hidden])
            mlpNorm = Ops.rmsNormRows(
                postAttn.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: cmd
            ).reshaped(to: [nRows, hidden])
        }

        // ── MLP — SwiGLU, batched ───────────────────────────────────────
        let gate = gateProj.callMany(mlpNorm, t: nRows, on: cmd, device: device)
        let up = upProj.callMany(mlpNorm, t: nRows, on: cmd, device: device)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj.callMany(mlpInner, t: nRows, on: cmd, device: device)

        return Ops.add(
            postAttn.reshaped(to: [nRows * hidden]),
            mlpOut.reshaped(to: [nRows * hidden]),
            on: cmd
        ).reshaped(to: [nRows, hidden])
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
    /// Cache scheme to use when `makeLayerCaches(...)` is called. Set at
    /// construction time from `LoadOptions.kvCache`.
    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction
    /// AURA decode-time attention path — `.compressed` (default) or
    /// `.dequantMirror` (Stage 1a A/B). Forwarded into every
    /// `AURAQuantizedKVCache` instantiated by `makeLayerCaches`.
    /// Ignored when `kvCacheKind != .auraQuantized(...)`.
    public let auraDecodePath: AURADecodePath

    init(embedTokens: AnyEmbedding, layers: [LlamaLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
         kvCacheKind: KVCacheKind = .raw,
         kvEviction: KVEviction = .unbounded,
         auraDecodePath: AURADecodePath = .compressed) {
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
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
        self.auraDecodePath = auraDecodePath
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

    /// Make a fresh KV cache for one inference session. The concrete
    /// type depends on `kvCacheKind` (set from LoadOptions at load
    /// time): raw `KVCache` for `.raw`, `AffineQuantizedKVCache` (with
    /// a shared working buffer pair) for `.affineQuantized`.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        let eviction = kvEviction
        switch kvCacheKind {
        case .raw:
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: eviction, device: device)
            }
        case .affineQuantized(let bits, let groupSize):
            // One shared working buffer pair across every layer's
            // cache — gives the real memory savings vs per-layer
            // working buffers.
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    eviction: eviction,
                    device: device
                )
            }
        case .auraQuantized(let scheme):
            // Codebooks are shared across layers; rotations are per-layer
            // (deterministic SRHT seeded by layer index). See Qwen3's
            // matching case for the longer explanation.
            let kCodebookData = AURACodebook.centroids(dim: headDim, bits: scheme.keyBits)
            let kBoundariesData = AURACodebook.boundaries(dim: headDim, bits: scheme.keyBits)
            let vCodebookData = AURACodebook.centroids(dim: headDim, bits: scheme.valueBits)
            let vBoundariesData = AURACodebook.boundaries(dim: headDim, bits: scheme.valueBits)

            let kCodebook = Tensor.empty(shape: [kCodebookData.count], dtype: .f32, device: device)
            kCodebook.copyIn(from: kCodebookData)
            let kBoundaries = Tensor.empty(shape: [kBoundariesData.count], dtype: .f32, device: device)
            kBoundaries.copyIn(from: kBoundariesData)
            let vCodebook = Tensor.empty(shape: [vCodebookData.count], dtype: .f32, device: device)
            vCodebook.copyIn(from: vCodebookData)
            let vBoundaries = Tensor.empty(shape: [vBoundariesData.count], dtype: .f32, device: device)
            vBoundaries.copyIn(from: vBoundariesData)

            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { i in
                let rot = AURAQuantizedKVCacheRotations.build(
                    headDim: headDim, layerIndex: i,
                    activationDtype: dtype, device: device)
                return AURAQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, scheme: scheme,
                    rotation: rot.rotation, rotationT: rot.rotationT,
                    rotationDtype: rot.rotationDtype, rotationDtypeT: rot.rotationDtypeT,
                    kCodebook: kCodebook, kBoundaries: kBoundaries,
                    vCodebook: vCodebook, vBoundaries: vBoundaries,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    eviction: eviction,
                    decodePath: auraDecodePath,
                    device: device
                )
            }
        }
    }

    /// Single-token forward. `tokenId` is the input token; `position` is
    /// its absolute sequence index. Returns logits [vocab].
    ///
    /// All N layers + embedding + final norm + lm head are queued on
    /// ONE MTLCommandBuffer with a single commit + wait at the end.
    /// Primitive: queue a single-token forward pass onto the caller's
    /// command buffer. No commit. Higher-level entry points
    /// (`forward`, `forwardSample`, `forwardSampleCategorical`) compose
    /// this with their respective output kernels on the same cmdbuf
    /// for a 1-commit-per-token decode step.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // `InspectTap` is a no-op when FFAI_INSPECT isn't set
        // (single bool compare per boundary). When enabled, it
        // runs everything on a private cmdbuf and prints per-layer
        // stats — see `Sources/FFAI/Inspect/InspectTap.swift`.
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        // Embedding
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        workCmd = tap.dumpLayerBoundary(h, label: "embed", layer: -1,
                                        cmd: workCmd, device: device)

        // Layers — all queued on the same command buffer, no syncs.
        // Tap fires at each layer's residual output. In production
        // mode `dumpLayerBoundary` returns its `cmd` parameter
        // unchanged (the optimizer folds the whole call to a no-op).
        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: workCmd, device: device)
            workCmd = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                            cmd: workCmd, device: device)
        }

        // Final norm + lm head
        let normed = finalNorm(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(normed, label: "final_norm", layer: -1,
                                        cmd: workCmd, device: device)
        let logits = lmHead(normed, on: workCmd)
        workCmd = tap.dumpLayerBoundary(logits, label: "logits", layer: -1,
                                        cmd: workCmd, device: device)

        // Flush the private cmdbuf when taps are active. Caller's
        // cmd has no work in that case (fast no-op commit).
        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    // `forwardSample` and `forwardSampleCategorical` come from
    // `LanguageModel`'s default extension — they compose
    // `forward(...on cmd:)` above with the appropriate output kernel on
    // the same command buffer. No family-specific override needed.

    /// Chunked multi-token forward — Phase 6.6 prefill fast path.
    /// Embeds `tokenIds.count` tokens, runs the layer stack with
    /// `LlamaLayer.forwardMulti` (one `Ops.sdpaMulti(causal: true)`
    /// per layer + batched projections), and returns the
    /// **tail-position logits only** — the contract `Generate.swift`
    /// consumes (only the final position is sampled).
    ///
    /// **Fallback.** AURA caches use a π-rotated K/V layout that the
    /// chunked path doesn't implement; if any layer's cache is
    /// `AURAQuantizedKVCache`, we fall back to the protocol-default
    /// per-token loop (still on one cmd, just one SDPA dispatch per
    /// token). Affine-quantized + raw KV caches take the fast path.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "LlamaModel.forwardMulti: tokenIds must be non-empty")

        // AURA fallback — see doc comment.
        let hasAura = caches.contains { $0 is AURAQuantizedKVCache }
        if hasAura {
            var logits: Tensor!
            for (i, tok) in tokenIds.enumerated() {
                logits = forward(tokenId: tok, position: position + i,
                                 caches: caches, on: cmd, device: device)
            }
            return logits
        }

        let n = tokenIds.count

        // Embedding — single gather for all N tokens, returns [N, hidden].
        let idsBuf = device.makeBuffer(length: n * 4)
        idsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: n) { p in
            for (i, t) in tokenIds.enumerated() { p[i] = UInt32(t) }
        }
        let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [n], dtype: .u32)
        var h = embedTokens(idsTensor, on: cmd)
        precondition(h.shape == [n, hidden],
                     "LlamaModel.forwardMulti: embedding shape \(h.shape) ≠ [n, hidden]")

        // Layers — each runs its chunked forward over [N, hidden].
        for (i, layer) in layers.enumerated() {
            h = layer.forwardMulti(
                h, startingAt: position,
                cache: caches[i] as! any KVCacheProtocol,
                cmd: cmd, device: device
            )
        }

        // Final norm + LM head on the LAST row only — we only need the
        // tail logits (only the final position is sampled).
        let tail = h.slicedRows(start: n - 1, count: 1).reshaped(to: [hidden])
        let normed = finalNorm(tail, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Embedding-input forward — the VLM splice path. Identical to
    /// `forward(tokenId:...)` minus the embedding gather: the `[hidden]`
    /// row is supplied directly (a vision-encoder token, or a text-token
    /// embedding the VL model looked up itself). Qwen 2 / 2.5 VL route
    /// their text backbone through the Llama dense engine, so the VL
    /// splice needs this primitive on `LlamaModel`.
    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "LlamaModel.forward(inputEmbedding:): expected [\(hidden)], "
                     + "got \(inputEmbedding.shape)")
        var h = inputEmbedding.reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: cmd, device: device)
        }
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Raw embedding-table lookup for one text token — the text-token
    /// half of the VLM splice stream.
    public func textEmbedding(tokenId: Int, device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let embed = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        cmd.commit()
        cmd.waitUntilCompleted()
        return embed
    }
}
