// NemotronDiffusion family — NVIDIA Nemotron-Labs-Diffusion, a
// "tri-mode" language model that runs the same weights as
// autoregressive (AR), block-wise diffusion, or self-speculation.
//
// The backbone is a standard dense Ministral/Llama transformer
// (RMSNorm, SwiGLU, GQA, RoPE, no biases). The only structural deltas
// vs Llama: weight keys are prefixed `encoder.`, the LM head is
// `diffusion_head.weight`, and head_dim is independent of hidden_size.
//
// What lives where:
//   - AR generation runs through the standard single-token `forward`
//     and the existing `Generate.swift` decode loop.
//   - Diffusion + self-speculation run through `forwardBlock` (multi-
//     token) and `GenerateDiffusion.swift`.
//
// Distinct from the planned `NemotronH` hybrid family (Mamba2 +
// attention) and its `NemotronCascade2` variant — different
// architecture, different `model_type`.
//
// RoPE: the checkpoint declares YaRN scaling, applied via the
// `ffai_rope_yarn` kernel (`Ops.ropeYaRN`). Correction-range bounds and
// the mscale attention factor are derived from `rope_parameters` at
// load time; a non-yarn checkpoint falls back to plain RoPE.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum NemotronDiffusion {
    // Both the text-only checkpoint and the VLM checkpoint share this
    // family — the VLM's text backbone uses the identical `encoder.*` /
    // `diffusion_head.weight` layout. Loading a VLM checkpoint here
    // brings up the tri-mode *text* backbone; the `vision_tower.*` /
    // `multi_modal_projector.*` tensors are left unreferenced until the
    // vision path lands.
    public static let modelTypes: Set<String> = [
        "nemotron_labs_diffusion", "nemotron_labs_diffusion_vlm",
    ]
    public static let architectures: Set<String> = [
        "NemotronDiffusionModel", "NemotronDiffusionVLMModel",
    ]

    public static func variant(
        for config: ModelConfig
    ) throws -> any NemotronDiffusionVariant.Type {
        return NemotronDiffusionDense.self
    }
}

public protocol NemotronDiffusionVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> NemotronDiffusionModel
}

public enum NemotronDiffusionError: Error, CustomStringConvertible {
    case missingConfig

    public var description: String {
        switch self {
        case .missingConfig: return "NemotronDiffusion: required config field missing"
        }
    }
}

// ─── NemotronDiffusionDense — the tri-mode dense transformer ──────

public struct NemotronDiffusionDense: NemotronDiffusionVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Defaults pulled from the Nemotron-Labs-Diffusion instruct
    /// checkpoints. Greedy (temp 0) is the safe default for the
    /// tri-mode decode paths until GPU sampling filters land.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> NemotronDiffusionModel {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let headDim = config.headDim,
              let vocab = config.vocabSize,
              let intermediate = config.intermediateSize,
              let eps = config.rmsNormEps
        else {
            throw NemotronDiffusionError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // rope_theta is nested under `rope_parameters` in this config.
        let theta: Float = {
            if let rp = config.nested("rope_parameters"),
               let t = (rp["rope_theta"] as? Double) ?? (rp["rope_theta"] as? Int).map(Double.init) {
                return Float(t)
            }
            return Float(config.ropeTheta ?? 1_000_000)
        }()
        // YaRN RoPE parameters from the `rope_parameters` block. Absent
        // or non-yarn configs fall back to plain RoPE (`factor == 1`).
        let yarn: Ops.RoPEYaRN = {
            guard let rp = config.nested("rope_parameters"),
                  (rp["rope_type"] as? String) == "yarn"
            else { return .plain }
            func f(_ key: String, _ fallback: Float) -> Float {
                if let d = rp[key] as? Double { return Float(d) }
                if let i = rp[key] as? Int { return Float(i) }
                return fallback
            }
            return Ops.RoPEYaRN.from(
                headDim: headDim, thetaBase: theta,
                factor: f("factor", 1),
                betaFast: f("beta_fast", 32),
                betaSlow: f("beta_slow", 1),
                originalMaxPosition: f("original_max_position_embeddings", 16384),
                mscale: f("mscale", 1), mscaleAllDim: f("mscale_all_dim", 1))
        }()

        // KV-cache depth. The checkpoint advertises a YaRN-extended
        // 262144 window — a cache that deep is ~27 GB across 26 layers,
        // so the default is a sane 8192. `LoadOptions.maxContextLength`
        // overrides it: pass the full `max_position_embeddings` to use
        // the entire advertised window, or a smaller value to bound
        // memory. Diffusion / self-speculation size their caches
        // explicitly per call regardless of this.
        let defaultMaxSeq = 8192
        let maxSeq = options.maxContextLength
            ?? min(config.int("max_position_embeddings") ?? defaultMaxSeq, defaultMaxSeq)
        let maskTokenId = config.int("mask_token_id") ?? 100
        let blockSize = config.int("block_size") ?? 32

        let quant = config.quantization

        let embedTokens = try loadEmbedding(
            base: "encoder.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        var layers: [NemotronDiffusionLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "encoder.layers.\(i)"
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
            layers.append(NemotronDiffusionLayer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, intermediate: intermediate,
                ropeTheta: theta, yarn: yarn
            ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "encoder.norm.weight"),
            eps: Float(eps))

        // LM head is `diffusion_head.weight` (untied). Fall back to a
        // tied embedding only if the head is genuinely absent.
        let lmHead: AnyLinear
        if weights.has("diffusion_head.weight") || weights.isQuantized("diffusion_head") {
            lmHead = try loadLinear(base: "diffusion_head", in: weights, quantization: quant)
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("encoder.embed_tokens"),
           let scales = try? weights.tensor(named: "encoder.embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return NemotronDiffusionModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype,
            maskTokenId: maskTokenId, blockSize: blockSize,
            eosTokenId: config.eosTokenId,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction
        )
    }
}

// ─── NemotronDiffusionLayer ──────────────────────────────────────

public final class NemotronDiffusionLayer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let yarn: Ops.RoPEYaRN
    let scale: Float

    /// LoRA adapter for `o_proj`, used during the self-speculation
    /// diffusion-draft phase only. `nil` unless an adapter was attached
    /// via `NemotronDiffusionModel.attachLoRA`. `loraB` already
    /// carries the adapter's alpha/rank scaling, so the delta is just
    /// `loraB · (loraA · x)`.
    var loraA: Tensor?
    var loraB: Tensor?

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float, yarn: Ops.RoPEYaRN) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.yarn = yarn
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

    /// Single-token forward (AR decode). Generic over the cache kind so
    /// AR mode works with raw / affine / AURA caches. Mirrors the
    /// Llama / Qwen3 single-token layer forward.
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRotated = Ops.ropeYaRN(q.reshaped(to: [nHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, yarn: yarn, on: cmd)
        let kRotated = Ops.ropeYaRN(k.reshaped(to: [nKVHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, yarn: yarn, on: cmd)

        cache.appendOnGPU(kFlat: kRotated,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)

        // AURA caches store K/V in Π-rotated space — rotate Q in, output
        // out. Raw / affine caches skip this. Same contract as Qwen3.
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

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)

        let attnReady: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            attnReady = Ops.auraRotatePerHead(
                attnOut.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtypeT,
                nHeads: nHeads, headDim: headDim, on: cmd)
        } else {
            attnReady = attnOut.reshaped(to: [nHeads * headDim])
        }
        let oOut = oProj(attnReady, on: cmd)

        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // (NemotronDiffusion hidden=3072 fits the ≤ 4096 cap).
        // Validator gate keeps the unfused path for any future wider
        // variant.
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

        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let mlpInner = Ops.mul(Ops.silu(gate, on: cmd), up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }

    /// Multi-token forward over a contiguous run of `N` tokens — the
    /// primitive diffusion / self-speculation build on. Requires a raw
    /// `KVCache` (the K/V scratch staging writes the buffer directly).
    ///
    /// - `append == false` (denoise draft): the N tokens' K/V are
    ///   staged into the cache's free region `[length, length+N)` and
    ///   `length` is left untouched, so the next denoise iteration
    ///   overwrites the same slots.
    /// - `append == true` (causal commit / prefill / verify): the N
    ///   tokens are appended, bumping `length` by N.
    /// - `causal == true`: query `r` attends `[0, length+r+1)`.
    ///   `causal == false`: every query attends the full
    ///   `[0, length+N)` (bidirectional within the block).
    func forwardTokens(_ h: Tensor, n: Int, positions: [Int], cache: KVCache,
                       append: Bool, causal: Bool, useLora: Bool,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let baseLength = cache.length
        let dt = h.dtype
        let qDim = nHeads * headDim
        let kvDim = nKVHeads * headDim

        // RMSNorm every row, then Q/K/V as one GEMM each — the weight is
        // read once and reused across the block's rows (vs N gemvs).
        let xNorm = Ops.rmsNormRows(h, weight: inputNorm.weight, eps: inputNorm.eps,
                                    nRows: n, rowSize: hidden, on: cmd)
        let qBlock = nemotronBlockProject(qProj, xNorm, nRows: n, on: cmd)   // [n, qDim]
        let kBlock = nemotronBlockProject(kProj, xNorm, nRows: n, on: cmd)   // [n, kvDim]
        let vBlock = nemotronBlockProject(vProj, xNorm, nRows: n, on: cmd)   // [n, kvDim]

        // RoPE per row (position-dependent) + K/V cache staging. RoPE
        // writes each rotated Q row straight into the contiguous `qAll`.
        let qAll = Tensor.empty(shape: [n, nHeads, headDim], dtype: dt)
        var kRot: [Tensor] = []; kRot.reserveCapacity(n)
        var vRows: [Tensor] = []; vRows.reserveCapacity(n)
        for r in 0..<n {
            let qRow = Tensor(buffer: qBlock.buffer, offset: qBlock.offset + r * qDim * dt.byteSize,
                              shape: [nHeads, headDim], dtype: dt)
            let kRow = Tensor(buffer: kBlock.buffer, offset: kBlock.offset + r * kvDim * dt.byteSize,
                              shape: [nKVHeads, headDim], dtype: dt)
            let vRow = Tensor(buffer: vBlock.buffer, offset: vBlock.offset + r * kvDim * dt.byteSize,
                              shape: [nKVHeads, headDim], dtype: dt)
            let qSlice = Tensor(buffer: qAll.buffer, offset: qAll.offset + r * qDim * dt.byteSize,
                                shape: [nHeads, headDim], dtype: dt)
            _ = Ops.ropeYaRN(qRow, position: positions[r], headDim: headDim,
                             thetaBase: ropeTheta, yarn: yarn, on: cmd, into: qSlice)
            kRot.append(Ops.ropeYaRN(kRow, position: positions[r], headDim: headDim,
                                     thetaBase: ropeTheta, yarn: yarn, on: cmd))
            vRows.append(vRow)
        }

        // Stage K/V into the cache buffer. `appendRangeOnGPU` writes
        // `[baseLength, baseLength+N)` and bumps `length`; the scratch
        // path writes the same slots without committing.
        if append {
            cache.appendRangeOnGPU(kRows: kRot, vRows: vRows, on: cmd)
        } else {
            for r in 0..<n {
                cache.writeTimestepOnGPU(kFlat: kRot[r], vFlat: vRows[r],
                                         atSlot: baseLength + r, on: cmd)
            }
        }

        // One multi-query SDPA over the whole block → [n, qDim].
        let attnAll = Ops.sdpaMulti(
            q: qAll, k: cache.kBuffer, v: cache.vBuffer,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            baseKV: baseLength, nQuery: n, kvStride: cache.maxSeq,
            causal: causal, scale: scale, on: cmd)

        // o_proj (GEMM), optional LoRA delta, residual — all batched.
        var oBlock = nemotronBlockProject(oProj, attnAll, nRows: n, on: cmd)   // [n, hidden]
        if useLora, let la = loraA, let lb = loraB {
            // o_proj(x) + loraB·(loraA·x); the alpha/rank scale is baked
            // into loraB. Each LoRA factor is one GEMM over the block.
            let aOut = Ops.gemm(weight: la, input: attnAll, nRows: n, on: cmd)
            let loraDelta = Ops.gemm(weight: lb, input: aOut, nRows: n, on: cmd)
            oBlock = Ops.add(oBlock, loraDelta, on: cmd)
        }
        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // over [n, hidden] rows (hidden=3072 fits the ≤ 4096 cap).
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, oBlock, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: n, rowSize: hidden, on: cmd)
            postAttn = fused.residual
            mlpNorm = fused.normed
        } else {
            postAttn = Ops.add(h, oBlock, on: cmd)   // [n, hidden]
            mlpNorm = Ops.rmsNormRows(postAttn, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                                      nRows: n, rowSize: hidden, on: cmd)
        }

        // SwiGLU MLP — gate/up/down as block GEMMs.
        let gate = nemotronBlockProject(gateProj, mlpNorm, nRows: n, on: cmd)
        let up = nemotronBlockProject(upProj, mlpNorm, nRows: n, on: cmd)
        let mlpInner = Ops.mul(Ops.silu(gate, on: cmd), up, on: cmd)
        let mlpOut = nemotronBlockProject(downProj, mlpInner, nRows: n, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }
}

/// Project a `[nRows, inDim]` activation block through a dense linear
/// layer in one `Ops.gemm`. Diffusion / self-speculation require a
/// non-quantized checkpoint — a quantized weight hits the precondition
/// (the block GEMM has no quantized path; AR mode handles quant fine).
private func nemotronBlockProject(_ proj: AnyLinear, _ input: Tensor, nRows: Int,
                                  on cmd: MTLCommandBuffer) -> Tensor {
    guard let lin = proj.inner as? Linear else {
        preconditionFailure("NemotronDiffusion diffusion / self-speculation require a "
            + "non-quantized checkpoint — the block GEMM has no quantized path")
    }
    precondition(lin.bias == nil,
                 "NemotronDiffusion: unexpected projection bias "
                 + "(config declares attention_bias / mlp_bias = false)")
    return Ops.gemm(weight: lin.weight, input: input, nRows: nRows, on: cmd)
}

// ─── NemotronDiffusionModel ──────────────────────────────────────

public final class NemotronDiffusionModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [NemotronDiffusionLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction

    /// Diffusion-mode parameters from `config.json`.
    public let maskTokenId: Int
    public let blockSize: Int
    public let eosTokenId: Int?

    /// Whether a `linear_spec_lora` adapter was attached — when true,
    /// self-speculation uses the LoRA-enhanced diffusion drafter.
    public private(set) var hasLoRA = false

    init(embedTokens: AnyEmbedding, layers: [NemotronDiffusionLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
         maskTokenId: Int, blockSize: Int, eosTokenId: Int?,
         kvCacheKind: KVCacheKind = .raw,
         kvEviction: KVEviction = .unbounded) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers; self.nHeads = nHeads
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.vocab = vocab
        self.maxSeq = maxSeq; self.ropeTheta = ropeTheta; self.dtype = dtype
        self.maskTokenId = maskTokenId; self.blockSize = blockSize
        self.eosTokenId = eosTokenId
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("encoder.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("encoder.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("encoder.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("diffusion_head.\(k)", v)) }
        return out
    }

    /// Load and attach a `linear_spec_lora` PEFT adapter. The adapter
    /// applies a rank-r LoRA to every layer's `o_proj`; self-speculation
    /// toggles it on for the diffusion-draft phase.
    ///
    /// `directory` may be a **model directory** (the adapter is resolved
    /// under `linear_spec_lora/`) or a directory that **directly holds**
    /// `adapter_model.safetensors` — so the same call hot-loads the
    /// checkpoint's bundled adapter or an external one. Safe to call at
    /// runtime: it replaces any currently-attached adapter. Silently
    /// no-ops when no adapter is found or the adapter is incomplete.
    /// Do not call while a generation is in flight.
    public func attachLoRA(from directory: URL, device: Device = .shared) {
        // Resolve the adapter directory — checkpoint subfolder or a
        // directory that holds the adapter outright.
        let fm = FileManager.default
        let subfolder = directory.appendingPathComponent("linear_spec_lora")
        let loraDir: URL
        if fm.fileExists(atPath: subfolder
            .appendingPathComponent("adapter_model.safetensors").path) {
            loraDir = subfolder
        } else if fm.fileExists(atPath: directory
            .appendingPathComponent("adapter_model.safetensors").path) {
            loraDir = directory
        } else {
            return
        }
        guard let bundle = try? SafeTensorsBundle(directory: loraDir, device: device)
        else { return }

        // LoRA scaling = lora_alpha / r (PEFT). Read from
        // adapter_config.json; fall back to the published 512/128 = 4.0.
        var scaling: Float = 4.0
        let cfgURL = loraDir.appendingPathComponent("adapter_config.json")
        if let data = try? Data(contentsOf: cfgURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let alpha = (obj["lora_alpha"] as? Double) ?? (obj["lora_alpha"] as? Int).map(Double.init)
            let rank = (obj["r"] as? Double) ?? (obj["r"] as? Int).map(Double.init)
            if let alpha, let rank, rank > 0 { scaling = Float(alpha / rank) }
        }

        for (i, layer) in layers.enumerated() {
            let prefix = "base_model.model.encoder.layers.\(i).self_attn.o_proj"
            guard let a = try? bundle.tensor(named: "\(prefix).lora_A.weight"),
                  let b = try? bundle.tensor(named: "\(prefix).lora_B.weight")
            else { return }   // incomplete adapter — leave the model adapter-free
            // Scaling is baked into loraB so the delta is just B·(A·x).
            layer.loraA = Self.convertWeight(a, to: dtype, scale: 1)
            layer.loraB = Self.convertWeight(b, to: dtype, scale: scaling)
        }
        hasLoRA = true
    }

    /// Detach the current LoRA adapter — clears every layer's o_proj
    /// adapter weights and flips `hasLoRA` off. The runtime counterpart
    /// to `attachLoRA`: call it to hot-unload, or call `attachLoRA`
    /// again to hot-swap. Do not call while a generation is in flight.
    public func detachLoRA() {
        for layer in layers {
            layer.loraA = nil
            layer.loraB = nil
        }
        hasLoRA = false
    }

    /// Convert an f32 adapter weight to the model dtype, optionally
    /// scaling. The PEFT adapter ships f32; the backbone (and the gemv
    /// path the LoRA delta runs through) is the model's dtype.
    private static func convertWeight(_ src: Tensor, to dtype: DType, scale: Float) -> Tensor {
        precondition(src.dtype == .f32, "LoRA adapter weight expected f32, got \(src.dtype)")
        let values = src.toArray(as: Float.self).map { $0 * scale }
        let out = Tensor.empty(shape: src.shape, dtype: dtype)
        switch dtype {
        case .f32:
            out.copyIn(from: values)
        case .f16:
            out.copyIn(from: values.map { Float16($0) })
        case .bf16:
            out.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
        default:
            fatalError("LoRA: unsupported model dtype \(dtype)")
        }
        return out
    }

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
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    eviction: eviction, device: device)
            }
        case .auraQuantized:
            // Diffusion / self-speculation require the raw KVCache (their
            // multi-token forward stages scratch K/V in the buffer).
            // AURA caches support AR mode only; fall back to raw so the
            // tri-mode paths keep working.
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: eviction, device: device)
            }
        }
    }

    /// Single-token AR forward — the `LanguageModel` primitive that
    /// `Generate.swift` composes for autoregressive decoding.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        workCmd = tap.dumpLayerBoundary(h, label: "embed", layer: -1,
                                        cmd: workCmd, device: device)

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: workCmd, device: device)
            workCmd = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                            cmd: workCmd, device: device)
        }

        let normed = finalNorm(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(normed, label: "final_norm", layer: -1,
                                        cmd: workCmd, device: device)
        let logits = lmHead(normed, on: workCmd)
        workCmd = tap.dumpLayerBoundary(logits, label: "logits", layer: -1,
                                        cmd: workCmd, device: device)

        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    /// Multi-token forward (LanguageModel protocol) — prefill
    /// fast path. NemotronDiffusion already ships a block-batched
    /// forward (`forwardBlock`) used by the diffusion + self-spec
    /// modes; this is the thin LanguageModel-protocol adapter that
    /// wraps it for the AR-prefill path.
    ///
    /// `forwardBlock` does the full chunked work — batched embed,
    /// batched RMSNorm + Linear projections, bidirectional / causal
    /// SDPA over the chunk, batched lm_head. The tail-position logits
    /// returned here are the final element of forwardBlock's
    /// per-position output array.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "NemotronDiffusionModel.forwardMulti: tokenIds must be non-empty")
        // forwardBlock requires raw KVCache per layer; if any cache
        // isn't raw, fall back to the per-token loop on the caller's
        // cmd (still gets the protocol's commit-count batching).
        let allRaw = caches.allSatisfy { $0 is KVCache }
        if !allRaw {
            var logits: Tensor!
            for (i, tok) in tokenIds.enumerated() {
                logits = forward(tokenId: tok, position: position + i,
                                 caches: caches, on: cmd, device: device)
            }
            return logits
        }
        let positions = Array(position ..< position + tokenIds.count)
        // forwardBlock commits its own command buffer; the returned
        // logits tensors are resident on the host's read path. The
        // outer driver in Generate.swift uses sampleNext on the final
        // prompt position separately, so we only need the tail logits
        // here for the post-prefill sampling step.
        let perPositionLogits = forwardBlock(
            tokenIds: tokenIds, positions: positions,
            caches: caches, append: true, useLora: false,
            device: device
        )
        return perPositionLogits.last!
    }

    // ─── VLM embedding-input path ────────────────────────────────────
    //
    // NemotronDiffusion is the text backbone of the Nemotron-Labs-
    // Diffusion VLM (see Models/Nemotron.swift `NemotronDiffusionVL`).
    // The VLM splice supplies a `[hidden]` row directly — either a
    // projected vision token or a text-token embedding the VL model
    // looked up. The forward is identical to `forward(tokenId:)`
    // minus the embedding gather. The diffusion + self-speculation
    // paths (`forwardBlock`) are unaffected — they take token IDs and
    // do their own batched embed.

    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "NemotronDiffusionModel.forward(inputEmbedding:): expected "
                     + "[\(hidden)], got \(inputEmbedding.shape)")
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        var h = inputEmbedding.reshaped(to: [hidden])
        workCmd = tap.dumpLayerBoundary(h, label: "embed_in", layer: -1,
                                        cmd: workCmd, device: device)

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: workCmd, device: device)
            workCmd = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                            cmd: workCmd, device: device)
        }

        let normed = finalNorm(h, on: workCmd)
        let logits = lmHead(normed, on: workCmd)
        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    /// Raw embedding-table lookup for one text token. The VLM splice
    /// calls this when filling in text positions of the multimodal
    /// prompt.
    public func textEmbedding(tokenId: Int, device: Device) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0,
                                 shape: [1], dtype: .u32)
        let embed = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        cmd.commit()
        cmd.waitUntilCompleted()
        return embed
    }

    /// Multi-token block forward — the primitive diffusion-denoising and
    /// self-speculation build on. Runs `tokenIds` (at absolute
    /// `positions`) through every layer and returns one `[vocab]` logits
    /// tensor per input position. Commits its own command buffer, so the
    /// returned tensors are valid on return.
    ///
    /// `append == false`: bidirectional denoise draft — K/V staged as
    /// scratch, cache `length` unchanged. `append == true`: causal
    /// commit — K/V appended, `length` bumped by `tokenIds.count`.
    ///
    /// Requires raw `KVCache` per layer (precondition).
    public func forwardBlock(tokenIds: [Int], positions: [Int],
                             caches: [any LayerCacheProtocol],
                             append: Bool, useLora: Bool = false,
                             device: Device = .shared) -> [Tensor] {
        precondition(tokenIds.count == positions.count,
                     "forwardBlock: tokenIds / positions count mismatch")
        let rawCaches: [KVCache] = caches.map {
            guard let kv = $0 as? KVCache else {
                preconditionFailure("NemotronDiffusion diffusion/self-speculation "
                    + "modes require a raw KVCache — load with LoadOptions.kvCache = .raw")
            }
            return kv
        }
        let n = tokenIds.count
        let cmd = device.makeCommandBuffer()

        // Embed all N tokens in one gather → contiguous [n, hidden].
        let tokBuf = device.makeBuffer(length: n * 4)
        let tokPtr = tokBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n { tokPtr[i] = UInt32(tokenIds[i]) }
        let tokTensor = Tensor(buffer: tokBuf, offset: 0, shape: [n], dtype: .u32)
        var hBlock = embedTokens(tokTensor, on: cmd)   // [n, hidden]

        // Causal commit appends K/V; bidirectional denoise stages scratch.
        let causal = append
        for (i, layer) in layers.enumerated() {
            hBlock = layer.forwardTokens(hBlock, n: n, positions: positions,
                                         cache: rawCaches[i],
                                         append: append, causal: causal, useLora: useLora,
                                         cmd: cmd, device: device)
        }

        // Final RMSNorm (rows) + LM head as one block GEMM → [n, vocab].
        let normed = Ops.rmsNormRows(hBlock, weight: finalNorm.weight, eps: finalNorm.eps,
                                     nRows: n, rowSize: hidden, on: cmd)
        let logitsBlock = nemotronBlockProject(lmHead, normed, nRows: n, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        let dt = logitsBlock.dtype
        var logits: [Tensor] = []
        logits.reserveCapacity(n)
        for r in 0..<n {
            logits.append(Tensor(buffer: logitsBlock.buffer,
                                  offset: logitsBlock.offset + r * vocab * dt.byteSize,
                                  shape: [vocab], dtype: dt))
        }
        return logits
    }
}
