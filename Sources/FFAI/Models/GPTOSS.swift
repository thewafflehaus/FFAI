// GPT-OSS family — OpenAI's GPT-OSS-20B mixture-of-experts transformer.
//
// Port of mlx-lm's `gpt_oss.py` (and mlx-swift-lm's `GPTOSS.swift`).
// GPT-OSS-20B is a 24-layer MoE transformer (~20B total params, ~3.6B
// active per token) with three structural features that distinguish it
// from the dense Llama / Qwen3 family files:
//
//   1. ─── Alternating layer schedule ───────────────────────────────
//      `config.layer_types` assigns each decoder layer one attention
//      kind: "sliding_attention" or "full_attention". The published
//      checkpoints alternate strictly (sliding, full, sliding, full,
//      …) but the schedule is read from the config, never assumed.
//      Sliding layers cap attention at `sliding_window` (128) recent
//      positions; full layers attend the whole context.
//
//   2. ─── Learned per-head attention sinks ─────────────────────────
//      Every attention layer carries a learned `self_attn.sinks`
//      vector — one scalar logit per query head. The sink is an extra
//      implicit column in the attention softmax denominator; its V is
//      zero so it contributes nothing to the output accumulator, it
//      only *attenuates* the real attention output. Math:
//
//        M  = max(max_t s_t, sink_h)
//        Z  = Σ_t exp(s_t - M)
//        O  = (Σ_t exp(s_t - M)·V_t) / Z          ← plain SDPA output
//        O' = O · Z / (Z + exp(sink_h - M))       ← sink-corrected
//
//      FFAI's `Ops.sdpaDecode` head_dim=64 kernel is dense-only — it
//      has no learned-sink-logit support (the `sinkEnd`/`windowStart`
//      params on the head_dim=128 variants are KV-*position* bounds,
//      a different feature). So GPT-OSS folds the sink as a per-head
//      *post-hoc rescale*: run plain SDPA for `O`, recover `M` and `Z`
//      per head via a CPU dot-product over the (small) K cache, then
//      multiply `O` by the per-head correction factor. See
//      `GPTOSSAttention.forward`.
//
//   3. ─── Bias-corrected K/V (and Q/O) projections ─────────────────
//      `config.attention_bias == true` — q/k/v/o projections all ship
//      `.bias` tensors. `Linear` applies them; `loadLinear` picks the
//      bias up automatically.
//
// ─── MoE FFN ─────────────────────────────────────────────────────────
//
// Every layer's feed-forward half is a block-sparse MoE: a biased
// router selects top-K of `num_local_experts` experts; the experts run
// a *clipped* SwiGLU (`swiglu_limit`-clamped, α=1.702 swish, with the
// `(linear + 1)` GPT-OSS gating form). The published checkpoints ship
// the experts MXFP4-quantized; `GPTOSSMoE.swift` re-packs MXFP4 to
// FFAI's affine-int4 format at load time. See that file for the codec
// + the command-buffer contract (`GPTOSSMoELayer.decode` commits).
//
// ─── Command-buffer discipline ───────────────────────────────────────
//
// Two things commit the command buffer mid-decode: (a) every layer's
// MoE FFN (`GPTOSSMoELayer.decode`, router CPU readback), and (b) every
// attention layer's sink correction (a CPU readback of K + Q). So a
// GPT-OSS layer ALWAYS commits the buffer it is handed. `GPTOSSModel.
// forward` therefore runs the embedding + every layer on internal
// `workCmd` buffers and queues ONLY the final norm + lm_head onto the
// caller's pristine `cmd` — the Jamba command-buffer discipline.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum GPTOSS {
    public static let modelTypes: Set<String> = ["gpt_oss"]
    public static let architectures: Set<String> = ["GptOssForCausalLM"]

    public static func variant(for _: ModelConfig) throws -> any GPTOSSVariant.Type {
        return GPTOSSMoEVariant.self
    }
}

public protocol GPTOSSVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> GPTOSSModel
}

public enum GPTOSSError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "GPT-OSS: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "GPT-OSS: unsupported config: \(m)"
        }
    }
}

// ─── Attention kind ──────────────────────────────────────────────────

/// The two attention kinds a `layer_types` entry can name.
public enum GPTOSSAttentionKind: Equatable, Sendable {
    case sliding   // "sliding_attention" — capped at `sliding_window`
    case full      // "full_attention"    — attends the whole context

    init(from name: String) throws {
        switch name {
        case "sliding_attention": self = .sliding
        case "full_attention": self = .full
        default:
            throw GPTOSSError.unsupportedConfig(
                "unknown layer_types entry '\(name)'")
        }
    }
}

// ─── GPTOSSMoEVariant — the single GPT-OSS-20B variant ───────────────

public struct GPTOSSMoEVariant: GPTOSSVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// GPT-OSS-20B greedy defaults — keeps the integration suite
    /// deterministic. 2048-token prefill chunk matches mlx-swift-lm's
    /// audited optimum (pure-attention model, no SSM bottleneck).
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 2048,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options _: LoadOptions,
        device: Device
    ) throws -> GPTOSSModel {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let vocab = config.vocabSize
        else {
            throw GPTOSSError.missingConfig(
                "hidden_size / num_hidden_layers / num_attention_heads / vocab_size")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // GPT-OSS ships an explicit head_dim (64) — it is NOT
        // hidden / nHeads (2880 / 64 = 45 ≠ 64). Read it, never derive.
        guard let headDim = config.headDim else {
            throw GPTOSSError.missingConfig("head_dim")
        }
        let eps = Float(config.rmsNormEps ?? 1e-5)
        let intermediate = config.intermediateSize ?? hidden
        let theta = Float(config.ropeTheta ?? 150_000)
        let maxSeq = config.int("max_position_embeddings") ?? 131_072
        let slidingWindow = config.int("sliding_window") ?? 128
        let swigluLimit = Float(config.float("swiglu_limit") ?? 7.0)
        let tieEmbed = config.tieWordEmbeddings

        // ── MoE geometry ──────────────────────────────────────────────
        guard let numExperts = config.int("num_local_experts"),
              let topK = config.int("num_experts_per_tok") ?? config.int("experts_per_token")
        else {
            throw GPTOSSError.missingConfig(
                "num_local_experts / num_experts_per_tok")
        }

        // ── Attention layer schedule ──────────────────────────────────
        // `layer_types` is mandatory on GPT-OSS — the alternating
        // sliding/full schedule is a structural property, not a
        // heuristic. Acting on it (vs silently ignoring it) is the
        // Gemma-4 `num_kv_shared_layers` lesson.
        guard let layerTypeNames = config.raw["layer_types"] as? [String],
              !layerTypeNames.isEmpty
        else {
            throw GPTOSSError.missingConfig("layer_types")
        }
        let attnKinds = try layerTypeNames.map { try GPTOSSAttentionKind(from: $0) }
        guard attnKinds.count == nLayers else {
            throw GPTOSSError.unsupportedConfig(
                "layer_types has \(attnKinds.count) entries, "
                + "num_hidden_layers is \(nLayers)")
        }

        // ── YaRN RoPE — context-extension scaling ─────────────────────
        // GPT-OSS rope_scaling is `rope_type: yarn`. `factor == 1` (no
        // rope_scaling block) collapses ropeYaRN to plain RoPE.
        var yarn = Ops.RoPEYaRN.plain
        if let rs = config.nested("rope_scaling"),
           ((rs["rope_type"] as? String) ?? (rs["type"] as? String)) == "yarn" {
            let factor = Float((rs["factor"] as? Double) ?? 1)
            let betaFast = Float((rs["beta_fast"] as? Double) ?? 32)
            let betaSlow = Float((rs["beta_slow"] as? Double) ?? 1)
            let origMax = Float(
                (rs["original_max_position_embeddings"] as? Int)
                ?? config.int("initial_context_length") ?? 4096)
            yarn = Ops.RoPEYaRN.from(
                headDim: headDim, thetaBase: theta, factor: factor,
                betaFast: betaFast, betaSlow: betaSlow,
                originalMaxPosition: origMax)
        }

        // GPT-OSS attention head_dim is 64 — the d64 SDPA kernel path.
        guard headDim == 64 || headDim == 128 else {
            throw GPTOSSError.unsupportedConfig(
                "head_dim \(headDim) — Ops.sdpaDecode supports {64, 128}")
        }

        // ── Mixed-precision quantization map ──────────────────────────
        // The published checkpoints are MIXED precision: the MoE
        // experts are MXFP4-quantized while the attention / router /
        // embedding / lm_head tensors are mlx *affine*-quantized (4- or
        // 8-bit, group 64). The per-tensor `quantization` block in
        // config.json carries the (bits, group_size) for each one.
        let quantMap = GPTOSSQuantMap(config: config)

        // ── Activation dtype ──────────────────────────────────────────
        // Quantized checkpoints carry activations in the scales' dtype;
        // a raw checkpoint uses the embedding table's dtype.
        let activationDtype: DType
        if weights.has("model.embed_tokens.scales") {
            activationDtype = try weights.tensor(
                named: "model.embed_tokens.scales").dtype
        } else {
            activationDtype = try weights.tensor(
                named: "model.embed_tokens.weight").dtype
        }
        precondition(
            activationDtype == .f16 || activationDtype == .bf16
            || activationDtype == .f32,
            "GPT-OSS: unexpected activation dtype \(activationDtype)")

        // ── Embedding — affine-quantized or raw ───────────────────────
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights, hidden: hidden,
            quantization: quantMap.config(for: "model.embed_tokens"))

        // ── Per-layer construction ────────────────────────────────────
        var layers: [GPTOSSLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in attnKinds.enumerated() {
            let p = "model.layers.\(i)"

            // Attention — q/k/v/o all carry projection biases
            // (`attention_bias`) and may be affine-quantized.
            let qProj = try loadGPTOSSBiasedLinear(
                base: "\(p).self_attn.q_proj", in: weights, quantMap: quantMap,
                dtype: activationDtype, device: device)
            let kProj = try loadGPTOSSBiasedLinear(
                base: "\(p).self_attn.k_proj", in: weights, quantMap: quantMap,
                dtype: activationDtype, device: device)
            let vProj = try loadGPTOSSBiasedLinear(
                base: "\(p).self_attn.v_proj", in: weights, quantMap: quantMap,
                dtype: activationDtype, device: device)
            let oProj = try loadGPTOSSBiasedLinear(
                base: "\(p).self_attn.o_proj", in: weights, quantMap: quantMap,
                dtype: activationDtype, device: device)

            // Learned per-head sink logits — [nHeads], read to host.
            let sinkTensor = try weights.tensor(named: "\(p).self_attn.sinks")
            precondition(sinkTensor.elementCount == nHeads,
                         "GPT-OSS: self_attn.sinks expected [\(nHeads)], "
                         + "got \(sinkTensor.shape)")
            let sinks = readGPTOSSFloats(sinkTensor)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: eps)
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: eps)

            let attn = GPTOSSAttention(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                sinks: sinks, kind: kind,
                nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                ropeTheta: theta, yarn: yarn)

            // MoE feed-forward — re-packs MXFP4 experts to affine int4.
            let moe = try buildGPTOSSMoE(
                prefix: "\(p).mlp", weights: weights, quantMap: quantMap,
                hidden: hidden, intermediate: intermediate,
                numExperts: numExperts, topK: topK,
                swigluLimit: swigluLimit,
                dtype: activationDtype, device: device)

            layers.append(GPTOSSLayer(
                attention: attn, moe: moe,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: hidden))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"), eps: eps)

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(
                base: "lm_head", in: weights,
                quantization: quantMap.config(for: "lm_head"))
        } else if weights.isQuantized("model.embed_tokens"),
                  let q = quantMap.config(for: "model.embed_tokens") {
            // Tied + quantized — reuse the embedding triplet as lm_head.
            let t = try weights.quantizedTriplet("model.embed_tokens")
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        return GPTOSSModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            attnKinds: attnKinds,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, slidingWindow: slidingWindow,
            dtype: activationDtype)
    }
}

// ─── GPTOSSQuantMap — per-tensor mixed-precision resolver ────────────
//
// GPT-OSS-20B checkpoints are MIXED precision: the MoE experts are
// MXFP4, every other quantized tensor (attention / router / embedding /
// lm_head) is mlx *affine*. config.json's `quantization` block carries
// a per-tensor `{ bits, group_size }` map; `GPTOSSQuantMap` resolves a
// tensor base to its affine `QuantizationConfig` (or `nil` when the
// tensor is unquantized — a raw bf16/f16 checkpoint).

struct GPTOSSQuantMap {
    /// Per-tensor `(bits, groupSize)` keyed by tensor base name.
    private let perTensor: [String: ModelConfig.QuantizationConfig]

    init(config: ModelConfig) {
        var map: [String: ModelConfig.QuantizationConfig] = [:]
        if let q = config.nested("quantization") {
            for (key, value) in q {
                guard let entry = value as? [String: Any],
                      let bits = entry["bits"] as? Int,
                      let group = (entry["group_size"] as? Int)
                          ?? (q["group_size"] as? Int)
                else { continue }
                map[key] = ModelConfig.QuantizationConfig(
                    bits: bits, groupSize: group)
            }
        }
        self.perTensor = map
    }

    /// The affine quant config for `base`, or `nil` if `base` is not an
    /// affine-quantized tensor (raw, or MXFP4 — handled separately).
    /// Only mlx-supported bit widths (3/4/5/6/8) are returned.
    func config(for base: String) -> ModelConfig.QuantizationConfig? {
        guard let c = perTensor[base],
              [3, 4, 5, 6, 8].contains(c.bits)
        else { return nil }
        return c
    }
}

// ─── GPTOSSBiasedLinear — quant/raw linear + projection bias ─────────
//
// GPT-OSS attention projections carry a separate `.bias` *projection*
// bias term (distinct from a quantizer's `.biases` zero-point). FFAI's
// `QuantizedLinear` has no bias field, so this wrapper holds an
// `AnyLinear` (quantized or raw) plus an optional projection bias and
// applies the bias after the matmul.

public final class GPTOSSBiasedLinear: Module {
    let linear: AnyLinear
    /// Optional projection bias `[outDim]`, in the activation dtype.
    let bias: Tensor?

    init(linear: AnyLinear, bias: Tensor?) {
        self.linear = linear
        self.bias = bias
    }

    public func parameters() -> [(String, Tensor)] {
        var out = linear.parameters()
        if let b = bias { out.append(("bias", b)) }
        return out
    }

    public func callAsFunction(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let y = linear(x, on: cmd)
        if let b = bias { return Ops.add(y, b, on: cmd) }
        return y
    }
}

/// Load a GPT-OSS attention projection: an affine-`QuantizedLinear` (or
/// raw `Linear`) plus its separate `.bias` projection bias. The bias is
/// cast to the activation dtype so the post-matmul `Ops.add` is dtype-
/// consistent (a quantized matmul outputs the activation dtype).
private func loadGPTOSSBiasedLinear(
    base: String, in weights: SafeTensorsBundle,
    quantMap: GPTOSSQuantMap, dtype: DType, device: Device
) throws -> GPTOSSBiasedLinear {
    let inner: AnyLinear
    if let q = quantMap.config(for: base), weights.isQuantized(base) {
        let t = try weights.quantizedTriplet(base)
        inner = AnyLinear(QuantizedLinear(
            weight: t.weight, scales: t.scales, biases: t.biases,
            bits: q.bits, groupSize: q.groupSize))
    } else {
        inner = AnyLinear(Linear(
            weight: try weights.tensor(named: "\(base).weight")))
    }

    var bias: Tensor? = nil
    if weights.has("\(base).bias") {
        let raw = try weights.tensor(named: "\(base).bias")
        bias = castGPTOSSTensor(raw, to: dtype, device: device)
    }
    return GPTOSSBiasedLinear(linear: inner, bias: bias)
}

/// Cast a tensor to `dtype` via a host round-trip if needed.
func castGPTOSSTensor(_ t: Tensor, to dtype: DType, device: Device) -> Tensor {
    if t.dtype == dtype { return t }
    let floats = readGPTOSSFloats(t)
    let out = Tensor.empty(shape: t.shape, dtype: dtype, device: device)
    writeGPTOSSFloats(floats, into: out)
    return out
}

/// Read an f16 / bf16 / f32 tensor into a host `[Float]`.
func readGPTOSSFloats(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    case .bf16:
        return t.toArray(as: UInt16.self).map {
            Float(bitPattern: UInt32($0) << 16)
        }
    default:
        fatalError("GPT-OSS: unsupported dtype for host read: \(t.dtype)")
    }
}

// ─── GPTOSSAttention — multi-head attention with learned sinks ───────
//
// Single-token decode attention. The sink fold runs as a per-head
// post-hoc rescale of the plain SDPA output (see the file header for
// why the d64 kernel can't do it natively, and the correction math).

public final class GPTOSSAttention: Module {
    let qProj, kProj, vProj, oProj: GPTOSSBiasedLinear
    /// Learned per-head sink logits, `[nHeads]`, host-resident.
    let sinks: [Float]
    let kind: GPTOSSAttentionKind
    let nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let yarn: Ops.RoPEYaRN
    let scale: Float

    init(qProj: GPTOSSBiasedLinear, kProj: GPTOSSBiasedLinear,
         vProj: GPTOSSBiasedLinear, oProj: GPTOSSBiasedLinear,
         sinks: [Float], kind: GPTOSSAttentionKind,
         nHeads: Int, nKVHeads: Int, headDim: Int,
         ropeTheta: Float, yarn: Ops.RoPEYaRN) {
        self.qProj = qProj; self.kProj = kProj
        self.vProj = vProj; self.oProj = oProj
        self.sinks = sinks; self.kind = kind
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.ropeTheta = ropeTheta; self.yarn = yarn
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        return out
    }

    /// Single-token attention forward. Queues the projections / RoPE /
    /// KV append / SDPA onto `cmd`, then COMMITS `cmd` to read back K
    /// and the rotated Q for the host-side sink correction. Returns the
    /// post-`o_proj` contribution on a fresh, locally-committed buffer
    /// so the result is fully resident (the residual add is the
    /// caller's job).
    func forward(_ xNorm: Tensor, position: Int, cache: KVCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── GPU phase: project, RoPE, append, plain SDPA ──────────────
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRot = Ops.ropeYaRN(q.reshaped(to: [nHeads * headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, yarn: yarn, on: cmd)
        let kRot = Ops.ropeYaRN(k.reshaped(to: [nKVHeads * headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, yarn: yarn, on: cmd)

        cache.appendOnGPU(kFlat: kRot.reshaped(to: [nKVHeads, headDim]),
                          vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let nKV = cache.length
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRot.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: nKV, kvStride: cache.maxSeq,
            scale: scale, on: cmd)

        // Commit so the host can read the rotated Q + the K cache for
        // the sink correction below.
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Host phase: per-head sink-correction factor ───────────────
        // For each query head h, recover the softmax max `M_h` and sum
        // `Z_h` over the nKV attended positions, then scale the SDPA
        // output by  Z_h / (Z_h + exp(sink_h - M_h)).  See the file
        // header for the derivation. K cache layout: [nKVHeads, maxSeq,
        // headDim]; query head h maps to kv head h / headsPerGroup.
        //
        // Only the LIVE [0, nKV) slice of each kv head is read back —
        // never the full maxSeq-capacity buffer (which can be 100k+
        // positions; copying it every token would dwarf the kernel).
        let qHost = readGPTOSSFloats(qRot)              // [nHeads*headDim]
        let headsPerGroup = nHeads / nKVHeads
        // Per-kv-head live K slice [nKV, headDim], read once and reused
        // across the headsPerGroup query heads that share it.
        var kLive = [[Float]](repeating: [], count: nKVHeads)
        if nKV > 0 {
            for kvHead in 0..<nKVHeads {
                // kBuffer is [nKVHeads, maxSeq, headDim]; slice head
                // kvHead, then its first nKV timesteps.
                let headSlab = cache.kBuffer
                    .slicedRows(start: kvHead, count: 1)
                    .reshaped(to: [cache.maxSeq, headDim])
                    .slicedRows(start: 0, count: nKV)
                kLive[kvHead] = readGPTOSSFloats(headSlab)
            }
        }

        // Online softmax over the nKV positions (initial running max =
        // sinks[h]; sink term folded in after the loop). Single pass
        // over the dot products — the previous implementation iterated
        // twice (once for the max, once for the sum of exp), doubling
        // the per-head FLOPs at long contexts. Each head's work is
        // independent (writes factors[h] only) so the head loop is
        // parallelised across (head) work items.
        var factors = [Float](repeating: 1, count: nHeads)
        factors.withUnsafeMutableBufferPointer { fBuf in
            let fPtr = fBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads) { h in
                let kvHead = h / headsPerGroup
                let qBase = h * headDim
                let kHead = kLive[kvHead]
                var m = sinks[h]
                var z: Float = 0
                for t in 0..<nKV {
                    let kBase = t * headDim
                    var dot: Float = 0
                    for d in 0..<headDim {
                        dot += qHost[qBase + d] * kHead[kBase + d]
                    }
                    let s = dot * scale
                    if s > m {
                        // Renormalise the running sum to the new max.
                        z = z * Foundation.exp(m - s)
                        m = s
                    }
                    z += Foundation.exp(s - m)
                }
                let sinkTerm = Foundation.exp(sinks[h] - m)
                // O' = O · Z / (Z + exp(sink - M)). Z > 0 whenever nKV > 0.
                fPtr[h] = z / (z + sinkTerm)
            }
        }

        // ── GPU phase 2: rescale O per-head, then o_proj ──────────────
        let phase2 = device.makeCommandBuffer()
        // Broadcast the per-head scalar across each head's headDim slice.
        let factorTensor = Tensor.empty(shape: [nHeads * headDim],
                                        dtype: attnOut.dtype, device: device)
        var factorFlat = [Float](repeating: 0, count: nHeads * headDim)
        for h in 0..<nHeads {
            for d in 0..<headDim { factorFlat[h * headDim + d] = factors[h] }
        }
        writeGPTOSSFloats(factorFlat, into: factorTensor)
        let corrected = Ops.mul(attnOut.reshaped(to: [nHeads * headDim]),
                                factorTensor, on: phase2)
        let result = oProj(corrected, on: phase2)
        phase2.commit()
        phase2.waitUntilCompleted()
        return result
    }
}

/// Write a host `[Float]` into an existing tensor, converting to its
/// dtype. bf16 uses round-to-nearest before truncating the low bits.
func writeGPTOSSFloats(_ values: [Float], into t: Tensor) {
    precondition(values.count == t.elementCount,
                 "GPT-OSS: writeFloats size mismatch")
    switch t.dtype {
    case .f32:
        t.copyIn(from: values)
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    case .bf16:
        t.copyIn(from: values.map { v -> UInt16 in
            let bits = v.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(rounded >> 16)
        })
    default:
        fatalError("GPT-OSS: unsupported dtype for host write: \(t.dtype)")
    }
}

// ─── GPTOSSLayer — one decoder layer (attention + MoE) ───────────────
//
// Pre-norm transformer block:
//
//   residual = h
//   h        = attention(input_layernorm(h))
//   h        = residual + h
//   residual = h
//   h        = moe(post_attention_layernorm(h))
//   out      = residual + h
//
// `GPTOSSAttention.forward` and `GPTOSSMoELayer.decode` both commit the
// command buffer (the sink correction + the MoE router both need a CPU
// sync). A layer therefore ALWAYS leaves the buffer it was handed in a
// committed state; `GPTOSSModel.forward` refreshes `workCmd` after
// every layer.

public final class GPTOSSLayer: Module {
    let attention: GPTOSSAttention
    let moe: GPTOSSMoELayer
    let inputNorm, postAttnNorm: RMSNorm
    let hidden: Int

    init(attention: GPTOSSAttention, moe: GPTOSSMoELayer,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm, hidden: Int) {
        self.attention = attention
        self.moe = moe
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        out.append(contentsOf: attention.parameters())
        for (k, v) in moe.parameters() { out.append(("mlp.\(k)", v)) }
        return out
    }

    /// Single-token decode. `cmd` is handed in by the host model; this
    /// layer commits it (both halves require a CPU sync) and returns a
    /// fully-resident hidden state on a fresh, locally-committed buffer.
    ///
    /// Input norm runs HOST-side: GPT-OSS `hidden = 2880` is not a
    /// multiple of 128, so the standalone GPU `Ops.rmsNorm` kernel
    /// (which needs a 128-aligned row for its 32-lane × 4-element
    /// reduction) cannot apply. A single-token norm over 2880 floats
    /// on the CPU is negligible next to the per-layer GPU gemms, and
    /// the layer already has CPU sync points (sinks + router) so the
    /// readback is free. The post-attention norm runs on the GPU
    /// because it is now fused into `mt_add_rms_norm`, whose row-
    /// width invariants (multiple of 4, ≤ 4096) GPT-OSS satisfies.
    func decode(_ h: Tensor, position: Int, cache: KVCache,
                cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── Attention half ────────────────────────────────────────────
        // `h` is resident on entry (the previous layer committed). The
        // input norm runs on the host and writes `xNorm` directly.
        let xNorm = gptOSSHostRMSNorm(h, weight: inputNorm.weight,
                                      eps: inputNorm.eps, device: device)
        // attention.forward commits `cmd`; the returned tensor is
        // resident on a fresh buffer it owns.
        let attnOut = attention.forward(xNorm, position: position,
                                        cache: cache, cmd: cmd, device: device)

        // The residual add + post-attention RMSNorm run as ONE fused
        // GPU dispatch via mt_add_rms_norm — replaces the previous
        // (Ops.add on GPU + gptOSSHostRMSNorm on CPU) pair. GPT-OSS's
        // hidden=2880 is a multiple of 4 and ≤ 4096, so it fits the
        // fused kernel's row-width invariants even though it does NOT
        // satisfy the legacy `mt_rms_norm`'s 128-element invariant
        // (which is why the standalone norm was on the host).
        precondition(OpsValidation.validateAddRmsNorm(n: hidden) == nil,
                     "GPTOSSLayer.decode: hidden=\(hidden) violates "
                     + "mt_add_rms_norm invariants")
        let addCmd = device.makeCommandBuffer()
        let fused = Ops.addAndRmsNorm(
            h, attnOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
            nRows: 1, rowSize: hidden, on: addCmd)
        let postAttn = fused.residual
        let ffnNorm = fused.normed
        addCmd.commit()
        addCmd.waitUntilCompleted()
        // moe.decode commits its own buffers; run the final residual
        // add on a fresh, locally-committed buffer so the result is
        // resident.
        let moeCmd = device.makeCommandBuffer()
        let moeOut = moe.decode(ffnNorm, cmd: moeCmd, device: device)
        let outCmd = device.makeCommandBuffer()
        let result = Ops.add(postAttn, moeOut, on: outCmd)
        outCmd.commit()
        outCmd.waitUntilCompleted()
        return result
    }
}

/// Host-side weighted RMSNorm of a resident `[n]` tensor. GPT-OSS's
/// `hidden = 2880` is not 128-aligned, so the GPU `Ops.rmsNorm` kernel
/// cannot run; a single-token norm is trivial on the CPU. Returns a new
/// resident tensor in `x`'s dtype.
///
/// `out[i] = x[i] / sqrt(mean(x²) + eps) · weight[i]`.
func gptOSSHostRMSNorm(_ x: Tensor, weight: Tensor, eps: Float,
                       device: Device) -> Tensor {
    let xs = readGPTOSSFloats(x)
    let ws = readGPTOSSFloats(weight)
    precondition(xs.count == ws.count,
                 "GPT-OSS RMSNorm: x (\(xs.count)) / weight (\(ws.count)) "
                 + "size mismatch")
    var sumSq: Float = 0
    for v in xs { sumSq += v * v }
    let inv = 1.0 / (sumSq / Float(xs.count) + eps).squareRoot()
    var out = [Float](repeating: 0, count: xs.count)
    for i in 0..<xs.count { out[i] = xs[i] * inv * ws[i] }
    let result = Tensor.empty(shape: x.shape, dtype: x.dtype, device: device)
    writeGPTOSSFloats(out, into: result)
    return result
}

// ─── GPTOSSModel ─────────────────────────────────────────────────────

public final class GPTOSSModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [GPTOSSLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    /// GPT-OSS 20B prefills 2048 tokens per chunk — the value tuned in
    /// `mlx-swift-lm`'s `Libraries/MLXLLM/Models/GPTOSS.swift`. Larger
    /// than the 1024 dense default because GPT-OSS's MoE FFN amortises
    /// well across more rows.
    public var defaultPrefillStepSize: Int { 2048 }

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    /// Sliding-window size for sliding-attention layers.
    public let slidingWindow: Int
    public let dtype: DType
    /// Attention kind per layer index — drives `makeLayerCaches`
    /// (sliding layers get a `.window` eviction cache).
    public let attnKinds: [GPTOSSAttentionKind]

    init(embedTokens: AnyEmbedding, layers: [GPTOSSLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         attnKinds: [GPTOSSAttentionKind],
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, slidingWindow: Int, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.attnKinds = attnKinds
        self.hidden = hidden; self.nLayers = nLayers; self.nHeads = nHeads
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.vocab = vocab
        self.maxSeq = maxSeq; self.slidingWindow = slidingWindow
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
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

    /// One KV cache per layer index. Sliding-attention layers get a
    /// `.window(slidingWindow)` eviction cache (the ring buffer
    /// physically caps the K/V at `slidingWindow` recent positions, so
    /// the plain dense `Ops.sdpaDecode` over the windowed cache is the
    /// correct sliding-window attention — no kernel-level window
    /// fast-path needed). Full-attention layers stay unbounded.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return attnKinds.map { kind in
            switch kind {
            case .sliding:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype,
                    eviction: .window(maxSize: min(slidingWindow, cap), keep: 0),
                    device: device)
            case .full:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, eviction: .unbounded, device: device)
            }
        }
    }

    /// Queue a single-token forward pass. **Does not commit `cmd`** —
    /// the `LanguageModel` default `forwardSample` extension composes
    /// the output kernel onto `cmd` and commits once.
    ///
    /// CRITICAL — command-buffer contract (the Jamba discipline). Every
    /// GPT-OSS layer commits the buffer it is handed (the sink
    /// correction + the MoE router both need a CPU sync). The caller's
    /// `cmd` must NEVER be handed to a layer — if it were, the first
    /// layer would commit it and the caller's later commit would
    /// double-commit. So the embedding + every layer run on internal
    /// `workCmd` buffers (refreshed after each committing layer) and
    /// ONLY the final norm + lm_head queue onto the caller's pristine
    /// `cmd`. The hidden state handed to the final norm is resident
    /// (the last layer committed its buffer).
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // Embedding runs on an internal buffer — never `cmd`. Commit it
        // so `h` is resident before layer 0's host-side input RMSNorm
        // reads it.
        let embedCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: embedCmd).reshaped(to: [hidden])
        embedCmd.commit()
        embedCmd.waitUntilCompleted()

        for (i, layer) in layers.enumerated() {
            guard let kv = caches[i] as? KVCache else {
                fatalError("GPTOSSModel: expected KVCache at layer \(i), "
                           + "got \(type(of: caches[i]))")
            }
            // Each layer runs on its own internal buffers (the attention
            // sink correction + the MoE router both commit), so it gets
            // a fresh buffer and returns a fully-resident `h`.
            let layerCmd = device.makeCommandBuffer()
            h = layer.decode(h, position: position, cache: kv,
                             cmd: layerCmd, device: device)
        }

        // `h` is resident (the last layer committed). The final RMSNorm
        // runs host-side (hidden=2880 is not 128-aligned); lm_head
        // queues onto the caller's pristine `cmd`.
        let normed = gptOSSHostRMSNorm(h, weight: finalNorm.weight,
                                       eps: finalNorm.eps, device: device)
        return lmHead(normed, on: cmd)
    }

    /// Multi-token forward — Phase 6.6 prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// GPT-OSS's per-token forward already commits the command buffer
    /// twice per layer (once for the host-side learned-sink correction
    /// in `GPTOSSAttention.forward`, once for the MoE-router CPU
    /// readback in `GPTOSSMoELayer.decode`). Collapsing the per-token
    /// SDPA into a chunked `sdpaMulti` would also require:
    /// 1. Porting the learned-sink fold to head_dim=64 metaltile-side
    ///    so the sink correction stays on-GPU (today it's host-side
    ///    per token; chunking would amplify the readback cost).
    /// 2. A batched MoE router + per-expert dispatch — today the
    ///    router commits the cmd, picks experts CPU-side, then
    ///    dispatches each.
    /// Until both land, this override is commit-count-batched only —
    /// the per-token forward stays as-is, with `Generate.driveGeneration`
    /// avoiding the per-token commit/wait on the outer prefill loop.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "GPTOSSModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(tokenId: tok, position: position + i,
                             caches: caches, on: cmd, device: device)
        }
        return logits
    }
}
