// Gemma 4 family — Google's Gemma 4 text decoder. A Phase 6 dense /
// PLE / MoE attention family. Three checkpoint shapes ship under the
// single `gemma4` / `gemma4_text` model_type, distinguished by config:
//
//   • Gemma4Dense  — 31B size. Gemma-style backbone, no PLE.
//   • Gemma4E      — E2B / E4B sizes. Adds Per-Layer Embeddings (PLE)
//                    and KV-sharing on the tail layers.
//   • Gemma4MoE    — 26B-A4B. Mixture-of-experts feed-forward.
//
// Variant detection is purely config-driven (see `Gemma4.variant`):
// `enable_moe_block` ⇒ MoE; `hidden_size_per_layer_input > 0` ⇒ E;
// otherwise Dense.
//
// ─── Architecture vs Gemma 3 ─────────────────────────────────────────
//
// Gemma 4 keeps Gemma 3's backbone shape — four per-block norms,
// per-head q/k norms, sqrt(hidden) embed scale, GELU MLP, tied
// embeddings — and adds:
//
//   0. **Plain RMSNorm.** Gemma 4 drops the Gemma 1/2/3 `(1 + weight)`
//      unit-offset convention: norm weight is initialised to ones and
//      applied as a plain `x/rms(x) · weight` (no `+1` fold at load).
//
//   1. **Two attention geometries.** `layer_types` labels each layer
//      `sliding_attention` or `full_attention`. Sliding layers use
//      `head_dim` (256) and standard RoPE; full (global) layers use
//      `global_head_dim` (512) and *ProportionalRoPE* — only the first
//      `partial_rotary_factor · global_head_dim` (128) dims rotate, and
//      the rotated pairs span the full head (`(i, i + 512/2)`), not the
//      first 128 dims.
//
//   2. **value RMSNorm.** Beyond q_norm / k_norm, the V projection is
//      passed through a *scale-free* RMSNorm (weight = 1) before SDPA.
//
//   3. **SDPA scale = 1.0** — Gemma 4 folds no `1/sqrt(d)` into the
//      score; the pre-attention scaling lives entirely in the q_norm.
//
//   4. **Per-layer scalar.** Each block's output is multiplied by a
//      learned `layer_scalar` (shape [1]).
//
//   5. **Per-Layer Embeddings (PLE) — Gemma4E only.** A second, small
//      embedding table (`embed_tokens_per_layer`, dim
//      `hidden_size_per_layer_input` per layer) is mixed into every
//      block: `h = h + post_norm(per_layer_projection(gelu(
//      per_layer_input_gate(h)) * per_layer_input))`.
//
//   6. **Logit soft-capping.** Final logits pass through
//      `softcap · tanh(logits / softcap)` (`final_logit_softcapping`,
//      30.0 across the family).
//
// ─── Attention path ─────────────────────────────────────────────────
//
// FFAI's `Ops.sdpaDecode` supports head_dim ∈ {64, 128, 256, 512}.
// Sliding layers (head_dim 256) take the d256 GPU kernel; global layers
// (head_dim 512) take the d512 specialization. Both attention paths are
// pure-GPU single-token decode dispatches — no host readback.
//
// ─── KV sharing (Gemma4E) ────────────────────────────────────────────
//
// `num_kv_shared_layers` makes the tail N layers reuse an earlier
// layer's K/V instead of computing their own. The donor is the last
// non-shared layer of the same `layer_types` kind. For first light we
// keep every layer's own KV cache and recompute K/V — correctness
// first; the sharing optimisation (skip k/v_proj on shared layers) is
// deferred. The shared layers' checkpoints still carry k_proj/v_proj
// weights, so recomputation is exact.

import Foundation
import Metal
import MetalTileSwift

public enum Gemma4 {
    public static let modelTypes: Set<String> = ["gemma4", "gemma4_text"]
    public static let architectures: Set<String> = [
        "Gemma4ForCausalLM", "Gemma4TextForCausalLM",
        "Gemma4ForConditionalGeneration",
    ]

    /// Resolve the concrete variant from config. MoE wins over PLE wins
    /// over plain dense.
    public static func variant(for config: ModelConfig) throws -> any Gemma4Variant.Type {
        let tc = Gemma4Config.textConfig(config)
        if (tc["enable_moe_block"] as? Bool) ?? false {
            return Gemma4MoE.self
        }
        if let ple = tc["hidden_size_per_layer_input"] as? Int, ple > 0 {
            return Gemma4E.self
        }
        return Gemma4Dense.self
    }
}

public enum Gemma4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedHeadDim(Int)
    case unalignedNorm(Int)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Gemma4: required config field missing: \(f)"
        case .unsupportedHeadDim(let d):
            return "Gemma4: head_dim \(d) unsupported (Ops.sdpaDecode needs 64/128/256/512)"
        case .unalignedNorm(let n):
            return "Gemma4: norm row size \(n) must be 128-aligned"
        }
    }
}

// MARK: - Config helpers

/// Gemma 4 ships its text fields under a nested `text_config` block
/// (the checkpoints are VLM conversions). This pulls the right
/// dictionary and exposes typed accessors with the family defaults.
enum Gemma4Config {
    /// The text-config dictionary — nested `text_config` if present,
    /// else the top-level config.
    static func textConfig(_ config: ModelConfig) -> [String: Any] {
        if let tc = config.nested("text_config") { return tc }
        return config.raw
    }
}

/// Strongly-typed view over a Gemma 4 text config.
struct Gemma4Params {
    let hidden: Int
    let nLayers: Int
    let nHeads: Int
    let kvHeads: Int          // sliding-layer KV head count
    let globalKvHeads: Int    // full-layer KV head count
    let headDim: Int          // sliding-layer head dim
    let globalHeadDim: Int    // full-layer head dim
    let intermediate: Int
    let eps: Double
    let vocab: Int
    let maxSeq: Int
    let slidingWindow: Int
    let layerTypes: [String]
    let ropeThetaSliding: Float
    let ropeThetaGlobal: Float
    let partialRotaryFactor: Float
    let finalLogitSoftcapping: Float?
    let tieEmbed: Bool
    let attentionKEqV: Bool
    let useDoubleWideMlp: Bool
    let numKvSharedLayers: Int
    // PLE (0 when absent)
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int
    // MoE (0 when absent)
    let numExperts: Int
    let topKExperts: Int
    let moeIntermediate: Int

    init(_ config: ModelConfig) throws {
        let tc = Gemma4Config.textConfig(config)
        func i(_ k: String) -> Int? { tc[k] as? Int }
        func f(_ k: String) -> Float? {
            if let d = tc[k] as? Double { return Float(d) }
            if let n = tc[k] as? Int { return Float(n) }
            return nil
        }
        func b(_ k: String) -> Bool? { tc[k] as? Bool }

        guard let hidden = i("hidden_size") else { throw Gemma4Error.missingConfig("hidden_size") }
        guard let nLayers = i("num_hidden_layers") else { throw Gemma4Error.missingConfig("num_hidden_layers") }
        guard let nHeads = i("num_attention_heads") else { throw Gemma4Error.missingConfig("num_attention_heads") }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw Gemma4Error.missingConfig("layer_types")
        }
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.kvHeads = i("num_key_value_heads") ?? nHeads
        // globalKvHeads: null in E2B/E4B → fall back to kvHeads.
        self.globalKvHeads = i("num_global_key_value_heads") ?? (i("num_key_value_heads") ?? nHeads)
        self.headDim = i("head_dim") ?? (hidden / nHeads)
        self.globalHeadDim = i("global_head_dim") ?? (i("head_dim") ?? (hidden / nHeads))
        self.intermediate = i("intermediate_size") ?? (4 * hidden)
        self.eps = (tc["rms_norm_eps"] as? Double) ?? 1e-6
        self.vocab = i("vocab_size") ?? 262144
        self.maxSeq = i("max_position_embeddings") ?? 131072
        self.slidingWindow = i("sliding_window") ?? 1024
        self.layerTypes = layerTypes
        self.finalLogitSoftcapping = f("final_logit_softcapping")
        self.tieEmbed = b("tie_word_embeddings") ?? true
        self.attentionKEqV = b("attention_k_eq_v") ?? false
        self.useDoubleWideMlp = b("use_double_wide_mlp") ?? false
        self.numKvSharedLayers = i("num_kv_shared_layers") ?? 0
        self.hiddenSizePerLayerInput = i("hidden_size_per_layer_input") ?? 0
        self.vocabSizePerLayerInput = i("vocab_size_per_layer_input") ?? 262144
        self.numExperts = i("num_experts") ?? 0
        self.topKExperts = i("top_k_experts") ?? 0
        self.moeIntermediate = i("moe_intermediate_size") ?? self.intermediate

        // RoPE: Gemma 4 nests theta under `rope_parameters.{sliding,full}_attention`.
        var thetaSliding: Float = 10_000
        var thetaGlobal: Float = 1_000_000
        var partial: Float = 0.25
        if let rp = tc["rope_parameters"] as? [String: Any] {
            if let s = rp["sliding_attention"] as? [String: Any],
               let t = s["rope_theta"] as? Double {
                thetaSliding = Float(t)
            }
            if let g = rp["full_attention"] as? [String: Any] {
                if let t = g["rope_theta"] as? Double { thetaGlobal = Float(t) }
                if let pr = g["partial_rotary_factor"] as? Double { partial = Float(pr) }
            }
        } else {
            if let t = f("rope_theta") { thetaSliding = t }
            if let t = f("global_rope_theta") { thetaGlobal = t }
            if let pr = f("partial_rotary_factor") { partial = pr }
        }
        self.ropeThetaSliding = thetaSliding
        self.ropeThetaGlobal = thetaGlobal
        self.partialRotaryFactor = partial
    }

    /// Layer `i` is a global (full-attention) layer.
    func isGlobal(_ i: Int) -> Bool { layerTypes[i] == "full_attention" }

    /// First layer index that shares KV with an earlier donor. Layers
    /// `[0, firstKvSharedIdx)` compute their own K/V; layers
    /// `[firstKvSharedIdx, nLayers)` reuse a donor's. Equals `nLayers`
    /// (no sharing) when `numKvSharedLayers == 0`.
    var firstKvSharedIdx: Int { nLayers - numKvSharedLayers }

    /// KV-sharing donor map: `previousKVs[i]` is the layer index whose
    /// K/V layer `i` reuses. `previousKVs[i] == i` means layer `i`
    /// computes its own K/V.
    ///
    /// Gemma 4's `num_kv_shared_layers` makes the final `numKvSharedLayers`
    /// layers reuse an earlier layer's K/V instead of projecting their
    /// own. The donor is the **last** non-shared layer of the *same
    /// attention geometry* (`sliding_attention` / `full_attention`) —
    /// a sliding shared layer reuses the last sliding donor, a global
    /// shared layer the last global donor. Matches `mlx_lm`'s
    /// `gemma4_text.py` `previous_kvs` construction.
    ///
    /// The shared layers' `k_proj` / `v_proj` / `k_norm` weights still
    /// ship in the checkpoint but are **dead** — the reference model
    /// (and `mlx_lm`) does not instantiate them. Projecting K/V from
    /// those dead weights instead of reusing the donor's cached K/V
    /// corrupts attention for every shared layer.
    var previousKVs: [Int] {
        var mapping = Array(0..<nLayers)
        guard numKvSharedLayers > 0 else { return mapping }
        // Last non-shared layer of each attention type.
        var lastDonorByType: [String: Int] = [:]
        for i in 0..<firstKvSharedIdx {
            lastDonorByType[layerTypes[i]] = i
        }
        for j in firstKvSharedIdx..<nLayers {
            if let donor = lastDonorByType[layerTypes[j]] {
                mapping[j] = donor
            }
        }
        return mapping
    }
}

// MARK: - Variant protocol

public protocol Gemma4Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma4Model
}

public extension Gemma4Variant {
    static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    static var defaultGenerationParameters: GenerationParameters {
        // Gemma 4: 4096-token prefill chunk is the audited family
        // optimum (pure-attention backbone, no SSM bottleneck).
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }
}

// MARK: - Gemma4Dense / Gemma4E / Gemma4MoE

/// Plain dense Gemma 4 (31B): Gemma-style backbone, no PLE, dense MLP.
public struct Gemma4Dense: Gemma4Variant {
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Gemma4Model {
        try Gemma4Loader.load(config: config, weights: weights,
                              options: options, device: device)
    }
}

/// Gemma 4 E (E2B / E4B): adds Per-Layer Embeddings.
public struct Gemma4E: Gemma4Variant {
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Gemma4Model {
        try Gemma4Loader.load(config: config, weights: weights,
                              options: options, device: device)
    }
}

/// Gemma 4 MoE (26B-A4B): mixture-of-experts feed-forward.
public struct Gemma4MoE: Gemma4Variant {
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Gemma4Model {
        try Gemma4Loader.load(config: config, weights: weights,
                              options: options, device: device)
    }
}

// MARK: - Loader

/// Shared loader for all three variants — they only differ in which
/// per-layer modules are populated, all of which are config-driven.
enum Gemma4Loader {
    /// Weight-key prefix. Gemma 4 checkpoints are VLM conversions that
    /// nest the text tower under `language_model.model.`; a plain text
    /// checkpoint would use `model.`. We probe for the embed tensor.
    static func resolvePrefix(_ weights: SafeTensorsBundle) -> String {
        if weights.has("language_model.model.embed_tokens.weight") {
            return "language_model.model."
        }
        return "model."
    }

    static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Gemma4Model {
        let p = try Gemma4Params(config)
        let prefix = resolvePrefix(weights)
        let quant = config.quantization

        // Every RMSNorm row goes through the `rmsNorm` / `rmsNormRows`
        // kernels, whose only row-width invariant is 128-alignment
        // (simdgroup granularity); rows wider than 4096 route to the
        // wide-row kernel automatically. Checked dims:
        //   • q/k/v head-dim norms — `headDim` / `globalHeadDim`.
        //   • the four per-block norms + final norm — `hidden`.
        for d in [p.headDim, p.globalHeadDim, p.hidden] {
            if d % 128 != 0 {
                throw Gemma4Error.unalignedNorm(d)
            }
        }
        // Both attention geometries go through Ops.sdpaDecode — the
        // sliding head_dim and the global head_dim must each be a
        // kernel-supported specialization (256 → d256, 512 → d512).
        for d in [p.headDim, p.globalHeadDim] {
            if !OpsValidation.supportedSdpaHeadDims.contains(d) {
                throw Gemma4Error.unsupportedHeadDim(d)
            }
        }

        let embedTokens = try loadEmbedding(
            base: "\(prefix)embed_tokens", in: weights,
            hidden: p.hidden, quantization: quant)

        // ── Per-Layer Embeddings (Gemma4E) ───────────────────────────
        var ple: Gemma4PLE? = nil
        if p.hiddenSizePerLayerInput > 0 {
            let plEmbed = try loadEmbedding(
                base: "\(prefix)embed_tokens_per_layer", in: weights,
                hidden: p.nLayers * p.hiddenSizePerLayerInput, quantization: quant)
            let plProj = try loadLinear(
                base: "\(prefix)per_layer_model_projection",
                in: weights, quantization: quant)
            let plProjNorm = try loadGemma4RMSNorm(
                base: "\(prefix)per_layer_projection_norm.weight",
                in: weights, eps: p.eps)
            ple = Gemma4PLE(
                embed: plEmbed, projection: plProj, projectionNorm: plProjNorm,
                hiddenSizePerLayerInput: p.hiddenSizePerLayerInput,
                hidden: p.hidden, nLayers: p.nLayers, device: device)
        }

        // ── Decoder layers ───────────────────────────────────────────
        var layers: [Gemma4Layer] = []
        layers.reserveCapacity(p.nLayers)
        let firstKvSharedIdx = p.nLayers - p.numKvSharedLayers
        for i in 0..<p.nLayers {
            let lp = "\(prefix)layers.\(i)"
            let isGlobal = p.isGlobal(i)
            let isKvShared = p.numKvSharedLayers > 0 && i >= firstKvSharedIdx
            let isDoubleWide = p.useDoubleWideMlp && isKvShared

            let qProj = try loadLinear(base: "\(lp).self_attn.q_proj",
                                       in: weights, quantization: quant)
            let kProj = try loadLinear(base: "\(lp).self_attn.k_proj",
                                       in: weights, quantization: quant)
            // attention_k_eq_v drops v_proj on global layers (V := K).
            let kEqV = p.attentionKEqV && isGlobal
            let vProj: AnyLinear? = kEqV
                ? nil
                : try loadLinear(base: "\(lp).self_attn.v_proj",
                                 in: weights, quantization: quant)
            let oProj = try loadLinear(base: "\(lp).self_attn.o_proj",
                                       in: weights, quantization: quant)

            let qNorm = try loadGemma4RMSNorm(
                base: "\(lp).self_attn.q_norm.weight", in: weights, eps: p.eps)
            let kNorm = try loadGemma4RMSNorm(
                base: "\(lp).self_attn.k_norm.weight", in: weights, eps: p.eps)

            let inputNorm = try loadGemma4RMSNorm(
                base: "\(lp).input_layernorm.weight", in: weights, eps: p.eps)
            let postAttnNorm = try loadGemma4RMSNorm(
                base: "\(lp).post_attention_layernorm.weight", in: weights, eps: p.eps)
            let preFFNorm = try loadGemma4RMSNorm(
                base: "\(lp).pre_feedforward_layernorm.weight", in: weights, eps: p.eps)
            let postFFNorm = try loadGemma4RMSNorm(
                base: "\(lp).post_feedforward_layernorm.weight", in: weights, eps: p.eps)

            // FFN: MoE block or dense GELU MLP.
            let ffn: Gemma4FFN
            if p.numExperts > 0 {
                ffn = .moe(try buildMoE(prefix: lp, weights: weights, p: p,
                                        quant: quant))
            } else {
                let effInter = isDoubleWide ? p.intermediate * 2 : p.intermediate
                let gateProj = try loadLinear(base: "\(lp).mlp.gate_proj",
                                              in: weights, quantization: quant)
                let upProj = try loadLinear(base: "\(lp).mlp.up_proj",
                                            in: weights, quantization: quant)
                let downProj = try loadLinear(base: "\(lp).mlp.down_proj",
                                              in: weights, quantization: quant)
                ffn = .dense(Gemma4DenseMLP(
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    intermediate: effInter))
            }

            // Per-layer scalar [1].
            let layerScalar = (try? weights.tensor(named: "\(lp).layer_scalar"))

            // PLE per-layer modules (Gemma4E).
            var plePerLayer: Gemma4LayerPLE? = nil
            if p.hiddenSizePerLayerInput > 0 {
                let gate = try loadLinear(
                    base: "\(lp).per_layer_input_gate",
                    in: weights, quantization: quant)
                let proj = try loadLinear(
                    base: "\(lp).per_layer_projection",
                    in: weights, quantization: quant)
                let norm = try loadGemma4RMSNorm(
                    base: "\(lp).post_per_layer_input_norm.weight",
                    in: weights, eps: p.eps)
                plePerLayer = Gemma4LayerPLE(gate: gate, projection: proj, norm: norm)
            }

            let ropeTheta = isGlobal ? p.ropeThetaGlobal : p.ropeThetaSliding
            let layerHeadDim = isGlobal ? p.globalHeadDim : p.headDim
            let layerKVHeads = isGlobal ? p.globalKvHeads : p.kvHeads
            // ProportionalRoPE rotated-dim count for global layers.
            let rotatedDim = isGlobal
                ? evenFloor(Int(Float(p.globalHeadDim) * p.partialRotaryFactor))
                : layerHeadDim

            layers.append(Gemma4Layer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                qNorm: qNorm, kNorm: kNorm,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                preFFNorm: preFFNorm, postFFNorm: postFFNorm,
                ffn: ffn, layerScalar: layerScalar, ple: plePerLayer,
                hidden: p.hidden, nHeads: p.nHeads, nKVHeads: layerKVHeads,
                headDim: layerHeadDim, isGlobal: isGlobal, kEqV: kEqV,
                isKvShared: isKvShared,
                ropeTheta: ropeTheta, rotatedDim: rotatedDim, eps: Float(p.eps),
                device: device))
        }

        let finalNorm = try loadGemma4RMSNorm(
            base: "\(prefix)norm.weight", in: weights, eps: p.eps)

        // lm_head: tied to embeddings unless an explicit head ships.
        let lmHead: AnyLinear
        if !p.tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("\(prefix)embed_tokens") {
            let t = try weights.quantizedTriplet("\(prefix)embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation dtype follows the embedding table (its scales for a
        // quantized table, the raw weight otherwise).
        let activationDtype: DType
        if weights.isQuantized("\(prefix)embed_tokens"),
           let scales = try? weights.tensor(named: "\(prefix)embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        // Pre-baked sqrt(hidden) embed-scale tensor (original Gemma
        // normalization, applied to the embedded row each forward).
        let embedScale = Tensor.empty(shape: [p.hidden], dtype: activationDtype, device: device)
        gemma4FillScalar(embedScale, scalar: Float(Double(p.hidden).squareRoot()),
                         dtype: activationDtype)

        return Gemma4Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead, embedScale: embedScale,
            ple: ple, params: p, dtype: activationDtype,
            kvCacheKind: options.kvCache, device: device)
    }

    /// Build a Gemma 4 MoE feed-forward block (26B-A4B). A MoE layer
    /// runs a shared dense MLP and a routed expert mixture in parallel,
    /// each with its own pre/post norm, then sums them. Experts ship as
    /// `[numExperts, outDim, inDim]` stacks under `experts.{gate,up,down}_proj`.
    private static func buildMoE(
        prefix lp: String, weights: SafeTensorsBundle, p: Gemma4Params,
        quant: ModelConfig.QuantizationConfig?
    ) throws -> Gemma4MoEFFN {
        // Shared dense GELU MLP — the `h1` branch, runs every token.
        let sharedGate = try loadLinear(base: "\(lp).mlp.gate_proj",
                                        in: weights, quantization: quant)
        let sharedUp = try loadLinear(base: "\(lp).mlp.up_proj",
                                      in: weights, quantization: quant)
        let sharedDown = try loadLinear(base: "\(lp).mlp.down_proj",
                                        in: weights, quantization: quant)
        let sharedMLP = Gemma4DenseMLP(
            gateProj: sharedGate, upProj: sharedUp, downProj: sharedDown,
            intermediate: p.intermediate)

        // Router: logit projection + the learned input-norm scale +
        // the per-expert combine-weight scale.
        let routerProj = try loadLinear(base: "\(lp).router.proj",
                                        in: weights, quantization: quant)
        let routerScale = try weights.tensor(named: "\(lp).router.scale")
        let perExpertScale = try weights.tensor(named: "\(lp).router.per_expert_scale")

        // Stacked experts → per-expert GELU-SwiGLU projections. Raw
        // Gemma 4 checkpoints key the expert stacks under
        // `experts.switch_glu.{gate,up,down}_proj` (mlx-lm strips the
        // `switch_glu` segment in its sanitize step; FFAI loads the raw
        // key directly).
        let gateProj = try sliceStacked(
            base: "\(lp).experts.switch_glu.gate_proj", in: weights,
            numExperts: p.numExperts, outDim: p.moeIntermediate, inDim: p.hidden,
            quant: quant)
        let upProj = try sliceStacked(
            base: "\(lp).experts.switch_glu.up_proj", in: weights,
            numExperts: p.numExperts, outDim: p.moeIntermediate, inDim: p.hidden,
            quant: quant)
        let downProj = try sliceStacked(
            base: "\(lp).experts.switch_glu.down_proj", in: weights,
            numExperts: p.numExperts, outDim: p.hidden, inDim: p.moeIntermediate,
            quant: quant)

        // The MoE block's three extra norms (plain Gemma 4 RMSNorm).
        let preNorm2 = try loadGemma4RMSNorm(
            base: "\(lp).pre_feedforward_layernorm_2.weight", in: weights, eps: p.eps)
        let postNorm1 = try loadGemma4RMSNorm(
            base: "\(lp).post_feedforward_layernorm_1.weight", in: weights, eps: p.eps)
        let postNorm2 = try loadGemma4RMSNorm(
            base: "\(lp).post_feedforward_layernorm_2.weight", in: weights, eps: p.eps)

        // Gemma 4 gating: top-K of the router logits, then softmax over
        // just those K (gemma4_text.py).
        let router = MoERouter(
            nExperts: p.numExperts, topK: p.topKExperts,
            gatingMode: .topKThenSoftmax)

        return Gemma4MoEFFN(
            sharedMLP: sharedMLP,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            routerProj: routerProj, routerScale: routerScale,
            rootSize: Float(pow(Double(p.hidden), -0.5)),
            routerEps: Float(p.eps), perExpertScale: perExpertScale,
            router: router,
            preNorm2: preNorm2, postNorm1: postNorm1, postNorm2: postNorm2,
            hidden: p.hidden)
    }

    /// Slice a stacked `[numExperts, outDim, inDim]` expert tensor into
    /// per-expert `AnyLinear`s (raw or mlx-quantized).
    private static func sliceStacked(
        base: String, in weights: SafeTensorsBundle,
        numExperts: Int, outDim: Int, inDim: Int,
        quant: ModelConfig.QuantizationConfig?
    ) throws -> [AnyLinear] {
        var out: [AnyLinear] = []
        out.reserveCapacity(numExperts)
        if let q = quant, weights.isQuantized(base) {
            let w = try weights.tensor(named: "\(base).weight")
            let s = try weights.tensor(named: "\(base).scales")
            let b = try weights.tensor(named: "\(base).biases")
            let packedCols = w.shape[w.shape.count - 1]
            let groupCols = s.shape[s.shape.count - 1]
            // The stacked tensor is one shape for all experts, so the
            // derived bit-width is uniform across the stack.
            let bits = deriveAffineQuantBits(
                weightPackedCols: packedCols, scaleCols: groupCols,
                groupSize: q.groupSize)
            precondition([3, 4, 5, 6, 8].contains(bits),
                         "sliceStacked: derived \(bits)-bit for \(base) — "
                         + "unsupported quantization bit-width")
            for e in 0..<numExperts {
                out.append(AnyLinear(QuantizedLinear(
                    weight: w.slicedRows(start: e, count: 1)
                        .reshaped(to: [outDim, packedCols]),
                    scales: s.slicedRows(start: e, count: 1)
                        .reshaped(to: [outDim, groupCols]),
                    biases: b.slicedRows(start: e, count: 1)
                        .reshaped(to: [outDim, groupCols]),
                    bits: bits, groupSize: q.groupSize)))
            }
        } else {
            let stacked = try weights.tensor(named: "\(base).weight")
            for e in 0..<numExperts {
                out.append(AnyLinear(Linear(weight:
                    stacked.slicedRows(start: e, count: 1)
                        .reshaped(to: [outDim, inDim]))))
            }
        }
        return out
    }
}

/// Largest even integer ≤ n. ProportionalRoPE rotates `rotatedDim` dims
/// in rotate-half pairs, so the count must be even.
private func evenFloor(_ n: Int) -> Int { n - (n % 2) }

// MARK: - Gemma 4 RMSNorm load (plain — NO (1 + weight) fold)

/// Load a Gemma 4 RMSNorm weight verbatim.
///
/// **Gemma 4 dropped the `(1 + weight)` RMSNorm convention.** Gemma 3
/// (and earlier Gemma) store norm weights centred near 0 and the norm
/// computes `x_normed · (1 + weight)`; FFAI's `Gemma3.loadGemmaRMSNorm`
/// folds the `+1` in at load time. Gemma 4's reference (`mlx-lm`
/// `gemma4_text`) uses a *plain* `nn.RMSNorm` — `x_normed · weight` —
/// and the checkpoints store the norm weights as the *direct*
/// multiplier (hidden-size norms have mean ≈ 10, the per-layer
/// projection norm has values spanning 0.09 … 5.5). Folding `+1` here
/// inflates every small weight — for the per-layer projection norm a
/// 0.09 weight becomes 1.09, a ~12× error — and was the cause of the
/// incoherent-generation bug. Load the weight as-is.
private func loadGemma4RMSNorm(
    base: String, in weights: SafeTensorsBundle, eps: Double
) throws -> RMSNorm {
    let raw = try weights.tensor(named: base)
    precondition(raw.shape.count == 1, "Gemma4 RMSNorm weight must be 1D, got \(raw.shape)")
    return RMSNorm(weight: raw, eps: Float(eps))
}

/// Fill a flat `[n]` tensor with a scalar (used for the embed-scale
/// tensor). Reuses Gemma 3's bit-conversion shims.
private func gemma4FillScalar(_ t: Tensor, scalar: Float, dtype: DType) {
    fillScalarForTest(t, scalar: scalar, dtype: dtype)
}

// MARK: - PLE (model-level)

/// Per-Layer Embedding inputs (Gemma4E). Computes, once per forward,
/// the `[nLayers, hiddenSizePerLayerInput]` per-layer-input tensor that
/// each block mixes into its residual.
public final class Gemma4PLE: Module {
    let embed: AnyEmbedding
    let projection: AnyLinear
    let projectionNorm: RMSNorm
    let hiddenSizePerLayerInput: Int
    let hidden: Int
    let nLayers: Int
    /// sqrt(hiddenSizePerLayerInput) — embed scaling.
    let embedScaleVec: Tensor
    /// hidden^(-0.5) — projection scaling.
    let projScaleVec: Tensor
    /// 2^(-0.5) ≈ 0.707 — combine scaling.
    let combineScaleVec: Tensor

    init(embed: AnyEmbedding, projection: AnyLinear, projectionNorm: RMSNorm,
         hiddenSizePerLayerInput: Int, hidden: Int, nLayers: Int, device: Device) {
        self.embed = embed
        self.projection = projection
        self.projectionNorm = projectionNorm
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.hidden = hidden
        self.nLayers = nLayers
        let plTotal = nLayers * hiddenSizePerLayerInput
        // The PLE scale vectors are float constants multiplied into the
        // (dequantized) per-layer embeddings. `embed.weight` is the
        // packed u32 table for a quantized checkpoint, so its dtype is
        // wrong here — use the projection norm's weight dtype, which is
        // always the model's float compute dtype (norm weights are
        // never quantized).
        let dtype = projectionNorm.weight.dtype
        self.embedScaleVec = Tensor.filled(
            Float(Double(max(hiddenSizePerLayerInput, 1)).squareRoot()),
            shape: [plTotal], dtype: dtype, device: device)
        self.projScaleVec = Tensor.filled(
            Float(pow(Double(hidden), -0.5)),
            shape: [plTotal], dtype: dtype, device: device)
        self.combineScaleVec = Tensor.filled(
            Float(pow(2.0, -0.5)),
            shape: [plTotal], dtype: dtype, device: device)
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embed.parameters() { out.append(("embed_tokens_per_layer.\(k)", v)) }
        for (k, v) in projection.parameters() { out.append(("per_layer_model_projection.\(k)", v)) }
        for (k, v) in projectionNorm.parameters() { out.append(("per_layer_projection_norm.\(k)", v)) }
        return out
    }

    /// Compute the per-layer-input tensor for one decode token.
    /// `tokenTensor` is the [1] u32 token id; `h` is the scaled
    /// embedding `[hidden]`. Returns a `[nLayers, hiddenSizePerLayerInput]`
    /// tensor (one row per decoder layer).
    func perLayerInputs(tokenTensor: Tensor, h: Tensor,
                        on cmd: MTLCommandBuffer) -> Tensor {
        let tap = InspectTap.fromEnvironment
        // 1. Embed token through the per-layer table, scale by
        //    sqrt(hiddenSizePerLayerInput).
        let plEmbed = embed(tokenTensor, on: cmd)
            .reshaped(to: [nLayers * hiddenSizePerLayerInput])
        let scaledEmbed = Ops.mul(plEmbed, embedScaleVec, on: cmd)

        // 2. Project the hidden state, scale by hidden^(-0.5), RMSNorm
        //    per-layer-input row, combine with the embedding.
        var plProj = projection(h, on: cmd)
        plProj = Ops.mul(plProj, projScaleVec, on: cmd)
        // projectionNorm is [hiddenSizePerLayerInput]-wide; apply per row.
        let normed = Ops.rmsNormRows(
            plProj, weight: projectionNorm.weight, eps: projectionNorm.eps,
            nRows: nLayers, rowSize: hiddenSizePerLayerInput, on: cmd)
        let combined = Ops.add(normed, scaledEmbed, on: cmd)
        let result = Ops.mul(combined, combineScaleVec, on: cmd)
            .reshaped(to: [nLayers, hiddenSizePerLayerInput])
        // Stash the substep tensors so `forward` can dump them AFTER it
        // commits the prep command buffer (committing here would
        // double-commit the caller's cmd).
        if tap.active {
            inspectSubsteps = [
                ("ple.plEmbed_raw", plEmbed),
                ("ple.plEmbed_scaled", scaledEmbed),
                ("ple.plProj_scaled", plProj),
                ("ple.plProj_normed", normed),
                ("ple.pli_final", result),
            ]
        }
        return result
    }

    /// Debug-only: substep tensors captured by the last `perLayerInputs`
    /// call when `InspectTap` is active. `forward` dumps these after the
    /// prep command buffer has been committed. Empty in production.
    var inspectSubsteps: [(String, Tensor)] = []
}

/// Per-layer PLE modules held on each decoder block (Gemma4E).
public final class Gemma4LayerPLE: Module {
    let gate: AnyLinear        // hidden → hiddenSizePerLayerInput
    let projection: AnyLinear  // hiddenSizePerLayerInput → hidden
    let norm: RMSNorm

    init(gate: AnyLinear, projection: AnyLinear, norm: RMSNorm) {
        self.gate = gate
        self.projection = projection
        self.norm = norm
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gate.parameters() { out.append(("per_layer_input_gate.\(k)", v)) }
        for (k, v) in projection.parameters() { out.append(("per_layer_projection.\(k)", v)) }
        for (k, v) in norm.parameters() { out.append(("post_per_layer_input_norm.\(k)", v)) }
        return out
    }
}

// MARK: - FFN sub-blocks

/// The feed-forward half of a Gemma 4 layer.
enum Gemma4FFN {
    case dense(Gemma4DenseMLP)
    case moe(Gemma4MoEFFN)
}

/// Dense GELU MLP: down(gelu(gate(x)) * up(x)).
public final class Gemma4DenseMLP: Module {
    let gateProj, upProj, downProj: AnyLinear
    let intermediate: Int

    init(gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         intermediate: Int) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.intermediate = intermediate
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        return out
    }

    func forward(_ xNorm: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let gate = gateProj(xNorm, on: cmd)
        let up = upProj(xNorm, on: cmd)
        let inner = Ops.mul(Ops.gelu(gate, on: cmd), up, on: cmd)
        return downProj(inner, on: cmd)
    }
}

/// Gemma 4 MoE feed-forward block (26B-A4B). A MoE layer runs a shared
/// dense GELU MLP and a routed expert mixture *in parallel* — each
/// branch with its own pre/post RMSNorm — and sums them:
///
///   h1 = post_ffn_norm_1(sharedMLP(pre_ffn_norm(h)))
///   h2 = post_ffn_norm_2( Σ wₑ · expertₑ(pre_ffn_norm_2(h)) )
///   ffnOut = h1 + h2
///
/// The router normalises its input with a learned `scale · hidden^(-0.5)`
/// weight, picks the top-K experts, softmaxes their logits, and scales
/// the combine weights by a learned per-expert factor. Mirrors
/// `gemma4_text.py`.
///
/// `forward` commits internally (the router needs its logits on the
/// host for top-K), so the caller must refresh its command buffer.
public final class Gemma4MoEFFN: Module {
    /// Shared dense MLP — the `h1` branch.
    let sharedMLP: Gemma4DenseMLP
    /// Per-expert GELU-SwiGLU projections (index-aligned with expert id).
    let gateProj, upProj, downProj: [AnyLinear]
    /// Router logit projection: hidden → numExperts.
    let routerProj: AnyLinear
    /// Router input-norm weight (the learned `router.scale`, [hidden]).
    let routerScale: Tensor
    /// `hidden^(-0.5)` — folded into the router logits. A positive
    /// scalar, so it leaves the top-K selection unchanged but sets the
    /// softmax temperature, matching the reference's `scale · hidden^-0.5`
    /// input norm.
    let rootSize: Float
    let routerEps: Float
    /// Per-expert combine-weight scale (`router.per_expert_scale`,
    /// [numExperts]).
    let perExpertScale: Tensor
    /// Top-K + softmax routing math.
    let router: MoERouter
    /// The MoE block's three extra norms.
    let preNorm2, postNorm1, postNorm2: RMSNorm
    let hidden: Int

    init(sharedMLP: Gemma4DenseMLP,
         gateProj: [AnyLinear], upProj: [AnyLinear], downProj: [AnyLinear],
         routerProj: AnyLinear, routerScale: Tensor, rootSize: Float,
         routerEps: Float, perExpertScale: Tensor, router: MoERouter,
         preNorm2: RMSNorm, postNorm1: RMSNorm, postNorm2: RMSNorm,
         hidden: Int) {
        self.sharedMLP = sharedMLP
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.routerProj = routerProj; self.routerScale = routerScale
        self.rootSize = rootSize; self.routerEps = routerEps
        self.perExpertScale = perExpertScale; self.router = router
        self.preNorm2 = preNorm2; self.postNorm1 = postNorm1
        self.postNorm2 = postNorm2; self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        out.append(contentsOf: sharedMLP.parameters())
        for (e, proj) in gateProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).gate_proj.\(k)", v)) }
        }
        for (e, proj) in upProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).up_proj.\(k)", v)) }
        }
        for (e, proj) in downProj.enumerated() {
            for (k, v) in proj.parameters() { out.append(("experts.\(e).down_proj.\(k)", v)) }
        }
        for (k, v) in routerProj.parameters() { out.append(("router.proj.\(k)", v)) }
        out.append(("router.scale", routerScale))
        out.append(("router.per_expert_scale", perExpertScale))
        for (k, v) in preNorm2.parameters() {
            out.append(("pre_feedforward_layernorm_2.\(k)", v))
        }
        for (k, v) in postNorm1.parameters() {
            out.append(("post_feedforward_layernorm_1.\(k)", v))
        }
        for (k, v) in postNorm2.parameters() {
            out.append(("post_feedforward_layernorm_2.\(k)", v))
        }
        return out
    }

    /// Run the Gemma 4 MoE FFN.
    /// - `preFFNormed`: `preFeedforwardLayerNorm(h)` — the shared-MLP
    ///   branch input.
    /// - `h`: the post-attention residual — the router and expert-branch
    ///   input (each applies its own norm).
    /// Returns `ffnOut = h1 + h2`; the caller adds `postFeedforwardLayerNorm`
    /// + the residual. Commits internally, so the caller must refresh its
    /// command buffer.
    func forward(preFFNormed: Tensor, h: Tensor,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── Router: rmsNorm(h, router.scale) → proj → logits ─────────
        let routerNormed = Ops.rmsNorm(h, weight: routerScale,
                                       eps: routerEps, on: cmd)
        let logitsTensor = routerProj(routerNormed, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // Fold hidden^(-0.5) into the logits — the reference normalises
        // the router input by `scale · hidden^-0.5`; `proj` is linear and
        // the factor is a positive scalar, so applying it to the logits
        // is equivalent and sets the softmax temperature.
        let logits = logitsTensor.toFloatArray().map { $0 * rootSize }
        let routing = router.route(logits: logits)
        let perExpert = perExpertScale.toFloatArray()
        // Combine weight = softmax weight × per-expert scale.
        let weights = routing.indices.enumerated().map { slot, expertId in
            routing.weights[slot] * perExpert[expertId]
        }

        // ── Both branches on a private command buffer ────────────────
        let work = device.makeCommandBuffer()

        // h1: shared dense MLP.
        var h1 = sharedMLP.forward(preFFNormed, on: work)
        h1 = postNorm1(h1, on: work)

        // h2: weighted sum of the selected experts (GELU-SwiGLU).
        let expertInput = preNorm2(h, on: work)
        var h2acc: Tensor?
        for (slot, expertId) in routing.indices.enumerated() {
            let g = gateProj[expertId](expertInput, on: work)
            let u = upProj[expertId](expertInput, on: work)
            let inner = Ops.mul(Ops.gelu(g, on: work), u, on: work)
            let expertOut = downProj[expertId](inner, on: work)
            let wTensor = Tensor.filled(weights[slot], shape: [hidden],
                                        dtype: h.dtype, device: device)
            let scaled = Ops.mul(expertOut, wTensor, on: work)
            h2acc = h2acc.map { Ops.add($0, scaled, on: work) } ?? scaled
        }
        // topK ≥ 1 ⇒ h2acc is non-nil.
        let h2 = postNorm2(h2acc!, on: work)

        let ffnOut = Ops.add(h1, h2, on: work)
        work.commit()
        work.waitUntilCompleted()
        return ffnOut
    }
}

// MARK: - Gemma4Layer

public final class Gemma4Layer: Module {
    let qProj, kProj, oProj: AnyLinear
    let vProj: AnyLinear?
    let qNorm, kNorm: RMSNorm
    let inputNorm, postAttnNorm, preFFNorm, postFFNorm: RMSNorm
    let ffn: Gemma4FFN
    /// Learned per-layer output scalar [1]; nil ⇒ identity.
    let layerScalar: Tensor?
    /// Per-layer PLE modules (Gemma4E); nil for dense / MoE.
    let ple: Gemma4LayerPLE?

    let hidden, nHeads, nKVHeads, headDim: Int
    public let isGlobal: Bool
    /// V := K (attention_k_eq_v on global layers).
    let kEqV: Bool
    /// True if this layer reuses a donor layer's K/V (Gemma 4
    /// `num_kv_shared_layers`). A shared layer projects only Q, applies
    /// RoPE to it, and runs SDPA against the donor's cache slab — it
    /// never projects or caches its own K/V. Its `kProj` / `vProj` /
    /// `kNorm` are the checkpoint's dead weights and go unused.
    public let isKvShared: Bool
    let ropeTheta: Float
    /// ProportionalRoPE rotated-dim count (== headDim for sliding layers,
    /// `partial_rotary_factor · globalHeadDim` for global layers).
    let rotatedDim: Int
    let eps: Float
    /// Scale-free [headDim] weight for the value RMSNorm (all-ones).
    let vNormWeight: Tensor
    /// True if this layer's FFN commits the command buffer (MoE).
    public let commitsCommandBuffer: Bool

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear?, oProj: AnyLinear,
         qNorm: RMSNorm, kNorm: RMSNorm,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         preFFNorm: RMSNorm, postFFNorm: RMSNorm,
         ffn: Gemma4FFN, layerScalar: Tensor?, ple: Gemma4LayerPLE?,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         isGlobal: Bool, kEqV: Bool, isKvShared: Bool,
         ropeTheta: Float, rotatedDim: Int,
         eps: Float, device: Device) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.preFFNorm = preFFNorm; self.postFFNorm = postFFNorm
        self.ffn = ffn
        self.layerScalar = layerScalar
        self.ple = ple
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.isGlobal = isGlobal; self.kEqV = kEqV
        self.isKvShared = isKvShared
        self.ropeTheta = ropeTheta; self.rotatedDim = rotatedDim; self.eps = eps
        // Value RMSNorm uses a unit weight (scale-free).
        self.vNormWeight = Tensor.filled(1.0, shape: [headDim],
                                         dtype: qNorm.weight.dtype, device: device)
        if case .moe = ffn { self.commitsCommandBuffer = true }
        else { self.commitsCommandBuffer = false }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        if let vp = vProj {
            for (k, v) in vp.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in qNorm.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        for (k, v) in kNorm.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        for (k, v) in preFFNorm.parameters() { out.append(("pre_feedforward_layernorm.\(k)", v)) }
        for (k, v) in postFFNorm.parameters() { out.append(("post_feedforward_layernorm.\(k)", v)) }
        switch ffn {
        case .dense(let mlp): out.append(contentsOf: mlp.parameters())
        case .moe(let moe): out.append(contentsOf: moe.parameters())
        }
        if let ls = layerScalar { out.append(("layer_scalar", ls)) }
        if let ple { out.append(contentsOf: ple.parameters()) }
        return out
    }

    /// A dense-FFN layer needs no mid-layer host sync — it runs entirely
    /// on a shared command buffer batched with its neighbours. Both the
    /// sliding (head_dim 256) and global (head_dim 512) attention paths
    /// are pure-GPU `Ops.sdpaDecode` dispatches. Only MoE layers
    /// (`MoELayer` host readback) must commit mid-layer, so they break
    /// the batch.
    var batchable: Bool {
        if case .moe = ffn { return false }
        return true
    }

    /// Single-token forward. `position` is the absolute sequence index.
    /// `perLayerInput` is this layer's PLE row (`[hiddenSizePerLayerInput]`),
    /// nil for dense / MoE variants. `cmd` is the caller's work buffer.
    ///
    /// `donorCache` is non-nil iff this is a KV-shared layer
    /// (`isKvShared`): it is the donor layer's cache, already updated
    /// with the current token by the time this layer runs. The shared
    /// layer reads K/V from it and never touches its own `cache`.
    ///
    /// Returns `(h, committed)`. A `batchable` layer queues onto `cmd`
    /// and returns `committed == false` — the caller keeps batching.
    /// A global / MoE layer commits internally (host readback) and
    /// returns `committed == true` — the caller must refresh its buffer.
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 perLayerInput: Tensor?, cmd: MTLCommandBuffer,
                 device: Device,
                 donorCache: (any KVCacheProtocol)? = nil)
        -> (h: Tensor, committed: Bool) {
        if batchable {
            return (forwardBatched(h, position: position, cache: cache,
                                   perLayerInput: perLayerInput, cmd: cmd,
                                   donorCache: donorCache),
                    false)
        }
        return (forwardCommitting(h, position: position, cache: cache,
                                  perLayerInput: perLayerInput, device: device,
                                  donorCache: donorCache),
                true)
    }

    /// Sliding + dense path — all work queues onto the shared `cmd`,
    /// nothing is committed. The caller commits the batch.
    private func forwardBatched(_ h: Tensor, position: Int,
                                cache: any KVCacheProtocol,
                                perLayerInput: Tensor?,
                                cmd: MTLCommandBuffer,
                                donorCache: (any KVCacheProtocol)? = nil)
        -> Tensor {
        let xNorm = inputNorm(h, on: cmd)
        let attnOut = isGlobal
            ? globalAttentionBatched(xNorm, position: position,
                                     cache: cache, cmd: cmd,
                                     donorCache: donorCache)
            : slidingAttentionBatched(xNorm, position: position,
                                      cache: cache, cmd: cmd,
                                      donorCache: donorCache)
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let normedAttn = postAttnNorm(oOut, on: cmd)
        var hOut = Ops.add(h, normedAttn, on: cmd)

        let ffnNorm = preFFNorm(hOut, on: cmd)
        guard case .dense(let mlp) = ffn else {
            fatalError("Gemma4Layer.forwardBatched: non-dense FFN")
        }
        let ffnOut = mlp.forward(ffnNorm, on: cmd)
        let normedFFN = postFFNorm(ffnOut, on: cmd)
        hOut = Ops.add(hOut, normedFFN, on: cmd)
        return applyPLEAndScalar(hOut, perLayerInput: perLayerInput, on: cmd)
    }

    /// Global / MoE path — needs a mid-layer host readback, so it runs
    /// on private buffers and commits before returning resident `h`.
    private func forwardCommitting(_ h: Tensor, position: Int,
                                   cache: any KVCacheProtocol,
                                   perLayerInput: Tensor?,
                                   device: Device,
                                   donorCache: (any KVCacheProtocol)? = nil)
        -> Tensor {
        let attnCmd = device.makeCommandBuffer()
        let xNorm = inputNorm(h, on: attnCmd)
        let attnOut: Tensor
        if isGlobal {
            attnOut = globalAttention(xNorm, position: position, cache: cache,
                                      cmd: attnCmd, device: device,
                                      donorCache: donorCache)
            // `globalAttention` already committed `attnCmd`.
        } else {
            // MoE layer with a sliding attention: run attention on its
            // own buffer (committed here) so the MoE FFN can sync.
            attnOut = slidingAttention(xNorm, position: position, cache: cache,
                                       cmd: attnCmd, donorCache: donorCache)
        }

        let postCmd = device.makeCommandBuffer()
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: postCmd)
        let normedAttn = postAttnNorm(oOut, on: postCmd)
        var hOut = Ops.add(h, normedAttn, on: postCmd)
        let ffnNorm = preFFNorm(hOut, on: postCmd)

        switch ffn {
        case .dense(let mlp):
            // Global + dense: finish the FFN on `postCmd`.
            let ffnOut = mlp.forward(ffnNorm, on: postCmd)
            let normedFFN = postFFNorm(ffnOut, on: postCmd)
            hOut = Ops.add(hOut, normedFFN, on: postCmd)
            hOut = applyPLEAndScalar(hOut, perLayerInput: perLayerInput, on: postCmd)
            postCmd.commit()
            postCmd.waitUntilCompleted()
            return hOut
        case .moe(let moe):
            // Gemma 4 MoE block: parallel shared MLP + routed experts.
            // It needs both the pre-FFN-normed input (shared MLP branch)
            // and the raw post-attention residual (router + expert
            // branch), and commits internally (host readback for top-K
            // routing). Sync `postCmd` so both inputs are resident.
            postCmd.commit()
            postCmd.waitUntilCompleted()
            let moeCmd = device.makeCommandBuffer()
            let ffnOut = moe.forward(preFFNormed: ffnNorm, h: hOut,
                                     cmd: moeCmd, device: device)
            let addCmd = device.makeCommandBuffer()
            let normedFFN = postFFNorm(ffnOut, on: addCmd)
            hOut = Ops.add(hOut, normedFFN, on: addCmd)
            hOut = applyPLEAndScalar(hOut, perLayerInput: perLayerInput, on: addCmd)
            addCmd.commit()
            addCmd.waitUntilCompleted()
            return hOut
        }
    }

    /// Per-Layer Embedding mix + learned per-layer output scalar.
    private func applyPLEAndScalar(_ h: Tensor, perLayerInput: Tensor?,
                                   on cmd: MTLCommandBuffer) -> Tensor {
        var hOut = h
        if let ple, let pli = perLayerInput {
            // h + post_norm(projection(gelu(gate(h)) * per_layer_input))
            let gated = Ops.mul(Ops.gelu(ple.gate(hOut, on: cmd), on: cmd),
                                pli, on: cmd)
            let projected = ple.projection(gated, on: cmd)
            let normed = ple.norm(projected, on: cmd)
            hOut = Ops.add(hOut, normed, on: cmd)
        }
        if let ls = layerScalar {
            // layer_scalar is [1]; broadcast-multiply via Ops needs a
            // [hidden] tensor — materialise it once from the scalar.
            let scalar = ls.toFloatArray().first ?? 1.0
            if scalar != 1.0 {
                let scaleVec = Tensor.filled(scalar, shape: [hidden],
                                             dtype: hOut.dtype, device: .shared)
                hOut = Ops.mul(hOut, scaleVec, on: cmd)
            }
        }
        return hOut
    }

    /// q / k / v projections + per-head RMSNorms. Returns the three
    /// `[…, headDim]` tensors (q `[nHeads,headDim]`, k/v `[nKVHeads,headDim]`).
    /// The value RMSNorm is scale-free; q/k norms carry the Gemma fold.
    private func projectAndNorm(_ xNorm: Tensor, on cmd: MTLCommandBuffer)
        -> (q: Tensor, k: Tensor, v: Tensor) {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v: Tensor = kEqV ? k : vProj!(xNorm, on: cmd)
        let qNormed = Ops.rmsNormRows(
            q, weight: qNorm.weight, eps: eps,
            nRows: nHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nHeads, headDim])
        let kNormed = Ops.rmsNormRows(
            k, weight: kNorm.weight, eps: eps,
            nRows: nKVHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nKVHeads, headDim])
        let vNormed = Ops.rmsNormRows(
            v, weight: vNormWeight, eps: eps,
            nRows: nKVHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nKVHeads, headDim])
        return (qNormed, kNormed, vNormed)
    }

    /// Project + RMSNorm only the query, leaving K/V untouched. Used by
    /// KV-shared layers, which reuse a donor's cached K/V and so never
    /// project their own. Returns `[nHeads, headDim]`.
    private func projectAndNormQuery(_ xNorm: Tensor, on cmd: MTLCommandBuffer)
        -> Tensor {
        let q = qProj(xNorm, on: cmd)
        return Ops.rmsNormRows(
            q, weight: qNorm.weight, eps: eps,
            nRows: nHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nHeads, headDim])
    }

    /// Sliding-layer attention queued onto a shared `cmd` — standard
    /// full RoPE + GPU `sdpaDecode` (head_dim 256). Does NOT commit;
    /// the caller batches and commits. Returns `[nHeads, headDim]`.
    ///
    /// `donorCache` is non-nil for a KV-shared layer: the donor layer
    /// (run earlier this token) has already appended the current K/V to
    /// `donorCache`, so the shared layer only projects Q, applies RoPE
    /// to it, and runs SDPA against the donor's slab — no K/V projection
    /// and no cache append.
    private func slidingAttentionBatched(_ xNorm: Tensor, position: Int,
                                         cache: any KVCacheProtocol,
                                         cmd: MTLCommandBuffer,
                                         donorCache: (any KVCacheProtocol)? = nil)
        -> Tensor {
        if let donorCache {
            // KV-shared layer: reuse the donor's already-updated slab.
            let q = projectAndNormQuery(xNorm, on: cmd)
            let qRotated = Ops.rope(q, position: position, headDim: headDim,
                                    thetaBase: ropeTheta, scaling: .none, on: cmd)
            let (cacheK, cacheV) = donorCache.prepareForAttention(on: cmd)
            return Ops.sdpaDecode(
                q: qRotated, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                nKV: donorCache.length, kvStride: donorCache.maxSeq,
                scale: 1.0, on: cmd)
        }
        let (q, k, v) = projectAndNorm(xNorm, on: cmd)
        let qRotated = Ops.rope(q, position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(k, position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: .none, on: cmd)
        cache.appendOnGPU(kFlat: kRotated, vFlat: v, on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        // Gemma 4 uses SDPA scale = 1.0 (pre-scaling lives in q_norm).
        return Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: 1.0, on: cmd)
    }

    /// Sliding attention that commits `cmd` and returns resident output
    /// — used by a MoE layer whose attention is sliding (the MoE FFN
    /// needs the attention result resident before its own host sync).
    private func slidingAttention(_ xNorm: Tensor, position: Int,
                                  cache: any KVCacheProtocol,
                                  cmd: MTLCommandBuffer,
                                  donorCache: (any KVCacheProtocol)? = nil)
        -> Tensor {
        let out = slidingAttentionBatched(xNorm, position: position,
                                          cache: cache, cmd: cmd,
                                          donorCache: donorCache)
        cmd.commit()
        cmd.waitUntilCompleted()
        return out
    }

    /// Global-layer (512-wide) attention queued onto a shared `cmd` —
    /// ProportionalRoPE + GPU `sdpaDecode` (head_dim 512). Does NOT
    /// commit; the caller batches and commits. Returns `[nHeads, headDim]`.
    ///
    /// `donorCache` is non-nil for a KV-shared global layer: it projects
    /// + RoPEs only Q and runs SDPA against the donor's already-updated
    /// slab — no K/V projection, no cache append.
    private func globalAttentionBatched(_ xNorm: Tensor, position: Int,
                                        cache: any KVCacheProtocol,
                                        cmd: MTLCommandBuffer,
                                        donorCache: (any KVCacheProtocol)? = nil)
        -> Tensor {
        let q: Tensor
        let attnCache: any KVCacheProtocol
        if let donorCache {
            // KV-shared layer: project + RoPE only the query.
            q = projectAndNormQuery(xNorm, on: cmd)
            Gemma4Ops.ropeProportional(
                q, position: position, headDim: headDim, rotatedDim: rotatedDim,
                thetaBase: ropeTheta, on: cmd)
            attnCache = donorCache
        } else {
            let (qOwn, k, v) = projectAndNorm(xNorm, on: cmd)
            // ProportionalRoPE on q and k (in-place): rotates pairs
            // `(i, i + headDim/2)` for `i ∈ [0, rotatedDim/2)` with
            // frequency `theta^(-2i/headDim)`. The unrotated tail passes
            // through.
            Gemma4Ops.ropeProportional(
                qOwn, position: position, headDim: headDim, rotatedDim: rotatedDim,
                thetaBase: ropeTheta, on: cmd)
            Gemma4Ops.ropeProportional(
                k, position: position, headDim: headDim, rotatedDim: rotatedDim,
                thetaBase: ropeTheta, on: cmd)
            cache.appendOnGPU(kFlat: k, vFlat: v, on: cmd)
            q = qOwn
            attnCache = cache
        }
        let (cacheK, cacheV) = attnCache.prepareForAttention(on: cmd)
        // Gemma 4 uses SDPA scale = 1.0 (pre-scaling lives in q_norm).
        // head_dim 512 routes to the d512 `sdpaDecode` specialization.
        return Ops.sdpaDecode(
            q: q, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: attnCache.length, kvStride: attnCache.maxSeq,
            scale: 1.0, on: cmd)
    }

    /// Global attention that commits `cmd` and returns resident output
    /// — used by a global + MoE layer whose MoE FFN needs the attention
    /// result resident before its own host sync.
    private func globalAttention(
        _ xNorm: Tensor, position: Int,
        cache: any KVCacheProtocol, cmd: MTLCommandBuffer, device _: Device,
        donorCache: (any KVCacheProtocol)? = nil
    ) -> Tensor {
        let out = globalAttentionBatched(xNorm, position: position,
                                         cache: cache, cmd: cmd,
                                         donorCache: donorCache)
        cmd.commit()
        cmd.waitUntilCompleted()
        return out
    }
}

// MARK: - Gemma4Ops — ProportionalRoPE wrapper

/// Gemma 4-specific Ops extensions. `ropeProportional` drives the
/// shared `ffai_rope_llama` RoPE kernel for Gemma 4's ProportionalRoPE.
enum Gemma4Ops {
    /// ProportionalRoPE: rotate the first `rotatedDim` dims of each
    /// `headDim`-strided head, pairing `(i, i + headDim/2)` for
    /// `i ∈ [0, rotatedDim/2)`, with frequency `theta^(-2i/headDim)`.
    /// Dims `[rotatedDim/2, headDim/2)` and their partners stay
    /// untouched, so this MUST run in-place — the buffer already holds
    /// the correct identity values for the unrotated dims.
    ///
    /// The shared `ffai_rope_llama` kernel computes
    /// `inv_freq = theta^(-i/half_dim)` and rotates `(base+i, base+i+half_dim)`.
    /// Driving it with `head_dim = headDim`, `half_dim = headDim/2`
    /// (true rotate-half offset + freq denominator) and a grid height of
    /// `rotatedDim/2` rotates exactly the proportional subset.
    static func ropeProportional(_ qk: Tensor, position: Int,
                                 headDim: Int, rotatedDim: Int,
                                 thetaBase: Float, on cmd: MTLCommandBuffer) {
        precondition(qk.elementCount % headDim == 0,
                     "ropeProportional: qk size must be a multiple of headDim")
        precondition(rotatedDim > 0 && rotatedDim <= headDim && rotatedDim % 2 == 0,
                     "ropeProportional: rotatedDim must be even and ≤ headDim")
        let nHeads = qk.elementCount / headDim
        let halfDim = headDim / 2          // pairing offset + freq denominator
        let rotatedPairs = rotatedDim / 2  // grid height — only these rotate
        let grid = MTLSize(width: nHeads, height: rotatedPairs, depth: 1)
        let tg = MTLSize(width: 1, height: 1, depth: 1)
        switch qk.dtype {
        case .f32:
            MetalTileKernels.ffai_rope_llama_f32(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                scale_factor: 1.0, low_freq_factor: 1.0,
                high_freq_factor: 1.0, original_max_position: 1.0,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .f16:
            MetalTileKernels.ffai_rope_llama_f16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                scale_factor: 1.0, low_freq_factor: 1.0,
                high_freq_factor: 1.0, original_max_position: 1.0,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        case .bf16:
            MetalTileKernels.ffai_rope_llama_bf16(
                qk: qk.buffer, qkOffset: qk.offset,
                out: qk.buffer, outOffset: qk.offset,
                head_dim: UInt32(headDim), half_dim: UInt32(halfDim),
                position: UInt32(position), theta_base: thetaBase,
                scale_factor: 1.0, low_freq_factor: 1.0,
                high_freq_factor: 1.0, original_max_position: 1.0,
                gridSize: grid, threadgroupSize: tg, on: cmd)
        default:
            fatalError("Gemma4Ops.ropeProportional: unsupported dtype \(qk.dtype)")
        }
    }
}

// MARK: - Gemma4Model

public final class Gemma4Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Gemma4Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear
    /// Pre-baked `[hidden]` sqrt(hidden) embed-scale tensor.
    public let embedScale: Tensor
    /// Per-Layer Embeddings (Gemma4E); nil for dense / MoE.
    public let ple: Gemma4PLE?

    let params: Gemma4Params

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let dtype: DType

    /// Gemma 4 prefills 4096 tokens per chunk — the value tuned in
    /// `mlx-swift-lm`'s `Libraries/MLXLLM/Models/Gemma4.swift`. The
    /// sliding-window every-other-layer schedule means alternating
    /// layers attend a tiny KV stripe, so amortising over more rows
    /// pays off bigger than the dense default.
    public var defaultPrefillStepSize: Int { 4096 }

    /// Gemma 4 is BOS-critical and its `tokenizer.json` post-processor's
    /// `single` template is bare (`[Sequence A]`, no `<bos>` special
    /// token) — unlike Gemma 3, whose post-processor lists `<bos>`. So
    /// `Tokenizer.encode` returns no leading BOS for Gemma 4, and
    /// `Generate.swift` must prepend it explicitly. Without the BOS the
    /// model generates degraded, incoherent text.
    public let requiresLeadingBOS = true

    let kvCacheKind: KVCacheKind
    /// Soft-cap value for the final logits; nil ⇒ no capping.
    let finalLogitSoftcapping: Float?

    init(embedTokens: AnyEmbedding, layers: [Gemma4Layer],
         finalNorm: RMSNorm, lmHead: AnyLinear, embedScale: Tensor,
         ple: Gemma4PLE?, params: Gemma4Params, dtype: DType,
         kvCacheKind: KVCacheKind, device: Device) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.embedScale = embedScale
        self.ple = ple
        self.params = params
        self.hidden = params.hidden
        self.nLayers = params.nLayers
        self.nHeads = params.nHeads
        // Report the sliding-layer KV head count (the dominant geometry).
        self.nKVHeads = params.kvHeads
        self.headDim = params.headDim
        self.vocab = params.vocab
        self.maxSeq = params.maxSeq
        self.dtype = dtype
        self.kvCacheKind = kvCacheKind
        self.finalLogitSoftcapping = params.finalLogitSoftcapping
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        if let ple {
            for (k, v) in ple.parameters() { out.append(("model.\(k)", v)) }
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

    /// Per-layer KV cache. Sliding layers cap at `slidingWindow`
    /// (`.window`); global layers stay `.unbounded`. Each layer's cache
    /// is sized to that layer's own head dim / KV-head count, because
    /// sliding (256) and global (512) layers differ.
    ///
    /// KV-shared layers (`num_kv_shared_layers`) reuse a donor layer's
    /// cache at decode time and never append to their own slot, so
    /// their cache is allocated at `maxSeq = 1` — just a placeholder to
    /// keep the per-layer array index aligned. The real K/V lives in
    /// the donor's cache (`params.previousKVs`).
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        guard kvCacheKind == .raw else {
            preconditionFailure(
                "Gemma4: only .raw KV cache supported today; got \(kvCacheKind)")
        }
        var caches: [any LayerCacheProtocol] = []
        caches.reserveCapacity(nLayers)
        for (i, layer) in layers.enumerated() {
            // A KV-shared layer never appends to its own cache — give it
            // a 1-slot placeholder so the array stays index-aligned
            // without allocating a full unused slab.
            let layerCap = layer.isKvShared ? 1 : cap
            let eviction: KVEviction = layer.isGlobal
                ? .unbounded
                : .window(maxSize: min(params.slidingWindow, layerCap), keep: 0)
            caches.append(KVCache(
                nKVHeads: layer.nKVHeads, headDim: layer.headDim, maxSeq: layerCap,
                dtype: dtype, eviction: eviction, device: device))
            _ = i
        }
        return caches
    }

    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // Each `Gemma4Layer.forward` is self-contained — it runs on its
        // own command buffers and commits before returning resident `h`.
        // The caller's pristine `cmd` is touched only by the embedding +
        // PLE prep and the final norm / lm_head / soft-cap.
        let tap = InspectTap.fromEnvironment
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // Embedding + sqrt(hidden) scale + PLE prep on a private buffer.
        let prepCmd = device.makeCommandBuffer()
        let h0 = embedTokens(tokenTensor, on: prepCmd).reshaped(to: [hidden])
        var h = Ops.mul(h0, embedScale, on: prepCmd)

        // Per-Layer Embeddings (Gemma4E): compute the per-layer-input
        // tensor once for this token.
        var perLayerInputs: Tensor? = nil
        if let ple {
            perLayerInputs = ple.perLayerInputs(
                tokenTensor: tokenTensor, h: h, on: prepCmd)
        }
        prepCmd.commit()
        prepCmd.waitUntilCompleted()
        if tap.active {
            _ = tap.dumpLayerBoundary(h, label: "embed*scale", layer: -1,
                                      cmd: device.makeCommandBuffer(), device: device)
            for (label, t) in ple?.inspectSubsteps ?? [] {
                _ = tap.dumpLayerBoundary(t, label: label, layer: -1,
                                          cmd: device.makeCommandBuffer(), device: device)
            }
            if let pli = perLayerInputs {
                _ = tap.dumpLayerBoundary(pli, label: "pli", layer: -1,
                                          cmd: device.makeCommandBuffer(), device: device)
            }
        }

        // Batchable (sliding + dense) layers queue onto a shared
        // `workCmd`; a global / MoE layer needs a mid-layer host sync,
        // so the pending batch is committed before it runs and a fresh
        // `workCmd` is started after. `batchPending` tracks whether
        // `workCmd` carries un-committed, un-waited layer work whose
        // output `h` the next consumer must wait for.
        // KV-sharing donor map: `donorMap[i]` is the layer whose cache
        // layer `i` reuses (== i for non-shared layers). Layers run in
        // index order, so a shared layer's donor (always an earlier
        // index) has already appended the current token's K/V to its
        // cache by the time the shared layer reads it.
        let donorMap = params.previousKVs
        var workCmd = device.makeCommandBuffer()
        var batchPending = false
        for (i, layer) in layers.enumerated() {
            let pli: Tensor? = perLayerInputs.map { plis in
                plis.slicedRows(start: i, count: 1)
                    .reshaped(to: [params.hiddenSizePerLayerInput])
            }
            if !layer.batchable && batchPending {
                // Flush the pending batch so `h` is resident before the
                // committing layer reads it.
                workCmd.commit()
                workCmd.waitUntilCompleted()
                batchPending = false
            }
            // KV-shared layer: hand it the donor's cache so it reuses
            // the donor's K/V instead of projecting its own.
            let donorCache: (any KVCacheProtocol)? =
                layer.isKvShared
                ? (caches[donorMap[i]] as? any KVCacheProtocol)
                : nil
            let (newH, committed) = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                perLayerInput: pli, cmd: workCmd, device: device,
                donorCache: donorCache)
            h = newH
            if committed {
                // The layer ran on its own buffers and left `h`
                // resident; start a fresh batch buffer.
                workCmd = device.makeCommandBuffer()
                batchPending = false
            } else {
                batchPending = true
            }
            if tap.shouldDump(layer: i) {
                // `h` is resident if the layer committed; otherwise the
                // pending batch must be flushed before reading it.
                if batchPending {
                    workCmd.commit()
                    workCmd.waitUntilCompleted()
                    workCmd = device.makeCommandBuffer()
                    batchPending = false
                }
                _ = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                          cmd: device.makeCommandBuffer(), device: device)
            }
        }
        // Flush any trailing batched work so `h` is resident.
        if batchPending {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        if tap.active {
            _ = tap.dumpLayerBoundary(h, label: "h_pre_finalnorm", layer: -1,
                                      cmd: device.makeCommandBuffer(), device: device)
        }

        // Final norm + lm_head. Soft-capping needs the logits on the
        // host, so the head runs on a private buffer that is committed
        // here; the caller's `cmd` is left pristine (its later commit in
        // `forwardSample` / `forward` becomes a no-op over already-
        // resident logits — the protocol contract still holds).
        if let softcap = finalLogitSoftcapping, softcap > 0 {
            let headCmd = device.makeCommandBuffer()
            let normed = finalNorm(h, on: headCmd)
            let logits = lmHead(normed, on: headCmd)
            return Gemma4Ops.softcap(logits, cap: softcap, on: headCmd)
        }
        // No soft-cap: queue norm + head onto the caller's `cmd` as the
        // protocol expects.
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    // ─── VLM embedding-input path ────────────────────────────────────
    //
    // Gemma 4 is a VL-target text backbone (Gemma4-VL wraps it). The
    // splice supplies a `[hidden]` row directly — a vision-encoder token
    // or a text-token embedding the VL model looked up. The forward
    // mirrors `forward(tokenId:...)` minus the embedding gather; the
    // `sqrt(hidden)` embed-scale is still applied here (image tokens in
    // Gemma 4 are scaled the same way as text tokens, matching the
    // Gemma 3 VL convention).
    //
    // Per-Layer Embeddings (Gemma4E) are token-id-derived and so cannot
    // be computed for a vision token — the dense Gemma 4 VL variant
    // carries no PLE, and a PLE-bearing variant simply runs its layers
    // with a nil per-layer input on the splice path (coherence-first).

    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "Gemma4Model.forward(inputEmbedding:): expected [\(hidden)], "
                     + "got \(inputEmbedding.shape)")

        // Embedding-scale on a private buffer (mirrors the token path).
        let prepCmd = device.makeCommandBuffer()
        var h = Ops.mul(inputEmbedding.reshaped(to: [hidden]), embedScale,
                        on: prepCmd)
        prepCmd.commit()
        prepCmd.waitUntilCompleted()

        // Layers run on their own self-committing buffers; `cmd` is left
        // pristine for the final norm / head (same contract as the
        // token-id forward). KV-sharing donor map applies unchanged.
        let donorMap = params.previousKVs
        var workCmd = device.makeCommandBuffer()
        var batchPending = false
        for (i, layer) in layers.enumerated() {
            if !layer.batchable && batchPending {
                workCmd.commit()
                workCmd.waitUntilCompleted()
                batchPending = false
            }
            let donorCache: (any KVCacheProtocol)? =
                layer.isKvShared
                ? (caches[donorMap[i]] as? any KVCacheProtocol)
                : nil
            let (newH, committed) = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                perLayerInput: nil, cmd: workCmd, device: device,
                donorCache: donorCache)
            h = newH
            if committed {
                workCmd = device.makeCommandBuffer()
                batchPending = false
            } else {
                batchPending = true
            }
        }
        if batchPending {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        // Final norm + head. Soft-capping needs the logits host-side, so
        // it runs on a private buffer (caller's `cmd` stays pristine).
        if let softcap = finalLogitSoftcapping, softcap > 0 {
            let headCmd = device.makeCommandBuffer()
            let normed = finalNorm(h, on: headCmd)
            let logits = lmHead(normed, on: headCmd)
            return Gemma4Ops.softcap(logits, cap: softcap, on: headCmd)
        }
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Raw embedding-table lookup for one text token — no embed-scale
    /// (that is applied inside `forward(inputEmbedding:...)`).
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

// MARK: - Soft-cap

extension Gemma4Ops {
    /// Logit soft-capping: `cap · tanh(x / cap)`. Implemented host-side
    /// after `cmd` completes — applied once per token to the `[vocab]`
    /// logit vector, which the sampler reads back anyway. Keeps the
    /// final-stage command buffer free of an extra dispatch.
    ///
    /// This commits + waits on `cmd`: it is the terminal op of
    /// `forward`, so the caller's commit becomes a no-op and the
    /// returned tensor is resident.
    static func softcap(_ logits: Tensor, cap: Float,
                        on cmd: MTLCommandBuffer) -> Tensor {
        cmd.commit()
        cmd.waitUntilCompleted()
        let n = logits.elementCount
        let ptr = logits.buffer.contents().advanced(by: logits.offset)
        switch logits.dtype {
        case .f32:
            let p = ptr.bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { p[i] = cap * tanh(p[i] / cap) }
        case .f16:
            let p = ptr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n {
                let f = halfBitsToFloatForTest(p[i])
                p[i] = floatToHalfBitsForTest(cap * tanh(f / cap))
            }
        case .bf16:
            let p = ptr.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n {
                let f = bf16BitsToFloatForTest(p[i])
                p[i] = floatToBf16BitsForTest(cap * tanh(f / cap))
            }
        default:
            fatalError("Gemma4Ops.softcap: unsupported dtype \(logits.dtype)")
        }
        return logits
    }
}
