// Qwen3.5 family — a Phase 5e *stack-interleaved hybrid* model that
// alternates a Gated Delta Net (GDN) recurrent mixer with full
// multi-head attention. It is the most-coupled of the Phase 5e hybrid
// families: the first FFAI consumer of the GDN kernel + `GDNStateCache`,
// and the first to combine a host-assisted recurrent mixer with an MoE
// feed-forward half.
//
// ─── Three variants, one layer pattern ───────────────────────────────
//
// Qwen3.5's `config.json` carries a `text_config` block (the checkpoint
// is a VLM wrapper — every weight is prefixed `language_model.model.`
// and a `vision_tower` is dropped at load). Three concrete variants:
//
//   * Qwen35Dense — GDN ↔ attention alternation, dense SwiGLU MLP on
//     every layer. `num_experts == 0`. (0.8B / 2B / 4B / 9B / 27B.)
//   * Qwen35MoE   — same GDN ↔ attention alternation, but the FFN half
//     is a block-sparse MoE (top-K SwiGLU experts) PLUS a sigmoid-gated
//     always-on shared expert. `num_experts > 0`. (35B-A3B.)
//   * Qwen35GDN   — the MoE/dense distinction is orthogonal; "GDN
//     variant" just names the hybrid checkpoints whose `layer_types`
//     interleaves `linear_attention` (GDN) and `full_attention`. Every
//     published Qwen3.5 checkpoint is a GDN hybrid, so the dense and
//     MoE variants ARE GDN hybrids — the variant axis the loader cares
//     about is dense-FFN vs MoE-FFN.
//
// The hybrid schedule is `(layerIdx + 1) % full_attention_interval == 0`
// → a `full_attention` layer, else a `linear_attention` (GDN) layer.
// Every published checkpoint also ships an explicit `layer_types` array
// that we honour when present (and fall back to the interval rule when
// it is absent).
//
// Per-layer dataflow (matches mlx-lm's `qwen3_5.py`):
//
//   residual = h
//   h        = input_layernorm(h)
//   h        = mixer(h)                    — GDN / attention
//   r        = residual + h
//   out      = r + feed_forward(post_attention_layernorm(r))
//
// ─── The GDN mixer is the STANDARD (non-fused) kernel ─────────────────
//
// `Ops.gatedDeltaStep` wraps the standard gated-delta kernel: it takes
// pre-normalised, pre-scaled `q` / `k`, pre-computed per-head gates `g`
// and `beta`, all in fp32. The fused decode kernel (which absorbs the
// q/k rmsNorm, `g = exp(-exp(A_log)·softplus(a + dt_bias))`, and
// `beta = sigmoid(b)`) was deliberately NOT ported. So `Qwen35GDNMixer`
// does that prep itself, host-side:
//
//   1. GPU: in_proj_qkv / in_proj_z / in_proj_b / in_proj_a, conv1d
//      causal step + SiLU, split q | k | v.
//   2. Commit + wait — the host needs q/k/v/z/a/b on the CPU.
//   3. Host: per-head unweighted RMSNorm of q (×invScale²) and k
//      (×invScale) with `invScale = headKDim^-0.5`; `g[hv] =
//      exp(-exp(A_log[hv]) · softplus(a[hv] + dt_bias[hv]))`;
//      `beta[hv] = sigmoid(b[hv])`. Write everything into fp32 tensors.
//   4. GPU phase 2: `Ops.gatedDeltaStep` → `y [Hv, Dv]` fp32; gated
//      RMSNorm `norm(y, gate: z)`; `out_proj`.
//
// The per-token host cost is tiny (q/k norm over `Hk·Dk` ≈ 2K elements,
// g/beta over `Hv` ≈ 16). The GPU still owns every projection, the
// conv, the GDN recurrence kernel, the gated norm, and the MLP/MoE.
// `GDNStateCache` is double-buffered: `current` is the kernel's
// `state_in`, `next` its `state_out`, and `swap()` ping-pongs them.
//
// ─── Command-buffer discipline ────────────────────────────────────────
//
// Every GDN layer commits the command buffer it is handed (the host
// q/k/g/beta prep needs a CPU sync); an MoE FFN also commits. So the
// caller's `cmd` must NEVER be handed to a layer. `Qwen35Model.forward`
// runs the embedding + every layer on internal `workCmd` buffers
// (committed by the layers / refreshed after each committing layer) and
// queues ONLY the final `norm` + `lm_head` onto the caller's pristine
// `cmd` — preserving the `LanguageModel` "does not commit the caller's
// `cmd`" contract so `forwardSample` / `forwardSampleCategorical`
// compose their output kernels cleanly. This is the Jamba pattern.
//
// ─── Attention: gated output, partial rotary, per-head q/k norm ───────
//
// Qwen3.5 attention (`attn_output_gate: true`) projects `q_proj` to
// `2 · n_heads · head_dim` — the first half is the queries, the second
// is a sigmoid gate applied to the attention output before `o_proj`.
// `q_norm` / `k_norm` are per-head RMSNorm over `head_dim`. RoPE is
// **partial** (`partial_rotary_factor = 0.25`): only the first
// `head_dim · 0.25` dims of each head are rotated — `Ops.ropePartial`.
// For text-only decode the config's mRoPE reduces to plain 1-D RoPE
// (mRoPE only differs for 2-D vision-token positions).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen35 {
    public static let modelTypes: Set<String> = [
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    ]
    public static let architectures: Set<String> = [
        "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM",
    ]

    public static func variant(for _: ModelConfig) throws -> any Qwen35Variant.Type {
        // A single variant covers all three forms — dense vs MoE is
        // decided per-checkpoint from `num_experts` inside `loadModel`.
        return Qwen35Hybrid.self
    }
}

public protocol Qwen35Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen35Model
}

public enum Qwen35Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Qwen3.5: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "Qwen3.5: unsupported config: \(m)"
        }
    }
}

// ─── Layer kind ──────────────────────────────────────────────────────

/// The two mixer kinds a Qwen3.5 hybrid layer can take.
enum Qwen35LayerKind: Equatable {
    case gdn         // "linear_attention" — Gated Delta Net recurrent mixer
    case attention   // "full_attention"   — multi-head attention

    init(from name: String) throws {
        switch name {
        case "linear_attention": self = .gdn
        case "full_attention": self = .attention
        default:
            throw Qwen35Error.unsupportedConfig(
                "unknown layer_types entry '\(name)'")
        }
    }
}

// ─── Qwen35Hybrid — the single variant ───────────────────────────────

public struct Qwen35Hybrid: Qwen35Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Qwen3.5 ships both base + instruction-tuned checkpoints. Greedy
    /// by default keeps the integration suite deterministic.
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
        options _: LoadOptions,
        device: Device
    ) throws -> Qwen35Model {
        // ── text_config — every text field lives under `text_config`
        //    on the VLM-wrapped checkpoints; fall back to the root for a
        //    hypothetical text-only conversion. ──────────────────────────
        let tc = config.nested("text_config")
        func tcInt(_ k: String) -> Int? {
            (tc?[k] as? Int) ?? config.int(k)
        }
        func tcBool(_ k: String) -> Bool? {
            (tc?[k] as? Bool) ?? config.bool(k)
        }
        func tcFloat(_ k: String) -> Double? {
            if let v = tc?[k] as? Double { return v }
            if let v = tc?[k] as? Int { return Double(v) }
            return config.float(k)
        }
        func tcIntArray(_ k: String) -> [Int]? {
            (tc?[k] as? [Int]) ?? config.intArray(k)
        }

        guard let hidden = tcInt("hidden_size"),
              let vocab = tcInt("vocab_size"),
              let nHeads = tcInt("num_attention_heads"),
              let nLayers = tcInt("num_hidden_layers")
        else {
            throw Qwen35Error.missingConfig(
                "hidden_size / vocab_size / num_attention_heads / num_hidden_layers")
        }
        let nKVHeads = tcInt("num_key_value_heads") ?? nHeads
        let headDim = tcInt("head_dim") ?? (hidden / nHeads)
        let eps = Float(tcFloat("rms_norm_eps") ?? 1e-6)
        let tieEmbed = (tcBool("tie_word_embeddings")
            ?? config.bool("tie_word_embeddings")) ?? false
        let intermediate = tcInt("intermediate_size") ?? (4 * hidden)
        let maxSeq = tcInt("max_position_embeddings") ?? 262_144
        let fullAttnInterval = tcInt("full_attention_interval") ?? 4
        // `attn_output_gate` — q_proj projects 2× heads (queries + gate).
        let attnOutputGate = tcBool("attn_output_gate") ?? true

        // ── RoPE: partial rotary + theta from `rope_parameters` ───────
        let ropeParams = tc?["rope_parameters"] as? [String: Any]
        let ropeTheta = Float(
            (ropeParams?["rope_theta"] as? Double)
            ?? (ropeParams?["rope_theta"] as? Int).map(Double.init)
            ?? tcFloat("rope_theta")
            ?? 10_000_000)
        let partialRotaryFactor = Float(
            (ropeParams?["partial_rotary_factor"] as? Double)
            ?? tcFloat("partial_rotary_factor")
            ?? 0.25)
        // rotaryDim must be even (rotate-half pairs); clamp to headDim.
        var rotaryDim = Int(Float(headDim) * partialRotaryFactor)
        rotaryDim = min(headDim, max(2, rotaryDim - (rotaryDim % 2)))

        // ── GDN (linear-attention) mixer geometry ─────────────────────
        guard let linearNumKeyHeads = tcInt("linear_num_key_heads"),
              let linearNumValueHeads = tcInt("linear_num_value_heads"),
              let linearKeyHeadDim = tcInt("linear_key_head_dim"),
              let linearValueHeadDim = tcInt("linear_value_head_dim")
        else {
            throw Qwen35Error.missingConfig(
                "linear_num_key_heads / linear_num_value_heads / "
                + "linear_key_head_dim / linear_value_head_dim")
        }
        let convKernel = tcInt("linear_conv_kernel_dim") ?? 4
        let keyDim = linearKeyHeadDim * linearNumKeyHeads
        let valueDim = linearValueHeadDim * linearNumValueHeads
        // convDim covers q | k | v: keyDim·2 + valueDim.
        let convDim = keyDim * 2 + valueDim

        // The GDN kernel takes (Dk, Dv, Hk, Hv) as runtime constexprs,
        // but still carries reduction-mode geometry invariants (Dk a
        // multiple of 32 and ≤ 256, Hv divisible by Hk) — reject any
        // violating config with a clear message rather than a freeze.
        if let reason = OpsValidation.validateGatedDeltaStep(
            keyHeadDim: linearKeyHeadDim, valueHeadDim: linearValueHeadDim,
            numKeyHeads: linearNumKeyHeads, numValueHeads: linearNumValueHeads
        ) {
            throw Qwen35Error.unsupportedConfig("GDN dims — \(reason)")
        }
        // The gated mixer RMSNorm runs over `linear_value_head_dim` rows;
        // the metaltile rms_norm_rows kernel needs the row size to be a
        // multiple of 128 and ≤ 4096.
        guard linearValueHeadDim % 128 == 0, linearValueHeadDim <= 4096 else {
            throw Qwen35Error.unsupportedConfig(
                "GDN gated-norm row size linear_value_head_dim = "
                + "\(linearValueHeadDim) must be a multiple of 128 and ≤ 4096 "
                + "(rmsNormRows kernel invariant)")
        }
        // SDPA full-attention head_dim must have an emitted kernel.
        if let reason = OpsValidation.validateSdpaDecode(
            headDim: headDim, nQHeads: nHeads, nKVHeads: nKVHeads,
            nKV: 1, kvStride: 1
        ) {
            throw Qwen35Error.unsupportedConfig("attention head_dim — \(reason)")
        }

        // ── MoE geometry (Qwen3.5 MoE / 3.6 only) ─────────────────────
        let numExperts = tcInt("num_experts") ?? 0
        let numExpertsPerToken = tcInt("num_experts_per_tok") ?? 0
        let moeIntermediate = tcInt("moe_intermediate_size") ?? intermediate
        let sharedExpertIntermediate = tcInt("shared_expert_intermediate_size") ?? 0
        let normTopkProb = tcBool("norm_topk_prob") ?? true
        let mlpOnlyLayers = Set(tcIntArray("mlp_only_layers") ?? [])
        let decoderSparseStep = tcInt("decoder_sparse_step") ?? 1
        let useMoE = numExperts > 0

        // ── Hybrid layer schedule ─────────────────────────────────────
        // Prefer an explicit `layer_types` array; fall back to the
        // `(i + 1) % full_attention_interval == 0` interval rule.
        let kinds: [Qwen35LayerKind]
        if let names = tc?["layer_types"] as? [String],
           !names.isEmpty {
            kinds = try names.map { try Qwen35LayerKind(from: $0) }
        } else if let names = config.raw["layer_types"] as? [String],
                  !names.isEmpty {
            kinds = try names.map { try Qwen35LayerKind(from: $0) }
        } else {
            kinds = (0..<nLayers).map { i in
                (i + 1) % fullAttnInterval == 0 ? .attention : .gdn
            }
        }
        guard kinds.count == nLayers else {
            throw Qwen35Error.unsupportedConfig(
                "layer_types has \(kinds.count) entries, "
                + "num_hidden_layers is \(nLayers)")
        }

        // ── Weight prefix — VLM-wrapped checkpoints prefix every text
        //    weight with `language_model.model.`; a text-only conversion
        //    would use `model.`. Detect from the embedding key. ──────────
        let prefixCandidates = ["language_model.model", "model"]
        guard let modelPrefix = prefixCandidates.first(where: {
            weights.has("\($0).embed_tokens.weight")
        }) else {
            throw Qwen35Error.missingConfig("embed_tokens.weight (model prefix)")
        }

        let quant = config.quantization

        // ── Embedding + activation dtype ──────────────────────────────
        // The embedding may be mlx-quantized (u32-packed `weight`); build
        // it through `loadEmbedding` (raw `Embedding` or `QuantizedEmbedding`)
        // and take the activation dtype from a raw tensor — the final
        // norm — not the packed embedding weight.
        let embedTokens = try loadEmbedding(
            base: "\(modelPrefix).embed_tokens", in: weights,
            hidden: hidden, quantization: quant)
        let activationDtype = try weights.tensor(
            named: "\(modelPrefix).norm.weight").dtype
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "Qwen3.5: unexpected activation dtype \(activationDtype)")
        let invKeyScale = Foundation.pow(Float(linearKeyHeadDim), -0.5)

        // ── Per-layer construction ────────────────────────────────────
        var layers: [any DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in kinds.enumerated() {
            let p = "\(modelPrefix).layers.\(i)"
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: eps)
            let postNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: eps)

            // ── Feed-forward half ─────────────────────────────────────
            // A layer in `mlp_only_layers`, or one off the sparse-step
            // cadence, gets a dense MLP even on an MoE checkpoint.
            let layerUsesMoE = useMoE
                && !mlpOnlyLayers.contains(i)
                && (i + 1) % decoderSparseStep == 0
            let ffn: Qwen35FFN
            if layerUsesMoE {
                ffn = .moe(try buildMoE(
                    prefix: "\(p).mlp", weights: weights,
                    hidden: hidden, moeIntermediate: moeIntermediate,
                    sharedIntermediate: sharedExpertIntermediate,
                    numExperts: numExperts, topK: numExpertsPerToken,
                    normTopkProb: normTopkProb, quant: quant))
            } else {
                let gate = try loadLinear(
                    base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
                let up = try loadLinear(
                    base: "\(p).mlp.up_proj", in: weights, quantization: quant)
                let down = try loadLinear(
                    base: "\(p).mlp.down_proj", in: weights, quantization: quant)
                ffn = .dense(Qwen35DenseMLP(
                    gateProj: gate, upProj: up, downProj: down))
            }

            switch kind {
            case .gdn:
                let mixer = try buildGDNMixer(
                    prefix: "\(p).linear_attn", weights: weights,
                    quantization: quant,
                    hidden: hidden,
                    numKeyHeads: linearNumKeyHeads, numValueHeads: linearNumValueHeads,
                    keyHeadDim: linearKeyHeadDim, valueHeadDim: linearValueHeadDim,
                    keyDim: keyDim, valueDim: valueDim, convDim: convDim,
                    convKernel: convKernel, eps: eps,
                    invKeyScale: invKeyScale,
                    dtype: activationDtype, device: device)
                layers.append(Qwen35GDNLayer(
                    inputNorm: inputNorm, postNorm: postNorm,
                    mixer: mixer, ffn: ffn, hidden: hidden))

            case .attention:
                let qProj = try loadLinear(
                    base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
                let kProj = try loadLinear(
                    base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
                let vProj = try loadLinear(
                    base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
                let oProj = try loadLinear(
                    base: "\(p).self_attn.o_proj", in: weights, quantization: quant)
                let qNorm = RMSNorm(
                    weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                    eps: eps)
                let kNorm = RMSNorm(
                    weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                    eps: eps)
                let mixer = Qwen35AttentionMixer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    qNorm: qNorm, kNorm: kNorm,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    rotaryDim: rotaryDim, ropeTheta: ropeTheta,
                    attnOutputGate: attnOutputGate)
                layers.append(Qwen35AttentionLayer(
                    inputNorm: inputNorm, postNorm: postNorm,
                    mixer: mixer, ffn: ffn, hidden: hidden))
            }
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "\(modelPrefix).norm.weight"), eps: eps)

        // lm_head — tied to the embedding table on every published
        // Qwen3.5 checkpoint; honour an untied `lm_head.weight` if a
        // checkpoint ships one.
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("\(modelPrefix).embed_tokens") {
            // Tied to a quantized embedding — reuse the embed triplet as a
            // QuantizedLinear (per-tensor bit-width via `deriveAffineQuantBits`).
            let t = try weights.quantizedTriplet("\(modelPrefix).embed_tokens")
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

        return Qwen35Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            numKeyHeads: linearNumKeyHeads, numValueHeads: linearNumValueHeads,
            keyHeadDim: linearKeyHeadDim, valueHeadDim: linearValueHeadDim,
            convDim: convDim, convKernel: convKernel,
            vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)
    }

    /// Build one GDN mixer. Reads the split QKV / Z / B / A projections,
    /// transposes the conv1d weight to the metaltile `[kernel, channels]`
    /// layout, and reads `A_log` + `dt_bias` to host arrays for the
    /// per-step gate computation.
    private static func buildGDNMixer(
        prefix p: String, weights: SafeTensorsBundle,
        quantization quant: ModelConfig.QuantizationConfig?,
        hidden: Int,
        numKeyHeads: Int, numValueHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        keyDim: Int, valueDim: Int, convDim: Int,
        convKernel: Int, eps: Float, invKeyScale: Float,
        dtype: DType, device: Device
    ) throws -> Qwen35GDNMixer {
        // Split projections — q|k|v stacked into `in_proj_qkv`; z, b, a
        // each their own projection. None ever carry bias on Qwen3.5.
        // `loadLinear` builds a QuantizedLinear when the checkpoint is
        // mlx-quantized, a plain Linear otherwise.
        let inProjQKV = try loadLinear(
            base: "\(p).in_proj_qkv", in: weights, quantization: quant)
        let inProjZ = try loadLinear(
            base: "\(p).in_proj_z", in: weights, quantization: quant)
        let inProjB = try loadLinear(
            base: "\(p).in_proj_b", in: weights, quantization: quant)
        let inProjA = try loadLinear(
            base: "\(p).in_proj_a", in: weights, quantization: quant)
        let outProj = try loadLinear(
            base: "\(p).out_proj", in: weights, quantization: quant)

        // conv1d.weight ships [conv_dim, kernel, 1]; the metaltile kernel
        // wants [kernel, conv_dim].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(convWSrc.elementCount == convDim * convKernel,
                     "Qwen3.5: conv1d.weight count mismatch: \(convWSrc.shape)")
        let convW = transposeConv1dWeight35(
            src: convWSrc, kernel: convKernel, channels: convDim,
            dtype: dtype, device: device)
        // conv1d carries no bias on Qwen3.5 — a zero bias keeps the
        // shared `conv1dCausalStep` kernel signature satisfied.
        let convB = zeroVector35(convDim, dtype: dtype, device: device)

        // Gated mixer RMSNorm — weighted, over `value_head_dim` (128).
        let mixerNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

        // A_log + dt_bias drive the per-step gate; read to host fp32.
        let aLog = readFloats35(try weights.tensor(named: "\(p).A_log"))
        let dtBias = readFloats35(try weights.tensor(named: "\(p).dt_bias"))
        precondition(aLog.count == numValueHeads,
                     "Qwen3.5: A_log expected [num_value_heads]")
        precondition(dtBias.count == numValueHeads,
                     "Qwen3.5: dt_bias expected [num_value_heads]")

        return Qwen35GDNMixer(
            inProjQKV: inProjQKV, inProjZ: inProjZ,
            inProjB: inProjB, inProjA: inProjA, outProj: outProj,
            convW: convW, convB: convB, mixerNorm: mixerNorm,
            aLog: aLog, dtBias: dtBias,
            hidden: hidden,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            keyDim: keyDim, valueDim: valueDim, convDim: convDim,
            convKernel: convKernel, eps: eps, invKeyScale: invKeyScale,
            dtype: dtype)
    }

    /// Build the MoE feed-forward block: a router + top-K SwiGLU experts
    /// (sliced from the stacked `switch_mlp` tensors) plus a sigmoid-
    /// gated always-on shared expert. Every projection may be quantized
    /// — `loadLinear` / per-slice `QuantizedLinear` route the
    /// dequant+gemv path through `AnyLinear` unchanged.
    private static func buildMoE(
        prefix p: String, weights: SafeTensorsBundle,
        hidden: Int, moeIntermediate: Int, sharedIntermediate: Int,
        numExperts: Int, topK: Int, normTopkProb: Bool,
        quant: ModelConfig.QuantizationConfig?
    ) throws -> Qwen35MoEFFN {
        // Router: hidden → numExperts logits (may be quantized).
        let gate = try loadLinear(
            base: "\(p).gate", in: weights, quantization: quant)

        // Per-expert SwiGLU sliced from the stacked switch_mlp tensors.
        // gate_proj / up_proj : [numExperts, moeIntermediate, hidden]
        // down_proj           : [numExperts, hidden, moeIntermediate]
        // For a quantized checkpoint the stacked weight is u32-packed and
        // the scales / biases stack identically along dim 0.
        let gateProj = try sliceStackedExperts(
            base: "\(p).switch_mlp.gate_proj", in: weights,
            numExperts: numExperts, outDim: moeIntermediate, inDim: hidden,
            quant: quant)
        let upProj = try sliceStackedExperts(
            base: "\(p).switch_mlp.up_proj", in: weights,
            numExperts: numExperts, outDim: moeIntermediate, inDim: hidden,
            quant: quant)
        let downProj = try sliceStackedExperts(
            base: "\(p).switch_mlp.down_proj", in: weights,
            numExperts: numExperts, outDim: hidden, inDim: moeIntermediate,
            quant: quant)

        // Qwen3.5 routing: softmax over ALL experts, then top-K of the
        // probabilities, then optional re-normalisation (`norm_topk_prob`).
        let router = MoERouter(
            nExperts: numExperts, topK: topK,
            gatingMode: .softmaxThenTopK, normTopKProb: normTopkProb)
        // The MoELayer carries only the routed experts — Qwen3.5's
        // shared expert is sigmoid-gated, which the plain MoELayer's
        // unconditional shared-expert add cannot express, so it is held
        // separately on the FFN wrapper.
        let moe = MoELayer(
            gate: gate,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            router: router, hidden: hidden)

        // Sigmoid-gated always-on shared expert.
        let sharedGate = try loadLinear(
            base: "\(p).shared_expert.gate_proj", in: weights, quantization: quant)
        let sharedUp = try loadLinear(
            base: "\(p).shared_expert.up_proj", in: weights, quantization: quant)
        let sharedDown = try loadLinear(
            base: "\(p).shared_expert.down_proj", in: weights, quantization: quant)
        let sharedExpertGate = try loadLinear(
            base: "\(p).shared_expert_gate", in: weights, quantization: quant)

        return Qwen35MoEFFN(
            moe: moe,
            sharedGateProj: sharedGate, sharedUpProj: sharedUp,
            sharedDownProj: sharedDown, sharedExpertGate: sharedExpertGate,
            hidden: hidden)
    }

    /// Slice a stacked `[numExperts, outDim, inDim]` expert tensor into
    /// `numExperts` per-expert `AnyLinear`s. Handles both raw and mlx-
    /// quantized stacks: the quantized stack's weight / scales / biases
    /// each slice along dim 0, so a per-expert `QuantizedLinear` is just
    /// three single-row slices reshaped to 2-D.
    private static func sliceStackedExperts(
        base: String, in weights: SafeTensorsBundle,
        numExperts: Int, outDim: Int, inDim: Int,
        quant: ModelConfig.QuantizationConfig?
    ) throws -> [AnyLinear] {
        var out: [AnyLinear] = []
        out.reserveCapacity(numExperts)

        if let q = quant, weights.isQuantized(base) {
            // Quantized stack: weight [E, outDim, inDim/packFactor] u32,
            // scales / biases [E, outDim, inDim/groupSize].
            let stackedW = try weights.tensor(named: "\(base).weight")
            let stackedS = try weights.tensor(named: "\(base).scales")
            let stackedB = try weights.tensor(named: "\(base).biases")
            let packedCols = stackedW.shape[stackedW.shape.count - 1]
            let groupCols = stackedS.shape[stackedS.shape.count - 1]
            // The stacked tensor is one shape for all experts, so the
            // derived bit-width is uniform across the stack.
            let bits = deriveAffineQuantBits(
                weightPackedCols: packedCols, scaleCols: groupCols,
                groupSize: q.groupSize)
            precondition([3, 4, 5, 6, 8].contains(bits),
                         "sliceStackedExperts: derived \(bits)-bit for "
                         + "\(base) — unsupported quantization bit-width")
            for e in 0..<numExperts {
                let w = stackedW.slicedRows(start: e, count: 1)
                    .reshaped(to: [outDim, packedCols])
                let s = stackedS.slicedRows(start: e, count: 1)
                    .reshaped(to: [outDim, groupCols])
                let b = stackedB.slicedRows(start: e, count: 1)
                    .reshaped(to: [outDim, groupCols])
                out.append(AnyLinear(QuantizedLinear(
                    weight: w, scales: s, biases: b,
                    bits: bits, groupSize: q.groupSize)))
            }
        } else {
            // Raw stack: weight [E, outDim, inDim].
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

// ─── FFN sub-block enum ──────────────────────────────────────────────

/// The feed-forward half of a Qwen3.5 layer — a dense SwiGLU MLP
/// (`num_experts == 0`) or a block-sparse MoE block with a sigmoid-
/// gated shared expert (`num_experts > 0`, commits the command buffer).
enum Qwen35FFN {
    case dense(Qwen35DenseMLP)
    case moe(Qwen35MoEFFN)
}

// ─── Qwen35DenseMLP — dense SwiGLU feed-forward ──────────────────────

public final class Qwen35DenseMLP: Module {
    let gateProj, upProj, downProj: AnyLinear

    init(gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear) {
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        return out
    }

    /// down(silu(gate(x)) * up(x)).
    func forward(_ xNorm: Tensor, cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(xNorm, on: cmd)
        let u = upProj(xNorm, on: cmd)
        let inner = Ops.mul(Ops.silu(g, on: cmd), u, on: cmd)
        return downProj(inner, on: cmd)
    }
}

// ─── Qwen35MoEFFN — MoE feed-forward with sigmoid-gated shared expert ─
//
// Wraps a `MoELayer` (the routed top-K experts) and applies the
// always-on shared expert separately so its output can be scaled by
// `sigmoid(shared_expert_gate(x))` — the plain `MoELayer` adds a shared
// expert unconditionally, which Qwen3.5's gate-scaled form needs.
//
// `MoELayer.decode` commits the command buffer; this wrapper therefore
// also commits, and runs the shared expert + the routed-combine add on
// fresh private buffers so the returned tensor is fully resident.

public final class Qwen35MoEFFN: Module {
    let moe: MoELayer
    let sharedGateProj, sharedUpProj, sharedDownProj: AnyLinear
    let sharedExpertGate: AnyLinear
    let hidden: Int

    init(moe: MoELayer,
         sharedGateProj: AnyLinear, sharedUpProj: AnyLinear,
         sharedDownProj: AnyLinear, sharedExpertGate: AnyLinear,
         hidden: Int) {
        self.moe = moe
        self.sharedGateProj = sharedGateProj
        self.sharedUpProj = sharedUpProj
        self.sharedDownProj = sharedDownProj
        self.sharedExpertGate = sharedExpertGate
        self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        // Re-key MoELayer's flat names into Qwen3.5's checkpoint layout.
        for (k, v) in moe.parameters() { out.append((qwen35MoEKey(k), v)) }
        for (k, v) in sharedGateProj.parameters() {
            out.append(("mlp.shared_expert.gate_proj.\(k)", v))
        }
        for (k, v) in sharedUpProj.parameters() {
            out.append(("mlp.shared_expert.up_proj.\(k)", v))
        }
        for (k, v) in sharedDownProj.parameters() {
            out.append(("mlp.shared_expert.down_proj.\(k)", v))
        }
        for (k, v) in sharedExpertGate.parameters() {
            out.append(("mlp.shared_expert_gate.\(k)", v))
        }
        return out
    }

    /// Run the MoE FFN. `MoELayer.decode` commits the passed `cmd`; the
    /// shared expert + the final add run on fresh private buffers, so
    /// the returned tensor never depends on the now-dead `cmd`.
    func forward(_ xNorm: Tensor, position: Int,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // Routed top-K experts — commits `cmd`.
        let routed = moe.decode(xNorm, position: position,
                                cache: StatelessLayerCache(),
                                cmd: cmd, device: device)
        // Shared expert on a fresh buffer: SwiGLU + scalar gate logit.
        let work = device.makeCommandBuffer()
        let sg = sharedGateProj(xNorm, on: work)
        let su = sharedUpProj(xNorm, on: work)
        let sharedInner = Ops.mul(Ops.silu(sg, on: work), su, on: work)
        let sharedOut = sharedDownProj(sharedInner, on: work)
        // shared_expert_gate → [1] logit. Commit so the host can read it.
        let gateLogit = sharedExpertGate(xNorm, on: work)
        work.commit()
        work.waitUntilCompleted()

        // Host: sigmoid(gateLogit) is a scalar; broadcast-scale the shared
        // expert output, then add to the routed combine. `sharedY =
        // sigmoid(shared_expert_gate(x)) * shared_expert(x)`.
        let result = scaleBySigmoidGate35(
            sharedOut, gateLogit: gateLogit,
            addTo: routed, hidden: hidden, device: device)
        return result
    }
}

/// `sigmoid(gateLogit) * value + base`, broadcasting the scalar
/// `gateLogit` ([1]) across the `[hidden]` `value` / `base` tensors.
///
/// `gateLogit` MUST already be resident (its producing command buffer
/// committed + waited) — this helper reads it host-side. The broadcast
/// scalar is materialised via `Tensor.filled` and the scaled add runs on
/// a fresh, locally-committed command buffer so the result is resident.
private func scaleBySigmoidGate35(_ value: Tensor, gateLogit: Tensor,
                                  addTo base: Tensor, hidden: Int,
                                  device: Device) -> Tensor {
    precondition(value.elementCount == hidden && base.elementCount == hidden,
                 "scaleBySigmoidGate35: value/base must be [hidden]")
    let logit = gateLogit.toFloatArray()
    precondition(logit.count == 1,
                 "scaleBySigmoidGate35: shared_expert_gate must project to [1]")
    let gate = sigmoid35(logit[0])
    let cmd = device.makeCommandBuffer()
    let gateVec = Tensor.filled(gate, shape: [hidden], dtype: value.dtype,
                                device: device)
    let scaled = Ops.mul(value, gateVec, on: cmd)
    let result = Ops.add(base, scaled, on: cmd)
    cmd.commit()
    cmd.waitUntilCompleted()
    return result
}

/// Re-key a flat `MoELayer` parameter name into Qwen3.5's checkpoint
/// layout. `MoELayer` emits `gate.*` / `experts.<e>.*`; Qwen3.5 stores
/// `mlp.gate.*` and stacked `mlp.switch_mlp.*` (the per-expert weights
/// are sliced from the stack at load time, so they keep the flat name).
private func qwen35MoEKey(_ k: String) -> String {
    if k.hasPrefix("gate.") {
        return "mlp.gate." + k.dropFirst("gate.".count)
    }
    return "mlp." + k
}

// ─── Qwen35GDNLayerCache — composite GDN-layer cache ─────────────────
//
// A Qwen3.5 GDN layer needs two pieces of per-token state: the rolling
// conv1d window over the `q|k|v` projection (`ConvStateCache`, GPU) and
// the Gated Delta Net recurrent matrix `S[Hv, Dv, Dk]` (`GDNStateCache`,
// double-buffered GPU). This bundles both behind `LayerCacheProtocol`
// so the heterogeneous decode loop can index it uniformly. Mirrors
// Jamba's `JambaMambaLayerCache`.

public final class Qwen35GDNLayerCache: LayerCacheProtocol, @unchecked Sendable {
    public let conv: ConvStateCache
    public let gdn: GDNStateCache

    public private(set) var length: Int = 0
    public let maxSeq: Int = .max

    public init(numKeyHeads: Int, numValueHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convDim: Int, convKernelSize: Int,
                dtype: DType, device: Device = .shared) {
        self.conv = ConvStateCache(nChannels: convDim,
                                   kernelSize: convKernelSize,
                                   dtype: dtype, device: device)
        self.gdn = GDNStateCache(numValueHeads: numValueHeads,
                                 valueHeadDim: valueHeadDim,
                                 keyHeadDim: keyHeadDim,
                                 device: device)
    }

    public func reset() {
        conv.reset()
        gdn.reset()
        length = 0
    }

    public func advance() { length += 1 }

    public var bytesAllocated: Int {
        conv.bytesAllocated + gdn.bytesAllocated
    }

    public var bytesInUse: Int {
        length == 0 ? 0 : bytesAllocated
    }
}

// ─── Qwen35GDNMixer — Gated Delta Net recurrent mixer ────────────────
//
// GPU owns every projection, the conv, the GDN recurrence kernel, the
// gated norm, and `out_proj`. The host owns only the per-head q/k
// RMSNorm + scale and the `[Hv]`-wide `g` / `beta` gate prep — tiny.
//
// Because the prep needs a CPU sync, `forward` commits the command
// buffer it is handed and returns a resident tensor on a fresh buffer
// (the Jamba mamba-mixer pattern).

public final class Qwen35GDNMixer: Module {
    let inProjQKV, inProjZ, inProjB, inProjA, outProj: AnyLinear
    let convW: Tensor        // [kernel, conv_dim]
    let convB: Tensor        // [conv_dim]
    let mixerNorm: RMSNorm   // gated mixer RMSNorm weight [value_head_dim]
    let aLog: [Float]        // [num_value_heads]   raw A_log
    let dtBias: [Float]      // [num_value_heads]
    let hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim: Int
    let keyDim, valueDim, convDim, convKernel: Int
    let eps, invKeyScale: Float
    let dtype: DType

    init(inProjQKV: AnyLinear, inProjZ: AnyLinear,
         inProjB: AnyLinear, inProjA: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor, mixerNorm: RMSNorm,
         aLog: [Float], dtBias: [Float],
         hidden: Int,
         numKeyHeads: Int, numValueHeads: Int,
         keyHeadDim: Int, valueHeadDim: Int,
         keyDim: Int, valueDim: Int, convDim: Int,
         convKernel: Int, eps: Float, invKeyScale: Float,
         dtype: DType) {
        self.inProjQKV = inProjQKV; self.inProjZ = inProjZ
        self.inProjB = inProjB; self.inProjA = inProjA; self.outProj = outProj
        self.convW = convW; self.convB = convB; self.mixerNorm = mixerNorm
        self.aLog = aLog; self.dtBias = dtBias
        self.hidden = hidden
        self.numKeyHeads = numKeyHeads; self.numValueHeads = numValueHeads
        self.keyHeadDim = keyHeadDim; self.valueHeadDim = valueHeadDim
        self.keyDim = keyDim; self.valueDim = valueDim
        self.convDim = convDim; self.convKernel = convKernel
        self.eps = eps; self.invKeyScale = invKeyScale
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inProjQKV.parameters() {
            out.append(("linear_attn.in_proj_qkv.\(k)", v))
        }
        for (k, v) in inProjZ.parameters() {
            out.append(("linear_attn.in_proj_z.\(k)", v))
        }
        for (k, v) in inProjB.parameters() {
            out.append(("linear_attn.in_proj_b.\(k)", v))
        }
        for (k, v) in inProjA.parameters() {
            out.append(("linear_attn.in_proj_a.\(k)", v))
        }
        for (k, v) in outProj.parameters() {
            out.append(("linear_attn.out_proj.\(k)", v))
        }
        for (k, v) in mixerNorm.parameters() {
            out.append(("linear_attn.norm.\(k)", v))
        }
        return out
    }

    /// Single-token GDN mixer forward. `xNorm` is the already-normalized
    /// layer input `[hidden]`. Commits `cmd` mid-way (the host gate prep
    /// needs the GPU projections on the CPU) and returns a resident
    /// `[hidden]` tensor produced on a fresh command buffer.
    func forward(_ xNorm: Tensor, cache: Qwen35GDNLayerCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── GPU phase 1: projections + conv + SiLU ────────────────────
        let qkv = inProjQKV(xNorm, on: cmd)        // [conv_dim]
        let z = inProjZ(xNorm, on: cmd)            // [value_dim]
        let bRaw = inProjB(xNorm, on: cmd)         // [num_value_heads]
        let aRaw = inProjA(xNorm, on: cmd)         // [num_value_heads]

        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: qkv, w: convW, b: convB,
            state: cache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)   // [conv_dim]

        // Commit so the host can read convAct / z / aRaw / bRaw.
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Host phase: split q|k|v, q/k norm + scale, g / beta ───────
        let convHost = convAct.toFloatArray()      // [conv_dim]
        let aHost = aRaw.toFloatArray()            // [num_value_heads]
        let bHost = bRaw.toFloatArray()            // [num_value_heads]

        // Split the conv output: q | k | v.
        let qFlat = Array(convHost[0..<keyDim])
        let kFlat = Array(convHost[keyDim..<(2 * keyDim)])
        let vFlat = Array(convHost[(2 * keyDim)..<(2 * keyDim + valueDim)])

        // Per-head unweighted RMSNorm (eps 1e-6) of q / k, then scale.
        // q ×invScale², k ×invScale  (invScale = key_head_dim^-0.5) —
        // matches mlx-lm's qwen3_5 standard (non-fused) GDN path.
        let qNormed = perHeadRMSNormScale35(
            qFlat, nHeads: numKeyHeads, headDim: keyHeadDim,
            scale: invKeyScale * invKeyScale)
        let kNormed = perHeadRMSNormScale35(
            kFlat, nHeads: numKeyHeads, headDim: keyHeadDim,
            scale: invKeyScale)

        // Per-value-head gates:
        //   g    = exp(-exp(A_log) · softplus(a + dt_bias))
        //   beta = sigmoid(b)
        var gHost = [Float](repeating: 0, count: numValueHeads)
        var betaHost = [Float](repeating: 0, count: numValueHeads)
        for hv in 0..<numValueHeads {
            let dt = softplus35(aHost[hv] + dtBias[hv])
            gHost[hv] = Foundation.exp(-Foundation.exp(aLog[hv]) * dt)
            betaHost[hv] = sigmoid35(bHost[hv])
        }

        // ── GPU phase 2: gatedDeltaStep on a fresh command buffer ─────
        // All GDN-kernel tensors are fp32 (the only emitted dtype).
        let phase2 = device.makeCommandBuffer()
        let qT = makeF32Tensor35(qNormed, device: device)
        let kT = makeF32Tensor35(kNormed, device: device)
        let vT = makeF32Tensor35(vFlat, device: device)
        let gT = makeF32Tensor35(gHost, device: device)
        let betaT = makeF32Tensor35(betaHost, device: device)
        let yT = Tensor.empty(shape: [numValueHeads, valueHeadDim],
                              dtype: .f32, device: device)
        Ops.gatedDeltaStep(
            q: qT, k: kT, v: vT, g: gT, beta: betaT,
            stateIn: cache.gdn.current, into: yT, stateOut: cache.gdn.next,
            numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
            on: phase2)
        phase2.commit()
        phase2.waitUntilCompleted()
        cache.gdn.swap()
        cache.advance()

        // ── Host phase 2: gated mixer RMSNorm ─────────────────────────
        // `Qwen3NextRMSNormGated`: y = rmsNorm(y, weight) · silu(z), the
        // RMSNorm taken per value-head row (`value_head_dim`). The GDN
        // kernel emits `y` in fp32; there is no GPU cast kernel, so the
        // (tiny: `Hv · Dv` ≈ 8K) norm + gate runs host-side and writes
        // the result back in the activation dtype for `out_proj`.
        let yHost = yT.toFloatArray()              // [value_dim] fp32
        let zHost = z.toFloatArray()               // [value_dim] activation
        let normW = readFloats35(mixerNorm.weight) // [value_head_dim]
        var yGatedHost = [Float](repeating: 0, count: valueDim)
        for hv in 0..<numValueHeads {
            let base = hv * valueHeadDim
            var sumSq: Float = 0
            for i in 0..<valueHeadDim {
                let v = yHost[base + i]; sumSq += v * v
            }
            let inv = 1.0 / (sumSq / Float(valueHeadDim) + eps).squareRoot()
            for i in 0..<valueHeadDim {
                let normed = yHost[base + i] * inv * normW[i]
                yGatedHost[base + i] = normed * siluScalar35(zHost[base + i])
            }
        }

        // ── GPU phase 3: out_proj on a fresh command buffer ───────────
        let phase3 = device.makeCommandBuffer()
        let yGatedT = Tensor.empty(shape: [valueDim], dtype: dtype, device: device)
        writeFloats35(yGatedHost, into: yGatedT)
        let result = outProj(yGatedT, on: phase3)
        phase3.commit()
        phase3.waitUntilCompleted()
        return result
    }
}

// ─── Qwen35AttentionMixer — gated multi-head attention ───────────────
//
// `attn_output_gate: true` — `q_proj` projects 2× heads (queries +
// gate); the attention output is multiplied by `sigmoid(gate)` before
// `o_proj`. `q_norm` / `k_norm` are per-head RMSNorm. RoPE is partial
// (`partial_rotary_factor`): only the first `rotaryDim` dims rotate.

public final class Qwen35AttentionMixer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let nHeads, nKVHeads, headDim, rotaryDim: Int
    let ropeTheta: Float
    let attnOutputGate: Bool
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         qNorm: RMSNorm, kNorm: RMSNorm,
         nHeads: Int, nKVHeads: Int, headDim: Int, rotaryDim: Int,
         ropeTheta: Float, attnOutputGate: Bool) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.rotaryDim = rotaryDim; self.ropeTheta = ropeTheta
        self.attnOutputGate = attnOutputGate
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
        return out
    }

    /// Single-token attention forward. Returns the post-o_proj
    /// contribution (residual add done by the enclosing layer). All work
    /// queued on `cmd`; no commit inside.
    func forward(_ xNorm: Tensor, position: Int, cache kv: KVCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // q_proj projects 2× heads when attn_output_gate is set: the
        // first `nHeads · headDim` elements are the queries, the second
        // half is the per-head sigmoid gate.
        let qOut = qProj(xNorm, on: cmd)
        let queries: Tensor
        let gate: Tensor?
        if attnOutputGate {
            // Layout is [nHeads, 2 · headDim] — per head the first
            // `headDim` is the query, the next `headDim` the gate.
            let q2 = qOut.reshaped(to: [nHeads, 2 * headDim])
            queries = sliceHeadHalves35(
                q2, nHeads: nHeads, headDim: headDim, takeFirst: true,
                on: cmd, device: device)
            gate = sliceHeadHalves35(
                q2, nHeads: nHeads, headDim: headDim, takeFirst: false,
                on: cmd, device: device)
        } else {
            queries = qOut
            gate = nil
        }
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head q_norm / k_norm (weighted RMSNorm over headDim).
        let qNormed = Ops.rmsNormRows(
            queries, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nHeads, rowSize: headDim, on: cmd)
        let kNormed = Ops.rmsNormRows(
            k, weight: kNorm.weight, eps: kNorm.eps,
            nRows: nKVHeads, rowSize: headDim, on: cmd)

        // Partial RoPE — rotate only the first `rotaryDim` dims of each
        // head, in place.
        Ops.ropePartial(qNormed, position: position,
                        headDim: headDim, rotaryDim: rotaryDim,
                        thetaBase: ropeTheta, on: cmd)
        Ops.ropePartial(kNormed, position: position,
                        headDim: headDim, rotaryDim: rotaryDim,
                        thetaBase: ropeTheta, on: cmd)

        // GPU KV cache update.
        kv.appendOnGPU(kFlat: kNormed.reshaped(to: [nKVHeads, headDim]),
                       vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qNormed.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.maxSeq,
            scale: scale, on: cmd)

        // Gated output: attnOut * sigmoid(gate), then o_proj.
        var attnFlat = attnOut.reshaped(to: [nHeads * headDim])
        if let gate {
            attnFlat = Ops.mul(attnFlat, Ops.sigmoid(gate, on: cmd), on: cmd)
        }
        return oProj(attnFlat, on: cmd)
    }
}

// ─── Qwen35GDNLayer — a "linear_attention" layer ─────────────────────
//
// One stack-interleaved hybrid layer with a GDN mixer. Conforms to
// `DecoderLayer`; its cache slot is a `GDNStateCache`.
//
// `commitsCommandBuffer` is ALWAYS true: the GDN mixer's host gate prep
// commits the command buffer mid-decode (an MoE FFN would also commit).
// `Qwen35Model.forward` refreshes the work buffer after this layer.

public final class Qwen35GDNLayer: Module, DecoderLayer {
    let inputNorm, postNorm: RMSNorm
    let mixer: Qwen35GDNMixer
    let ffn: Qwen35FFN
    let hidden: Int

    public let commitsCommandBuffer: Bool = true

    init(inputNorm: RMSNorm, postNorm: RMSNorm,
         mixer: Qwen35GDNMixer, ffn: Qwen35FFN, hidden: Int) {
        self.inputNorm = inputNorm; self.postNorm = postNorm
        self.mixer = mixer; self.ffn = ffn; self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        out.append(contentsOf: mixer.parameters())
        out.append(contentsOf: qwen35FFNParameters(ffn))
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `GDNStateCache`.
    /// IMPORTANT: commits `cmd` (the GDN host gate prep + an MoE FFN
    /// both need a CPU sync). The host model refreshes `cmd` afterwards.
    public func decode(_ h: Tensor, position: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let gc = cache as? Qwen35GDNLayerCache else {
            fatalError("Qwen35GDNLayer: expected Qwen35GDNLayerCache, "
                       + "got \(type(of: cache))")
        }
        // ── Mixer half — pre-norm + GDN mixer + residual add ──────────
        let xNorm = inputNorm(h, on: cmd)
        // mixer.forward commits `cmd` and returns a resident tensor.
        let mixerOut = mixer.forward(xNorm, cache: gc, cmd: cmd, device: device)

        // `h` was produced on the now-committed `cmd`; it is resident.
        // The residual add + FFN run on a fresh command buffer.
        let ffnCmd = device.makeCommandBuffer()
        let postMix = Ops.add(h, mixerOut, on: ffnCmd)
        return qwen35ApplyFFN(ffn, postMix: postMix, postNorm: postNorm,
                              position: position, cmd: ffnCmd,
                              commitCmd: true, device: device)
    }
}

// ─── Qwen35AttentionLayer — a "full_attention" layer ─────────────────
//
// One stack-interleaved hybrid layer with a multi-head attention mixer.
// Conforms to `DecoderLayer`; its cache slot is a `KVCache`.
//
// `commitsCommandBuffer` is true only when the FFN is an MoE block
// (attention itself is pure GPU).

public final class Qwen35AttentionLayer: Module, DecoderLayer {
    let inputNorm, postNorm: RMSNorm
    let mixer: Qwen35AttentionMixer
    let ffn: Qwen35FFN
    let hidden: Int

    public let commitsCommandBuffer: Bool

    init(inputNorm: RMSNorm, postNorm: RMSNorm,
         mixer: Qwen35AttentionMixer, ffn: Qwen35FFN, hidden: Int) {
        self.inputNorm = inputNorm; self.postNorm = postNorm
        self.mixer = mixer; self.ffn = ffn; self.hidden = hidden
        if case .moe = ffn { self.commitsCommandBuffer = true }
        else { self.commitsCommandBuffer = false }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        out.append(contentsOf: mixer.parameters())
        out.append(contentsOf: qwen35FFNParameters(ffn))
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `KVCache`. Commits
    /// `cmd` only when the FFN is an MoE block.
    public func decode(_ h: Tensor, position: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("Qwen35AttentionLayer: expected KVCache, "
                       + "got \(type(of: cache))")
        }
        // ── Mixer half — pre-norm + attention + residual add ──────────
        let xNorm = inputNorm(h, on: cmd)
        let mixerOut = mixer.forward(xNorm, position: position, cache: kv,
                                     cmd: cmd, device: device)
        let postMix = Ops.add(h, mixerOut, on: cmd)

        // ── Feed-forward half ─────────────────────────────────────────
        // `cmd` is the host model's `workCmd`; the model owns its commit
        // (or swaps it after an MoE FFN). This layer does not commit it
        // for the dense path.
        return qwen35ApplyFFN(ffn, postMix: postMix, postNorm: postNorm,
                              position: position, cmd: cmd,
                              commitCmd: false, device: device)
    }
}

// ─── Shared FFN helpers ──────────────────────────────────────────────

/// Collect the `(name, tensor)` parameters of a layer's FFN half.
private func qwen35FFNParameters(_ ffn: Qwen35FFN) -> [(String, Tensor)] {
    switch ffn {
    case .dense(let mlp): return mlp.parameters()
    case .moe(let moe): return moe.parameters()
    }
}

/// Apply the feed-forward half of a Qwen3.5 layer: post-attention norm,
/// FFN, and the residual add.
///
/// - `commitCmd`: when `true`, this function owns `cmd` and commits it
///   (the GDN-layer path, handed a fresh buffer). When `false`, the
///   caller owns `cmd`'s commit (the attention-layer dense path, where
///   `cmd` is the host model's `workCmd`). When the FFN is an MoE block,
///   it commits `cmd` regardless; the residual add then runs on a fresh,
///   locally-committed buffer so the returned tensor is resident.
private func qwen35ApplyFFN(_ ffn: Qwen35FFN, postMix: Tensor, postNorm: RMSNorm,
                            position: Int, cmd: MTLCommandBuffer,
                            commitCmd: Bool, device: Device) -> Tensor {
    let ffnNorm = postNorm(postMix, on: cmd)
    switch ffn {
    case .dense(let mlp):
        let ffnOut = mlp.forward(ffnNorm, cmd: cmd)
        let result = Ops.add(postMix, ffnOut, on: cmd)
        if commitCmd {
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        return result
    case .moe(let moe):
        // Qwen35MoEFFN.forward commits `cmd`; run the residual add on a
        // fresh buffer so the returned tensor does not depend on a dead
        // command buffer.
        let ffnOut = moe.forward(ffnNorm, position: position,
                                 cmd: cmd, device: device)
        let addCmd = device.makeCommandBuffer()
        let result = Ops.add(postMix, ffnOut, on: addCmd)
        addCmd.commit()
        addCmd.waitUntilCompleted()
        return result
    }
}

// ─── Qwen35Model ─────────────────────────────────────────────────────

public final class Qwen35Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a GDN or attention
    /// hybrid layer, ordered by `layer_types`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    /// GDN mixer geometry.
    public let numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim: Int
    public let convDim, convKernel: Int
    public let dtype: DType

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [Qwen35LayerKind]
    /// True when any layer carries an MoE FFN (purely informational —
    /// every GDN layer commits the command buffer regardless).
    public let hasMoE: Bool

    init(embedTokens: AnyEmbedding, layers: [any DecoderLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         numKeyHeads: Int, numValueHeads: Int,
         keyHeadDim: Int, valueHeadDim: Int,
         convDim: Int, convKernel: Int,
         vocab: Int, maxSeq: Int, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.numKeyHeads = numKeyHeads; self.numValueHeads = numValueHeads
        self.keyHeadDim = keyHeadDim; self.valueHeadDim = valueHeadDim
        self.convDim = convDim; self.convKernel = convKernel
        self.vocab = vocab; self.maxSeq = maxSeq; self.dtype = dtype
        self.layerKinds = layers.map { layer in
            switch layer {
            case is Qwen35GDNLayer: return .gdn
            case is Qwen35AttentionLayer: return .attention
            default: return .gdn
            }
        }
        self.hasMoE = layers.contains { layer in
            if let g = layer as? Qwen35GDNLayer, case .moe = g.ffn { return true }
            if let a = layer as? Qwen35AttentionLayer, case .moe = a.ffn { return true }
            return false
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            let params: [(String, Tensor)]
            switch layer {
            case let l as Qwen35GDNLayer: params = l.parameters()
            case let l as Qwen35AttentionLayer: params = l.parameters()
            default: params = []
            }
            for (k, v) in params { out.append(("model.layers.\(i).\(k)", v)) }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One cache per layer index, matching the layer kind:
    ///   GDN → GDNStateCache, attention → KVCache.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return layerKinds.map { kind in
            switch kind {
            case .gdn:
                return Qwen35GDNLayerCache(
                    numKeyHeads: numKeyHeads,
                    numValueHeads: numValueHeads,
                    keyHeadDim: keyHeadDim,
                    valueHeadDim: valueHeadDim,
                    convDim: convDim, convKernelSize: convKernel,
                    dtype: dtype, device: device)
            case .attention:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, device: device)
            }
        }
    }

    /// Queue a single-token forward pass onto `cmd`. **Does not commit
    /// `cmd`** — the protocol contract holds, so the default
    /// `forwardSample` / `forwardSampleCategorical` extensions compose
    /// their output kernels onto `cmd` and commit once.
    ///
    /// CRITICAL — command-buffer contract. Every Qwen3.5 *GDN* layer
    /// commits the command buffer it is handed (the host gate prep runs
    /// CPU-side); an MoE FFN also commits. So the caller's `cmd` must
    /// NEVER be handed to a layer — if it were, the first GDN layer
    /// would commit it and the caller's later commit would
    /// double-commit. Instead the embedding + every layer run on
    /// internal `workCmd` buffers (committed by the layers themselves /
    /// refreshed after each committing layer), and ONLY the final
    /// `norm` + `lm_head` queue onto the caller's pristine `cmd`.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // The embedding + layers run on internal buffers — never `cmd`.
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.decode(h, position: position, cache: caches[i],
                             cmd: workCmd, device: device)
            // Refresh `workCmd` if the layer committed it.
            let committed: Bool
            switch layer {
            case let l as Qwen35GDNLayer: committed = l.commitsCommandBuffer
            case let l as Qwen35AttentionLayer: committed = l.commitsCommandBuffer
            default: committed = false
            }
            if committed { workCmd = device.makeCommandBuffer() }
        }

        // If the last layer was a non-committing attention layer with a
        // dense FFN, `workCmd` still carries its uncommitted work —
        // commit it so `h` is resident before the caller's `cmd` reads
        // it. (A GDN or MoE-bearing last layer already left `h`
        // resident and a fresh `workCmd` pending.)
        if let last = layers.last,
           !((last as? Qwen35GDNLayer)?.commitsCommandBuffer ?? false),
           !((last as? Qwen35AttentionLayer)?.commitsCommandBuffer ?? false) {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        // Final norm + lm_head queue onto the caller's pristine `cmd`.
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    // ─── VLM embedding-input path ────────────────────────────────────
    //
    // Qwen3.5 is a VL-target text backbone (Qwen3-VL-MoE wraps it). The
    // splice supplies a `[hidden]` row directly — a vision-encoder token
    // or a text-token embedding the VL model looked up — so the forward
    // is identical to `forward(tokenId:...)` minus the embedding gather.
    // The same command-buffer contract holds: layers run on internal
    // `workCmd` buffers; only `norm` + `lm_head` touch the caller's
    // pristine `cmd`.

    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "Qwen35Model.forward(inputEmbedding:): expected [\(hidden)], "
                     + "got \(inputEmbedding.shape)")
        var h = inputEmbedding.reshaped(to: [hidden])

        // Layers run on internal buffers — never the caller's `cmd`.
        var workCmd = device.makeCommandBuffer()
        for (i, layer) in layers.enumerated() {
            h = layer.decode(h, position: position, cache: caches[i],
                             cmd: workCmd, device: device)
            let committed: Bool
            switch layer {
            case let l as Qwen35GDNLayer: committed = l.commitsCommandBuffer
            case let l as Qwen35AttentionLayer: committed = l.commitsCommandBuffer
            default: committed = false
            }
            if committed { workCmd = device.makeCommandBuffer() }
        }
        // Flush a trailing non-committing layer's pending work so `h` is
        // resident before the caller's `cmd` reads it.
        if let last = layers.last,
           !((last as? Qwen35GDNLayer)?.commitsCommandBuffer ?? false),
           !((last as? Qwen35AttentionLayer)?.commitsCommandBuffer ?? false) {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Raw embedding-table lookup for one text token.
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

// ─── Load-time + host-prep helpers ───────────────────────────────────

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
private func readFloats35(_ t: Tensor) -> [Float] {
    t.toFloatArray()
}

/// Build a fresh fp32 tensor from a `[Float]` host array.
private func makeF32Tensor35(_ values: [Float], device: Device) -> Tensor {
    let t = Tensor.empty(shape: [values.count], dtype: .f32, device: device)
    t.copyIn(from: values)
    return t
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVector35(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}

/// Transpose HF conv1d.weight `[C, K, 1]` → `[K, C]` for the metaltile
/// conv kernel. The trailing `1` is the depthwise group dim; row-major
/// the source is `[c][k]`-ordered.
private func transposeConv1dWeight35(src: Tensor, kernel K: Int, channels C: Int,
                                     dtype: DType, device: Device) -> Tensor {
    let floats = readFloats35(src)
    precondition(floats.count == K * C, "Qwen3.5: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C {
        for k in 0..<K { dst[k * C + c] = floats[c * K + k] }
    }
    let t = Tensor.empty(shape: [K, C], dtype: dtype, device: device)
    writeFloats35(dst, into: t)
    return t
}

/// Write a `[Float]` into an existing tensor, converting to its dtype.
private func writeFloats35(_ values: [Float], into t: Tensor) {
    precondition(values.count == t.elementCount, "Qwen3.5: writeFloats size mismatch")
    switch t.dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(from: values.map { v -> UInt16 in
            // Round-to-nearest before truncating the low 16 bits.
            let bits = v.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(rounded >> 16)
        })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("Qwen3.5: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Per-head unweighted RMSNorm (eps 1e-6) of a flat `[nHeads · headDim]`
/// host vector, scaled by `scale`. Matches the standard (non-fused) GDN
/// path's `scale · rmsNorm(x, weight=None)`.
private func perHeadRMSNormScale35(_ x: [Float], nHeads: Int, headDim: Int,
                                   scale: Float) -> [Float] {
    precondition(x.count == nHeads * headDim,
                 "Qwen3.5: perHeadRMSNorm size mismatch")
    let eps: Float = 1e-6
    var out = [Float](repeating: 0, count: x.count)
    for h in 0..<nHeads {
        let base = h * headDim
        var sumSq: Float = 0
        for i in 0..<headDim { let v = x[base + i]; sumSq += v * v }
        let inv = scale / (sumSq / Float(headDim) + eps).squareRoot()
        for i in 0..<headDim { out[base + i] = x[base + i] * inv }
    }
    return out
}

/// Scalar softplus: `log(1 + exp(x))`, numerically stable.
private func softplus35(_ x: Float) -> Float {
    if x > 20 { return x }
    if x < -20 { return Foundation.exp(x) }
    return Foundation.log1p(Foundation.exp(x))
}

/// Scalar logistic sigmoid.
private func sigmoid35(_ x: Float) -> Float {
    1.0 / (1.0 + Foundation.exp(-x))
}

/// Scalar SiLU / swish: `x · sigmoid(x)`.
private func siluScalar35(_ x: Float) -> Float {
    x * sigmoid35(x)
}

/// Extract the per-head query OR gate half of the gated `q_proj` output.
///
/// `attn_output_gate` makes `q_proj` emit `2 · headDim` per head, laid
/// out `[query(headDim) | gate(headDim)]` (mlx-lm reshapes the flat
/// projection to `[nHeads, 2·headDim]` then `split(parts: 2, axis: -1)`).
/// Reinterpreting `q2` as a `[nHeads · 2, headDim]` row table, the query
/// rows are the even indices `2h` and the gate rows the odd indices
/// `2h + 1`. `Ops.gather` row-copies them into a contiguous
/// `[nHeads · headDim]` result on the GPU — no host sync, no commit.
private func sliceHeadHalves35(_ q2: Tensor, nHeads: Int, headDim: Int,
                               takeFirst: Bool,
                               on cmd: MTLCommandBuffer,
                               device: Device) -> Tensor {
    precondition(q2.elementCount == nHeads * 2 * headDim,
                 "sliceHeadHalves35: q2 must be [nHeads, 2·headDim]")
    let table = q2.reshaped(to: [nHeads * 2, headDim])
    // Row indices: even rows = queries, odd rows = gates.
    var rows = [UInt32](repeating: 0, count: nHeads)
    for h in 0..<nHeads { rows[h] = UInt32(2 * h + (takeFirst ? 0 : 1)) }
    let idxBuf = device.makeBuffer(length: nHeads * 4)
    rows.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, nHeads * 4) }
    let idx = Tensor(buffer: idxBuf, offset: 0, shape: [nHeads], dtype: .u32)
    let gathered = Ops.gather(table: table, tokenIds: idx, on: cmd)
    return gathered.reshaped(to: [nHeads * headDim])
}
