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
        // ── lm_head prefix — VLM-wrapped Qwen3.6 ships lm_head under
        //    `language_model.lm_head.*` (NOT under `language_model.model`).
        //    A text-only conversion drops the wrapper entirely.
        let lmHeadCandidates = ["language_model.lm_head", "lm_head"]
        let lmHeadPrefix = lmHeadCandidates.first(where: {
            weights.has("\($0).weight")
        })

        let quant = config.quantization

        // ── Activation dtype — from the embedding table ───────────────
        // Qwen3.6 ships a *quantized* embedding (uint32-packed weight +
        // bf16 scales/biases). The activation dtype must come from the
        // scales tensor in that case — the weight is a u32 pack table.
        let embedW = try weights.tensor(named: "\(modelPrefix).embed_tokens.weight")
        let activationDtype: DType
        if weights.isQuantized("\(modelPrefix).embed_tokens") {
            let sc = try weights.tensor(named: "\(modelPrefix).embed_tokens.scales")
            activationDtype = sc.dtype
        } else {
            activationDtype = embedW.dtype
        }
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "Qwen3.5: unexpected activation dtype \(activationDtype)")
        // `loadEmbedding` picks Embedding vs QuantizedEmbedding based on
        // the bundle's quant triplet — Qwen3.6's quantized embed is
        // routed through `dequantGather` automatically.
        let embedTokens = try loadEmbedding(
            base: "\(modelPrefix).embed_tokens", in: weights,
            hidden: hidden, quantization: quant)

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

        // lm_head — most Qwen3.5 dense checkpoints tie to the embedding
        // table; Qwen3.6 ships an UNTIED quantized lm_head under
        // `language_model.lm_head.*`. Honour an explicit lm_head if the
        // checkpoint provides one (located via `lmHeadPrefix` above),
        // otherwise tie to the (raw) embedding weight.
        let lmHead: AnyLinear
        if !tieEmbed, let lmPrefix = lmHeadPrefix {
            // Explicit lm_head (Qwen3.6 + most production checkpoints).
            // `lmHeadPrefix` handles both top-level `lm_head` and
            // `language_model.lm_head` (multimodal nested).
            lmHead = try loadLinear(base: lmPrefix, in: weights, quantization: quant)
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
            // Tied to a raw (unquantized) embedding — straight Linear.
            lmHead = AnyLinear(Linear(weight: embedW))
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
    ///
    /// Qwen3.5 dense ships the linear-attention projections in raw bf16;
    /// Qwen3.6 ships them affine-quantized (mlx layout, scales/biases
    /// alongside the u32-packed weight). `loadLinear` picks the right
    /// `Linear` vs `QuantizedLinear` variant per-tensor.
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

        // Batched gather BGEMM fast path: when the experts are mlx
        // int4-quantized AND the shapes satisfy the bm16 tile contract
        // (N%32 == 0, K%32 == 0), capture the stacked weight tensors so
        // `MoELayer.decode` can dispatch `mt_moe_gather_qmm_mma_int4_bm16_*`
        // — one launch per projection instead of `topK` sequential
        // SwiGLU triplets (24 → 3 dispatches at `topK=8`).
        let stackedInt4 = tryLoadStackedInt4(
            base: "\(p).switch_mlp",
            weights: weights, quant: quant,
            numExperts: numExperts, moeIntermediate: moeIntermediate,
            hidden: hidden)
        // The MoELayer carries only the routed experts — Qwen3.5's
        // shared expert is sigmoid-gated, which the plain MoELayer's
        // unconditional shared-expert add cannot express, so it is held
        // separately on the FFN wrapper.
        let moe = MoELayer(
            gate: gate,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            router: router, hidden: hidden,
            stackedInt4Experts: stackedInt4)

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

    /// Try to load the stacked-int4-experts triplet for the batched
    /// gather BGEMM fast path in `MoELayer.decode`. Returns nil unless
    /// the checkpoint is mlx affine 4-bit AND the gate/up/down shapes
    /// satisfy the `mt_moe_gather_qmm_mma_int4_bm16_*` tile contract
    /// (`hidden % 32 == 0`, `moeIntermediate % 32 == 0`, derived bits
    /// per-tensor == 4). The returned tensors are direct handles into
    /// the safetensors mmap, so no extra allocations.
    ///
    /// On Qwen3.6-A3B (256 experts, hidden=2048, moeIntermediate=512,
    /// groupSize=64, bits=4) this is the hot path — at decode `topK=8`
    /// it replaces 24 sequential per-expert SwiGLU dispatches with 3
    /// batched kernel launches + a handful of element-wise ops.
    private static func tryLoadStackedInt4(
        base: String, weights: SafeTensorsBundle,
        quant: ModelConfig.QuantizationConfig?,
        numExperts: Int, moeIntermediate: Int, hidden: Int
    ) -> MoELayer.StackedInt4Experts? {
        guard let q = quant,
              weights.isQuantized("\(base).gate_proj"),
              weights.isQuantized("\(base).up_proj"),
              weights.isQuantized("\(base).down_proj")
        else { return nil }
        // bm16 tile contract — N must be multiple of 32, K must be
        // multiple of 32. Gate / up: N=moeIntermediate K=hidden. Down:
        // N=hidden K=moeIntermediate. Both reduce to the same
        // divisibility test, since `moeIntermediate % 32 == 0 &&
        // hidden % 32 == 0` implies both directions.
        guard hidden % 32 == 0, moeIntermediate % 32 == 0 else { return nil }
        // bm16 / bgemm fast path is int4-only — derive bits per-tensor
        // from the stacked shapes; bail if any of the three is not 4.
        do {
            let gw = try weights.tensor(named: "\(base).gate_proj.weight")
            let gs = try weights.tensor(named: "\(base).gate_proj.scales")
            let gb = try weights.tensor(named: "\(base).gate_proj.biases")
            let uw = try weights.tensor(named: "\(base).up_proj.weight")
            let us = try weights.tensor(named: "\(base).up_proj.scales")
            let ub = try weights.tensor(named: "\(base).up_proj.biases")
            let dw = try weights.tensor(named: "\(base).down_proj.weight")
            let ds = try weights.tensor(named: "\(base).down_proj.scales")
            let db = try weights.tensor(named: "\(base).down_proj.biases")
            // Derived bit-width must be 4 for all three projections.
            let gateBits = deriveAffineQuantBits(
                weightPackedCols: gw.shape[gw.shape.count - 1],
                scaleCols: gs.shape[gs.shape.count - 1],
                groupSize: q.groupSize)
            let upBits = deriveAffineQuantBits(
                weightPackedCols: uw.shape[uw.shape.count - 1],
                scaleCols: us.shape[us.shape.count - 1],
                groupSize: q.groupSize)
            let downBits = deriveAffineQuantBits(
                weightPackedCols: dw.shape[dw.shape.count - 1],
                scaleCols: ds.shape[ds.shape.count - 1],
                groupSize: q.groupSize)
            guard gateBits == 4, upBits == 4, downBits == 4 else { return nil }
            // Activations dtype matches scales / biases (mlx convention).
            let dtype = gs.dtype
            guard dtype == .f16 || dtype == .bf16 || dtype == .f32 else { return nil }
            return MoELayer.StackedInt4Experts(
                gateWeight: gw, gateScales: gs, gateBiases: gb,
                upWeight: uw, upScales: us, upBiases: ub,
                downWeight: dw, downScales: ds, downBiases: db,
                numExperts: numExperts, moeIntermediate: moeIntermediate,
                hidden: hidden, groupSize: q.groupSize, dtype: dtype)
        } catch {
            // Any missing tensor → silently fall back. The per-expert
            // sliced path will load these same tensors via the regular
            // `sliceStackedExperts` route.
            return nil
        }
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

    /// T-batched: down(silu(gate(x)) * up(x)) over T rows. Returns
    /// `[T, hidden]` flat. Three batched projections + one elementwise
    /// SwiGLU pass.
    func forwardMany(_ xNormFlat: Tensor, t: Int,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let xRows = xNormFlat.reshaped(to: [t, xNormFlat.elementCount / t])
        let g = gateProj.callMany(xRows, t: t, on: cmd, device: device)
        let u = upProj.callMany(xRows, t: t, on: cmd, device: device)
        let inner = Ops.mul(Ops.silu(g, on: cmd), u, on: cmd)
        return downProj.callMany(inner, t: t, on: cmd, device: device)
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
        // The fused-sigmoid kernel keeps the scalar on the GPU, so this
        // buffer no longer needs a `waitUntilCompleted` — it just needs
        // to be in flight so the FMA can hazard-track its dependencies.
        let work = device.makeCommandBuffer()
        let sg = sharedGateProj(xNorm, on: work)
        let su = sharedUpProj(xNorm, on: work)
        let sharedInner = Ops.mul(Ops.silu(sg, on: work), su, on: work)
        let sharedOut = sharedDownProj(sharedInner, on: work)
        let gateLogit = sharedExpertGate(xNorm, on: work)
        work.commit()

        // GPU fan-out: `out = routed + sigmoid(gateLogit) * sharedOut`
        // in one dispatch. Replaces the prior host detour
        // (`gateLogit.toFloatArray()` + host sigmoid + `Tensor.filled`
        // broadcast + mul + add + commit + wait) — saves one host stall
        // per MoE layer per token. Fires on all 40 Qwen3.6-A3B layers.
        let fmaCmd = device.makeCommandBuffer()
        let result = Tensor.empty(shape: [hidden], dtype: routed.dtype, device: device)
        Ops.sigmoidScalarFMA(
            gate: gateLogit, value: sharedOut, base: routed,
            into: result, on: fmaCmd)
        fmaCmd.commit()
        return result
    }

    /// T-batched MoE FFN. `xNormFlat` is `[T, hidden]` flat; returns
    /// `[T, hidden]` flat. `MoELayer.decodeMany` commits the caller's
    /// `cmd`; the shared-expert SwiGLU runs on a fresh `work` cmd and
    /// the per-row sigmoidScalarFMA fan-out on fresh `fmaCmd`s. Mirrors
    /// the single-token `forward`'s commit pattern; the returned
    /// `outFlat` is in-flight on the last committed `fmaCmd` and
    /// downstream reads hazard-track against it.
    func forwardMany(_ xNormFlat: Tensor, t: Int,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let dt = xNormFlat.dtype
        let dtBytes = dt.byteSize

        // ── Routed top-K experts — commits `cmd`. ────────────────────────
        let routed = moe.decodeMany(xNormFlat, t: t, cmd: cmd, device: device)
        // routed: `[T, hidden]` flat, on `work`-internal-committed cmd.

        // ── Shared expert batched: SwiGLU + scalar gate logit ────────────
        let work = device.makeCommandBuffer()
        let xRows = xNormFlat.reshaped(to: [t, hidden])
        let sgAll = sharedGateProj.callMany(xRows, t: t, on: work, device: device)
        let suAll = sharedUpProj.callMany(xRows, t: t, on: work, device: device)
        let sharedInnerAll = Ops.mul(Ops.silu(sgAll, on: work), suAll, on: work)
        let sharedOutAll = sharedDownProj.callMany(sharedInnerAll, t: t,
                                                    on: work, device: device)
        // sharedExpertGate is `hidden → 1`; per row this is one scalar.
        let gateLogitsAll = sharedExpertGate.callMany(xRows, t: t,
                                                      on: work, device: device)
        work.commit()

        // ── Per-row sigmoidScalarFMA chained on one cmd ──────────────────
        // `Ops.sigmoidScalarFMA` requires a `[1]` scalar gate, so the
        // T-row fan-out stays as T dispatches (each per-row), but ALL
        // share one command buffer. Previously each row spun up its own
        // `device.makeCommandBuffer() + commit()` — T·40 layers = 20 480
        // tiny cmd allocations at Qwen3.6-A3B T=512 prefill. Single cmd
        // keeps Metal's dispatcher pipelining contiguous and saves the
        // per-buffer encode/finalise overhead.
        let outFlat = Tensor.empty(shape: [t * hidden], dtype: dt, device: device)
        let fmaCmd = device.makeCommandBuffer()
        for r in 0..<t {
            let gateRow = Tensor(buffer: gateLogitsAll.buffer,
                                 offset: gateLogitsAll.offset + r * dtBytes,
                                 shape: [1], dtype: dt)
            let sharedRow = Tensor(buffer: sharedOutAll.buffer,
                                   offset: sharedOutAll.offset + r * hidden * dtBytes,
                                   shape: [hidden], dtype: dt)
            let routedRow = Tensor(buffer: routed.buffer,
                                   offset: routed.offset + r * hidden * dtBytes,
                                   shape: [hidden], dtype: dt)
            let outRow = Tensor(buffer: outFlat.buffer,
                                offset: outFlat.offset + r * hidden * dtBytes,
                                shape: [hidden], dtype: dt)
            Ops.sigmoidScalarFMA(
                gate: gateRow, value: sharedRow, base: routedRow,
                into: outRow, on: fmaCmd)
        }
        fmaCmd.commit()
        return outFlat
    }
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

    /// Set the position counter directly without touching tensor state.
    /// Spec-decode uses this after manually restoring conv + gdn state
    /// from a snapshot: tensors are restored separately; this fixes the
    /// position counter back to the snapshotted value. Going through
    /// `reset()` would zero the just-restored tensors.
    public func setLength(_ length: Int) {
        precondition(length >= 0, "Qwen35GDNLayerCache.setLength: must be ≥ 0")
        self.length = length
    }

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

    // ─── Fused `mt_gated_delta_prep_step` path constants (fp32) ───────
    //
    // The fused kernel takes its tensors all in a single dtype. The GDN
    // recurrence state stays fp32 (see `GDNStateCache` header — bf16
    // drift over long decodes is intolerable on this recurrence), so we
    // run the fused step in fp32 and cast bf16 model activations
    // (convAct / aRaw / bRaw) up to fp32 on the GPU before dispatch via
    // `Ops.castToF32`.
    //
    // Layout / values are pinned to the kernel's contract — see
    // `crates/metaltile-std/src/ffai/gated_delta_prep.rs`:
    //   qNormWeightF32 : [Hk·Dk] fp32, filled with `invKeyScale²` —
    //                    folds the per-head scale into the weight so
    //                    the kernel runs the *unweighted*
    //                    `perHeadRMSNorm` path while reusing its
    //                    weighted-RMSNorm inner loop.
    //   kNormWeightF32 : [Hk·Dk] fp32, filled with `invKeyScale`.
    //   aLogTF32       : [Hv]    fp32.
    //   dtBiasTF32     : [Hv]    fp32.
    let qNormWeightF32, kNormWeightF32: Tensor
    let aLogTF32, dtBiasTF32: Tensor

    /// fp32 epsilon as a 1-element buffer for `Ops.gatedMixerNorm`.
    /// Built once in init so the GPU-side phase-2 path doesn't pay an
    /// allocation per token.
    let epsBufFused: Tensor

    /// `FFAI_GDN_FUSED_PREP=1` route through `mt_gated_delta_prep_step`
    /// + `mt_gated_mixer_norm`. Cached at init so per-token env lookups
    /// disappear and the GDN *layer* (not just the mixer) can branch on
    /// it — when fused, the mixer keeps `cmd` in-flight so the residual
    /// add + FFN chain onto the same command buffer.
    let fused: Bool

    /// Pre-allocated per-call scratch tensors. The fused GDN path
    /// writes / reads these inside one command buffer per decode token;
    /// the engine's `workCmd.commit()` + caller wait between tokens
    /// guarantees the GPU has finished the previous token's writes
    /// before the next token starts writing the same scratch. Metal's
    /// default hazard tracking serialises any in-flight reads against
    /// the next write. Replaces six per-decode-token `Tensor.empty`
    /// allocations (× 30 GDN layers on Qwen3.6-A3B = 180 MTLBuffer
    /// allocs / token recovered).
    let convOutScratch: Tensor
    let convActF32Scratch: Tensor
    let aRawF32Scratch: Tensor
    let bRawF32Scratch: Tensor
    let yF32Scratch: Tensor
    let yGatedScratch: Tensor

    init(inProjQKV: AnyLinear, inProjZ: AnyLinear,
         inProjB: AnyLinear, inProjA: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor, mixerNorm: RMSNorm,
         aLog: [Float], dtBias: [Float],
         hidden: Int,
         numKeyHeads: Int, numValueHeads: Int,
         keyHeadDim: Int, valueHeadDim: Int,
         keyDim: Int, valueDim: Int, convDim: Int,
         convKernel: Int, eps: Float, invKeyScale: Float,
         dtype: DType,
         device: Device = .shared) {
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

        // ── Pre-build the fused-path constant tensors (fp32) ────────
        let hkDk = numKeyHeads * keyHeadDim
        let qScale = invKeyScale * invKeyScale          // unweighted RMSNorm × invKeyScale²
        let kScale = invKeyScale                        // unweighted RMSNorm × invKeyScale
        self.qNormWeightF32 = makeF32Tensor35(
            [Float](repeating: qScale, count: hkDk), device: device)
        self.kNormWeightF32 = makeF32Tensor35(
            [Float](repeating: kScale, count: hkDk), device: device)
        self.aLogTF32 = makeF32Tensor35(aLog, device: device)
        self.dtBiasTF32 = makeF32Tensor35(dtBias, device: device)
        self.epsBufFused = makeF32Tensor35([eps], device: device)
        self.fused = ProcessInfo.processInfo.environment["FFAI_GDN_FUSED_PREP"] != nil

        // Per-decode-token scratch — pre-allocated once at init so the
        // fused GDN path doesn't pay 6 × MTLBuffer allocations per call.
        // See the comments next to the field declarations for the
        // cross-token safety argument (Metal hazard tracking + the
        // engine's per-token cmd wait).
        self.convOutScratch = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        self.convActF32Scratch = Tensor.empty(shape: [convDim], dtype: .f32, device: device)
        self.aRawF32Scratch = Tensor.empty(shape: [numValueHeads], dtype: .f32, device: device)
        self.bRawF32Scratch = Tensor.empty(shape: [numValueHeads], dtype: .f32, device: device)
        self.yF32Scratch = Tensor.empty(shape: [numValueHeads, valueHeadDim],
                                        dtype: .f32, device: device)
        self.yGatedScratch = Tensor.empty(shape: [valueDim], dtype: dtype, device: device)
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

        Ops.conv1dCausalStep(
            x: qkv, w: convW, b: convB,
            state: cache.conv.state, into: convOutScratch,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        // `convAct` is the silu of convOutScratch in the same dtype as
        // the model (bf16/f16). In the fused path below we then cast
        // it to f32 — so when `fused` is on, we route through the
        // fused `siluCastToF32` instead, saving 1 elementwise silu
        // dispatch + 1 castToF32 dispatch (= 1 fused dispatch) per
        // GDN layer. Outside the fused path we still need the bf16
        // silu output for the legacy host phase.
        let convAct: Tensor
        if fused {
            // Output goes DIRECTLY into the f32 scratch — fused kernel
            // reads bf16/f16, computes silu at f32 precision, writes f32.
            convAct = convOutScratch  // unused later in fused branch
            Ops.siluCastToF32(convOutScratch, into: convActF32Scratch, on: cmd)
        } else {
            convAct = Ops.silu(convOutScratch, on: cmd)   // [conv_dim]
        }

        // ── Fused GDN prep + recurrence path (opt-in) ─────────────────
        //
        // `FFAI_GDN_FUSED_PREP=1` routes through `mt_gated_delta_prep_step`,
        // which absorbs the per-head q/k RMSNorm + scale + g / beta math
        // + the recurrence kernel into ONE dispatch. Eliminates the
        // phase-1 host commit+wait (the host no longer needs convAct /
        // aRaw / bRaw) and folds the phase-2 dispatch into the same
        // command buffer as phase 1 — so it costs **one** commit+wait per
        // GDN layer instead of two.
        //
        // Tradeoff: the kernel takes all tensors in one dtype, so the
        // cache state stays in `dtype` (bf16 for Qwen3.6) via the
        // separate `gdn.currentBf16` / `nextBf16` slots. The legacy fp32
        // state path is untouched. bf16 state precision drifts faster
        // than fp32 over long decodes (per GDNStateCache header) — for a
        // demo + bench this is acceptable; the proper fix (GPU cast
        // kernel so the kernel runs fp32 against a bf16 model) is
        // tracked on the roadmap.
        let yT: Tensor
        if fused {
            // GPU casts: bf16 activations → fp32 scratch, all on the
            // same command buffer as phase 1. No host round-trip; the
            // fused step runs the recurrence in fp32 against the
            // existing fp32 state slots, matching the canonical
            // precision of the legacy path.
            // convAct → convActF32Scratch was already done above via
            // siluCastToF32. Just batch the remaining 2 casts.
            Ops.castToF32Two(
                aRaw, into: aRawF32Scratch,
                bRaw, into: bRawF32Scratch,
                on: cmd)

            Ops.gatedDeltaPrepStep(
                convOut: convActF32Scratch,
                aLog: aLogTF32, dtBias: dtBiasTF32,
                aRaw: aRawF32Scratch, bRaw: bRawF32Scratch,
                qNormWeight: qNormWeightF32, kNormWeight: kNormWeightF32,
                stateIn: cache.gdn.current, stateOut: cache.gdn.next,
                y: yF32Scratch,
                batchSize: 1, dk: keyHeadDim, dv: valueHeadDim,
                hv: numValueHeads, hk: numKeyHeads,
                on: cmd)
            cache.gdn.swap()
            cache.advance()
            yT = yF32Scratch
            // NOTE: no commit + wait here. The fused branch keeps phase
            // 1 + the fused step + the upcoming gatedMixerNorm + out_proj
            // all on `cmd`. Metal serialises dispatches inside a single
            // command buffer in submission order — `out_proj` reads the
            // gated-norm output, gated-norm reads `yT`, the kernel reads
            // the cast outputs of phase 1. One commit at the bottom of
            // this function covers everything; the caller waits naturally
            // when it next commits its own work after this mixer.
        } else {
            // ── Legacy path: host-prep + Ops.gatedDeltaStep ───────────
            // Commit so the host can read convAct / z / aRaw / bRaw.
            cmd.commit()
            cmd.waitUntilCompleted()

            // ── Host phase: split q|k|v, q/k norm + scale, g / beta ───
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

            // ── GPU phase 2: gatedDeltaStep on a fresh command buffer ─
            // All GDN-kernel tensors are fp32 (the only emitted dtype).
            let phase2 = device.makeCommandBuffer()
            let qT = makeF32Tensor35(qNormed, device: device)
            let kT = makeF32Tensor35(kNormed, device: device)
            let vTen = makeF32Tensor35(vFlat, device: device)
            let gT = makeF32Tensor35(gHost, device: device)
            let betaT = makeF32Tensor35(betaHost, device: device)
            let yLegacy = Tensor.empty(shape: [numValueHeads, valueHeadDim],
                                       dtype: .f32, device: device)
            Ops.gatedDeltaStep(
                q: qT, k: kT, v: vTen, g: gT, beta: betaT,
                stateIn: cache.gdn.current, into: yLegacy, stateOut: cache.gdn.next,
                numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
                keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim,
                on: phase2)
            phase2.commit()
            phase2.waitUntilCompleted()
            cache.gdn.swap()
            cache.advance()
            yT = yLegacy
        }

        // ── Phase 2: gated mixer RMSNorm — GPU (fused) or host (legacy) ─
        //
        // Fused: `Ops.gatedMixerNorm` computes `out = rms_norm(y, w) · silu(z)`
        // per `[Hv, Dv]` row in a single dispatch on the SAME command
        // buffer as phase 1 + the fused step. Eliminates the host
        // round-trip + the per-token `Tensor.filled` allocation, recovering
        // ~30 host commit+waits per Qwen3.6-A3B decode token (one per GDN
        // layer). `out_proj` runs on the same `cmd` afterwards; Metal
        // serialises in submission order so no fence is needed.
        //
        // Legacy: keep the per-element host loop. The GDN kernel emits
        // `y` in fp32 and the host needs `silu(z) · norm(y, w)` to feed
        // `out_proj` — same shape it has always been. A new command
        // buffer (`phase3`) is required because the legacy phase-2 was
        // committed.
        let result: Tensor
        if fused {
            Ops.gatedMixerNorm(
                y: yT, z: z, weight: mixerNorm.weight,
                epsBuf: epsBufFused,
                into: yGatedScratch,
                numValueHeads: numValueHeads, valueHeadDim: valueHeadDim,
                on: cmd)
            result = outProj(yGatedScratch, on: cmd)
            // NOTE: no commit here. The fused path keeps `cmd` in-flight
            // so `Qwen35GDNLayer.decode` can chain the residual add +
            // FFN onto the same command buffer. The FFN (MoE branch)
            // commits `cmd` itself when it needs the gate logits on the
            // CPU; the dense branch commits at the end of
            // `qwen35ApplyFFN`. Net effect for Qwen3.6-A3B: one commit
            // per GDN layer instead of two (mixer + FFN), saving ~30
            // commits per decode token.
        } else {
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
            let phase3 = device.makeCommandBuffer()
            let yGatedT = Tensor.empty(shape: [valueDim], dtype: dtype, device: device)
            writeFloats35(yGatedHost, into: yGatedT)
            result = outProj(yGatedT, on: phase3)
            phase3.commit()
            phase3.waitUntilCompleted()
        }
        return result
    }

    /// T-batched GDN mixer forward. Returns `[T, hidden]` flat. Requires
    /// `FFAI_GDN_FUSED_PREP=1` (fused path) — the legacy path commits
    /// mid-way for host gate-prep and can't compose into a single
    /// in-flight `cmd` across the T-loop.
    ///
    /// Architecture:
    ///   1. Five projections (in_proj_qkv / z / b / a / out) all fan
    ///      `T·gemv → 1·gemm` via `AnyLinear.callMany`.
    ///   2. Per-token: conv1d step + silu + cast-to-f32 + fused
    ///      `gatedDeltaPrepStep` + `gatedMixerNorm` write per-row into a
    ///      `[T, value_dim]` assembly buffer. The recurrence STAYS
    ///      per-token — GDN state crosses tokens, so each step depends
    ///      on the previous step's state out. Scratch tensors are
    ///      reused across iters; Metal serialises dispatches in
    ///      submission order inside the single `cmd`, so the
    ///      write-then-read pattern stays correct.
    ///   3. `outProj.callMany` on the assembled `[T, value_dim]` →
    ///      `[T, hidden]`.
    ///
    /// Wins delivered: 5 batched projections × 30 GDN layers × (T-1)
    /// gemv launches saved at prefill. The recurrence cost is unchanged
    /// (a future `Ops.gatedDeltaChunk` rewrite of the recurrence T-loop
    /// folds that into one kernel; `gatedDeltaChunk` is already emitted
    /// and tested in metaltile-ffai PR #115).
    ///
    /// All work queued on `cmd`; no commit inside. Mirrors the fused
    /// single-token path's cmd ownership.
    func forwardMany(_ xNormFlat: Tensor, t: Int,
                     cache: Qwen35GDNLayerCache,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(fused,
                     "Qwen35GDNMixer.forwardMany: requires FFAI_GDN_FUSED_PREP=1; legacy host-prep path can't run on a single in-flight cmd. Set the env or fall back to a per-token loop in the caller.")
        precondition(t > 0, "Qwen35GDNMixer.forwardMany: T must be positive")
        precondition(xNormFlat.elementCount == t * hidden,
                     "Qwen35GDNMixer.forwardMany: xNormFlat size \(xNormFlat.elementCount) ≠ T·hidden = \(t * hidden)")

        // ── Chunked prep+recurrence — now the default route ──────────────
        //
        // Replaces the per-token T-loop body (conv1d_step + silu_cast +
        // prep_step + mixer_norm + state swap) with: T conv1d_step + T
        // silu_cast + ONE `Ops.gatedDeltaPrepChunk` dispatch (state
        // register-resident across the full T-sweep) + ONE batched
        // mixer norm. State traffic per layer drops from
        // `T·(load + store)` to `1·(load + store)`.
        //
        // Bench wins (Qwen3.6-A3B M5 Max, 5-run median):
        //   T=32:  96.29 → 189.26 tps  +97% (1.97×)
        //   T=128: 136.52 → 295.16 tps +116% (2.16×)
        //   T=512: 203.11 → 253.99 tps +22% / best 252 → 328 tps +30%
        //
        // Opt out via `FFAI_GDN_NO_PREP_CHUNK=1` for A/B benching or in
        // case a future silicon family regresses on the chunked path
        // (M2 mini soak not yet run as of flip).
        if ProcessInfo.processInfo.environment["FFAI_GDN_NO_PREP_CHUNK"] == nil {
            return forwardManyChunked(xNormFlat, t: t, cache: cache,
                                      cmd: cmd, device: device)
        }

        let dt = xNormFlat.dtype
        let dtBytes = dt.byteSize

        // ── Projections — five gemms, T-batched ──────────────────────────
        let qkvAll = inProjQKV.callMany(xNormFlat, t: t, on: cmd, device: device)
        let zAll = inProjZ.callMany(xNormFlat, t: t, on: cmd, device: device)
        let bRawAll = inProjB.callMany(xNormFlat, t: t, on: cmd, device: device)
        let aRawAll = inProjA.callMany(xNormFlat, t: t, on: cmd, device: device)

        // ── Batched aRaw/bRaw → f32 cast. The recurrence needs aRaw/bRaw
        // in f32 (gatedDeltaPrepStep is f32-only). Casting EACH row inside
        // the loop fires T·2·30 layers = 30 720 dispatches at T=512;
        // casting the whole `[T, numValueHeads]` once collapses that to 2
        // dispatches per layer. The per-row slices below alias into the
        // pre-cast f32 storage — no extra copy.
        let aRawF32All = Tensor.empty(shape: [t * numValueHeads], dtype: .f32, device: device)
        let bRawF32All = Tensor.empty(shape: [t * numValueHeads], dtype: .f32, device: device)
        Ops.castToF32(aRawAll, into: aRawF32All, on: cmd)
        Ops.castToF32(bRawAll, into: bRawF32All, on: cmd)

        // ── Per-token recurrence — scratches reused, T-loop on `cmd`. ────
        let yGatedAll = Tensor.empty(shape: [t * valueDim], dtype: dt, device: device)
        let f32Bytes = DType.f32.byteSize
        for r in 0..<t {
            let qkvRow = Tensor(buffer: qkvAll.buffer,
                                offset: qkvAll.offset + r * convDim * dtBytes,
                                shape: [convDim], dtype: dt)
            let zRow = Tensor(buffer: zAll.buffer,
                              offset: zAll.offset + r * valueDim * dtBytes,
                              shape: [valueDim], dtype: dt)
            // Pre-cast f32 slices — kernel reads these directly, no per-row
            // cast dispatch fired in the loop body.
            let aRawF32Row = Tensor(buffer: aRawF32All.buffer,
                                    offset: aRawF32All.offset + r * numValueHeads * f32Bytes,
                                    shape: [numValueHeads], dtype: .f32)
            let bRawF32Row = Tensor(buffer: bRawF32All.buffer,
                                    offset: bRawF32All.offset + r * numValueHeads * f32Bytes,
                                    shape: [numValueHeads], dtype: .f32)

            Ops.conv1dCausalStep(
                x: qkvRow, w: convW, b: convB,
                state: cache.conv.state, into: convOutScratch,
                nChannels: convDim, kernelSize: convKernel, on: cmd)
            // Fused silu + cast-to-f32. Replaces the prior two-dispatch
            // chain `silu(convOutScratch) → castToF32(...)` with one
            // dispatch reading bf16/f16 conv output, applying silu at
            // fp32 precision, writing fp32 directly into the prep
            // scratch. Saves T·30 dispatches per Qwen3.6-A3B prefill.
            if convOutScratch.dtype == .f32 {
                // f32 conv → silu in place, no cast needed.
                _ = Ops.silu(convOutScratch, on: cmd, into: convOutScratch)
                Ops.castToF32(convOutScratch, into: convActF32Scratch, on: cmd)
            } else {
                Ops.siluCastToF32(convOutScratch, into: convActF32Scratch, on: cmd)
            }

            Ops.gatedDeltaPrepStep(
                convOut: convActF32Scratch,
                aLog: aLogTF32, dtBias: dtBiasTF32,
                aRaw: aRawF32Row, bRaw: bRawF32Row,
                qNormWeight: qNormWeightF32, kNormWeight: kNormWeightF32,
                stateIn: cache.gdn.current, stateOut: cache.gdn.next,
                y: yF32Scratch,
                batchSize: 1, dk: keyHeadDim, dv: valueHeadDim,
                hv: numValueHeads, hk: numKeyHeads,
                on: cmd)
            cache.gdn.swap()
            cache.advance()

            let yGatedRow = Tensor(buffer: yGatedAll.buffer,
                                   offset: yGatedAll.offset + r * valueDim * dtBytes,
                                   shape: [valueDim], dtype: dt)
            Ops.gatedMixerNorm(
                y: yF32Scratch, z: zRow, weight: mixerNorm.weight,
                epsBuf: epsBufFused,
                into: yGatedRow,
                numValueHeads: numValueHeads, valueHeadDim: valueHeadDim,
                on: cmd)
        }

        // ── Batched output projection ────────────────────────────────────
        let yGatedRows = yGatedAll.reshaped(to: [t, valueDim])
        return outProj.callMany(yGatedRows, t: t, on: cmd, device: device)
    }

    /// Chunked GDN forward — replaces the per-token T-loop body with one
    /// `Ops.gatedDeltaPrepChunk` dispatch. The recurrence state stays in
    /// per-lane registers across all T tokens; the kernel loads state
    /// once, sweeps T tokens, stores state once. State traffic per layer
    /// drops from `T × (load + store)` to `1 × (load + store)` — at T=512
    /// on Qwen3.6-A3B's `[Hv=16, Dv=128, Dk=128]` state that's ~512×
    /// reduction in state-pass bandwidth.
    ///
    /// Conv1d still loops T times because the depthwise causal state
    /// crosses tokens (each step depends on the previous step's state),
    /// but each call dispatches onto the same `cmd` — no host sync, no
    /// commit boundary.
    ///
    /// The output projection + mixer norm fan-out match the per-token
    /// `forwardMany` route. Mixer norm runs T per-row dispatches onto
    /// the same `cmd` (cheap; could batch in a follow-up if it surfaces
    /// in a profile).
    ///
    /// Gated by `FFAI_GDN_PREP_CHUNK=1` from `forwardMany`. All work
    /// queued on `cmd`; no commit inside.
    func forwardManyChunked(_ xNormFlat: Tensor, t: Int,
                            cache: Qwen35GDNLayerCache,
                            cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let dt = xNormFlat.dtype
        let dtBytes = dt.byteSize
        let f32Bytes = DType.f32.byteSize

        // ── Five T-batched projections (same as forwardMany) ─────────────
        let qkvAll = inProjQKV.callMany(xNormFlat, t: t, on: cmd, device: device)
        let zAll = inProjZ.callMany(xNormFlat, t: t, on: cmd, device: device)
        let bRawAll = inProjB.callMany(xNormFlat, t: t, on: cmd, device: device)
        let aRawAll = inProjA.callMany(xNormFlat, t: t, on: cmd, device: device)

        // a_raw / b_raw → f32 once (instead of per-row).
        let aRawF32All = Tensor.empty(shape: [t * numValueHeads],
                                       dtype: .f32, device: device)
        let bRawF32All = Tensor.empty(shape: [t * numValueHeads],
                                       dtype: .f32, device: device)
        Ops.castToF32(aRawAll, into: aRawF32All, on: cmd)
        Ops.castToF32(bRawAll, into: bRawF32All, on: cmd)

        // ── Conv1d + silu+cast → `[T, convDim]` f32 staging buffer ──────
        //
        // Conv1d_causal_step is per-token state-carrying — must loop T
        // times. silu+cast stays per-row INTENTIONALLY: per-row keeps
        // conv1d output in L1 cache for the immediately-following
        // silu_cast read; materialising a full `[T, convDim]` activation
        // buffer between conv and silu_cast forces a device-memory
        // round-trip (memory: feedback_gdn_inner_loop_already_bandwidth_optimal —
        // tried batched silu_cast 2026-05-22, lost -4.8% median).
        let convOutAllF32 = Tensor.empty(shape: [t * convDim],
                                         dtype: .f32, device: device)
        for r in 0..<t {
            let qkvRow = Tensor(
                buffer: qkvAll.buffer,
                offset: qkvAll.offset + r * convDim * dtBytes,
                shape: [convDim], dtype: dt)
            Ops.conv1dCausalStep(
                x: qkvRow, w: convW, b: convB,
                state: cache.conv.state, into: convOutScratch,
                nChannels: convDim, kernelSize: convKernel, on: cmd)
            let convOutRowF32 = Tensor(
                buffer: convOutAllF32.buffer,
                offset: convOutAllF32.offset + r * convDim * f32Bytes,
                shape: [convDim], dtype: .f32)
            if convOutScratch.dtype == .f32 {
                _ = Ops.silu(convOutScratch, on: cmd, into: convOutScratch)
                Ops.castToF32(convOutScratch, into: convOutRowF32, on: cmd)
            } else {
                Ops.siluCastToF32(convOutScratch, into: convOutRowF32, on: cmd)
            }
        }

        // ── ONE chunked prep+recurrence dispatch over all T tokens ──────
        //
        // State `cache.gdn.current` is read once; updated state written
        // to `cache.gdn.next`; swap once at the end (vs T swaps in
        // forwardMany).
        let tLenBuf = device.makeBuffer(length: 4)
        var tLenU32 = UInt32(t)
        memcpy(tLenBuf.contents(), &tLenU32, 4)
        let tLenScalar = Tensor(buffer: tLenBuf, offset: 0,
                                shape: [1], dtype: .u32)

        let yF32All = Tensor.empty(
            shape: [t * numValueHeads * valueHeadDim],
            dtype: .f32, device: device)

        Ops.gatedDeltaPrepChunk(
            convOut: convOutAllF32,
            aLog: aLogTF32, dtBias: dtBiasTF32,
            aRaw: aRawF32All, bRaw: bRawF32All,
            qNormWeight: qNormWeightF32, kNormWeight: kNormWeightF32,
            stateIn: cache.gdn.current, stateOut: cache.gdn.next,
            y: yF32All,
            tLen: tLenScalar,
            batchSize: 1, dk: keyHeadDim, dv: valueHeadDim,
            hv: numValueHeads, hk: numKeyHeads,
            on: cmd)
        cache.gdn.swap()
        for _ in 0..<t { cache.advance() }

        // ── Mixer norm T-batched (one dispatch across T·Hv rows) ────────
        //
        // `mt_gated_mixer_norm` decodes its row from
        // `program_id::<0>()`, so dispatching `T·Hv` rows instead of
        // `Hv` runs the same per-row math over every token in one
        // dispatch. Saves T-1 per-layer dispatches × 30 GDN layers =
        // 15330 dispatches at T=512.
        let yGatedAll = Tensor.empty(shape: [t * valueDim],
                                     dtype: dt, device: device)
        Ops.gatedMixerNormMany(
            y: yF32All, z: zAll, weight: mixerNorm.weight,
            epsBuf: epsBufFused,
            into: yGatedAll,
            t: t, numValueHeads: numValueHeads, valueHeadDim: valueHeadDim,
            on: cmd)

        // ── Batched output projection ───────────────────────────────────
        let yGatedRows = yGatedAll.reshaped(to: [t, valueDim])
        return outProj.callMany(yGatedRows, t: t, on: cmd, device: device)
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
        // ITER 12: shared encoder saves 1 begin/end pair per attn
        // layer × 10 = ~170 µs/decode token.
        let qNormed = Tensor.empty(shape: queries.shape, dtype: queries.dtype)
        let kNormed = Tensor.empty(shape: k.shape, dtype: k.dtype)
        Ops.rmsNormRowsTwo(
            queries, weight: qNorm.weight, eps1: qNorm.eps,
            nRows1: nHeads, rowSize1: headDim, into: qNormed,
            k, weight: kNorm.weight, eps2: kNorm.eps,
            nRows2: nKVHeads, rowSize2: headDim, into: kNormed,
            on: cmd)

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
        // ITER 11: fused sigmoid+mul via mt_sigmoid_mul saves 1 encoder
        // per attn layer × 10 attn layers = ~170 µs/decode token.
        var attnFlat = attnOut.reshaped(to: [nHeads * headDim])
        if let gate {
            attnFlat = Ops.sigmoidMul(attnFlat, gate, on: cmd)
        }
        return oProj(attnFlat, on: cmd)
    }

    /// T-batched attention forward. `xNormFlat` is `[T, hidden]` flat
    /// (T·hidden elements). `startPosition` is the position of `xNorm[0]`;
    /// `xNorm[t]` runs at `startPosition + t`. KV cache is appended for all
    /// T tokens in order. Returns `[T, hidden]` flat (the post-o_proj
    /// contribution; residual add belongs to the enclosing layer).
    ///
    /// Architecture: all four projections + the gate-split gather + per-
    /// head Q/K norm fan through a single dispatch each (vs T serial
    /// dispatches in `forward`). RoPE + KV append still loop T times but
    /// dispatch onto the same `cmd` — no host commit/wait. SDPA uses
    /// `sdpaMulti` (one causal-batched dispatch over `[T, nHeads, headDim]`
    /// queries vs the existing `[startPos, startPos+T)` window plus the
    /// just-appended KV rows).
    ///
    /// The "many" wins: 1 gemm per projection vs T gemv calls, 1 gather vs
    /// T gathers, 1 rmsNormRows vs T rmsNormRows, 1 sdpaMulti vs T
    /// sdpaDecode, 1 oProj.gemm vs T oProj.gemv. Per-token kernels (RoPE,
    /// KV append) keep their counts but ride the same in-flight buffer.
    ///
    /// All work queued on `cmd`; no commit inside.
    func forwardMany(_ xNormFlat: Tensor, t: Int, startPosition: Int,
                     cache kv: KVCache,
                     cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(t > 0, "Qwen35AttentionMixer.forwardMany: T must be positive")
        let dt = xNormFlat.dtype
        let dtBytes = dt.byteSize
        let qDim = nHeads * headDim
        let kvDim = nKVHeads * headDim

        // ── Projections — one gemm/qmm each, T-batched ───────────────────
        let qOut = qProj.callMany(xNormFlat, t: t, on: cmd, device: device)
        let kOut = kProj.callMany(xNormFlat, t: t, on: cmd, device: device)
        let vOut = vProj.callMany(xNormFlat, t: t, on: cmd, device: device)

        // ── Gate split — one gather (vs T) when attnOutputGate ───────────
        let queriesFlat: Tensor   // `[T, nHeads, headDim]` flat
        let gateFlat: Tensor?     // `[T, nHeads, headDim]` flat
        if attnOutputGate {
            let q2T = qOut.reshaped(to: [t, nHeads, 2 * headDim])
            queriesFlat = sliceHeadHalvesMany35(
                q2T, t: t, nHeads: nHeads, headDim: headDim, takeFirst: true,
                on: cmd, device: device)
            gateFlat = sliceHeadHalvesMany35(
                q2T, t: t, nHeads: nHeads, headDim: headDim, takeFirst: false,
                on: cmd, device: device)
        } else {
            queriesFlat = qOut
            gateFlat = nil
        }

        // ── Q/K norm — one rmsNormRows over (T*nHeads) and (T*nKVHeads)
        // rows. rmsNormRows is already row-wise, so the T-batched form is
        // a count change, not a kernel change.
        let qNormed = Ops.rmsNormRows(
            queriesFlat, weight: qNorm.weight, eps: qNorm.eps,
            nRows: t * nHeads, rowSize: headDim, on: cmd)
        let kNormed = Ops.rmsNormRows(
            kOut, weight: kNorm.weight, eps: kNorm.eps,
            nRows: t * nKVHeads, rowSize: headDim, on: cmd)

        // ── RoPE per token + KV append — T dispatches each, all on `cmd`.
        // The single-token RoPE kernel grids `[nHeads, halfRotary]` —
        // small enough that T launches at T≤256 cost less than one big
        // qmm. KV append uses appendRangeOnGPU for the single length-lock
        // path; each step is one Ops.kvCacheUpdate dispatch.
        var kRows: [Tensor] = []; kRows.reserveCapacity(t)
        var vRows: [Tensor] = []; vRows.reserveCapacity(t)
        for r in 0..<t {
            let qRow = Tensor(buffer: qNormed.buffer,
                              offset: qNormed.offset + r * qDim * dtBytes,
                              shape: [qDim], dtype: dt)
            let kRow = Tensor(buffer: kNormed.buffer,
                              offset: kNormed.offset + r * kvDim * dtBytes,
                              shape: [kvDim], dtype: dt)
            let vRow = Tensor(buffer: vOut.buffer,
                              offset: vOut.offset + r * kvDim * dtBytes,
                              shape: [kvDim], dtype: dt)
            Ops.ropePartial(qRow, position: startPosition + r,
                            headDim: headDim, rotaryDim: rotaryDim,
                            thetaBase: ropeTheta, on: cmd)
            Ops.ropePartial(kRow, position: startPosition + r,
                            headDim: headDim, rotaryDim: rotaryDim,
                            thetaBase: ropeTheta, on: cmd)
            kRows.append(kRow.reshaped(to: [nKVHeads, headDim]))
            vRows.append(vRow.reshaped(to: [nKVHeads, headDim]))
        }
        kv.appendRangeOnGPU(kRows: kRows, vRows: vRows, on: cmd)

        // ── SDPA — per-token loop over `sdpaDecode`. `Ops.sdpaMulti`
        // would consolidate the T calls into one but it's head_dim-128
        // only today; Qwen3.6 attention is head_dim-256. The per-token
        // sdpaDecode loop preserves correctness; the projection and
        // norm wins above still amortise. A future head_dim-256
        // `sdpaMulti` variant (or transpose-to-prefill_mma) collapses
        // these T launches into one.
        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnAll = Tensor.empty(shape: [t * qDim], dtype: dt, device: device)
        for r in 0..<t {
            let qRow = Tensor(buffer: qNormed.buffer,
                              offset: qNormed.offset + r * qDim * dtBytes,
                              shape: [nHeads, headDim], dtype: dt)
            let outRow = Tensor(buffer: attnAll.buffer,
                                offset: attnAll.offset + r * qDim * dtBytes,
                                shape: [nHeads, headDim], dtype: dt)
            _ = Ops.sdpaDecode(
                q: qRow, k: cacheK, v: cacheV,
                nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                nKV: startPosition + r + 1, kvStride: kv.maxSeq,
                scale: scale, on: cmd, into: outRow)
        }

        // ── Gated output * o_proj ────────────────────────────────────────
        var attnFlat = attnAll
        if let gateFlat {
            attnFlat = Ops.mul(attnFlat, Ops.sigmoid(gateFlat, on: cmd), on: cmd)
        }
        let attnRows = attnFlat.reshaped(to: [t, qDim])
        return oProj.callMany(attnRows, t: t, on: cmd, device: device)
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
        // In fused mode `mixer.forward` leaves `cmd` in-flight and the
        // residual add + FFN ride the same command buffer. In legacy
        // mode `mixer.forward` commits `cmd` (host needs to read
        // intermediate buffers for the phase-2 norm loop) and the FFN
        // runs on a fresh `ffnCmd`.
        let mixerOut = mixer.forward(xNorm, cache: gc, cmd: cmd, device: device)

        let ffnCmd = mixer.fused ? cmd : device.makeCommandBuffer()
        let postMix = Ops.add(h, mixerOut, on: ffnCmd)
        return qwen35ApplyFFN(ffn, postMix: postMix, postNorm: postNorm,
                              position: position, cmd: ffnCmd,
                              commitCmd: true, device: device)
    }

    /// T-batched layer forward for batched prefill. Mirrors the attention
    /// layer's `decodeMany`: pre-norm + batched mixer + residual add + FFN
    /// per-row loop. Requires `mixer.fused == true` — the legacy host-
    /// prep GDN path commits mid-mixer and can't compose on a single
    /// in-flight `cmd`.
    ///
    /// Command-buffer ownership matches single-token `decode` for the
    /// fused branch: mixer leaves `cmd` in-flight; residual add stays on
    /// `cmd`; FFN per-row loop commits the per-row MoE buffers and
    /// rotates `workCmd`. `commitsCommandBuffer = true` always (the GDN
    /// layer's contract — MoE FFN commits inside, dense FFN ends with a
    /// fresh fully-committed addCmd).
    public func decodeMany(_ hFlat: Tensor, t: Int, startPosition: Int,
                           cache: any LayerCacheProtocol,
                           cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let gc = cache as? Qwen35GDNLayerCache else {
            fatalError("Qwen35GDNLayer.decodeMany: expected Qwen35GDNLayerCache, "
                       + "got \(type(of: cache))")
        }
        precondition(mixer.fused,
                     "Qwen35GDNLayer.decodeMany: requires FFAI_GDN_FUSED_PREP=1; legacy GDN path commits cmd mid-way and can't compose into a single in-flight cmd across the T-loop.")
        precondition(t > 0, "Qwen35GDNLayer.decodeMany: T must be positive")
        precondition(hFlat.elementCount == t * hidden,
                     "Qwen35GDNLayer.decodeMany: hFlat size \(hFlat.elementCount) ≠ T·hidden = \(t * hidden)")

        let dt = hFlat.dtype
        let dtBytes = dt.byteSize

        // ── Pre-norm — one rmsNormRows over T rows ──────────────────────
        let xNormFlat = Ops.rmsNormRows(
            hFlat, weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: t, rowSize: hidden, on: cmd)

        // ── Batched GDN mixer ───────────────────────────────────────────
        let mixerOutFlat = mixer.forwardMany(
            xNormFlat, t: t, cache: gc, cmd: cmd, device: device
        ).reshaped(to: [t * hidden])

        // ── Residual add ────────────────────────────────────────────────
        let postMix = Ops.add(hFlat, mixerOutFlat, on: cmd)

        // ── FFN half — T-batched via qwen35ApplyFFNMany. ─────────────────
        // For MoE FFN this dispatches `moe.decodeMany` (one BGEMM per
        // gate/up/down at mTotal=T·topK + scatter-sum). For dense FFN
        // it dispatches `mlp.forwardMany` (3 gemms). Mirrors the
        // attention-layer's FFN dispatch.
        return qwen35ApplyFFNMany(ffn, postMix: postMix, t: t,
                                  postNorm: postNorm, hidden: hidden,
                                  cmd: cmd, device: device)
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

    /// T-batched layer forward for batched prefill. `hFlat` is
    /// `[T, hidden]` flat (T·hidden elements). `startPosition` is the
    /// position of `hFlat[0]`. Returns `[T, hidden]` flat.
    ///
    /// Architecture: pre-norm + mixer fan through one batched dispatch
    /// each (`Ops.rmsNormRows` over T rows, `Qwen35AttentionMixer.
    /// forwardMany`). The residual add over `[T, hidden]` is also one
    /// kernel. The FFN half STILL loops per token — `MoELayer.decodeMany`
    /// is multi-day work (per Decode Plan §4) and unblocking the mixer
    /// piece is the lower-risk first ship. The expensive dispatches
    /// (qmm / sdpa / gemm) all amortise once.
    ///
    /// Command-buffer ownership matches single-token `decode`:
    ///   - Dense FFN: `cmd` stays in-flight with all T row writes;
    ///     caller's `commitsCommandBuffer = false` contract holds.
    ///   - MoE FFN: iter 0's `moe.forward` commits `cmd`; subsequent
    ///     iters spin fresh `workCmd`s. Each iter's residual add runs
    ///     on a fresh `addCmd` that commits immediately. Output's writes
    ///     are scattered across T committed `addCmd`s — caller's
    ///     `commitsCommandBuffer = true` contract triggers the model-
    ///     level `workCmd` refresh; the next layer reads via Metal hazard
    ///     tracking.
    public func decodeMany(_ hFlat: Tensor, t: Int, startPosition: Int,
                           cache: any LayerCacheProtocol,
                           cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("Qwen35AttentionLayer.decodeMany: expected KVCache, "
                       + "got \(type(of: cache))")
        }
        precondition(t > 0, "Qwen35AttentionLayer.decodeMany: T must be positive")
        precondition(hFlat.elementCount == t * hidden,
                     "Qwen35AttentionLayer.decodeMany: hFlat size \(hFlat.elementCount) "
                     + "≠ T·hidden = \(t * hidden)")

        let dt = hFlat.dtype
        let dtBytes = dt.byteSize

        // ── Pre-norm — one rmsNormRows over T rows ──────────────────────
        let xNormFlat = Ops.rmsNormRows(
            hFlat, weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: t, rowSize: hidden, on: cmd)

        // ── Batched attention mixer ─────────────────────────────────────
        // mixer.forwardMany returns shape `[T, hidden]` (oProj.callMany's
        // 2D shape). Flatten to match `hFlat` for the elementwise add.
        let mixerOutFlat = mixer.forwardMany(
            xNormFlat, t: t, startPosition: startPosition,
            cache: kv, cmd: cmd, device: device
        ).reshaped(to: [t * hidden])

        // ── Residual add — one Ops.add over T·hidden elements ───────────
        let postMix = Ops.add(hFlat, mixerOutFlat, on: cmd)

        // ── FFN half — T-batched via qwen35ApplyFFNMany. ─────────────────
        // For MoE FFN this dispatches `moe.decodeMany` (one BGEMM per
        // gate/up/down at mTotal=T·topK + scatter-sum). For dense FFN
        // it dispatches `mlp.forwardMany` (3 gemms).
        return qwen35ApplyFFNMany(ffn, postMix: postMix, t: t,
                                  postNorm: postNorm, hidden: hidden,
                                  cmd: cmd, device: device)
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

/// T-batched FFN application: post-attention norm + FFN.forwardMany +
/// residual add over `[T, hidden]` rows. Returns `[T, hidden]` flat.
/// Internally commits cmds for the MoE path (matching the per-row
/// `qwen35ApplyFFN` behaviour); the returned buffer is in-flight on
/// the last committed cmd, downstream reads hazard-track.
///
/// Replaces the per-row FFN loop in the layer-level `decodeMany`s
/// when both FFN type and dtype support a batched form. Today: dense
/// and MoE branches both have `forwardMany`. Layers always commit
/// (MoE) or leave the layer's residual on the caller's cmd (dense —
/// the dense-FFN add lives on `cmd` here, mirror commit-Cmd contract
/// of `qwen35ApplyFFN`).
private func qwen35ApplyFFNMany(_ ffn: Qwen35FFN, postMix: Tensor, t: Int,
                                postNorm: RMSNorm, hidden: Int,
                                cmd: MTLCommandBuffer, device: Device) -> Tensor {
    let dt = postMix.dtype
    // Post-norm over T rows of [hidden]. One rmsNormRows kernel.
    let ffnNorm = Ops.rmsNormRows(
        postMix, weight: postNorm.weight, eps: postNorm.eps,
        nRows: t, rowSize: hidden, on: cmd)
    switch ffn {
    case .dense(let mlp):
        let ffnOut = mlp.forwardMany(ffnNorm, t: t, cmd: cmd, device: device)
        return Ops.add(postMix, ffnOut.reshaped(to: [t * hidden]), on: cmd)
    case .moe(let moe):
        // moe.forwardMany commits cmd; returns `[T, hidden]` flat on a
        // fresh in-flight buffer. Run the residual add on a fresh
        // addCmd so the returned tensor is independent of the
        // already-committed cmd chain.
        let ffnOut = moe.forwardMany(ffnNorm, t: t, cmd: cmd, device: device)
        let addCmd = device.makeCommandBuffer()
        let resultFlat = Ops.add(postMix, ffnOut, on: addCmd)
        addCmd.commit()
        let _ = resultFlat
        return resultFlat
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
            // NOTE: commit without waitUntilCompleted. The returned
            // `result` tensor is on the in-flight cmd; the engine spins
            // a fresh `workCmd` for the next layer and Metal's default
            // hazard tracking serialises reads against the prior write.
            // Pipelining up to maxCommandBufferCount buffers in flight
            // recovers the per-layer host stall (~1-2 ms × 40 layers).
            cmd.commit()
        }
        return result
    case .moe(let moe):
        // Qwen35MoEFFN.forward commits `cmd`; run the residual add on a
        // fresh buffer. We commit without waiting: the residual add
        // tensor is the layer's output; the next layer queues onto a
        // fresh workCmd and Metal hazard-tracks the read.
        let ffnOut = moe.forward(ffnNorm, position: position,
                                 cmd: cmd, device: device)
        let addCmd = device.makeCommandBuffer()
        let result = Ops.add(postMix, ffnOut, on: addCmd)
        addCmd.commit()
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
        var h = Profile.time("forward.embed") {
            embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        }

        for (i, layer) in layers.enumerated() {
            if let attn = layer as? Qwen35AttentionLayer {
                h = Profile.time("forward.attn_layer") {
                    attn.decode(h, position: position, cache: caches[i],
                                cmd: workCmd, device: device)
                }
                if attn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else if let gdn = layer as? Qwen35GDNLayer {
                h = Profile.time("forward.gdn_layer") {
                    gdn.decode(h, position: position, cache: caches[i],
                               cmd: workCmd, device: device)
                }
                if gdn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else {
                h = layer.decode(h, position: position, cache: caches[i],
                                 cmd: workCmd, device: device)
            }
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

    /// Like `forward(...)` but also returns hidden states. Used by the
    /// ANE MTP drafter — the drafter consumes the last decode step's
    /// hidden state as input to predict the next position's hidden
    /// state, which we then project via `lmHead` to get a candidate
    /// token.
    ///
    /// Returns `(hiddenPreNorm, hiddenPostNorm, logits)`. The MTP head
    /// can take either depending on training convention; expose both
    /// so the drafter can A/B compare. Tensors are on the caller's
    /// `cmd` and become resident once it commits.
    public func forwardWithBothHiddens(tokenId: Int, position: Int,
                                       caches: [any LayerCacheProtocol],
                                       on cmd: MTLCommandBuffer,
                                       device: Device) -> (preNorm: Tensor, postNorm: Tensor, logits: Tensor) {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            if let attn = layer as? Qwen35AttentionLayer {
                h = attn.decode(h, position: position, cache: caches[i],
                                cmd: workCmd, device: device)
                if attn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else if let gdn = layer as? Qwen35GDNLayer {
                h = gdn.decode(h, position: position, cache: caches[i],
                               cmd: workCmd, device: device)
                if gdn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else {
                h = layer.decode(h, position: position, cache: caches[i],
                                 cmd: workCmd, device: device)
            }
        }
        // Commit + wait so `h` (pre-finalNorm) is resident on the
        // host. Otherwise the drafter can't snapshot it.
        workCmd.commit()
        workCmd.waitUntilCompleted()

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)
        return (h, normed, logits)
    }

    /// Like `forward(...)` but also returns the post-final-norm hidden
    /// state. Used by the ANE MTP drafter — the drafter consumes the
    /// last decode step's hidden state as input to predict the next
    /// position's hidden state, which we then project via `lmHead` to
    /// get a candidate token.
    ///
    /// Returns `(hidden, logits)`. Both tensors are on the caller's
    /// `cmd` and become resident once it commits.
    public func forwardWithHidden(tokenId: Int, position: Int,
                                  caches: [any LayerCacheProtocol],
                                  on cmd: MTLCommandBuffer,
                                  device: Device) -> (hidden: Tensor, logits: Tensor) {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            if let attn = layer as? Qwen35AttentionLayer {
                h = attn.decode(h, position: position, cache: caches[i],
                                cmd: workCmd, device: device)
                if attn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else if let gdn = layer as? Qwen35GDNLayer {
                h = gdn.decode(h, position: position, cache: caches[i],
                               cmd: workCmd, device: device)
                if gdn.commitsCommandBuffer { workCmd = device.makeCommandBuffer() }
            } else {
                h = layer.decode(h, position: position, cache: caches[i],
                                 cmd: workCmd, device: device)
            }
        }
        if let last = layers.last,
           !((last as? Qwen35GDNLayer)?.commitsCommandBuffer ?? false),
           !((last as? Qwen35AttentionLayer)?.commitsCommandBuffer ?? false) {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)
        return (normed, logits)
    }

    /// Project a [hidden]-shape post-final-norm tensor through
    /// `lmHead` to produce [vocab] logits. Used by the ANE MTP drafter
    /// after MTP predicts `hidden_next` — project then argmax for the
    /// candidate token.
    public func projectHiddenToLogits(_ hidden: Tensor,
                                      on cmd: MTLCommandBuffer) -> Tensor {
        precondition(hidden.elementCount == self.hidden,
                     "projectHiddenToLogits: hidden has \(hidden.elementCount) elements, expected \(self.hidden)")
        return lmHead(hidden, on: cmd)
    }

    /// Multi-token forward over `tokenIds[startPosition .. startPosition+T)`
    /// for prefill. Returns the logits of the *last* token only — every
    /// preceding token's logits is consumed only by its KV/GDN cache
    /// write. Mirrors mlx-lm's chunked-prefill driver.
    ///
    /// ─── Current state (Phase 0 — API + per-token loop) ─────────────────
    ///
    /// This is the entry point for batched prefill in Qwen3.5/3.6. The
    /// per-token-state architecture (single-token decode kernels for
    /// projections, conv1d, GDN, and attention) means a real batched-T
    /// implementation requires:
    ///
    ///   1. A batched-T variant of every projection (`dequantGemv` /
    ///      `gemv` → batched `dequantGemm` / `gemm`). The existing
    ///      `mt_qmm_mma_m16` is BM=16 fixed; a dynamic-M qmm + chunking
    ///      dispatcher is needed.
    ///   2. A batched-T conv1d_causal (the GDN input projection's conv1d
    ///      is rolling over `convKernel`; chunked form exists in
    ///      metaltile-ffai's `mt_gated_delta_chunk` for the *recurrence*
    ///      but conv1d still has to step T times).
    ///   3. Per-layer `decodeMany(_ inputs: [T, hidden], ...)` on
    ///      `Qwen35GDNLayer` / `Qwen35AttentionLayer` that uses
    ///      `Ops.gatedDeltaChunk` + `Ops.sdpaPrefillMma` (both newly
    ///      added) on the recurrence / SDPA hot path.
    ///
    /// Until those land, this entry point loops per-token through the
    /// existing decode path. The shape of the API is the same, so a
    /// future agent can swap the loop body without changing callers
    /// (the bench harness, the `Generate.swift` prefill driver).
    public func forwardMany(tokenIds: [Int], startPosition: Int,
                            caches: [any LayerCacheProtocol],
                            on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "Qwen35Model.forwardMany: tokenIds must not be empty")
        if tokenIds.count == 1 {
            return forward(tokenId: tokenIds[0], position: startPosition,
                           caches: caches, on: cmd, device: device)
        }
        // ─── Batched-T forward (mixed dispatch) ──────────────────────────
        // Attention layers dispatch `Qwen35AttentionLayer.decodeMany`
        // (mixer T-batched, FFN per-row). GDN layers still loop per-token
        // because `Qwen35GDNMixer.forwardMany` (chunked-T GDN) hasn't
        // landed yet; their per-row results are blitted back into the
        // running `[T, hidden]` buffer via `Ops.copy`.
        //
        // The wins delivered today: attention-layer mixer projections
        // fan from T·gemv to 1·gemm, SDPA from T·sdpaDecode to 1·
        // sdpaMulti, gate-split gather from T to 1. Compounds across
        // every attention layer in the stack.
        if ProcessInfo.processInfo.environment["FFAI_LEGACY_FORWARDMANY"] != nil {
            return _forwardManyPerTokenLegacy(tokenIds: tokenIds,
                                              startPosition: startPosition,
                                              caches: caches,
                                              on: cmd, device: device)
        }
        return _forwardManyBatched(tokenIds: tokenIds,
                                   startPosition: startPosition,
                                   caches: caches,
                                   returnAllLogits: false,
                                   on: cmd, device: device)
    }

    /// Multi-token forward returning logits at EVERY position — `[T, vocab]`.
    /// Same KV/GDN-cache-writing semantics as `forwardMany`; differs only
    /// in the output: instead of slicing the last row's hidden, applies
    /// final RMSNorm + lm_head row-wise to all T positions.
    ///
    /// Spec-decode driver: a drafter proposes `γ` candidate tokens, the
    /// target verifies via `forwardManyAllLogits(prefix + candidates)` to
    /// get the per-position logits, then accepts up to the first reject.
    public func forwardManyAllLogits(tokenIds: [Int], startPosition: Int,
                                     caches: [any LayerCacheProtocol],
                                     on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "Qwen35Model.forwardManyAllLogits: tokenIds must not be empty")
        return _forwardManyBatched(tokenIds: tokenIds,
                                   startPosition: startPosition,
                                   caches: caches,
                                   returnAllLogits: true,
                                   on: cmd, device: device)
    }

    /// Legacy per-token-loop path retained behind `FFAI_LEGACY_FORWARDMANY=1`
    /// for A/B benching of the batched mixer wiring.
    private func _forwardManyPerTokenLegacy(
        tokenIds: [Int], startPosition: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Per-token loop for the (T-1) prefix — each call commits its own
        // work buffers, so `cmd` stays pristine. The final token queues
        // onto the caller's `cmd` so the default `forwardSample` /
        // `forwardSampleCategorical` overlay their output kernels.
        let prefix = tokenIds.count - 1
        for i in 0..<prefix {
            let stepCmd = device.makeCommandBuffer()
            _ = forward(tokenId: tokenIds[i], position: startPosition + i,
                        caches: caches, on: stepCmd, device: device)
            stepCmd.commit()
            stepCmd.waitUntilCompleted()
        }
        return forward(tokenId: tokenIds[prefix], position: startPosition + prefix,
                       caches: caches, on: cmd, device: device)
    }

    /// Mixed-dispatch batched forwardMany. Attention layers fan to
    /// `decodeMany`; GDN (and any other) layer types stay per-token with
    /// a per-row `Ops.copy` blit-back into the running `[T, hidden]`
    /// buffer. Embedding gathers all T tokens in one dispatch. Final
    /// norm + lm_head run only on the LAST row, on the caller's `cmd`.
    private func _forwardManyBatched(
        tokenIds: [Int], startPosition: Int,
        caches: [any LayerCacheProtocol],
        returnAllLogits: Bool,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let t = tokenIds.count

        // ── Embed all T tokens in one gather ─────────────────────────────
        let idsBuf = device.makeBuffer(length: t * 4)
        let idsHost = tokenIds.map { UInt32($0) }
        idsHost.withUnsafeBytes { _ = memcpy(idsBuf.contents(), $0.baseAddress!, t * 4) }
        let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [t], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        var h = Profile.time("forwardMany.embed") {
            embedTokens(idsTensor, on: workCmd).reshaped(to: [t * hidden])
        }
        // h is `[T, hidden]` flat. Dtype follows the embedding output —
        // for QuantizedEmbedding that's `scales.dtype` (model running
        // dtype), NOT `weight.dtype` (which would be u32 packed). Read
        // it off `h` directly to stay correct under both Embedding and
        // QuantizedEmbedding variants.
        let dt = h.dtype
        let dtBytes = dt.byteSize

        for (i, layer) in layers.enumerated() {
            if let attn = layer as? Qwen35AttentionLayer {
                h = Profile.time("forwardMany.attn_layer") {
                    attn.decodeMany(h, t: t, startPosition: startPosition,
                                    cache: caches[i],
                                    cmd: workCmd, device: device)
                }
                // attn.decodeMany commits workCmd if MoE FFN. Refresh.
                if attn.commitsCommandBuffer {
                    workCmd = device.makeCommandBuffer()
                }
            } else if let gdn = layer as? Qwen35GDNLayer, gdn.mixer.fused {
                h = Profile.time("forwardMany.gdn_layer") {
                    gdn.decodeMany(h, t: t, startPosition: startPosition,
                                   cache: caches[i],
                                   cmd: workCmd, device: device)
                }
                // GDN's commitsCommandBuffer is always true; refresh.
                workCmd = device.makeCommandBuffer()
            } else {
                // GDN (legacy non-fused) or other — per-token loop with
                // blit-back.
                // Each layer.decode is the existing single-token path;
                // we slice `h[r]` as input, get a fresh result tensor
                // back, and blit it into `h[r]`. All T blits queue onto
                // one `blitCmd` that commits at the end of this layer.
                let blitCmd = device.makeCommandBuffer()
                for r in 0..<t {
                    let hRow = Tensor(buffer: h.buffer,
                                      offset: h.offset + r * hidden * dtBytes,
                                      shape: [hidden], dtype: dt)
                    let rowOut = layer.decode(hRow, position: startPosition + r,
                                              cache: caches[i],
                                              cmd: workCmd, device: device)
                    Ops.copy(rowOut, into: hRow, on: blitCmd)
                    // GDN layers commit workCmd inside .decode; refresh.
                    let committed: Bool
                    switch layer {
                    case let l as Qwen35GDNLayer: committed = l.commitsCommandBuffer
                    case let l as Qwen35AttentionLayer: committed = l.commitsCommandBuffer
                    default: committed = false
                    }
                    if committed { workCmd = device.makeCommandBuffer() }
                }
                blitCmd.commit()
            }
        }

        // Make sure all in-flight writes to `h` are resident before the
        // caller's `cmd` reads `h[T-1]` for the final norm / lm_head.
        // If the last layer was non-committing (dense FFN attention),
        // its work still sits on `workCmd`. Commit + wait.
        if let last = layers.last,
           !((last as? Qwen35GDNLayer)?.commitsCommandBuffer ?? false),
           !((last as? Qwen35AttentionLayer)?.commitsCommandBuffer ?? false) {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        // ── Final norm + lm_head ─────────────────────────────────────────
        //
        // Two output shapes:
        //   * `returnAllLogits == false` (default, prefill driver): logits
        //     of the LAST row only — `[vocab]`. The (T-1) preceding rows'
        //     logits are consumed only by the KV / GDN cache writes inside
        //     the layer loop. This is what `Generate.swift` uses for
        //     batched prefill that only needs the sample-from-last-token.
        //   * `returnAllLogits == true` (spec-decode verify): logits at
        //     EVERY position — `[T, vocab]`. The spec-decode driver needs
        //     the per-position logits to verify each candidate draft token.
        if !returnAllLogits {
            let lastRow = Tensor(buffer: h.buffer,
                                 offset: h.offset + (t - 1) * hidden * dtBytes,
                                 shape: [hidden], dtype: dt)
            let normed = finalNorm(lastRow, on: cmd)
            return lmHead(normed, on: cmd)
        }
        // All-T path: batched RMSNorm over T rows + batched lm_head.
        let normedAll = Ops.rmsNormRows(
            h, weight: finalNorm.weight, eps: finalNorm.eps,
            nRows: t, rowSize: hidden, on: cmd)
        return lmHead.callMany(normedAll.reshaped(to: [t, hidden]),
                               t: t, on: cmd, device: device)
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

/// T-batched variant of `sliceHeadHalves35`. Input `q2T` is
/// `[T, nHeads, 2·headDim]` — each of T rows has its own
/// `[nHeads, 2·headDim]` block. Returns `[T · nHeads · headDim]` flat
/// (caller reshapes), gathered in ONE dispatch.
///
/// The single-token helper builds a per-row idx buffer + dispatches
/// `Ops.gather`. The T-batched version builds one flat `[T · nHeads]`
/// index buffer (each entry points into the flattened
/// `[T · nHeads · 2, headDim]` table) and runs the same gather kernel
/// once. Saves T launch boundaries vs the per-row loop and avoids the
/// in-flight-cmd dependency-ordering problem the loop would surface.
///
/// Unblocks `Qwen35AttentionMixer.forwardMany` for the
/// `attnOutputGate=true` path (every shipped Qwen3.6-A3B config).
private func sliceHeadHalvesMany35(_ q2T: Tensor, t: Int,
                                   nHeads: Int, headDim: Int,
                                   takeFirst: Bool,
                                   on cmd: MTLCommandBuffer,
                                   device: Device) -> Tensor {
    precondition(q2T.elementCount == t * nHeads * 2 * headDim,
                 "sliceHeadHalvesMany35: q2T must be [T, nHeads, 2·headDim] (got elements=\(q2T.elementCount), expected \(t * nHeads * 2 * headDim))")
    let table = q2T.reshaped(to: [t * nHeads * 2, headDim])
    let nIdx = t * nHeads
    var rows = [UInt32](repeating: 0, count: nIdx)
    let half: UInt32 = takeFirst ? 0 : 1
    for r in 0..<t {
        let base = UInt32(r * 2 * nHeads)
        let rowBase = r * nHeads
        for h in 0..<nHeads {
            rows[rowBase + h] = base + UInt32(2 * h) + half
        }
    }
    let idxBuf = device.makeBuffer(length: nIdx * 4)
    rows.withUnsafeBytes { _ = memcpy(idxBuf.contents(), $0.baseAddress!, nIdx * 4) }
    let idx = Tensor(buffer: idxBuf, offset: 0, shape: [nIdx], dtype: .u32)
    let gathered = Ops.gather(table: table, tokenIds: idx, on: cmd)
    // gathered shape: [nIdx, headDim] = [T*nHeads, headDim], row-major.
    // Flatten for caller convenience.
    return gathered.reshaped(to: [nIdx * headDim])
}
