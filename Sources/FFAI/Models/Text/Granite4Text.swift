// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Granite4 text — concrete variants + the hybrid decoder for IBM's
// Granite 4 family. The family enum (`enum Granite4`), variant
// protocol (`Granite4Variant`), and error type (`Granite4Error`) live
// in `Models/Granite4.swift` (the family root / main interface). This
// file holds the text-only impl:
//
//   • `Granite4Hybrid` — `Granite4Variant` conformance + the per-
//     variant `loadModel` entry,
//   • `Granite4LayerKind`, `Granite4MambaLayer`,
//     `Granite4AttentionLayer`, `Granite4Model` — the per-layer + full-
//     model impl. `Granite4Model.forward` runs all per-layer work on
//     internal `workCmd` buffers so a committing MoE FFN never double-
//     commits the caller's `cmd`. No RoPE.

import Foundation
import Metal

// ─── Layer kind ──────────────────────────────────────────────────────

/// The two mixer kinds a `layer_types` entry can name.
enum Granite4LayerKind: Equatable {
    case mamba  // "mamba"
    case attention  // "attention"

    init(from name: String) throws {
        switch name {
        case "mamba": self = .mamba
        case "attention": self = .attention
        default:
            throw Granite4Error.unsupportedConfig(
                "unknown layer_types entry '\(name)'")
        }
    }
}

// ─── Granite4Hybrid — the single variant ─────────────────────

public struct Granite4Hybrid: Granite4Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Granite-4 ships both `-base` and instruction-tuned checkpoints.
    /// Greedy by default keeps the integration suite deterministic.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 256,
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
    ) throws -> Granite4Model {
        guard let hidden = config.hiddenSize,
            let vocab = config.vocabSize,
            let nHeads = config.numAttentionHeads
        else {
            throw Granite4Error.missingConfig(
                "hidden / vocab / num_attention_heads")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let eps = Float(config.rmsNormEps ?? 1e-5)
        let tieEmbed = config.tieWordEmbeddings

        // ── Hybrid layer schedule ─────────────────────────────────────
        guard let layerTypeNames = config.raw["layer_types"] as? [String],
            !layerTypeNames.isEmpty
        else { throw Granite4Error.missingConfig("layer_types") }
        let kinds = try layerTypeNames.map { try Granite4LayerKind(from: $0) }
        let nLayers = kinds.count

        // ── Scalar multipliers ────────────────────────────────────────
        // Granite4 keeps all four as runtime config values
        // (mlx-lm's sanitize folds none of them) — no double-fold risk.
        let embeddingMultiplier = Float(config.float("embedding_multiplier") ?? 1.0)
        let residualMultiplier = Float(config.float("residual_multiplier") ?? 1.0)
        let logitsScaling = Float(config.float("logits_scaling") ?? 1.0)
        // attention_multiplier replaces the usual 1/sqrt(head_dim) scale.
        let attentionScale = Float(
            config.float("attention_multiplier")
                ?? (1.0 / Double(headDim).squareRoot()))

        // ── Mamba 2 mixer geometry ────────────────────────────────────
        guard let mambaNHeads = config.int("mamba_n_heads")
        else { throw Granite4Error.missingConfig("mamba_n_heads") }
        guard let mambaHeadDim = config.int("mamba_d_head")
        else { throw Granite4Error.missingConfig("mamba_d_head") }
        guard let stateDim = config.int("mamba_d_state")
        else { throw Granite4Error.missingConfig("mamba_d_state") }
        let convKernel = config.int("mamba_d_conv") ?? 4
        let nGroups = config.int("mamba_n_groups") ?? 1
        let useConvBias = config.bool("mamba_conv_bias") ?? true

        // d_inner taken directly from the Mamba head decomposition.
        let dInner = mambaNHeads * mambaHeadDim
        guard mambaNHeads % nGroups == 0 else {
            throw Granite4Error.unsupportedConfig(
                "mamba_n_heads (\(mambaNHeads)) must be a multiple of "
                    + "n_groups (\(nGroups))")
        }
        // Granite's gated mixer RMSNorm is a single full-width RMSNorm
        // over d_inner (NOT per-group like NemotronH). The metaltile
        // rms_norm reduction kernel requires the row size to be a
        // multiple of 128 and ≤ 4096.
        guard dInner % 128 == 0, dInner <= 4096 else {
            throw Granite4Error.unsupportedConfig(
                "gated mixer RMSNorm row size d_inner = \(dInner) must be "
                    + "a multiple of 128 and ≤ 4096 (rmsNorm kernel invariant)")
        }
        let convDim = dInner + 2 * nGroups * stateDim

        // time_step_limit clamps softplus(dt). Granite ships none; honour
        // one if a checkpoint sets it.
        let tsLimit = config.raw["time_step_limit"] as? [Double]
        let tsMin = Float(tsLimit?.first ?? 0.0)
        let tsMax: Float = {
            guard let hi = tsLimit?.dropFirst().first else { return .infinity }
            return hi.isFinite ? Float(hi) : .infinity
        }()

        // ── Feed-forward geometry ─────────────────────────────────────
        let numLocalExperts = config.int("num_local_experts") ?? 0
        let numExpertsPerToken = config.int("num_experts_per_tok") ?? 0
        let useMoE = numLocalExperts > 0
        // Dense FFN intermediate is `intermediate_size`; MoE uses the
        // same field as the per-expert intermediate.
        let intermediate = config.intermediateSize ?? (4 * hidden)
        // shared_intermediate_size sizes the always-on shared expert
        // (MoE checkpoints only).
        let sharedIntermediate = config.int("shared_intermediate_size") ?? intermediate

        // ── Activation dtype — taken from the embedding table ─────────
        // Granite4 4-bit conversions (e.g. mlx-community/granite-4.0-h-350m-4bit)
        // do NOT pre-bake `embedding_multiplier` or `residual_multiplier`
        // into the packed weights — verified by dequantizing the embed
        // triplet and comparing against the raw bf16 checkpoint. The
        // loader instead folds the multipliers into the quantized
        // triplet's `scales` and `biases` tensors at load time:
        //
        //     dequant = nibble * scale + bias
        //     dequant * m = nibble * (m·scale) + (m·bias)
        //
        // Multiplying both `scales` and `biases` by the multiplier is
        // mathematically identical to scaling every dequantized output,
        // and costs one vector-scale per triplet at load. The packed
        // u32 `weight` is untouched. Same arithmetic as the raw path's
        // `scaleTensorGMH`; the FalconH1 / Jamba / LFM2 / NemotronH
        // siblings either ship pre-baked multipliers (FalconH1) or
        // have no µP-style multipliers (the rest) and can skip this
        // step entirely. Granite4 is the first family where the
        // multipliers must be folded into the quantized triplet.
        let quant = config.quantization
        let isQuantized = quant != nil
        let activationDtype: DType
        let embedTokens: AnyEmbedding
        // For tied lm_head: raw path keeps the *unscaled* embed table
        // (see file header). Quantized path keeps the *unscaled*
        // triplet (multiplier-scaled version goes to the embedding
        // lookup only).
        let embedWRawForTiedLmHead: Tensor?
        let embedTripletForTiedLmHead: SafeTensorsBundle.QuantizedTriplet?
        if isQuantized, weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
            let triplet = try weights.quantizedTriplet("model.embed_tokens")
            let scaledScales = scaleTensorGMH(
                triplet.scales, by: embeddingMultiplier, device: device)
            let scaledBiases = scaleTensorGMH(
                triplet.biases, by: embeddingMultiplier, device: device)
            let bits = deriveAffineQuantBits(
                weightPackedCols: triplet.weight.shape[triplet.weight.shape.count - 1],
                scaleCols: triplet.scales.shape[triplet.scales.shape.count - 1],
                groupSize: quant!.groupSize)
            embedTokens = AnyEmbedding(
                QuantizedEmbedding(
                    weight: triplet.weight, scales: scaledScales, biases: scaledBiases,
                    hidden: hidden, bits: bits, groupSize: quant!.groupSize))
            embedWRawForTiedLmHead = nil
            embedTripletForTiedLmHead = triplet
        } else {
            let embedWRaw = try weights.tensor(named: "model.embed_tokens.weight")
            activationDtype = embedWRaw.dtype
            let embedW = scaleTensorGMH(embedWRaw, by: embeddingMultiplier, device: device)
            embedTokens = AnyEmbedding(Embedding(weight: embedW))
            embedWRawForTiedLmHead = embedWRaw
            embedTripletForTiedLmHead = nil
        }
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "Granite4: unexpected activation dtype \(activationDtype)")

        // ── Per-layer construction ────────────────────────────────────
        var layers: [any DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in kinds.enumerated() {
            let p = "model.layers.\(i)"
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: eps)
            let postNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: eps)

            // ── Mixer ─────────────────────────────────────────────────
            let mixer: Granite4Mixer
            switch kind {
            case .mamba:
                mixer = .mamba(
                    try buildMambaMixer(
                        prefix: "\(p).mamba", weights: weights,
                        dInner: dInner, convDim: convDim,
                        nHeads: mambaNHeads, headDim: mambaHeadDim, stateDim: stateDim,
                        nGroups: nGroups, convKernel: convKernel,
                        useConvBias: useConvBias, eps: eps,
                        tsMin: tsMin, tsMax: tsMax,
                        residualMultiplier: residualMultiplier,
                        dtype: activationDtype, device: device,
                        quantization: quant))
            case .attention:
                // o_proj folds residual_multiplier so the residual add
                // stays a plain Ops.add. On the quantized path the
                // fold goes into the scales/biases of the o_proj
                // triplet (loadLinearScaled); the packed u32 weight
                // stays untouched.
                let qProj: AnyLinear
                let kProj: AnyLinear
                let vProj: AnyLinear
                let oProj: AnyLinear
                if isQuantized {
                    qProj = try loadLinear(
                        base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
                    kProj = try loadLinear(
                        base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
                    vProj = try loadLinear(
                        base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
                    oProj = try loadLinearScaledGMH(
                        base: "\(p).self_attn.o_proj", in: weights,
                        quantization: quant!, by: residualMultiplier, device: device)
                } else {
                    qProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(p).self_attn.q_proj.weight")))
                    kProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(p).self_attn.k_proj.weight")))
                    vProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(p).self_attn.v_proj.weight")))
                    let oW = scaleTensorGMH(
                        try weights.tensor(named: "\(p).self_attn.o_proj.weight"),
                        by: residualMultiplier, device: device)
                    oProj = AnyLinear(Linear(weight: oW))
                }
                mixer = .attention(
                    Granite4AttentionMixer(
                        qProj: qProj, kProj: kProj, vProj: vProj,
                        oProj: oProj,
                        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                        scale: attentionScale))
            }

            // ── Feed-forward ──────────────────────────────────────────
            let ffn: Granite4FFN
            if useMoE {
                ffn = .moe(
                    try buildMoE(
                        prefix: p, weights: weights,
                        hidden: hidden, moeIntermediate: intermediate,
                        sharedIntermediate: sharedIntermediate,
                        numExperts: numLocalExperts, topK: numExpertsPerToken,
                        residualMultiplier: residualMultiplier, device: device,
                        quantization: quant))
            } else {
                // Dense SwiGLU MLP. down_proj folds residual_multiplier
                // on the raw path; quantized bakes it in.
                let gateProj: AnyLinear
                let upProj: AnyLinear
                let downProj: AnyLinear
                if isQuantized {
                    gateProj = try loadLinear(
                        base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
                    upProj = try loadLinear(
                        base: "\(p).mlp.up_proj", in: weights, quantization: quant)
                    downProj = try loadLinearScaledGMH(
                        base: "\(p).mlp.down_proj", in: weights,
                        quantization: quant!, by: residualMultiplier, device: device)
                } else {
                    let gateW = try weights.tensor(named: "\(p).mlp.gate_proj.weight")
                    let upW = try weights.tensor(named: "\(p).mlp.up_proj.weight")
                    let downW = scaleTensorGMH(
                        try weights.tensor(named: "\(p).mlp.down_proj.weight"),
                        by: residualMultiplier, device: device)
                    gateProj = AnyLinear(Linear(weight: gateW))
                    upProj = AnyLinear(Linear(weight: upW))
                    downProj = AnyLinear(Linear(weight: downW))
                }
                ffn = .dense(
                    Granite4DenseMLP(
                        gateProj: gateProj,
                        upProj: upProj,
                        downProj: downProj))
            }

            layers.append(
                Granite4Layer(
                    inputNorm: inputNorm, postNorm: postNorm,
                    mixer: mixer, ffn: ffn, hidden: hidden))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"), eps: eps)

        // lm_head — tied + raw uses the *unscaled* embed table; tied +
        // quantized wraps the *unscaled* embed triplet as a
        // QuantizedLinear (the embedding lookup above used a copy with
        // embedding_multiplier folded into scales/biases; the tied
        // lm_head must use the original unscaled triplet).
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            if isQuantized, weights.isQuantized("lm_head") {
                lmHead = try loadLinear(
                    base: "lm_head", in: weights, quantization: quant)
            } else {
                lmHead = AnyLinear(
                    Linear(weight: try weights.tensor(named: "lm_head.weight")))
            }
        } else if let q = quant, let t = embedTripletForTiedLmHead {
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize))
        } else if let embedWRaw = embedWRawForTiedLmHead {
            // Tied + raw: the head shares the *unscaled* embedding table.
            lmHead = AnyLinear(Linear(weight: embedWRaw))
        } else {
            throw Granite4Error.unsupportedConfig(
                "quantized Granite4 checkpoint missing both an explicit "
                    + "lm_head.weight and a quantized embed_tokens triplet")
        }

        let maxSeq = config.int("max_position_embeddings") ?? 8192
        return Granite4Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            mambaNHeads: mambaNHeads, mambaHeadDim: mambaHeadDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            nGroups: nGroups, dInner: dInner,
            vocab: vocab, maxContextWindow: maxSeq,
            logitsScaling: logitsScaling, dtype: activationDtype,
            kvCacheKind: options.kvCache, kvEviction: options.kvEviction)
    }

    /// Build one Mamba 2 mixer. Reads + derives the per-head SSM
    /// parameters and transposes the conv1d weight.
    private static func buildMambaMixer(
        prefix p: String, weights: SafeTensorsBundle,
        dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, useConvBias: Bool, eps: Float,
        tsMin: Float, tsMax: Float, residualMultiplier: Float,
        dtype: DType, device: Device,
        quantization: ModelConfig.QuantizationConfig?
    ) throws -> Granite4MambaMixer {
        // in_proj / out_proj quantized on 4-bit checkpoints. out_proj's
        // residual_multiplier folds into the scales/biases on the
        // quantized path (see loader header for the dequant arithmetic);
        // raw path folds into the weight directly. Conv1d, A_log, D,
        // dt_bias, and the gated mixer RMSNorm weight stay raw on both
        // paths.
        let inProj: AnyLinear
        let outProj: AnyLinear
        if let q = quantization, weights.isQuantized("\(p).in_proj") {
            inProj = try loadLinear(
                base: "\(p).in_proj", in: weights, quantization: q)
            outProj = try loadLinearScaledGMH(
                base: "\(p).out_proj", in: weights, quantization: q,
                by: residualMultiplier, device: device)
        } else {
            inProj = AnyLinear(
                Linear(weight: try weights.tensor(named: "\(p).in_proj.weight")))
            let outW = scaleTensorGMH(
                try weights.tensor(named: "\(p).out_proj.weight"),
                by: residualMultiplier, device: device)
            outProj = AnyLinear(Linear(weight: outW))
        }

        // conv1d.weight ships [conv_dim, 1, kernel]; the metaltile kernel
        // wants [kernel, conv_dim].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(
            convWSrc.elementCount == convDim * convKernel,
            "Granite4: conv1d.weight count mismatch: \(convWSrc.shape)")
        let convW = transposeConv1dWeightGMH(
            src: convWSrc, kernel: convKernel, channels: convDim,
            dtype: dtype, device: device)
        let convB: Tensor = {
            if useConvBias, weights.has("\(p).conv1d.bias") {
                return castVectorGMH(
                    (try? weights.tensor(named: "\(p).conv1d.bias"))
                        ?? zeroVectorGMH(convDim, dtype: dtype, device: device),
                    count: convDim, dtype: dtype, device: device)
            }
            return zeroVectorGMH(convDim, dtype: dtype, device: device)
        }()

        // A_eff = -exp(A_log); dt_bias per head; D tiled across head_dim.
        let aEff = computeAEffGMH(
            aLog: try weights.tensor(named: "\(p).A_log"),
            nHeads: nHeads, dtype: dtype, device: device)
        let dtBias = castVectorGMH(
            try weights.tensor(named: "\(p).dt_bias"),
            count: nHeads, dtype: dtype, device: device)
        let dTiled = tileDGMH(
            d: try weights.tensor(named: "\(p).D"),
            nHeads: nHeads, headDim: headDim, dtype: dtype, device: device)

        // Gated mixer RMSNorm weight — full [d_inner].
        let mixerNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

        return Granite4MambaMixer(
            inProj: inProj, outProj: outProj,
            convW: convW, convB: convB,
            aEff: aEff, dtBias: dtBias, dTiled: dTiled,
            mixerNorm: mixerNorm,
            dInner: dInner, convDim: convDim,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim,
            nGroups: nGroups, convKernel: convKernel, dtype: dtype)
    }

    /// Build the MoE feed-forward block: top-K SwiGLU experts plus an
    /// always-on shared SwiGLU expert. The per-expert weights ship in
    /// the Granite4-specific "half-stacked" layout:
    ///
    ///   - `input_linear.weight`  : `[E, 2·moeI, hidden]`  — gate AND
    ///     up fused along dim 1 for every expert.
    ///   - `output_linear.weight` : `[E, hidden, moeI]`     — per-
    ///     expert down only.
    ///   - `shared_mlp.input_linear.weight`  : `[2·sharedI, hidden]` —
    ///     same gate+up fuse on the single shared expert.
    ///   - `shared_mlp.output_linear.weight` : `[hidden, sharedI]`.
    ///
    /// Raw + quantized layouts handled symmetrically:
    ///   - Raw: slice per-expert then split dim 0 for gate/up.
    ///   - Quantized: same axis-0 slicing on packed u32 weight +
    ///     bf16/f16 scales+biases (group/pack are along the inner dim
    ///     so axis-0 slicing preserves their boundaries). The
    ///     down_proj triplet additionally folds Granite4's
    ///     `residual_multiplier` into scales+biases via the same
    ///     `loadLinearScaledGMH` math the dense path uses.
    private static func buildMoE(
        prefix p: String, weights: SafeTensorsBundle,
        hidden: Int, moeIntermediate: Int, sharedIntermediate: Int,
        numExperts: Int, topK: Int, residualMultiplier: Float,
        device: Device,
        quantization: ModelConfig.QuantizationConfig?
    ) throws -> MoELayer {
        // Router: hidden → numExperts logits. Small (one weight per
        // expert per channel) — never quantized in the mlx-community
        // conversions.
        let gate = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "\(p).block_sparse_moe.router.layer.weight")))

        var gateProj: [AnyLinear] = []
        var upProj: [AnyLinear] = []
        var downProj: [AnyLinear] = []
        gateProj.reserveCapacity(numExperts)
        upProj.reserveCapacity(numExperts)
        downProj.reserveCapacity(numExperts)

        let inputBase = "\(p).block_sparse_moe.input_linear"
        let outputBase = "\(p).block_sparse_moe.output_linear"

        if let q = quantization, weights.isQuantized(inputBase) {
            // ── Quantized path ────────────────────────────────────────
            // input_linear triplet: weight u32 [E, 2·moeI, hidden/pf],
            // scales/biases bf16 [E, 2·moeI, hidden/gs]. Slice axis 0
            // per expert → [2·moeI, ...]; then axis-0 again into gate
            // (first moeI rows) and up (last moeI rows).
            let inW = try weights.tensor(named: "\(inputBase).weight")
            let inS = try weights.tensor(named: "\(inputBase).scales")
            let inB = try weights.tensor(named: "\(inputBase).biases")
            let inPackedCols = inW.shape[inW.shape.count - 1]
            let inGroupCols = inS.shape[inS.shape.count - 1]
            let inBits = deriveAffineQuantBits(
                weightPackedCols: inPackedCols, scaleCols: inGroupCols,
                groupSize: q.groupSize)
            precondition(
                [2, 3, 4, 5, 6, 8].contains(inBits),
                "Granite4: input_linear derived \(inBits)-bit — unsupported")
            // output_linear triplet: weight u32 [E, hidden, moeI/pf],
            // scales/biases bf16 [E, hidden, moeI/gs]. Per-expert
            // slice + multiplier-fold for the residual_multiplier.
            let outW = try weights.tensor(named: "\(outputBase).weight")
            let outS = try weights.tensor(named: "\(outputBase).scales")
            let outB = try weights.tensor(named: "\(outputBase).biases")
            let outPackedCols = outW.shape[outW.shape.count - 1]
            let outGroupCols = outS.shape[outS.shape.count - 1]
            let outBits = deriveAffineQuantBits(
                weightPackedCols: outPackedCols, scaleCols: outGroupCols,
                groupSize: q.groupSize)
            precondition(
                [2, 3, 4, 5, 6, 8].contains(outBits),
                "Granite4: output_linear derived \(outBits)-bit — unsupported")

            for e in 0 ..< numExperts {
                // Per-expert input slice: [2·moeI, hidden/{pf,gs}].
                let perExpertInW = inW.slicedRows(start: e, count: 1)
                    .reshaped(to: [2 * moeIntermediate, inPackedCols])
                let perExpertInS = inS.slicedRows(start: e, count: 1)
                    .reshaped(to: [2 * moeIntermediate, inGroupCols])
                let perExpertInB = inB.slicedRows(start: e, count: 1)
                    .reshaped(to: [2 * moeIntermediate, inGroupCols])
                // gate = first moeI rows, up = last moeI rows. Axis-0
                // split preserves group/pack alignment.
                gateProj.append(
                    AnyLinear(
                        QuantizedLinear(
                            weight: perExpertInW.slicedRows(start: 0, count: moeIntermediate),
                            scales: perExpertInS.slicedRows(start: 0, count: moeIntermediate),
                            biases: perExpertInB.slicedRows(start: 0, count: moeIntermediate),
                            bits: inBits, groupSize: q.groupSize)))
                upProj.append(
                    AnyLinear(
                        QuantizedLinear(
                            weight: perExpertInW.slicedRows(
                                start: moeIntermediate, count: moeIntermediate),
                            scales: perExpertInS.slicedRows(
                                start: moeIntermediate, count: moeIntermediate),
                            biases: perExpertInB.slicedRows(
                                start: moeIntermediate, count: moeIntermediate),
                            bits: inBits, groupSize: q.groupSize)))

                // Per-expert output slice + residual_multiplier fold
                // into scales/biases (same trick `loadLinearScaledGMH`
                // uses on the dense path).
                let perExpertOutS = outS.slicedRows(start: e, count: 1)
                    .reshaped(to: [hidden, outGroupCols])
                let perExpertOutB = outB.slicedRows(start: e, count: 1)
                    .reshaped(to: [hidden, outGroupCols])
                let scaledS = scaleTensorGMH(
                    perExpertOutS, by: residualMultiplier, device: device)
                let scaledB = scaleTensorGMH(
                    perExpertOutB, by: residualMultiplier, device: device)
                downProj.append(
                    AnyLinear(
                        QuantizedLinear(
                            weight: outW.slicedRows(start: e, count: 1)
                                .reshaped(to: [hidden, outPackedCols]),
                            scales: scaledS, biases: scaledB,
                            bits: outBits, groupSize: q.groupSize)))
            }
        } else {
            // ── Raw path (unchanged) ──────────────────────────────────
            let inputLinear = try weights.tensor(named: "\(inputBase).weight")
            let outputLinear = try weights.tensor(named: "\(outputBase).weight")
            precondition(
                inputLinear.shape == [numExperts, 2 * moeIntermediate, hidden],
                "Granite4: block_sparse_moe.input_linear shape "
                    + "\(inputLinear.shape) ≠ [\(numExperts), \(2 * moeIntermediate), \(hidden)]"
            )
            precondition(
                outputLinear.shape == [numExperts, hidden, moeIntermediate],
                "Granite4: block_sparse_moe.output_linear shape "
                    + "\(outputLinear.shape) ≠ [\(numExperts), \(hidden), \(moeIntermediate)]"
            )
            for e in 0 ..< numExperts {
                let stacked = inputLinear.slicedRows(start: e, count: 1)
                    .reshaped(to: [2 * moeIntermediate, hidden])
                gateProj.append(
                    AnyLinear(
                        Linear(
                            weight: stacked.slicedRows(start: 0, count: moeIntermediate))))
                upProj.append(
                    AnyLinear(
                        Linear(
                            weight: stacked.slicedRows(
                                start: moeIntermediate, count: moeIntermediate))))
                let downRaw = outputLinear.slicedRows(start: e, count: 1)
                    .reshaped(to: [hidden, moeIntermediate])
                downProj.append(
                    AnyLinear(
                        Linear(
                            weight: scaleTensorGMH(
                                downRaw, by: residualMultiplier, device: device))))
            }
        }

        // Shared expert — a plain SwiGLU at `shared_mlp.*`. Shape is
        // `[2·sharedI, hidden]` for input (NOT stacked-by-expert; it's
        // a single shared expert). Quantized + raw handled the same way.
        let sharedInputBase = "\(p).shared_mlp.input_linear"
        let sharedOutputBase = "\(p).shared_mlp.output_linear"

        let sharedGate: AnyLinear
        let sharedUp: AnyLinear
        let sharedDown: AnyLinear

        if let q = quantization, weights.isQuantized(sharedInputBase) {
            // Quantized shared input: axis-0 split into gate/up
            // (preserves group/pack alignment). down folds the
            // residual_multiplier into scales/biases.
            let inT = try weights.quantizedTriplet(sharedInputBase)
            let inBits = deriveAffineQuantBits(
                weightPackedCols: inT.weight.shape[inT.weight.shape.count - 1],
                scaleCols: inT.scales.shape[inT.scales.shape.count - 1],
                groupSize: q.groupSize)
            sharedGate = AnyLinear(
                QuantizedLinear(
                    weight: inT.weight.slicedRows(start: 0, count: sharedIntermediate),
                    scales: inT.scales.slicedRows(start: 0, count: sharedIntermediate),
                    biases: inT.biases.slicedRows(start: 0, count: sharedIntermediate),
                    bits: inBits, groupSize: q.groupSize))
            sharedUp = AnyLinear(
                QuantizedLinear(
                    weight: inT.weight.slicedRows(
                        start: sharedIntermediate, count: sharedIntermediate),
                    scales: inT.scales.slicedRows(
                        start: sharedIntermediate, count: sharedIntermediate),
                    biases: inT.biases.slicedRows(
                        start: sharedIntermediate, count: sharedIntermediate),
                    bits: inBits, groupSize: q.groupSize))
            sharedDown = try loadLinearScaledGMH(
                base: sharedOutputBase, in: weights,
                quantization: q, by: residualMultiplier, device: device)
        } else {
            let sharedInput = try weights.tensor(named: "\(sharedInputBase).weight")
            let sharedOutput = try weights.tensor(named: "\(sharedOutputBase).weight")
            precondition(
                sharedInput.shape == [2 * sharedIntermediate, hidden],
                "Granite4: shared_mlp.input_linear shape "
                    + "\(sharedInput.shape) ≠ [\(2 * sharedIntermediate), \(hidden)]")
            precondition(
                sharedOutput.shape == [hidden, sharedIntermediate],
                "Granite4: shared_mlp.output_linear shape "
                    + "\(sharedOutput.shape) ≠ [\(hidden), \(sharedIntermediate)]")
            sharedGate = AnyLinear(
                Linear(
                    weight: sharedInput.slicedRows(start: 0, count: sharedIntermediate)))
            sharedUp = AnyLinear(
                Linear(
                    weight: sharedInput.slicedRows(
                        start: sharedIntermediate, count: sharedIntermediate)))
            sharedDown = AnyLinear(
                Linear(
                    weight: scaleTensorGMH(
                        sharedOutput, by: residualMultiplier, device: device)))
        }

        // Granite4 routing is top-K of the raw logits, then a
        // softmax over just those K (`.topKThenSoftmax`) — always
        // normalised, so `normTopKProb` does not apply.
        let router = MoERouter(
            nExperts: numExperts, topK: topK,
            gatingMode: .topKThenSoftmax)
        return MoELayer(
            gate: gate,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            sharedGateProj: sharedGate, sharedUpProj: sharedUp,
            sharedDownProj: sharedDown,
            router: router, hidden: hidden)
    }
}

// ─── Mixer + FFN sub-block enums ─────────────────────────────────────

/// The mixer half of a Granite4 layer — Mamba 2 or attention.
enum Granite4Mixer {
    case mamba(Granite4MambaMixer)
    case attention(Granite4AttentionMixer)
}

/// The feed-forward half of a Granite4 layer — block-sparse MoE
/// (commits the command buffer) or a dense SwiGLU MLP.
enum Granite4FFN {
    case moe(MoELayer)
    case dense(Granite4DenseMLP)
}

// ─── Granite4MambaMixer ──────────────────────────────────────
//
// The Mamba 2 selective-SSM mixer half. `out_proj` has had
// residual_multiplier folded in at load time; the gated mixer RMSNorm
// is a single full-width RMSNorm over d_inner.

public final class Granite4MambaMixer: Module {
    let inProj, outProj: AnyLinear
    let convW: Tensor  // [kernel, conv_dim]
    let convB: Tensor  // [conv_dim]
    let aEff: Tensor  // [n_heads]   = -exp(A_log)
    let dtBias: Tensor  // [n_heads]
    let dTiled: Tensor  // [d_inner]   D[h] tiled across head_dim
    let mixerNorm: RMSNorm  // gated mixer RMSNorm weight [d_inner]
    let dInner, convDim, nHeads, headDim, stateDim, nGroups, convKernel: Int
    let dtype: DType
    /// Heads sharing one B/C group.
    let headsPerGroup: Int

    init(
        inProj: AnyLinear, outProj: AnyLinear,
        convW: Tensor, convB: Tensor,
        aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
        mixerNorm: RMSNorm,
        dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, dtype: DType
    ) {
        self.inProj = inProj
        self.outProj = outProj
        self.convW = convW
        self.convB = convB
        self.aEff = aEff
        self.dtBias = dtBias
        self.dTiled = dTiled
        self.mixerNorm = mixerNorm
        self.dInner = dInner
        self.convDim = convDim
        self.nHeads = nHeads
        self.headDim = headDim
        self.stateDim = stateDim
        self.nGroups = nGroups
        self.convKernel = convKernel
        self.dtype = dtype
        self.headsPerGroup = nHeads / nGroups
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inProj.parameters() { out.append(("mamba.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mamba.out_proj.\(k)", v)) }
        for (k, v) in mixerNorm.parameters() { out.append(("mamba.norm.\(k)", v)) }
        return out
    }

    /// Single-token mixer forward. `xNorm` is the already-normalized
    /// layer input. Returns the post-out_proj mixer contribution
    /// (residual add done by the enclosing layer), shape [hidden].
    func forward(
        _ xNorm: Tensor, cache: Mamba2LayerCache,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // in_proj → split into z (gate) / xBC / dt_raw.
        // in_proj output layout: [d_inner | conv_dim | n_heads].
        let proj = inProj(xNorm, on: cmd)
        let z = proj.slicedRows(start: 0, count: dInner)
        let xBC = proj.slicedRows(start: dInner, count: convDim)
        let dtRaw = proj.slicedRows(start: dInner + convDim, count: nHeads)

        // conv1d causal step (rolling state) + SiLU.
        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: xBC, w: convW, b: convB,
            state: cache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)

        // split conv output → x / B / C.
        // conv layout: [d_inner | n_groups*state_dim | n_groups*state_dim].
        let x = convAct.slicedRows(start: 0, count: dInner)
        let bAll = convAct.slicedRows(start: dInner, count: nGroups * stateDim)
            .reshaped(to: [nGroups, stateDim])
        let cAll = convAct.slicedRows(
            start: dInner + nGroups * stateDim,
            count: nGroups * stateDim
        )
        .reshaped(to: [nGroups, stateDim])

        // dt = softplus(dt_raw + dt_bias).
        let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
        let dt = Ops.softplus(dtSum, on: cmd)

        // selective scan — dispatched per group so the shipped
        // single-group ssm_step kernel handles grouped B/C. With
        // n_groups = 1 (every published Granite-4 checkpoint) this is a
        // single dispatch; the loop also covers a future n_groups > 1
        // checkpoint without a kernel change.
        let y = Tensor.empty(shape: [nHeads, headDim], dtype: dtype, device: device)
        let xHeads = x.reshaped(to: [nHeads, headDim])
        let stateHeads = cache.ssm.h  // [nHeads, headDim, stateDim]
        for g in 0 ..< nGroups {
            let h0 = g * headsPerGroup
            let xg = xHeads.slicedRows(start: h0, count: headsPerGroup)
                .reshaped(to: [headsPerGroup * headDim])
            let yg = y.slicedRows(start: h0, count: headsPerGroup)
            let stateG = stateHeads.slicedRows(start: h0, count: headsPerGroup)
            let aG = aEff.slicedRows(start: h0, count: headsPerGroup)
            let dtG = dt.slicedRows(start: h0, count: headsPerGroup)
            let bG = bAll.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
            let cG = cAll.slicedRows(start: g, count: 1).reshaped(to: [stateDim])
            Ops.ssmStep(
                x: xg, a: aG, b: bG, c: cG, dt: dtG,
                state: stateG, into: yg,
                nHeads: headsPerGroup, headDim: headDim, stateDim: stateDim,
                on: cmd)
        }
        let yFlat = y.reshaped(to: [dInner])

        // skip: y += D_tiled * x.
        let dx = Ops.mul(dTiled, x, on: cmd)
        let ySkip = Ops.add(yFlat, dx, on: cmd)

        // gated mixer RMSNorm: y *= silu(z), then a single full-width
        // RMSNorm over d_inner. Matches Granite4RMSNormGated.
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)
        let yNormed = Ops.rmsNorm(
            yGated, weight: mixerNorm.weight, eps: mixerNorm.eps, on: cmd)

        // out_proj → [hidden] (residual_multiplier already folded in).
        return outProj(yNormed, on: cmd)
    }
}

// ─── Granite4AttentionMixer ──────────────────────────────────
//
// Multi-head attention with NO positional embedding (no RoPE — every
// Granite-4 "-H" checkpoint ships position_embedding_type "nope").
// `scale` is the config's attention_multiplier; `o_proj` has had
// residual_multiplier folded in at load time.

public final class Granite4AttentionMixer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let nHeads, nKVHeads, headDim: Int
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        nHeads: Int, nKVHeads: Int, headDim: Int, scale: Float
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.scale = scale
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        return out
    }

    /// Single-token attention forward. Returns the post-o_proj
    /// contribution (residual add done by the enclosing layer).
    func forward(
        _ xNorm: Tensor, cache kv: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // No RoPE — Granite "-H" attention attends without positional
        // rotation. K/V go straight into the cache unrotated.
        kv.appendOnGPU(
            kFlat: k.reshaped(to: [nKVHeads, headDim]),
            vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        // AURA caches store K/V Π-rotated — rotate Q in / un-rotate out
        // (no-op for raw / affine).
        let qForSdpa = auraRotatedQuery(
            q.reshaped(to: [nHeads, headDim]), cache: kv,
            nHeads: nHeads, headDim: headDim, on: cmd)
        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.capacity,
            scale: scale, on: cmd)
        let outFlat = auraUnrotatedOutput(
            attnOut, cache: kv, nHeads: nHeads, headDim: headDim, on: cmd)

        return oProj(outFlat, on: cmd)
    }
}

// ─── Granite4DenseMLP ────────────────────────────────────────
//
// The dense feed-forward path (Granite-4 "-H" checkpoints with
// num_local_experts == 0). A plain SwiGLU; down_proj has had
// residual_multiplier folded in at load time.

public final class Granite4DenseMLP: Module {
    let gateProj, upProj, downProj: AnyLinear

    init(gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        return out
    }

    /// down(silu(gate(x)) * up(x)). Returns the FFN contribution
    /// (residual_multiplier already folded into down_proj).
    func forward(_ xNorm: Tensor, cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(xNorm, on: cmd)
        let u = upProj(xNorm, on: cmd)
        // Fused silu(gate) * up — one kernel, intermediate silu(g)
        // stays in registers instead of round-tripping to DRAM.
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        return downProj(inner, on: cmd)
    }
}

// ─── Granite4Layer ───────────────────────────────────────────
//
// One stack-interleaved hybrid layer: a mixer (Mamba 2 OR attention)
// with `input_layernorm`, then a feed-forward (MoE+shared OR dense MLP)
// with `post_attention_layernorm`. Both residual adds are plain
// Ops.add — residual_multiplier was folded into the output projections
// at load time.
//
// `commitsCommandBuffer` is true when the FFN is an `MoELayer`: the MoE
// router commits the command buffer mid-layer, so the host model must
// allocate a fresh one after this layer's `decode` returns.

public final class Granite4Layer: Module, DecoderLayer {
    let inputNorm, postNorm: RMSNorm
    let mixer: Granite4Mixer
    let ffn: Granite4FFN
    let hidden: Int

    /// True when this layer's FFN commits the command buffer it is given
    /// (MoE-bearing layers only). The host decode loop refreshes `cmd`
    /// after any layer for which this is true.
    public let commitsCommandBuffer: Bool

    init(
        inputNorm: RMSNorm, postNorm: RMSNorm,
        mixer: Granite4Mixer, ffn: Granite4FFN, hidden: Int
    ) {
        self.inputNorm = inputNorm
        self.postNorm = postNorm
        self.mixer = mixer
        self.ffn = ffn
        self.hidden = hidden
        if case .moe = ffn {
            self.commitsCommandBuffer = true
        } else {
            self.commitsCommandBuffer = false
        }
    }

    /// The Mamba 2 mixer cache slot is a `Mamba2LayerCache`; the
    /// attention cache slot is a `KVCache`. Either way the FFN holds no
    /// per-token state.
    var kind: Granite4LayerKind {
        switch mixer {
        case .mamba: return .mamba
        case .attention: return .attention
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        switch mixer {
        case .mamba(let m): out.append(contentsOf: m.parameters())
        case .attention(let a): out.append(contentsOf: a.parameters())
        }
        switch ffn {
        case .moe(let moe):
            // Re-key the MoELayer parameters into Granite's checkpoint
            // layout (`block_sparse_moe.*` / `shared_mlp.*`).
            for (k, v) in moe.parameters() {
                out.append((graniteMoEKey(k), v))
            }
        case .dense(let mlp):
            out.append(contentsOf: mlp.parameters())
        }
        return out
    }

    /// `DecoderLayer` conformance — layer-local single-token decode.
    ///
    /// IMPORTANT: when the FFN is an `MoELayer`, this commits the passed
    /// `cmd` (the router needs the gate logits on the CPU). The host
    /// model checks `commitsCommandBuffer` and refreshes `cmd`
    /// afterwards. See `Granite4Model.forward`.
    public func decode(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── Mixer half — pre-norm + mixer + residual add ──────────────
        let xNorm = inputNorm(h, on: cmd)
        let mixerOut: Tensor
        switch mixer {
        case .mamba(let m):
            guard let mc = cache as? Mamba2LayerCache else {
                fatalError(
                    "Granite4Layer: mamba layer expected "
                        + "Mamba2LayerCache, got \(type(of: cache))")
            }
            mixerOut = m.forward(xNorm, cache: mc, cmd: cmd, device: device)
            mc.advance()
        case .attention(let a):
            guard let kv = cache as? any KVCacheProtocol else {
                fatalError(
                    "Granite4Layer: attention layer expected "
                        + "KVCache, got \(type(of: cache))")
            }
            mixerOut = a.forward(xNorm, cache: kv, cmd: cmd, device: device)
        }
        // residual_multiplier already folded into the mixer output proj.
        // Fused residual add + post-mixer RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096), but ONLY on the attention-mixer branch —
        // the task carve-out is "attention/FFN residuals, NOT the
        // SSM recurrence paths." Validator gate handles wider variants.
        let postMix: Tensor
        let ffnNorm: Tensor
        if case .attention = mixer,
            OpsValidation.validateAddRmsNorm(n: hidden) == nil
        {
            let fused = Ops.addAndRmsNorm(
                h, mixerOut, weight: postNorm.weight, eps: postNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postMix = fused.residual
            ffnNorm = fused.normed
        } else {
            postMix = Ops.add(h, mixerOut, on: cmd)
            // ── Feed-forward half — pre-norm + FFN + residual add ─────────
            ffnNorm = postNorm(postMix, on: cmd)
        }
        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forward(ffnNorm, cmd: cmd)
            return Ops.add(postMix, ffnOut, on: cmd)
        case .moe(let moe):
            // MoELayer.decode commits `cmd` and runs the experts on its
            // own private buffer; it returns a fully-resident tensor.
            // The FFN includes the always-on shared expert. The host
            // model refreshes `cmd` after this layer (see the header).
            let ffnOut = moe.decode(
                ffnNorm, position: position,
                cache: StatelessLayerCache(),
                cmd: cmd, device: device)
            // postMix is already resident (cmd was committed by the MoE
            // layer, which waited for completion). The add queues onto a
            // fresh private buffer here so the returned tensor is valid
            // without depending on the now-dead `cmd`.
            let addCmd = device.makeCommandBuffer()
            let result = Ops.add(postMix, ffnOut, on: addCmd)
            addCmd.commit()
            addCmd.waitUntilCompleted()
            return result
        }
    }
}

/// Re-key a flat `MoELayer` parameter name into Granite4's
/// checkpoint layout. `MoELayer` emits `gate.*` / `experts.<e>.*` /
/// `shared_expert.*`; Granite stores `block_sparse_moe.router.layer.*`
/// / stacked `block_sparse_moe.*` / `shared_mlp.*`.
private func graniteMoEKey(_ k: String) -> String {
    if k.hasPrefix("gate.") {
        return "block_sparse_moe.router.layer." + k.dropFirst("gate.".count)
    }
    if k.hasPrefix("shared_expert.") {
        return "shared_mlp." + k.dropFirst("shared_expert.".count)
    }
    // Per-expert weights are sliced from a stacked tensor at load time;
    // there is no 1:1 checkpoint key, so keep the MoELayer-flat name.
    return "block_sparse_moe." + k
}

// ─── Granite4Model ───────────────────────────────────────────

public final class Granite4Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a Mamba or attention
    /// hybrid layer, ordered by `layer_types`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxContextWindow: Int
    public let mambaNHeads, mambaHeadDim, stateDim, convDim, convKernel, nGroups, dInner: Int
    /// Final logits are divided by this (Granite's `logits_scaling`).
    public let logitsScaling: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [Granite4LayerKind]
    /// True when this model has any MoE-bearing layer. Purely
    /// informational — `forward` uses the uniform internal-`workCmd`
    /// discipline regardless of whether any layer commits.
    public let hasMoE: Bool

    init(
        embedTokens: AnyEmbedding, layers: [any DecoderLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        mambaNHeads: Int, mambaHeadDim: Int, stateDim: Int,
        convDim: Int, convKernel: Int, nGroups: Int, dInner: Int,
        vocab: Int, maxContextWindow: Int, logitsScaling: Float, dtype: DType,
        kvCacheKind: KVCacheKind = .raw,
        kvEviction: KVEviction = .unbounded
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.mambaNHeads = mambaNHeads
        self.mambaHeadDim = mambaHeadDim
        self.stateDim = stateDim
        self.convDim = convDim
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.dInner = dInner
        self.vocab = vocab
        self.maxContextWindow = maxContextWindow
        self.logitsScaling = logitsScaling
        self.dtype = dtype
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
        self.layerKinds = layers.map { layer in
            (layer as? Granite4Layer)?.kind ?? .mamba
        }
        self.hasMoE = layers.contains {
            ($0 as? Granite4Layer)?.commitsCommandBuffer ?? false
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            if let l = layer as? Granite4Layer {
                for (k, v) in l.parameters() {
                    out.append(("model.layers.\(i).\(k)", v))
                }
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One cache per layer index, matching the layer kind:
    ///   mamba → Mamba2LayerCache, attention → KVCache.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxContextWindow
        // raw / affine / AURA all run through the shared factory; one
        // dequant scratch is shared across every attention layer.
        let scratch = makeAttentionScratch(
            kind: kvCacheKind, nKVHeads: nKVHeads, headDim: headDim,
            contextLength: cap, dtype: dtype, device: device)
        return layerKinds.enumerated().map { (i, layerKind) in
            switch layerKind {
            case .mamba:
                return Mamba2LayerCache(
                    nHeads: mambaNHeads, stateDim: stateDim, headDim: mambaHeadDim,
                    convChannels: convDim, convKernelSize: convKernel,
                    dtype: dtype, device: device)
            case .attention:
                return makeAttentionCache(
                    kind: kvCacheKind, scratch: scratch,
                    nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                    dtype: dtype, eviction: kvEviction, layerIndex: i, device: device)
            }
        }
    }

    /// Queue a single-token forward pass onto `cmd`. **Does not commit
    /// `cmd`** — the protocol contract holds, so the default
    /// `forwardSample` / `forwardSampleCategorical` extensions compose
    /// their output kernels onto `cmd` and commit once, exactly like
    /// every other family.
    ///
    /// CRITICAL — command-buffer contract. When a layer's FFN is an
    /// `MoELayer` its `decode` commits the command buffer it is handed
    /// (the router reads the gate logits back on the CPU). So the
    /// caller's `cmd` must NEVER be handed to a layer — if it were, the
    /// first MoE-bearing layer would commit it and the caller's later
    /// commit would double-commit. Instead the embedding + every layer
    /// run on internal `workCmd` buffers (committed by the layers
    /// themselves / refreshed after each committing layer), and ONLY the
    /// final `norm` + `lm_head` + logits_scaling queue onto the caller's
    /// pristine `cmd`.
    ///
    /// This discipline is uniform across the dense and MoE checkpoints:
    /// dense Granite-4 "-H" stacks (H-350M / H-1B, `num_local_experts =
    /// 0`) have no committing layer, so the loop commits `workCmd` once
    /// after the stack to make `h` resident before the caller's `cmd`
    /// reads it; MoE stacks (H-Tiny / H-Small) have `workCmd` committed +
    /// refreshed by each MoE layer. Either way the caller's single
    /// commit of `cmd` produces correct final logits.
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        // The embedding + layers run on internal buffers — never `cmd`.
        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.decode(
                h, position: position, cache: caches[i],
                cmd: workCmd, device: device)
            // If the layer committed `workCmd` (MoE FFN), swap in a
            // fresh buffer for the next layer.
            if let g = layer as? Granite4Layer, g.commitsCommandBuffer {
                workCmd = device.makeCommandBuffer()
            }
        }

        // After a committing layer `workCmd` is a fresh, empty buffer and
        // `h` is already resident. After a non-committing layer (the
        // dense path, or an MoE stack ending on a dense layer) `workCmd`
        // still carries that layer's uncommitted work — commit it so `h`
        // is resident before the caller's `cmd` reads it.
        let lastCommitted =
            (layers.last as? Granite4Layer)?
            .commitsCommandBuffer ?? false
        if !lastCommitted {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        // Final norm + lm_head queue onto the caller's pristine `cmd`.
        let normed = finalNorm(h, on: cmd)
        let logits = lmHead(normed, on: cmd)

        // Apply logits_scaling (logits = logits / logits_scaling). The
        // scale divide queues onto the caller's `cmd` too, so the
        // caller's single commit produces correct final logits.
        if logitsScaling != 1.0 {
            let invScale = Tensor.filled(
                1.0 / logitsScaling, shape: logits.shape,
                dtype: logits.dtype, device: device)
            return Ops.mul(logits, invScale, on: cmd)
        }
        return logits
    }

    /// Multi-token forward — prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// Granite4 interleaves Mamba 2 + MoE-FFN + attention
    /// layers. The MoE-FFN router commits mid-layer for CPU readback;
    /// a per-attention-layer `decodeMulti` override will need to
    /// preserve that commit pattern across the chunk. Today this
    /// override is commit-count-batched only.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "Granite4Model.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }
}

// ─── Load-time host helpers ──────────────────────────────────────────
//
// Small CPU-side derivations done once at load — the cost is in the
// noise. Mirror the FalconH1 / NemotronH helpers.

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
private func readFloatsGMH(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("Granite4: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloatsGMH(
    _ values: [Float], shape: [Int],
    dtype: DType, device: Device
) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("Granite4: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// Multiply every element of `t` by a scalar, returning a fresh tensor
/// in `t`'s dtype. Identity-fast-path: returns `t` unchanged when the
/// multiplier is exactly 1.0.
private func scaleTensorGMH(_ t: Tensor, by m: Float, device: Device) -> Tensor {
    if m == 1.0 { return t }
    let floats = readFloatsGMH(t).map { $0 * m }
    return writeFloatsGMH(floats, shape: t.shape, dtype: t.dtype, device: device)
}

/// Load a quantized Linear with a scalar fold-in into its
/// `scales`/`biases` — mathematically equivalent to multiplying every
/// dequantized output by `m`:
///
///     dequant      = nibble * scale + bias
///     dequant * m  = nibble * (m·scale) + (m·bias)
///
/// Used to fold Granite4's `residual_multiplier` into the out_proj /
/// o_proj / down_proj triplets on the quantized path (the raw path
/// folds the multiplier directly into the weight via scaleTensorGMH).
/// The packed u32 `weight` is untouched.
///
/// Falls back to the regular raw `Linear` path when the bundle isn't
/// quantized at this base (defensive — every Granite4 4-bit
/// checkpoint we ship the o_proj / out_proj / mlp.down_proj as
/// quantized triplets).
private func loadLinearScaledGMH(
    base: String, in weights: SafeTensorsBundle,
    quantization q: ModelConfig.QuantizationConfig,
    by m: Float, device: Device
) throws -> AnyLinear {
    guard weights.isQuantized(base) else {
        let w = scaleTensorGMH(
            try weights.tensor(named: "\(base).weight"),
            by: m, device: device)
        return AnyLinear(Linear(weight: w))
    }
    let t = try weights.quantizedTriplet(base)
    let scaledScales = scaleTensorGMH(t.scales, by: m, device: device)
    let scaledBiases = scaleTensorGMH(t.biases, by: m, device: device)
    let bits = deriveAffineQuantBits(
        weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
        scaleCols: t.scales.shape[t.scales.shape.count - 1],
        groupSize: q.groupSize)
    return AnyLinear(
        QuantizedLinear(
            weight: t.weight, scales: scaledScales, biases: scaledBiases,
            bits: bits, groupSize: q.groupSize))
}

/// A_eff = -exp(A_log), per head, in the activation dtype.
private func computeAEffGMH(
    aLog: Tensor, nHeads: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsGMH(aLog)
    precondition(floats.count == nHeads, "Granite4: A_log expected [n_heads]")
    return writeFloatsGMH(
        floats.map { -Foundation.exp($0) },
        shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head / per-channel vector to the activation dtype.
private func castVectorGMH(
    _ src: Tensor, count: Int,
    dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsGMH(src)
    precondition(floats.count == count, "Granite4: vector size mismatch")
    return writeFloatsGMH(floats, shape: [count], dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels → `[n_heads * head_dim]`.
private func tileDGMH(
    d: Tensor, nHeads: Int, headDim: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsGMH(d)
    precondition(floats.count == nHeads, "Granite4: D expected [n_heads]")
    var tiled: [Float] = []
    tiled.reserveCapacity(nHeads * headDim)
    for h in 0 ..< nHeads {
        for _ in 0 ..< headDim { tiled.append(floats[h]) }
    }
    return writeFloatsGMH(tiled, shape: [nHeads * headDim], dtype: dtype, device: device)
}

/// Transpose HF conv1d.weight `[C, 1, K]` → `[K, C]` for the metaltile
/// conv kernel.
private func transposeConv1dWeightGMH(
    src: Tensor, kernel K: Int, channels C: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsGMH(src)
    precondition(floats.count == K * C, "Granite4: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0 ..< C {
        for k in 0 ..< K { dst[k * C + c] = floats[c * K + k] }
    }
    return writeFloatsGMH(dst, shape: [K, C], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVectorGMH(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}
