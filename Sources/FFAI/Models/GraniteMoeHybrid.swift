// GraniteMoeHybrid family — a Phase 5e *stack-interleaved hybrid* model
// with a mixture-of-experts feed-forward block.
//
// GraniteMoeHybrid (IBM's Granite 4.0 "-H" series — H-350M / H-1B /
// H-Tiny / H-Small) is a **stack-interleaved** hybrid like NemotronH: a
// `layer_types` array assigns each decoder layer exactly ONE mixer kind
// — "mamba" or "attention" — and the kinds vary down the stack (the
// published checkpoints are mostly Mamba with a sparse handful of
// attention layers). Unlike NemotronH, the FEED-FORWARD half of every
// layer is identical across the stack:
//
//   * `num_local_experts > 0`  → block-sparse MoE (top-K SwiGLU experts)
//                                 PLUS an always-on shared SwiGLU expert.
//   * `num_local_experts == 0` → a plain dense SwiGLU MLP.
//
// H-350M / H-1B ship `num_local_experts = 0` (dense FFN); H-Tiny /
// H-Small ship the 64-expert MoE FFN. Both FFN shapes are supported.
//
// Per-layer dataflow (matches mlx-lm's `granitemoehybrid.py`):
//
//   residual = h
//   h        = input_layernorm(h)                 [hidden]
//   h        = mixer(h)                            [hidden]  — mamba / attn
//   h        = residual + h * residual_multiplier
//   residual = h
//   n        = post_attention_layernorm(h)         [hidden]
//   ffn      = MoE(n) + shared_mlp(n)   (or)  dense_mlp(n)
//   out      = residual + ffn * residual_multiplier
//
// ─── No RoPE ─────────────────────────────────────────────────────────
//
// Every published Granite-4 "-H" checkpoint ships
// `position_embedding_type: "nope"` — the attention layers attend
// WITHOUT positional rotation (the Mamba layers carry sequence order).
// `GraniteMoeHybridAttentionLayer` therefore skips the `Ops.rope` call.
//
// ─── Scalar multipliers — applied at runtime / folded, never doubled ─
//
// Granite scatters four scalar multipliers (`embedding_multiplier`,
// `attention_multiplier`, `residual_multiplier`, `logits_scaling`).
// Unlike FalconH1's µP vectors, mlx-lm's GraniteMoeHybrid `sanitize`
// does NOT fold any of them — they live as runtime config values and
// the reference applies them live in `callAsFunction`. mlx-community
// conversions preserve that (their `sanitize` only transposes conv1d
// and splits the stacked MoE weights), so there is no double-fold
// hazard. We:
//   * fold `embedding_multiplier` into a dedicated scaled copy of the
//     embedding table (the tied lm_head keeps the unscaled table);
//   * fold `residual_multiplier` into every mixer `out_proj` and every
//     FFN down-projection so the decode hot path stays a plain
//     `residual + mixerOut` with zero runtime scalar ops;
//   * keep `attention_multiplier` as the SDPA scale;
//   * keep `logits_scaling` as a final divide on the logits.
//
// ─── MoE commits the command buffer ──────────────────────────────────
//
// `MoELayer.decode` commits the command buffer it is handed (the router
// needs the gate logits on the CPU). A GraniteMoeHybrid layer whose FFN
// is an `MoELayer` therefore commits mid-layer. `GraniteMoeHybridModel.
// forward` keeps ALL per-layer work on internal self-managed command
// buffers — never the caller's `cmd` — so a committing layer can never
// double-commit the caller's buffer. It refreshes the internal `workCmd`
// after each committing layer (`commitsCommandBuffer` flag) and queues
// only the final norm + lm_head onto the caller's pristine `cmd`. See
// the `forward` doc comment and the MoELayer file header.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum GraniteMoeHybrid {
    public static let modelTypes: Set<String> = ["granitemoehybrid"]
    public static let architectures: Set<String> = ["GraniteMoeHybridForCausalLM"]

    public static func variant(for _: ModelConfig) throws -> any GraniteMoeHybridVariant.Type {
        return GraniteMoeHybridHybrid.self
    }
}

public protocol GraniteMoeHybridVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> GraniteMoeHybridModel
}

public enum GraniteMoeHybridError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "GraniteMoeHybrid: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "GraniteMoeHybrid: unsupported config: \(m)"
        }
    }
}

// ─── Layer kind ──────────────────────────────────────────────────────

/// The two mixer kinds a `layer_types` entry can name.
enum GraniteMoeHybridLayerKind: Equatable {
    case mamba       // "mamba"
    case attention   // "attention"

    init(from name: String) throws {
        switch name {
        case "mamba": self = .mamba
        case "attention": self = .attention
        default:
            throw GraniteMoeHybridError.unsupportedConfig(
                "unknown layer_types entry '\(name)'")
        }
    }
}

// ─── GraniteMoeHybridHybrid — the single variant ─────────────────────

public struct GraniteMoeHybridHybrid: GraniteMoeHybridVariant {
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
        options _: LoadOptions,
        device: Device
    ) throws -> GraniteMoeHybridModel {
        guard let hidden = config.hiddenSize,
              let vocab = config.vocabSize,
              let nHeads = config.numAttentionHeads
        else {
            throw GraniteMoeHybridError.missingConfig(
                "hidden / vocab / num_attention_heads")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let eps = Float(config.rmsNormEps ?? 1e-5)
        let tieEmbed = config.tieWordEmbeddings

        // ── Hybrid layer schedule ─────────────────────────────────────
        guard let layerTypeNames = config.raw["layer_types"] as? [String],
              !layerTypeNames.isEmpty
        else { throw GraniteMoeHybridError.missingConfig("layer_types") }
        let kinds = try layerTypeNames.map { try GraniteMoeHybridLayerKind(from: $0) }
        let nLayers = kinds.count

        // ── Scalar multipliers ────────────────────────────────────────
        // GraniteMoeHybrid keeps all four as runtime config values
        // (mlx-lm's sanitize folds none of them) — no double-fold risk.
        let embeddingMultiplier = Float(config.float("embedding_multiplier") ?? 1.0)
        let residualMultiplier = Float(config.float("residual_multiplier") ?? 1.0)
        let logitsScaling = Float(config.float("logits_scaling") ?? 1.0)
        // attention_multiplier replaces the usual 1/sqrt(head_dim) scale.
        let attentionScale = Float(config.float("attention_multiplier")
            ?? (1.0 / Double(headDim).squareRoot()))

        // ── Mamba 2 mixer geometry ────────────────────────────────────
        guard let mambaNHeads = config.int("mamba_n_heads")
        else { throw GraniteMoeHybridError.missingConfig("mamba_n_heads") }
        guard let mambaHeadDim = config.int("mamba_d_head")
        else { throw GraniteMoeHybridError.missingConfig("mamba_d_head") }
        guard let stateDim = config.int("mamba_d_state")
        else { throw GraniteMoeHybridError.missingConfig("mamba_d_state") }
        let convKernel = config.int("mamba_d_conv") ?? 4
        let nGroups = config.int("mamba_n_groups") ?? 1
        let useConvBias = config.bool("mamba_conv_bias") ?? true

        // d_inner taken directly from the Mamba head decomposition.
        let dInner = mambaNHeads * mambaHeadDim
        guard mambaNHeads % nGroups == 0 else {
            throw GraniteMoeHybridError.unsupportedConfig(
                "mamba_n_heads (\(mambaNHeads)) must be a multiple of "
                + "n_groups (\(nGroups))")
        }
        // Granite's gated mixer RMSNorm is a single full-width RMSNorm
        // over d_inner (NOT per-group like NemotronH). The metaltile
        // rms_norm reduction kernel requires the row size to be a
        // multiple of 128 and ≤ 4096.
        guard dInner % 128 == 0, dInner <= 4096 else {
            throw GraniteMoeHybridError.unsupportedConfig(
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
        let embedWRaw = try weights.tensor(named: "model.embed_tokens.weight")
        let activationDtype = embedWRaw.dtype
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "GraniteMoeHybrid: unexpected activation dtype \(activationDtype)")
        guard config.quantization == nil else {
            throw GraniteMoeHybridError.unsupportedConfig(
                "quantized GraniteMoeHybrid checkpoints not yet supported — "
                + "load a raw bf16/f16 variant")
        }

        // Embedding table folds embedding_multiplier (the tied lm_head
        // keeps the unscaled table — see file header).
        let embedW = scaleTensorGMH(embedWRaw, by: embeddingMultiplier, device: device)
        let embedTokens = AnyEmbedding(Embedding(weight: embedW))

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
            let mixer: GraniteMoeHybridMixer
            switch kind {
            case .mamba:
                mixer = .mamba(try buildMambaMixer(
                    prefix: "\(p).mamba", weights: weights,
                    dInner: dInner, convDim: convDim,
                    nHeads: mambaNHeads, headDim: mambaHeadDim, stateDim: stateDim,
                    nGroups: nGroups, convKernel: convKernel,
                    useConvBias: useConvBias, eps: eps,
                    tsMin: tsMin, tsMax: tsMax,
                    residualMultiplier: residualMultiplier,
                    dtype: activationDtype, device: device))
            case .attention:
                // o_proj folds residual_multiplier so the residual add
                // stays a plain Ops.add.
                let qProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.q_proj.weight")))
                let kProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.k_proj.weight")))
                let vProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.v_proj.weight")))
                let oW = scaleTensorGMH(
                    try weights.tensor(named: "\(p).self_attn.o_proj.weight"),
                    by: residualMultiplier, device: device)
                mixer = .attention(GraniteMoeHybridAttentionMixer(
                    qProj: qProj, kProj: kProj, vProj: vProj,
                    oProj: AnyLinear(Linear(weight: oW)),
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    scale: attentionScale))
            }

            // ── Feed-forward ──────────────────────────────────────────
            let ffn: GraniteMoeHybridFFN
            if useMoE {
                ffn = .moe(try buildMoE(
                    prefix: p, weights: weights,
                    hidden: hidden, moeIntermediate: intermediate,
                    sharedIntermediate: sharedIntermediate,
                    numExperts: numLocalExperts, topK: numExpertsPerToken,
                    residualMultiplier: residualMultiplier, device: device))
            } else {
                // Dense SwiGLU MLP. down_proj folds residual_multiplier.
                let gateW = try weights.tensor(named: "\(p).mlp.gate_proj.weight")
                let upW = try weights.tensor(named: "\(p).mlp.up_proj.weight")
                let downW = scaleTensorGMH(
                    try weights.tensor(named: "\(p).mlp.down_proj.weight"),
                    by: residualMultiplier, device: device)
                ffn = .dense(GraniteMoeHybridDenseMLP(
                    gateProj: AnyLinear(Linear(weight: gateW)),
                    upProj: AnyLinear(Linear(weight: upW)),
                    downProj: AnyLinear(Linear(weight: downW))))
            }

            layers.append(GraniteMoeHybridLayer(
                inputNorm: inputNorm, postNorm: postNorm,
                mixer: mixer, ffn: ffn, hidden: hidden))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"), eps: eps)

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = AnyLinear(Linear(weight: try weights.tensor(named: "lm_head.weight")))
        } else {
            // Tied: the head shares the *unscaled* embedding table.
            lmHead = AnyLinear(Linear(weight: embedWRaw))
        }

        let maxSeq = config.int("max_position_embeddings") ?? 8192
        return GraniteMoeHybridModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            mambaNHeads: mambaNHeads, mambaHeadDim: mambaHeadDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            nGroups: nGroups, dInner: dInner,
            vocab: vocab, maxSeq: maxSeq,
            logitsScaling: logitsScaling, dtype: activationDtype)
    }

    /// Build one Mamba 2 mixer. Reads + derives the per-head SSM
    /// parameters and transposes the conv1d weight.
    private static func buildMambaMixer(
        prefix p: String, weights: SafeTensorsBundle,
        dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, useConvBias: Bool, eps: Float,
        tsMin: Float, tsMax: Float, residualMultiplier: Float,
        dtype: DType, device: Device
    ) throws -> GraniteMoeHybridMambaMixer {
        let inProj = AnyLinear(Linear(
            weight: try weights.tensor(named: "\(p).in_proj.weight")))
        // out_proj folds residual_multiplier — the layer-level residual
        // add stays a plain Ops.add.
        let outW = scaleTensorGMH(
            try weights.tensor(named: "\(p).out_proj.weight"),
            by: residualMultiplier, device: device)
        let outProj = AnyLinear(Linear(weight: outW))

        // conv1d.weight ships [conv_dim, 1, kernel]; the metaltile kernel
        // wants [kernel, conv_dim].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(convWSrc.elementCount == convDim * convKernel,
                     "GraniteMoeHybrid: conv1d.weight count mismatch: \(convWSrc.shape)")
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

        return GraniteMoeHybridMambaMixer(
            inProj: inProj, outProj: outProj,
            convW: convW, convB: convB,
            aEff: aEff, dtBias: dtBias, dTiled: dTiled,
            mixerNorm: mixerNorm,
            dInner: dInner, convDim: convDim,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim,
            nGroups: nGroups, convKernel: convKernel, dtype: dtype)
    }

    /// Build the MoE feed-forward block: top-K SwiGLU experts plus an
    /// always-on shared SwiGLU expert. The per-expert weights ship
    /// stacked; the router + shared expert ship as plain 2D weights.
    private static func buildMoE(
        prefix p: String, weights: SafeTensorsBundle,
        hidden: Int, moeIntermediate: Int, sharedIntermediate: Int,
        numExperts: Int, topK: Int, residualMultiplier: Float,
        device: Device
    ) throws -> MoELayer {
        // Router: hidden → numExperts logits.
        let gate = AnyLinear(Linear(
            weight: try weights.tensor(named: "\(p).block_sparse_moe.router.layer.weight")))

        // Per-expert SwiGLU. `input_linear.weight` ships stacked
        // [numExperts, 2*moeIntermediate, hidden] — slice expert e, then
        // split dim-1 into gate / up. `output_linear.weight` ships
        // [numExperts, hidden, moeIntermediate].
        let inputLinear = try weights.tensor(
            named: "\(p).block_sparse_moe.input_linear.weight")
        let outputLinear = try weights.tensor(
            named: "\(p).block_sparse_moe.output_linear.weight")
        precondition(inputLinear.shape == [numExperts, 2 * moeIntermediate, hidden],
                     "GraniteMoeHybrid: block_sparse_moe.input_linear shape "
                     + "\(inputLinear.shape) ≠ [\(numExperts), \(2 * moeIntermediate), \(hidden)]")
        precondition(outputLinear.shape == [numExperts, hidden, moeIntermediate],
                     "GraniteMoeHybrid: block_sparse_moe.output_linear shape "
                     + "\(outputLinear.shape) ≠ [\(numExperts), \(hidden), \(moeIntermediate)]")

        var gateProj: [AnyLinear] = []
        var upProj: [AnyLinear] = []
        var downProj: [AnyLinear] = []
        gateProj.reserveCapacity(numExperts)
        upProj.reserveCapacity(numExperts)
        downProj.reserveCapacity(numExperts)
        for e in 0..<numExperts {
            // [1, 2*moeIntermediate, hidden] → [2*moeIntermediate, hidden].
            let stacked = inputLinear.slicedRows(start: e, count: 1)
                .reshaped(to: [2 * moeIntermediate, hidden])
            gateProj.append(AnyLinear(Linear(
                weight: stacked.slicedRows(start: 0, count: moeIntermediate))))
            upProj.append(AnyLinear(Linear(
                weight: stacked.slicedRows(start: moeIntermediate, count: moeIntermediate))))
            // down_proj folds residual_multiplier so the layer residual
            // add stays a plain Ops.add (the routed combine sums all the
            // pre-scaled expert outputs).
            let downRaw = outputLinear.slicedRows(start: e, count: 1)
                .reshaped(to: [hidden, moeIntermediate])
            downProj.append(AnyLinear(Linear(
                weight: scaleTensorGMH(downRaw, by: residualMultiplier, device: device))))
        }

        // Shared expert — a plain SwiGLU. `input_linear.weight` ships
        // [2*sharedIntermediate, hidden] (gate/up stacked along dim 0).
        let sharedInput = try weights.tensor(
            named: "\(p).shared_mlp.input_linear.weight")
        let sharedOutput = try weights.tensor(
            named: "\(p).shared_mlp.output_linear.weight")
        precondition(sharedInput.shape == [2 * sharedIntermediate, hidden],
                     "GraniteMoeHybrid: shared_mlp.input_linear shape "
                     + "\(sharedInput.shape) ≠ [\(2 * sharedIntermediate), \(hidden)]")
        precondition(sharedOutput.shape == [hidden, sharedIntermediate],
                     "GraniteMoeHybrid: shared_mlp.output_linear shape "
                     + "\(sharedOutput.shape) ≠ [\(hidden), \(sharedIntermediate)]")
        let sharedGate = AnyLinear(Linear(
            weight: sharedInput.slicedRows(start: 0, count: sharedIntermediate)))
        let sharedUp = AnyLinear(Linear(
            weight: sharedInput.slicedRows(start: sharedIntermediate, count: sharedIntermediate)))
        let sharedDown = AnyLinear(Linear(
            weight: scaleTensorGMH(sharedOutput, by: residualMultiplier, device: device)))

        // GraniteMoeHybrid routing is top-K of the raw logits, then a
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

/// The mixer half of a GraniteMoeHybrid layer — Mamba 2 or attention.
enum GraniteMoeHybridMixer {
    case mamba(GraniteMoeHybridMambaMixer)
    case attention(GraniteMoeHybridAttentionMixer)
}

/// The feed-forward half of a GraniteMoeHybrid layer — block-sparse MoE
/// (commits the command buffer) or a dense SwiGLU MLP.
enum GraniteMoeHybridFFN {
    case moe(MoELayer)
    case dense(GraniteMoeHybridDenseMLP)
}

// ─── GraniteMoeHybridMambaMixer ──────────────────────────────────────
//
// The Mamba 2 selective-SSM mixer half. `out_proj` has had
// residual_multiplier folded in at load time; the gated mixer RMSNorm
// is a single full-width RMSNorm over d_inner.

public final class GraniteMoeHybridMambaMixer: Module {
    let inProj, outProj: AnyLinear
    let convW: Tensor        // [kernel, conv_dim]
    let convB: Tensor        // [conv_dim]
    let aEff: Tensor         // [n_heads]   = -exp(A_log)
    let dtBias: Tensor       // [n_heads]
    let dTiled: Tensor       // [d_inner]   D[h] tiled across head_dim
    let mixerNorm: RMSNorm   // gated mixer RMSNorm weight [d_inner]
    let dInner, convDim, nHeads, headDim, stateDim, nGroups, convKernel: Int
    let dtype: DType
    /// Heads sharing one B/C group.
    let headsPerGroup: Int

    init(inProj: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor,
         aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
         mixerNorm: RMSNorm,
         dInner: Int, convDim: Int,
         nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
         convKernel: Int, dtype: DType) {
        self.inProj = inProj; self.outProj = outProj
        self.convW = convW; self.convB = convB
        self.aEff = aEff; self.dtBias = dtBias; self.dTiled = dTiled
        self.mixerNorm = mixerNorm
        self.dInner = dInner; self.convDim = convDim
        self.nHeads = nHeads; self.headDim = headDim; self.stateDim = stateDim
        self.nGroups = nGroups; self.convKernel = convKernel; self.dtype = dtype
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
    func forward(_ xNorm: Tensor, cache: Mamba2LayerCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
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
        let cAll = convAct.slicedRows(start: dInner + nGroups * stateDim,
                                      count: nGroups * stateDim)
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
        let stateHeads = cache.ssm.h        // [nHeads, headDim, stateDim]
        for g in 0..<nGroups {
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
        // RMSNorm over d_inner. Matches GraniteMoeHybridRMSNormGated.
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)
        let yNormed = Ops.rmsNorm(
            yGated, weight: mixerNorm.weight, eps: mixerNorm.eps, on: cmd)

        // out_proj → [hidden] (residual_multiplier already folded in).
        return outProj(yNormed, on: cmd)
    }
}

// ─── GraniteMoeHybridAttentionMixer ──────────────────────────────────
//
// Multi-head attention with NO positional embedding (no RoPE — every
// Granite-4 "-H" checkpoint ships position_embedding_type "nope").
// `scale` is the config's attention_multiplier; `o_proj` has had
// residual_multiplier folded in at load time.

public final class GraniteMoeHybridAttentionMixer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let nHeads, nKVHeads, headDim: Int
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         nHeads: Int, nKVHeads: Int, headDim: Int, scale: Float) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
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
    func forward(_ xNorm: Tensor, cache kv: KVCache,
                 cmd: MTLCommandBuffer, device _: Device) -> Tensor {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // No RoPE — Granite "-H" attention attends without positional
        // rotation. K/V go straight into the cache unrotated.
        kv.appendOnGPU(kFlat: k.reshaped(to: [nKVHeads, headDim]),
                       vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: q.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.maxSeq,
            scale: scale, on: cmd)

        return oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
    }
}

// ─── GraniteMoeHybridDenseMLP ────────────────────────────────────────
//
// The dense feed-forward path (Granite-4 "-H" checkpoints with
// num_local_experts == 0). A plain SwiGLU; down_proj has had
// residual_multiplier folded in at load time.

public final class GraniteMoeHybridDenseMLP: Module {
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

    /// down(silu(gate(x)) * up(x)). Returns the FFN contribution
    /// (residual_multiplier already folded into down_proj).
    func forward(_ xNorm: Tensor, cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(xNorm, on: cmd)
        let u = upProj(xNorm, on: cmd)
        let inner = Ops.mul(Ops.silu(g, on: cmd), u, on: cmd)
        return downProj(inner, on: cmd)
    }
}

// ─── GraniteMoeHybridLayer ───────────────────────────────────────────
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

public final class GraniteMoeHybridLayer: Module, DecoderLayer {
    let inputNorm, postNorm: RMSNorm
    let mixer: GraniteMoeHybridMixer
    let ffn: GraniteMoeHybridFFN
    let hidden: Int

    /// True when this layer's FFN commits the command buffer it is given
    /// (MoE-bearing layers only). The host decode loop refreshes `cmd`
    /// after any layer for which this is true.
    public let commitsCommandBuffer: Bool

    init(inputNorm: RMSNorm, postNorm: RMSNorm,
         mixer: GraniteMoeHybridMixer, ffn: GraniteMoeHybridFFN, hidden: Int) {
        self.inputNorm = inputNorm; self.postNorm = postNorm
        self.mixer = mixer; self.ffn = ffn; self.hidden = hidden
        if case .moe = ffn { self.commitsCommandBuffer = true }
        else { self.commitsCommandBuffer = false }
    }

    /// The Mamba 2 mixer cache slot is a `Mamba2LayerCache`; the
    /// attention cache slot is a `KVCache`. Either way the FFN holds no
    /// per-token state.
    var kind: GraniteMoeHybridLayerKind {
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
    /// afterwards. See `GraniteMoeHybridModel.forward`.
    public func decode(_ h: Tensor, position: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // ── Mixer half — pre-norm + mixer + residual add ──────────────
        let xNorm = inputNorm(h, on: cmd)
        let mixerOut: Tensor
        switch mixer {
        case .mamba(let m):
            guard let mc = cache as? Mamba2LayerCache else {
                fatalError(
                    "GraniteMoeHybridLayer: mamba layer expected "
                    + "Mamba2LayerCache, got \(type(of: cache))")
            }
            mixerOut = m.forward(xNorm, cache: mc, cmd: cmd, device: device)
            mc.advance()
        case .attention(let a):
            guard let kv = cache as? KVCache else {
                fatalError(
                    "GraniteMoeHybridLayer: attention layer expected "
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
            let ffnOut = moe.decode(ffnNorm, position: position,
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

/// Re-key a flat `MoELayer` parameter name into GraniteMoeHybrid's
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

// ─── GraniteMoeHybridModel ───────────────────────────────────────────

public final class GraniteMoeHybridModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a Mamba or attention
    /// hybrid layer, ordered by `layer_types`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let mambaNHeads, mambaHeadDim, stateDim, convDim, convKernel, nGroups, dInner: Int
    /// Final logits are divided by this (Granite's `logits_scaling`).
    public let logitsScaling: Float
    public let dtype: DType

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [GraniteMoeHybridLayerKind]
    /// True when this model has any MoE-bearing layer. Purely
    /// informational — `forward` uses the uniform internal-`workCmd`
    /// discipline regardless of whether any layer commits.
    public let hasMoE: Bool

    init(embedTokens: AnyEmbedding, layers: [any DecoderLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         mambaNHeads: Int, mambaHeadDim: Int, stateDim: Int,
         convDim: Int, convKernel: Int, nGroups: Int, dInner: Int,
         vocab: Int, maxSeq: Int, logitsScaling: Float, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.mambaNHeads = mambaNHeads; self.mambaHeadDim = mambaHeadDim
        self.stateDim = stateDim; self.convDim = convDim
        self.convKernel = convKernel; self.nGroups = nGroups; self.dInner = dInner
        self.vocab = vocab; self.maxSeq = maxSeq
        self.logitsScaling = logitsScaling; self.dtype = dtype
        self.layerKinds = layers.map { layer in
            (layer as? GraniteMoeHybridLayer)?.kind ?? .mamba
        }
        self.hasMoE = layers.contains {
            ($0 as? GraniteMoeHybridLayer)?.commitsCommandBuffer ?? false
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            if let l = layer as? GraniteMoeHybridLayer {
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
        let cap = maxSeq ?? self.maxSeq
        return layerKinds.map { kind in
            switch kind {
            case .mamba:
                return Mamba2LayerCache(
                    nHeads: mambaNHeads, stateDim: stateDim, headDim: mambaHeadDim,
                    convChannels: convDim, convKernelSize: convKernel,
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
            // If the layer committed `workCmd` (MoE FFN), swap in a
            // fresh buffer for the next layer.
            if let g = layer as? GraniteMoeHybridLayer, g.commitsCommandBuffer {
                workCmd = device.makeCommandBuffer()
            }
        }

        // After a committing layer `workCmd` is a fresh, empty buffer and
        // `h` is already resident. After a non-committing layer (the
        // dense path, or an MoE stack ending on a dense layer) `workCmd`
        // still carries that layer's uncommitted work — commit it so `h`
        // is resident before the caller's `cmd` reads it.
        let lastCommitted = (layers.last as? GraniteMoeHybridLayer)?
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

    /// Multi-token forward — Phase 6.6 prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// GraniteMoeHybrid interleaves Mamba 2 + MoE-FFN + attention
    /// layers. The MoE-FFN router commits mid-layer for CPU readback;
    /// a per-attention-layer `decodeMulti` override will need to
    /// preserve that commit pattern across the chunk. Today this
    /// override is commit-count-batched only.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "GraniteMoeHybridModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(tokenId: tok, position: position + i,
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
        fatalError("GraniteMoeHybrid: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloatsGMH(_ values: [Float], shape: [Int],
                            dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: shape, dtype: dtype, device: device)
    switch dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(from: values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("GraniteMoeHybrid: unsupported dtype for host conversion: \(dtype)")
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

/// A_eff = -exp(A_log), per head, in the activation dtype.
private func computeAEffGMH(aLog: Tensor, nHeads: Int,
                            dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsGMH(aLog)
    precondition(floats.count == nHeads, "GraniteMoeHybrid: A_log expected [n_heads]")
    return writeFloatsGMH(floats.map { -Foundation.exp($0) },
                          shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head / per-channel vector to the activation dtype.
private func castVectorGMH(_ src: Tensor, count: Int,
                           dtype: DType, device: Device) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsGMH(src)
    precondition(floats.count == count, "GraniteMoeHybrid: vector size mismatch")
    return writeFloatsGMH(floats, shape: [count], dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels → `[n_heads * head_dim]`.
private func tileDGMH(d: Tensor, nHeads: Int, headDim: Int,
                      dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsGMH(d)
    precondition(floats.count == nHeads, "GraniteMoeHybrid: D expected [n_heads]")
    var tiled: [Float] = []
    tiled.reserveCapacity(nHeads * headDim)
    for h in 0..<nHeads {
        for _ in 0..<headDim { tiled.append(floats[h]) }
    }
    return writeFloatsGMH(tiled, shape: [nHeads * headDim], dtype: dtype, device: device)
}

/// Transpose HF conv1d.weight `[C, 1, K]` → `[K, C]` for the metaltile
/// conv kernel.
private func transposeConv1dWeightGMH(src: Tensor, kernel K: Int, channels C: Int,
                                      dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsGMH(src)
    precondition(floats.count == K * C, "GraniteMoeHybrid: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C {
        for k in 0..<K { dst[k * C + c] = floats[c * K + k] }
    }
    return writeFloatsGMH(dst, shape: [K, C], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVectorGMH(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}
