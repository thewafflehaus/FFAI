// FalconH1 family — the first Phase 5e *hybrid* model and the proving
// ground for the `DecoderLayer` scaffolding.
//
// FalconH1 (TII's Falcon-H1 series — Tiny-90M / 0.5B / 1.5B / 3B / 7B)
// is a **parallel hybrid**: every decoder layer runs BOTH a Mamba 2
// selective-SSM mixer AND a grouped-query attention path on the *same*
// normalized input, sums their outputs into the residual, then applies
// a SwiGLU MLP. There is no layer-schedule interleave — all `nLayers`
// layers are identical in shape (unlike Jamba / NemotronH, which
// alternate Mamba and attention blocks). The hybrid-ness is *within*
// the layer, not across the stack.
//
// Per-layer dataflow (matches mlx-lm's `falcon_h1.py` reference):
//
//   residual = h
//   h        = input_layernorm(h)              [hidden]
//   mambaH   = mamba(h)                         [hidden]   — SSM mixer
//   attnH    = self_attn(h)                     [hidden]   — GQA + RoPE
//   h        = residual + mambaH + attnH
//   residual = h
//   h        = pre_ff_layernorm(h)              [hidden]
//   h        = feed_forward(h)                  [hidden]   — SwiGLU
//   out      = residual + h
//
// **Scalar multipliers.** FalconH1 scatters ~10 scalar multipliers
// across the architecture (`embedding_multiplier`, `lm_head_multiplier`,
// `attention_{in,out}_multiplier`, `key_multiplier`, `mlp_multipliers`,
// `ssm_{in,out}_multiplier`, `ssm_multipliers` — a per-channel µP
// vector). The HF reference folds them into projection weights at
// load time. **mlx-community checkpoints are PRE-SANITIZED** — the
// conversion tool already folded the multipliers into the saved
// weights, so re-applying them on load would double-fold and corrupt
// the activations. The loader detects a pre-sanitized checkpoint via
// a conv1d-weight-shape probe (see `preSanitized` in `loadModel`) and
// skips all folding in that case. For a genuine HF-original checkpoint
// the multipliers ARE folded — `scaleTensor` / `scaleRows` /
// `computeMupVector` do the CPU-side arithmetic so the hot decode path
// stays a plain `residual + mixer` with zero runtime scalar ops.
//
// **Mamba sizing quirk.** Unlike plain Mamba 2 (`d_inner = expand *
// hidden`), FalconH1 takes the SSM inner width *directly* from
// `mamba_d_ssm`. For Tiny-90M that is 768 = `mamba_n_heads(24) *
// mamba_d_head(32)`, which is NOT `mamba_expand(2) * hidden(512)`. The
// loader uses `mamba_d_ssm` and asserts the head decomposition.
//
// **Cache.** `makeLayerCaches` returns one `FalconH1LayerCache` per
// layer — a thin bundle of a `Mamba2LayerCache` (SSM + conv state) and
// a `KVCache` (attention K/V). Both mixers in a layer step in lockstep
// off the same per-layer cache slot.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum FalconH1 {
    public static let modelTypes: Set<String> = ["falcon_h1"]
    public static let architectures: Set<String> = ["FalconH1ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any FalconH1Variant.Type {
        return FalconH1Hybrid.self
    }
}

public protocol FalconH1Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> FalconH1Model
}

public enum FalconH1Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "FalconH1: required config field missing: \(f)"
        case .unsupportedConfig(let m): return "FalconH1: unsupported config: \(m)"
        }
    }
}

// ─── FalconH1Hybrid — the single (and only) variant ──────────────────

public struct FalconH1Hybrid: FalconH1Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// FalconH1 ships chat-tuned `-Instruct` checkpoints. Greedy by
    /// default keeps the integration suite deterministic; users can
    /// override temperature/top-p as usual.
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
    ) throws -> FalconH1Model {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let vocab = config.vocabSize,
              let nHeads = config.numAttentionHeads
        else { throw FalconH1Error.missingConfig("hidden / layers / vocab / num_attention_heads") }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let intermediate = config.intermediateSize ?? (4 * hidden)
        let eps = Float(config.rmsNormEps ?? 1e-5)
        let theta = Float(config.ropeTheta ?? 100_000)
        let maxSeq = config.int("max_position_embeddings") ?? 8192
        let tieEmbed = config.tieWordEmbeddings

        // ── Mamba 2 mixer geometry ────────────────────────────────────
        // FalconH1 takes the SSM inner width directly from mamba_d_ssm
        // (NOT expand*hidden — see file header).
        guard let dSSM = config.int("mamba_d_ssm")
        else { throw FalconH1Error.missingConfig("mamba_d_ssm") }
        guard let mambaNHeads = config.int("mamba_n_heads")
        else { throw FalconH1Error.missingConfig("mamba_n_heads") }
        guard let mambaHeadDim = config.int("mamba_d_head")
        else { throw FalconH1Error.missingConfig("mamba_d_head") }
        guard let stateDim = config.int("mamba_d_state")
        else { throw FalconH1Error.missingConfig("mamba_d_state") }
        let convKernel = config.int("mamba_d_conv") ?? 4
        let nGroups = config.int("mamba_n_groups") ?? 1
        let useConvBias = config.bool("mamba_conv_bias") ?? true
        let mambaRMSNorm = config.bool("mamba_rms_norm") ?? false

        guard nGroups == 1 else {
            throw FalconH1Error.unsupportedConfig(
                "mamba_n_groups > 1 not yet supported (got \(nGroups))")
        }
        guard dSSM == mambaNHeads * mambaHeadDim else {
            throw FalconH1Error.unsupportedConfig(
                "mamba_d_ssm (\(dSSM)) must equal mamba_n_heads*mamba_d_head "
                + "(\(mambaNHeads * mambaHeadDim))")
        }
        // The shipped Mamba 2 SSM kernels gate y with `silu(z)` directly.
        // FalconH1 checkpoints with `mamba_rms_norm=true` apply a gated
        // RMSNorm on y instead — not yet wired here. Tiny-90M / 0.5B /
        // 1.5B all ship `mamba_rms_norm=false`.
        guard !mambaRMSNorm else {
            throw FalconH1Error.unsupportedConfig(
                "mamba_rms_norm=true (gated mixer RMSNorm) not yet supported")
        }
        let convDim = dSSM + 2 * nGroups * stateDim

        // ── Scalar multipliers ────────────────────────────────────────
        // FalconH1 scatters ~10 scalar multipliers across the
        // architecture. The HF reference (`falcon_h1.py`) folds them
        // into projection weights at load time inside `sanitize`.
        //
        // BUT: mlx-community's conversion tool runs `sanitize` ONCE at
        // conversion time and saves the *already-folded* weights. Its
        // `sanitize` then detects an already-converted checkpoint via a
        // conv1d-weight-shape probe and returns the weights untouched
        // on every subsequent load. We replicate that probe exactly:
        // an HF-original checkpoint ships conv1d.weight as `[C, 1, K]`
        // (`dim(-1) = K > dim(1) = 1`); a pre-sanitized one ships
        // `[C, K, 1]` (`dim(-1) = 1 <= dim(1) = K`). When pre-sanitized,
        // applying the config multipliers again would DOUBLE-fold them
        // and corrupt the residual-stream magnitudes — so we skip all
        // folding and treat every multiplier as 1.0.
        let conv1dShape = (try weights.tensor(named: "model.layers.0.mamba.conv1d.weight")).shape
        let preSanitized = conv1dShape.count >= 2
            && conv1dShape[conv1dShape.count - 1] <= conv1dShape[1]

        func mult(_ key: String, default def: Float = 1.0) -> Float {
            preSanitized ? 1.0 : Float(config.float(key) ?? Double(def))
        }
        let embeddingMultiplier = mult("embedding_multiplier")
        let lmHeadMultiplier = mult("lm_head_multiplier")
        let attnInMultiplier = mult("attention_in_multiplier")
        let attnOutMultiplier = mult("attention_out_multiplier")
        let keyMultiplier = mult("key_multiplier")
        let ssmInMultiplier = mult("ssm_in_multiplier")
        let ssmOutMultiplier = mult("ssm_out_multiplier")
        let mlpMultipliers: [Float] = preSanitized
            ? [1.0, 1.0]
            : (config.raw["mlp_multipliers"] as? [Double])?.map { Float($0) } ?? [1.0, 1.0]
        let mlpGateMultiplier = mlpMultipliers.first ?? 1.0
        let mlpDownMultiplier = mlpMultipliers.count > 1 ? mlpMultipliers[1] : 1.0
        let ssmMultipliers: [Float] = preSanitized
            ? [1.0, 1.0, 1.0, 1.0, 1.0]
            : (config.raw["ssm_multipliers"] as? [Double])?.map { Float($0) }
                ?? [1.0, 1.0, 1.0, 1.0, 1.0]

        let quant = config.quantization
        guard quant == nil else {
            // Quantized FalconH1 checkpoints store the per-channel µP
            // scaling already baked into their packed weights/scales;
            // folding the config multipliers on top would double-apply.
            // Raw bf16/f16 checkpoints are the supported path for now.
            throw FalconH1Error.unsupportedConfig(
                "quantized FalconH1 checkpoints not yet supported — load a raw bf16/f16 variant")
        }

        // ── Activation dtype — taken from the embedding table ─────────
        let embedWRaw = try weights.tensor(named: "model.embed_tokens.weight")
        let activationDtype = embedWRaw.dtype
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "FalconH1: unexpected activation dtype \(activationDtype)")

        // Embedding table folds in `embedding_multiplier`.
        let embedW = scaleTensor(embedWRaw, by: embeddingMultiplier, device: device)
        let embedTokens = AnyEmbedding(Embedding(weight: embedW))

        // The µP vector multiplies the in_proj weight per *output* row.
        // It segments the in_proj output [gate | conv(x|B|C) | dt] and
        // applies one ssm_multiplier per segment, all scaled by
        // ssm_in_multiplier. See computeMupVector.
        let mupVector = computeMupVector(
            dSSM: dSSM, groupsStateSize: nGroups * stateDim,
            nHeads: mambaNHeads, ssmMultipliers: ssmMultipliers,
            ssmInMultiplier: ssmInMultiplier)

        // ── Per-layer construction ────────────────────────────────────
        var layers: [FalconH1DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            // ── Attention path ────────────────────────────────────────
            // q_proj / k_proj fold attention_in_multiplier; k_proj also
            // folds key_multiplier; o_proj folds attention_out_multiplier.
            let qW = scaleTensor(try weights.tensor(named: "\(p).self_attn.q_proj.weight"),
                                 by: attnInMultiplier, device: device)
            let kW = scaleTensor(try weights.tensor(named: "\(p).self_attn.k_proj.weight"),
                                 by: attnInMultiplier * keyMultiplier, device: device)
            let vW = try weights.tensor(named: "\(p).self_attn.v_proj.weight")
            let oW = scaleTensor(try weights.tensor(named: "\(p).self_attn.o_proj.weight"),
                                 by: attnOutMultiplier, device: device)
            let qProj = AnyLinear(Linear(weight: qW))
            let kProj = AnyLinear(Linear(weight: kW))
            let vProj = AnyLinear(Linear(weight: vW))
            let oProj = AnyLinear(Linear(weight: oW))

            // ── Mamba 2 mixer path ────────────────────────────────────
            // in_proj folds the per-row µP vector; out_proj folds
            // ssm_out_multiplier.
            let inProjRaw = try weights.tensor(named: "\(p).mamba.in_proj.weight")
            let inProjW = scaleRows(inProjRaw, byRowVector: mupVector,
                                    dtype: activationDtype, device: device)
            let outProjW = scaleTensor(try weights.tensor(named: "\(p).mamba.out_proj.weight"),
                                       by: ssmOutMultiplier, device: device)
            let inProj = AnyLinear(Linear(weight: inProjW))
            let outProj = AnyLinear(Linear(weight: outProjW))

            // conv1d.weight ships [conv_dim, 1, kernel]; the metaltile
            // kernel wants [kernel, conv_dim]. Same transpose Mamba 2 does.
            let convWSrc = try weights.tensor(named: "\(p).mamba.conv1d.weight")
            precondition(convWSrc.elementCount == convDim * convKernel,
                         "FalconH1: conv1d.weight count mismatch: \(convWSrc.shape)")
            let convW = transposeConv1dWeightFH1(src: convWSrc,
                                                 kernel: convKernel, channels: convDim,
                                                 dtype: activationDtype, device: device)
            let convB: Tensor = {
                if useConvBias, weights.has("\(p).mamba.conv1d.bias") {
                    return (try? weights.tensor(named: "\(p).mamba.conv1d.bias"))
                        ?? zeroVector(convDim, dtype: activationDtype, device: device)
                }
                return zeroVector(convDim, dtype: activationDtype, device: device)
            }()

            // A_eff = -exp(A_log); dt_bias per head; D tiled across head_dim.
            let aLog = try weights.tensor(named: "\(p).mamba.A_log")
            let dVec = try weights.tensor(named: "\(p).mamba.D")
            let dtBiasSrc = try weights.tensor(named: "\(p).mamba.dt_bias")
            let aEff = computeAEffFH1(aLog: aLog, nHeads: mambaNHeads,
                                      dtype: activationDtype, device: device)
            let dtBias = castVectorFH1(dtBiasSrc, count: mambaNHeads,
                                       dtype: activationDtype, device: device)
            let dTiled = tileDFH1(d: dVec, nHeads: mambaNHeads, headDim: mambaHeadDim,
                                  dtype: activationDtype, device: device)

            // ── MLP (SwiGLU) — gate/down fold mlp_multipliers ─────────
            let gateW = scaleTensor(try weights.tensor(named: "\(p).feed_forward.gate_proj.weight"),
                                    by: mlpGateMultiplier, device: device)
            let upW = try weights.tensor(named: "\(p).feed_forward.up_proj.weight")
            let downW = scaleTensor(try weights.tensor(named: "\(p).feed_forward.down_proj.weight"),
                                    by: mlpDownMultiplier, device: device)
            let gateProj = AnyLinear(Linear(weight: gateW))
            let upProj = AnyLinear(Linear(weight: upW))
            let downProj = AnyLinear(Linear(weight: downW))

            // ── Norms ─────────────────────────────────────────────────
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"), eps: eps)
            let preFfNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).pre_ff_layernorm.weight"), eps: eps)

            // Mamba 2 mixer block (no mixer RMSNorm — falcon_h1 Tiny/0.5B/
            // 1.5B all use mamba_rms_norm=false → plain silu(z) gating).
            let mamba = FalconH1MambaMixer(
                inProj: inProj, outProj: outProj,
                convW: convW, convB: convB,
                aEff: aEff, dtBias: dtBias, dTiled: dTiled,
                dInner: dSSM, convDim: convDim,
                nHeads: mambaNHeads, headDim: mambaHeadDim, stateDim: stateDim,
                convKernel: convKernel, dtype: activationDtype)

            layers.append(FalconH1DecoderLayer(
                inputNorm: inputNorm, preFfNorm: preFfNorm,
                mamba: mamba,
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, intermediate: intermediate,
                ropeTheta: theta))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.final_layernorm.weight"), eps: eps)

        // lm_head folds lm_head_multiplier. When tied, the head shares
        // the *unscaled* embed table — so we build a separate scaled
        // copy rather than reusing the embedding-multiplier-scaled one.
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            let lmW = scaleTensor(try weights.tensor(named: "lm_head.weight"),
                                  by: lmHeadMultiplier, device: device)
            lmHead = AnyLinear(Linear(weight: lmW))
        } else {
            let lmW = scaleTensor(embedWRaw, by: lmHeadMultiplier, device: device)
            lmHead = AnyLinear(Linear(weight: lmW))
        }

        return FalconH1Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            mambaNHeads: mambaNHeads, mambaHeadDim: mambaHeadDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            dSSM: dSSM, vocab: vocab, maxSeq: maxSeq,
            dtype: activationDtype)
    }
}

// ─── FalconH1MambaMixer ──────────────────────────────────────────────
//
// The SSM half of a FalconH1 layer. Structurally identical to
// `Mamba2Layer`'s mixer body but (a) takes its normalized input from
// the shared layer-level `input_layernorm` rather than owning a
// pre-norm, and (b) has no mixer RMSNorm (FalconH1 gates with silu(z)
// directly). Returns the post-out_proj mixer contribution; the residual
// add is done by the enclosing `FalconH1DecoderLayer`.

public final class FalconH1MambaMixer: Module {
    let inProj, outProj: AnyLinear
    let convW: Tensor        // [kernel, conv_dim]
    let convB: Tensor        // [conv_dim]
    let aEff: Tensor         // [n_heads]   = -exp(A_log)
    let dtBias: Tensor       // [n_heads]
    let dTiled: Tensor       // [d_inner]   D[h] tiled across head_dim
    let dInner, convDim, nHeads, headDim, stateDim, convKernel: Int
    let dtype: DType

    init(inProj: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor,
         aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
         dInner: Int, convDim: Int,
         nHeads: Int, headDim: Int, stateDim: Int,
         convKernel: Int, dtype: DType) {
        self.inProj = inProj; self.outProj = outProj
        self.convW = convW; self.convB = convB
        self.aEff = aEff; self.dtBias = dtBias; self.dTiled = dTiled
        self.dInner = dInner; self.convDim = convDim
        self.nHeads = nHeads; self.headDim = headDim; self.stateDim = stateDim
        self.convKernel = convKernel; self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inProj.parameters() { out.append(("mamba.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mamba.out_proj.\(k)", v)) }
        return out
    }

    /// Single-token mixer forward. `xNorm` is the already-normalized
    /// layer input. All work queued on `cmd`; cache mutations are
    /// kernel-driven (no host sync). Returns the mixer contribution
    /// (pre-residual-add), shape [hidden].
    func forward(_ xNorm: Tensor, cache: Mamba2LayerCache,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // in_proj → split into z (gate) / xBC / dt_raw
        let proj = inProj(xNorm, on: cmd)
        let z = proj.slicedRows(start: 0, count: dInner)
        let xBC = proj.slicedRows(start: dInner, count: convDim)
        let dtRaw = proj.slicedRows(start: dInner + convDim, count: nHeads)

        // conv1d causal step (rolling state) + SiLU
        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: xBC, w: convW, b: convB,
            state: cache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)

        // split conv output → x / B / C
        let x = convAct.slicedRows(start: 0, count: dInner)
        let bVec = convAct.slicedRows(start: dInner, count: stateDim)
        let cVec = convAct.slicedRows(start: dInner + stateDim, count: stateDim)

        // dt = softplus(dt_raw + dt_bias)
        let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
        let dt = Ops.softplus(dtSum, on: cmd)

        // selective scan step
        let y = Tensor.empty(shape: [dInner], dtype: dtype, device: device)
        Ops.ssmStep(
            x: x, a: aEff, b: bVec, c: cVec, dt: dt,
            state: cache.ssm.h, into: y,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim, on: cmd)

        // skip: y += D_tiled * x
        let dx = Ops.mul(dTiled, x, on: cmd)
        let ySkip = Ops.add(y, dx, on: cmd)

        // gating: y *= silu(z)
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)

        // out_proj → [hidden]
        return outProj(yGated, on: cmd)
    }
}

// ─── FalconH1DecoderLayer ────────────────────────────────────────────
//
// One parallel-hybrid layer: Mamba 2 mixer + GQA attention on the same
// normalized input, summed into the residual, then a SwiGLU MLP.
// Conforms to `DecoderLayer` so a future heterogeneous hybrid (Jamba,
// NemotronH) can mix this with other layer kinds in one stack — FalconH1
// itself is homogeneous but exercises the protocol end-to-end.

public final class FalconH1DecoderLayer: Module, DecoderLayer {
    let inputNorm, preFfNorm: RMSNorm
    let mamba: FalconH1MambaMixer
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(inputNorm: RMSNorm, preFfNorm: RMSNorm,
         mamba: FalconH1MambaMixer,
         qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float) {
        self.inputNorm = inputNorm; self.preFfNorm = preFfNorm
        self.mamba = mamba
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in preFfNorm.parameters() { out.append(("pre_ff_layernorm.\(k)", v)) }
        out.append(contentsOf: mamba.parameters())
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("feed_forward.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("feed_forward.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("feed_forward.down_proj.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance — layer-local single-token decode.
    /// The cache slot is a `FalconH1LayerCache` (Mamba state + KV).
    public func decode(_ h: Tensor, position: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let layerCache = cache as? FalconH1LayerCache else {
            fatalError("FalconH1DecoderLayer: expected FalconH1LayerCache, got \(type(of: cache))")
        }

        // Shared pre-mixer norm feeds BOTH the SSM and attention paths.
        let xNorm = inputNorm(h, on: cmd)

        // Mamba 2 mixer contribution.
        let mambaH = mamba.forward(xNorm, cache: layerCache.mamba,
                                   cmd: cmd, device: device)

        // GQA attention contribution.
        let attnH = attention(xNorm, position: position,
                               cache: layerCache.kv, cmd: cmd, device: device)

        // Parallel-hybrid join: residual + mixer + attention.
        let postMix = Ops.add(Ops.add(h, mambaH, on: cmd), attnH, on: cmd)

        // SwiGLU MLP with its own pre-norm + residual.
        let mlpNorm = preFfNorm(postMix, on: cmd)
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postMix, mlpOut, on: cmd)
    }

    /// GQA attention path: project → RoPE → KV-cache append → SDPA →
    /// o_proj. No per-head q/k norm (FalconH1 has none). Returns the
    /// pre-residual-add attention contribution, shape [hidden].
    private func attention(_ xNorm: Tensor, position: Int,
                           cache: KVCache,
                           cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRotated = Ops.rope(q.reshaped(to: [nHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(k.reshaped(to: [nKVHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: .none, on: cmd)

        cache.appendOnGPU(kFlat: kRotated,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)

        return oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
    }
}

// ─── FalconH1LayerCache ──────────────────────────────────────────────
//
// One per-layer cache for a parallel-hybrid FalconH1 layer: a Mamba 2
// SSM+conv state bundle AND an attention KV cache, both live and
// stepped together every decode token. `length` / `maxSeq` follow the
// attention cache (the SSM half is constant-size and reports `.max`).

public final class FalconH1LayerCache: LayerCacheProtocol, @unchecked Sendable {
    public let mamba: Mamba2LayerCache
    public let kv: KVCache

    public init(mamba: Mamba2LayerCache, kv: KVCache) {
        self.mamba = mamba
        self.kv = kv
    }

    /// Attention KV cache drives the user-visible length (it grows with
    /// the sequence; the SSM state is constant-size).
    public var length: Int { kv.length }
    public var maxSeq: Int { kv.maxSeq }
    public var bytesAllocated: Int { mamba.bytesAllocated + kv.bytesAllocated }
    public var bytesInUse: Int { mamba.bytesInUse + kv.bytesInUse }

    public func reset() {
        mamba.reset()
        kv.reset()
    }
}

// ─── FalconH1Model ───────────────────────────────────────────────────

public final class FalconH1Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous-capable layer stack. FalconH1 is homogeneous, but
    /// the array is `[any DecoderLayer]` so the decode loop exercises
    /// the protocol exactly as a Jamba / NemotronH stack would.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let mambaNHeads, mambaHeadDim, stateDim, convDim, convKernel, dSSM: Int
    public let dtype: DType

    init(embedTokens: AnyEmbedding, layers: [any DecoderLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         mambaNHeads: Int, mambaHeadDim: Int, stateDim: Int,
         convDim: Int, convKernel: Int, dSSM: Int,
         vocab: Int, maxSeq: Int, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.mambaNHeads = mambaNHeads; self.mambaHeadDim = mambaHeadDim
        self.stateDim = stateDim; self.convDim = convDim
        self.convKernel = convKernel; self.dSSM = dSSM
        self.vocab = vocab; self.maxSeq = maxSeq; self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("model.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            // Concrete-type cast: every FalconH1 layer is a
            // FalconH1DecoderLayer (homogeneous stack).
            if let l = layer as? FalconH1DecoderLayer {
                for (k, v) in l.parameters() {
                    out.append(("model.layers.\(i).\(k)", v))
                }
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.final_layernorm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One `FalconH1LayerCache` per layer — Mamba state + KV cache. The
    /// SSM half is constant-size; the KV half is sized to `maxSeq`.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return (0..<nLayers).map { _ in
            let mamba = Mamba2LayerCache(
                nHeads: mambaNHeads, stateDim: stateDim, headDim: mambaHeadDim,
                convChannels: convDim, convKernelSize: convKernel,
                dtype: dtype, device: device)
            let kv = KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                dtype: dtype, device: device)
            return FalconH1LayerCache(mamba: mamba, kv: kv)
        }
    }

    /// Queue a single-token forward pass onto `cmd`. No commit. The
    /// `LanguageModel` default extension composes this with the output
    /// kernel for the 1-commit-per-token decode path.
    ///
    /// Walks `[any DecoderLayer]` in lockstep with the per-layer caches —
    /// the exact heterogeneous-hybrid decode loop the `DecoderLayer`
    /// protocol exists to support.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.decode(h, position: position, cache: caches[i],
                             cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }
}

// ─── Load-time host helpers ──────────────────────────────────────────
//
// FalconH1 folds every config scalar multiplier into a projection
// weight at load time so the decode hot path stays scalar-op-free.
// These helpers do the CPU-side arithmetic; cost is in the load-time
// noise (small per-head vectors + one pass over each weight).

/// Read an f32 / bf16 / f16 tensor into `[Float]` for CPU-side
/// derivation work. Only used at load time, never on the hot path.
private func readFloatsFH1(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("FalconH1: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloatsFH1(_ values: [Float], shape: [Int],
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
        fatalError("FalconH1: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// Multiply every element of `t` by a scalar, returning a fresh tensor
/// in `t`'s dtype. Identity-fast-path: returns `t` unchanged when the
/// multiplier is exactly 1.0 (the common case for Tiny-90M's all-1.0
/// attention/mlp multipliers — avoids a needless copy per layer).
private func scaleTensor(_ t: Tensor, by m: Float, device: Device) -> Tensor {
    if m == 1.0 { return t }
    let floats = readFloatsFH1(t).map { $0 * m }
    return writeFloatsFH1(floats, shape: t.shape, dtype: t.dtype, device: device)
}

/// Multiply each *row* of a 2D weight `[outFeatures, inFeatures]` by the
/// per-row scalar in `rowVector` (`[outFeatures]`). FalconH1's in_proj
/// folds the µP vector this way — one multiplier per output channel.
private func scaleRows(_ t: Tensor, byRowVector rowVector: [Float],
                       dtype: DType, device: Device) -> Tensor {
    precondition(t.shape.count == 2, "scaleRows: weight must be 2D")
    let outF = t.shape[0], inF = t.shape[1]
    precondition(rowVector.count == outF,
                 "scaleRows: rowVector count \(rowVector.count) ≠ outFeatures \(outF)")
    var floats = readFloatsFH1(t)
    for r in 0..<outF {
        let m = rowVector[r]
        if m == 1.0 { continue }
        let base = r * inF
        for c in 0..<inF { floats[base + c] *= m }
    }
    return writeFloatsFH1(floats, shape: t.shape, dtype: dtype, device: device)
}

/// Build the µP (maximal-update-parametrization) vector that scales the
/// Mamba in_proj weight per output row. The in_proj output is the
/// concatenation `[gate | x | B | C | dt]` with segment sizes
/// `[dSSM, dSSM, groupsStateSize, groupsStateSize, nHeads]`. Each segment
/// gets one `ssm_multipliers` entry, all multiplied by `ssm_in_multiplier`.
private func computeMupVector(dSSM: Int, groupsStateSize: Int, nHeads: Int,
                              ssmMultipliers: [Float],
                              ssmInMultiplier: Float) -> [Float] {
    let sizes = [dSSM, dSSM, groupsStateSize, groupsStateSize, nHeads]
    precondition(ssmMultipliers.count == sizes.count,
                 "computeMupVector: ssm_multipliers must have \(sizes.count) entries")
    var vec: [Float] = []
    vec.reserveCapacity(sizes.reduce(0, +))
    for (size, mult) in zip(sizes, ssmMultipliers) {
        vec.append(contentsOf: Array(repeating: mult * ssmInMultiplier, count: size))
    }
    return vec
}

/// A_eff = -exp(A_log), per head, in the activation dtype.
private func computeAEffFH1(aLog: Tensor, nHeads: Int,
                            dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsFH1(aLog)
    precondition(floats.count == nHeads, "FalconH1: A_log expected [n_heads]")
    return writeFloatsFH1(floats.map { -Foundation.exp($0) },
                          shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head vector to the activation dtype, preserving values.
private func castVectorFH1(_ src: Tensor, count: Int,
                           dtype: DType, device: Device) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsFH1(src)
    precondition(floats.count == count, "FalconH1: vector size mismatch")
    return writeFloatsFH1(floats, shape: [count], dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels → `[n_heads * head_dim]` so the
/// SSM skip connection (`y += D * x`) reuses `Ops.mul`.
private func tileDFH1(d: Tensor, nHeads: Int, headDim: Int,
                      dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsFH1(d)
    precondition(floats.count == nHeads, "FalconH1: D expected [n_heads]")
    var tiled: [Float] = []
    tiled.reserveCapacity(nHeads * headDim)
    for h in 0..<nHeads {
        for _ in 0..<headDim { tiled.append(floats[h]) }
    }
    return writeFloatsFH1(tiled, shape: [nHeads * headDim], dtype: dtype, device: device)
}

/// Transpose HF conv1d.weight `[C, 1, K]` (channel-major) → `[K, C]`
/// (kernel-tap-major) for the metaltile conv kernel.
private func transposeConv1dWeightFH1(src: Tensor, kernel K: Int, channels C: Int,
                                      dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsFH1(src)
    precondition(floats.count == K * C, "FalconH1: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C {
        for k in 0..<K { dst[k * C + c] = floats[c * K + k] }
    }
    return writeFloatsFH1(dst, shape: [K, C], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype (conv bias fallback).
private func zeroVector(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}
