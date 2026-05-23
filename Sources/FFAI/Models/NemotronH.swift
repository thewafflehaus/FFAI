// NemotronH family — a Phase 5e *stack-interleaved hybrid* model.
//
// Unlike FalconH1 (a *parallel* hybrid where every layer runs Mamba
// AND attention together), NemotronH is a **stack-interleaved** hybrid:
// a `hybrid_override_pattern` string assigns each decoder layer exactly
// ONE mixer kind, and the kinds genuinely vary down the stack:
//
//   "M"  → Mamba 2 selective-SSM mixer       (NemotronHMambaLayer)
//   "*"  → multi-head attention              (NemotronHAttentionLayer)
//   "-"  → dense squared-ReLU MLP            (NemotronHMLPLayer)
//   "E"  → mixture-of-experts feed-forward   (rejected — see below)
//
// e.g. NemotronH-4B-Base ships
//   "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
// — 52 layers, 4 of them attention, the rest split between Mamba and
// dense MLP. The layer array is therefore *heterogeneous*: it is held
// as `[any DecoderLayer]` and the decode loop walks it in lockstep
// with a per-index `[any LayerCacheProtocol]`.
//
// Per-layer dataflow — every layer kind shares the same pre-norm +
// residual skeleton (matches mlx-lm's `nemotron_h.py`):
//
//   residual = h
//   h        = norm(h)               [hidden]  — RMSNorm, one per layer
//   h        = mixer(h)              [hidden]  — M / * / - / E
//   out      = residual + h
//
// There is NO separate pre-FF norm and NO second residual: a NemotronH
// layer is "one norm, one mixer, one residual add", and the MLP / MoE
// layers are themselves a layer kind in the stack rather than a
// trailing block inside every layer.
//
// ─── Attention has no RoPE ───────────────────────────────────────────
//
// NemotronH attention uses *no* positional embedding — no RoPE, no ALiBi.
// The Mamba layers carry the sequence-order information; the sparse
// attention layers attend without rotation. `NemotronHAttentionLayer`
// therefore skips the `Ops.rope` call that every other attention family
// in FFAI makes.
//
// ─── Mamba 2 with n_groups > 1 + gated mixer RMSNorm ─────────────────
//
// Every published NemotronH checkpoint ships `n_groups = 8` and a gated
// mixer RMSNorm (`mixer.norm`). The shared `Mamba2Layer` does not
// support either, so NemotronH carries its own `NemotronHMambaLayer`:
//
//   * **Grouped B/C** — the SSM `B`/`C` tensors are shared across
//     `heads_per_group = n_heads / n_groups` heads. We reuse the
//     shipped scalar `Ops.ssmStep` kernel by dispatching it once per
//     group over a contiguous head sub-slab — no kernel change needed.
//   * **Gated RMSNorm** — `y` is gated by `silu(z)` BEFORE a per-group
//     (unweighted) RMSNorm, then scaled by the learned `mixer.norm`
//     weight. Implemented with `Ops.rmsNormRows` (one row per group) +
//     `Ops.mul`.
//
// ─── MoE layers are rejected ─────────────────────────────────────────
//
// The "E" layer kind (Nemotron-Cascade-2 / Nemotron-3 MoE checkpoints)
// uses squared-ReLU experts with sigmoid + group-expert-select routing
// — both diverge from the SwiGLU-only `MoELayer` and its softmax /
// top-K-then-softmax routers. No small published checkpoint exercises
// "E", so rather than ship an untested MoE path the loader rejects any
// pattern containing "E" with a clear `unsupportedConfig` error.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────

public enum NemotronH {
    public static let modelTypes: Set<String> = ["nemotron_h"]
    public static let architectures: Set<String> = ["NemotronHForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any NemotronHVariant.Type {
        return NemotronHHybrid.self
    }
}

public protocol NemotronHVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> NemotronHModel
}

public enum NemotronHError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "NemotronH: required config field missing: \(f)"
        case .unsupportedConfig(let m): return "NemotronH: unsupported config: \(m)"
        }
    }
}

// ─── Layer kind ──────────────────────────────────────────────────────

/// The four mixer kinds a `hybrid_override_pattern` character can name.
enum NemotronHLayerKind: Equatable {
    case mamba       // "M"
    case attention   // "*"
    case mlp         // "-"
    case moe         // "E"

    init(from char: Character) throws {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default:
            throw NemotronHError.unsupportedConfig(
                "unknown hybrid_override_pattern character '\(char)'")
        }
    }
}

// ─── NemotronHHybrid — the single variant ────────────────────────────

public struct NemotronHHybrid: NemotronHVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// NemotronH ships both `-Base` and `-Instruct` checkpoints. Greedy
    /// by default keeps the integration suite deterministic.
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
    ) throws -> NemotronHModel {
        guard let hidden = config.hiddenSize,
              let vocab = config.vocabSize,
              let nHeads = config.numAttentionHeads
        else { throw NemotronHError.missingConfig("hidden / vocab / num_attention_heads") }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // NemotronH attention can carry an explicit head_dim
        // (attention_head_dim on the 4B config, head_dim elsewhere) that
        // is NOT hidden / n_heads.
        let headDim = config.int("attention_head_dim")
            ?? config.int("head_dim")
            ?? (hidden / nHeads)
        let intermediate = config.intermediateSize ?? (4 * hidden)
        let eps = Float(config.float("layer_norm_epsilon")
            ?? config.float("rms_norm_eps") ?? 1e-5)
        let tieEmbed = config.tieWordEmbeddings

        // ── Hybrid layer schedule ─────────────────────────────────────
        guard let pattern = config.string("hybrid_override_pattern"), !pattern.isEmpty
        else { throw NemotronHError.missingConfig("hybrid_override_pattern") }
        let kinds = try Array(pattern).map { try NemotronHLayerKind(from: $0) }
        let nLayers = kinds.count
        guard !kinds.contains(.moe) else {
            throw NemotronHError.unsupportedConfig(
                "MoE ('E') layers not yet supported — NemotronH MoE uses "
                + "squared-ReLU experts + sigmoid group-expert routing, which "
                + "diverge from the shipped SwiGLU MoELayer. Load a "
                + "Mamba/attention/MLP-only checkpoint (e.g. Nemotron-H-4B-Base).")
        }

        // ── Mamba 2 mixer geometry ────────────────────────────────────
        guard let mambaNHeads = config.int("mamba_num_heads")
        else { throw NemotronHError.missingConfig("mamba_num_heads") }
        guard let mambaHeadDim = config.int("mamba_head_dim")
        else { throw NemotronHError.missingConfig("mamba_head_dim") }
        guard let stateDim = config.int("ssm_state_size")
        else { throw NemotronHError.missingConfig("ssm_state_size") }
        let convKernel = config.int("conv_kernel") ?? 4
        let nGroups = config.int("n_groups") ?? 1
        let useConvBias = config.bool("use_conv_bias") ?? true

        // d_inner is taken directly from the Mamba head decomposition
        // (NemotronH does not use the `expand * hidden` relation).
        let dInner = mambaNHeads * mambaHeadDim
        guard mambaNHeads % nGroups == 0 else {
            throw NemotronHError.unsupportedConfig(
                "mamba_num_heads (\(mambaNHeads)) must be a multiple of "
                + "n_groups (\(nGroups))")
        }
        // The gated mixer RMSNorm runs per group via Ops.rmsNormRows,
        // whose kernel requires the row size (group size) to be a
        // multiple of 128 and ≤ 4096. dInner / nGroups is that row size.
        let groupSize = dInner / nGroups
        guard groupSize % 128 == 0, groupSize <= 4096 else {
            throw NemotronHError.unsupportedConfig(
                "per-group mixer RMSNorm row size d_inner/n_groups = "
                + "\(groupSize) must be a multiple of 128 and ≤ 4096 "
                + "(rmsNormRows kernel invariant)")
        }
        let convDim = dInner + 2 * nGroups * stateDim

        // time_step_limit clamps softplus(dt). NemotronH-4B ships
        // [0.0, Infinity] (no clamp); honour it if a checkpoint sets one.
        let tsLimit = config.raw["time_step_limit"] as? [Double]
        let tsMin = Float(tsLimit?.first ?? 0.0)
        let tsMax: Float = {
            guard let hi = tsLimit?.dropFirst().first else { return .infinity }
            return hi.isFinite ? Float(hi) : .infinity
        }()

        // ── Activation dtype — taken from the embedding table ─────────
        let embedWRaw = try weights.tensor(named: "backbone.embeddings.weight")
        let activationDtype = embedWRaw.dtype
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "NemotronH: unexpected activation dtype \(activationDtype)")
        guard config.quantization == nil else {
            throw NemotronHError.unsupportedConfig(
                "quantized NemotronH checkpoints not yet supported — load a raw bf16/f16 variant")
        }
        let embedTokens = AnyEmbedding(Embedding(weight: embedWRaw))

        // ── Per-layer construction ────────────────────────────────────
        var layers: [any DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in kinds.enumerated() {
            let p = "backbone.layers.\(i)"
            // Every layer kind shares one pre-mixer RMSNorm.
            let norm = RMSNorm(
                weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

            switch kind {
            case .mamba:
                layers.append(try buildMambaLayer(
                    prefix: "\(p).mixer", norm: norm, weights: weights,
                    hidden: hidden, dInner: dInner, convDim: convDim,
                    nHeads: mambaNHeads, headDim: mambaHeadDim, stateDim: stateDim,
                    nGroups: nGroups, convKernel: convKernel,
                    useConvBias: useConvBias, eps: eps,
                    tsMin: tsMin, tsMax: tsMax,
                    dtype: activationDtype, device: device))

            case .attention:
                let qProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.q_proj.weight")))
                let kProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.k_proj.weight")))
                let vProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.v_proj.weight")))
                let oProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.o_proj.weight")))
                layers.append(NemotronHAttentionLayer(
                    norm: norm,
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim))

            case .mlp:
                let upProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.up_proj.weight")))
                let downProj = AnyLinear(Linear(
                    weight: try weights.tensor(named: "\(p).mixer.down_proj.weight")))
                layers.append(NemotronHMLPLayer(
                    norm: norm, upProj: upProj, downProj: downProj,
                    hidden: hidden, intermediate: intermediate))

            case .moe:
                // Already rejected above; unreachable.
                throw NemotronHError.unsupportedConfig("MoE layer reached construction")
            }
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "backbone.norm_f.weight"), eps: eps)

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = AnyLinear(Linear(weight: try weights.tensor(named: "lm_head.weight")))
        } else {
            lmHead = AnyLinear(Linear(weight: embedWRaw))
        }

        let maxSeq = config.int("max_position_embeddings") ?? 8192
        return NemotronHModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            mambaNHeads: mambaNHeads, mambaHeadDim: mambaHeadDim,
            stateDim: stateDim, convDim: convDim, convKernel: convKernel,
            nGroups: nGroups, dInner: dInner,
            vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)
    }

    /// Build one Mamba 2 mixer layer (`"M"`). Reads + derives the
    /// per-head SSM parameters and transposes the conv1d weight, the
    /// same load-time arithmetic `Mamba2`/`FalconH1` do.
    private static func buildMambaLayer(
        prefix p: String, norm: RMSNorm, weights: SafeTensorsBundle,
        hidden: Int, dInner: Int, convDim: Int,
        nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
        convKernel: Int, useConvBias: Bool, eps: Float,
        tsMin: Float, tsMax: Float,
        dtype: DType, device: Device
    ) throws -> NemotronHMambaLayer {
        let inProj = AnyLinear(Linear(
            weight: try weights.tensor(named: "\(p).in_proj.weight")))
        let outProj = AnyLinear(Linear(
            weight: try weights.tensor(named: "\(p).out_proj.weight")))

        // conv1d.weight ships [conv_dim, 1, kernel]; the metaltile kernel
        // wants [kernel, conv_dim].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(convWSrc.elementCount == convDim * convKernel,
                     "NemotronH: conv1d.weight count mismatch: \(convWSrc.shape)")
        let convW = transposeConv1dWeightNH(
            src: convWSrc, kernel: convKernel, channels: convDim,
            dtype: dtype, device: device)
        let convB: Tensor = {
            if useConvBias, weights.has("\(p).conv1d.bias") {
                return castVectorNH((try? weights.tensor(named: "\(p).conv1d.bias"))
                    ?? zeroVectorNH(convDim, dtype: dtype, device: device),
                    count: convDim, dtype: dtype, device: device)
            }
            return zeroVectorNH(convDim, dtype: dtype, device: device)
        }()

        // A_eff = -exp(A_log); dt_bias per head; D tiled across head_dim.
        let aEff = computeAEffNH(
            aLog: try weights.tensor(named: "\(p).A_log"),
            nHeads: nHeads, dtype: dtype, device: device)
        let dtBias = castVectorNH(
            try weights.tensor(named: "\(p).dt_bias"),
            count: nHeads, dtype: dtype, device: device)
        let dTiled = tileDNH(
            d: try weights.tensor(named: "\(p).D"),
            nHeads: nHeads, headDim: headDim, dtype: dtype, device: device)

        // Gated mixer RMSNorm weight — full [d_inner].
        let mixerNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).norm.weight"), eps: eps)

        return NemotronHMambaLayer(
            norm: norm, inProj: inProj, outProj: outProj,
            convW: convW, convB: convB,
            aEff: aEff, dtBias: dtBias, dTiled: dTiled,
            mixerNorm: mixerNorm,
            hidden: hidden, dInner: dInner, convDim: convDim,
            nHeads: nHeads, headDim: headDim, stateDim: stateDim,
            nGroups: nGroups, convKernel: convKernel,
            tsMin: tsMin, tsMax: tsMax, dtype: dtype)
    }
}

// ─── NemotronHMambaLayer — "M" ───────────────────────────────────────
//
// A Mamba 2 selective-SSM mixer with `n_groups > 1` grouped B/C and a
// gated mixer RMSNorm. Conforms to `DecoderLayer`; its cache slot is a
// `Mamba2LayerCache` (SSM state + conv state).

public final class NemotronHMambaLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let inProj, outProj: AnyLinear
    let convW: Tensor        // [kernel, conv_dim]
    let convB: Tensor        // [conv_dim]
    let aEff: Tensor         // [n_heads]   = -exp(A_log)
    let dtBias: Tensor       // [n_heads]
    let dTiled: Tensor       // [d_inner]   D[h] tiled across head_dim
    let mixerNorm: RMSNorm   // gated mixer RMSNorm weight [d_inner]
    let hidden, dInner, convDim, nHeads, headDim, stateDim, nGroups, convKernel: Int
    let tsMin, tsMax: Float
    let dtype: DType
    /// Heads sharing one B/C group.
    let headsPerGroup: Int
    /// Per-group gated-RMSNorm row size = d_inner / n_groups.
    let groupSize: Int
    /// `[groupSize]` ones vector — the "no learned scale" weight for the
    /// per-group unweighted RMSNorm pass (the learned `mixer.norm`
    /// weight is applied afterwards via `Ops.mul`). Built once at load.
    let onesWeight: Tensor

    init(norm: RMSNorm, inProj: AnyLinear, outProj: AnyLinear,
         convW: Tensor, convB: Tensor,
         aEff: Tensor, dtBias: Tensor, dTiled: Tensor,
         mixerNorm: RMSNorm,
         hidden: Int, dInner: Int, convDim: Int,
         nHeads: Int, headDim: Int, stateDim: Int, nGroups: Int,
         convKernel: Int, tsMin: Float, tsMax: Float, dtype: DType) {
        self.norm = norm
        self.inProj = inProj; self.outProj = outProj
        self.convW = convW; self.convB = convB
        self.aEff = aEff; self.dtBias = dtBias; self.dTiled = dTiled
        self.mixerNorm = mixerNorm
        self.hidden = hidden; self.dInner = dInner; self.convDim = convDim
        self.nHeads = nHeads; self.headDim = headDim; self.stateDim = stateDim
        self.nGroups = nGroups; self.convKernel = convKernel
        self.tsMin = tsMin; self.tsMax = tsMax; self.dtype = dtype
        self.headsPerGroup = nHeads / nGroups
        self.groupSize = dInner / nGroups
        self.onesWeight = Tensor.filled(1.0, shape: [dInner / nGroups], dtype: dtype)
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in inProj.parameters() { out.append(("mixer.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mixer.out_proj.\(k)", v)) }
        for (k, v) in mixerNorm.parameters() { out.append(("mixer.norm.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `Mamba2LayerCache`.
    public func decode(_ h: Tensor, position _: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device: Device) -> Tensor {
        guard let layerCache = cache as? Mamba2LayerCache else {
            fatalError("NemotronHMambaLayer: expected Mamba2LayerCache, got \(type(of: cache))")
        }

        // (1) pre-mixer RMSNorm.
        let xNorm = norm(h, on: cmd)

        // (2) in_proj → split into z (gate) / xBC / dt_raw.
        //     in_proj output layout: [d_inner | conv_dim | n_heads].
        let proj = inProj(xNorm, on: cmd)
        let z = proj.slicedRows(start: 0, count: dInner)
        let xBC = proj.slicedRows(start: dInner, count: convDim)
        let dtRaw = proj.slicedRows(start: dInner + convDim, count: nHeads)

        // (3) conv1d causal step (rolling state) + SiLU.
        let convOut = Tensor.empty(shape: [convDim], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: xBC, w: convW, b: convB,
            state: layerCache.conv.state, into: convOut,
            nChannels: convDim, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)

        // (4) split conv output → x / B / C.
        //     conv layout: [d_inner | n_groups*state_dim | n_groups*state_dim].
        let x = convAct.slicedRows(start: 0, count: dInner)
        let bAll = convAct.slicedRows(start: dInner, count: nGroups * stateDim)
            .reshaped(to: [nGroups, stateDim])
        let cAll = convAct.slicedRows(start: dInner + nGroups * stateDim,
                                      count: nGroups * stateDim)
            .reshaped(to: [nGroups, stateDim])

        // (5) dt = softplus(dt_raw + dt_bias).
        let dtSum = Ops.add(dtRaw, dtBias, on: cmd)
        let dt = Ops.softplus(dtSum, on: cmd)

        // (6) selective scan — dispatched per group so the shipped
        //     single-group ssm_step kernel handles grouped B/C. Each
        //     group owns a contiguous head sub-slab in x / state / y.
        let y = Tensor.empty(shape: [nHeads, headDim], dtype: dtype, device: device)
        let xHeads = x.reshaped(to: [nHeads, headDim])
        let stateHeads = layerCache.ssm.h        // [nHeads, headDim, stateDim]
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

        // (7) skip: y += D_tiled * x.
        let dx = Ops.mul(dTiled, x, on: cmd)
        let ySkip = Ops.add(yFlat, dx, on: cmd)

        // (8) gated mixer RMSNorm: y *= silu(z), then per-group
        //     (unweighted) RMSNorm, then scale by the learned weight.
        //     Matches mlx-lm's NemotronHRMSNormGated exactly.
        let zAct = Ops.silu(z, on: cmd)
        let yGated = Ops.mul(ySkip, zAct, on: cmd)
        let yNormed = Ops.rmsNormRows(
            yGated, weight: onesWeight,
            eps: mixerNorm.eps, nRows: nGroups, rowSize: groupSize, on: cmd)
        let yScaled = Ops.mul(yNormed, mixerNorm.weight, on: cmd)

        // (9) out_proj + residual.
        let yOut = outProj(yScaled, on: cmd)
        let result = Ops.add(h, yOut, on: cmd)

        layerCache.advance()
        return result
    }
}

// ─── NemotronHAttentionLayer — "*" ───────────────────────────────────
//
// Multi-head attention with NO positional embedding (no RoPE). Cache
// slot is a raw `KVCache`.

public final class NemotronHAttentionLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let qProj, kProj, vProj, oProj: AnyLinear
    let nHeads, nKVHeads, headDim: Int
    let scale: Float

    init(norm: RMSNorm,
         qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         nHeads: Int, nKVHeads: Int, headDim: Int) {
        self.norm = norm
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in qProj.parameters() { out.append(("mixer.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("mixer.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("mixer.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("mixer.o_proj.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `KVCache`.
    public func decode(_ h: Tensor, position _: Int,
                       cache: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device _: Device) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("NemotronHAttentionLayer: expected KVCache, got \(type(of: cache))")
        }
        let xNorm = norm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // No RoPE — NemotronH attention attends without positional
        // rotation. K/V go straight into the cache unrotated.
        kv.appendOnGPU(kFlat: k.reshaped(to: [nKVHeads, headDim]),
                       vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let (cacheK, cacheV) = kv.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: q.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.maxSeq,
            scale: scale, on: cmd)

        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        return Ops.add(h, oOut, on: cmd)
    }
}

// ─── NemotronHMLPLayer — "-" ─────────────────────────────────────────
//
// Dense feed-forward block with a squared-ReLU activation and a single
// up-projection (no gate — unlike SwiGLU). Cache slot is a
// `StatelessLayerCache` (no per-token state).

public final class NemotronHMLPLayer: Module, DecoderLayer {
    let norm: RMSNorm
    let upProj, downProj: AnyLinear
    let hidden, intermediate: Int

    init(norm: RMSNorm, upProj: AnyLinear, downProj: AnyLinear,
         hidden: Int, intermediate: Int) {
        self.norm = norm
        self.upProj = upProj; self.downProj = downProj
        self.hidden = hidden; self.intermediate = intermediate
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in norm.parameters() { out.append(("norm.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mixer.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mixer.down_proj.\(k)", v)) }
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `StatelessLayerCache`
    /// and is ignored — a dense MLP holds no per-token state.
    public func decode(_ h: Tensor, position _: Int,
                       cache _: any LayerCacheProtocol,
                       cmd: MTLCommandBuffer, device _: Device) -> Tensor {
        let xNorm = norm(h, on: cmd)
        // down( relu(up(x))^2 ) — squared-ReLU activation.
        let up = upProj(xNorm, on: cmd)
        let r = Ops.relu(up, on: cmd)
        let r2 = Ops.mul(r, r, on: cmd)
        let mlpOut = downProj(r2, on: cmd)
        return Ops.add(h, mlpOut, on: cmd)
    }
}

// ─── NemotronHModel ──────────────────────────────────────────────────

public final class NemotronHModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a Mamba / attention /
    /// MLP layer, ordered by the `hybrid_override_pattern`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let mambaNHeads, mambaHeadDim, stateDim, convDim, convKernel, nGroups, dInner: Int
    public let dtype: DType

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [NemotronHLayerKind]

    init(embedTokens: AnyEmbedding, layers: [any DecoderLayer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         mambaNHeads: Int, mambaHeadDim: Int, stateDim: Int,
         convDim: Int, convKernel: Int, nGroups: Int, dInner: Int,
         vocab: Int, maxSeq: Int, dtype: DType) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers
        self.nHeads = nHeads; self.nKVHeads = nKVHeads; self.headDim = headDim
        self.mambaNHeads = mambaNHeads; self.mambaHeadDim = mambaHeadDim
        self.stateDim = stateDim; self.convDim = convDim
        self.convKernel = convKernel; self.nGroups = nGroups; self.dInner = dInner
        self.vocab = vocab; self.maxSeq = maxSeq; self.dtype = dtype
        self.layerKinds = layers.map { layer in
            switch layer {
            case is NemotronHMambaLayer: return .mamba
            case is NemotronHAttentionLayer: return .attention
            case is NemotronHMLPLayer: return .mlp
            default: return .moe   // unreachable — MoE is rejected at load
            }
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("backbone.embeddings.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            let params: [(String, Tensor)]
            switch layer {
            case let l as NemotronHMambaLayer: params = l.parameters()
            case let l as NemotronHAttentionLayer: params = l.parameters()
            case let l as NemotronHMLPLayer: params = l.parameters()
            default: params = []
            }
            for (k, v) in params { out.append(("backbone.layers.\(i).\(k)", v)) }
        }
        for (k, v) in finalNorm.parameters() { out.append(("backbone.norm_f.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One cache per layer index, matching the layer kind:
    ///   M → Mamba2LayerCache, * → KVCache, - → StatelessLayerCache.
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
            case .mlp, .moe:
                return StatelessLayerCache()
            }
        }
    }

    /// Queue a single-token forward pass. Walks the heterogeneous
    /// `[any DecoderLayer]` in lockstep with the per-layer caches.
    ///
    /// NOTE: NemotronH currently has no command-buffer-committing layer
    /// kind (MoE is rejected), so the whole forward queues onto `cmd`
    /// exactly like FalconH1 / Qwen3. If a future MoE-bearing variant
    /// lands, this loop must refresh `cmd` after any layer that commits
    /// — see the MoELayer file header for the pattern.
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

    /// Multi-token forward — Phase 6.6 prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// NemotronH is a layer-type-string hybrid (`M`/`*`/`E`/`-`
    /// alternation). Per-attention-layer chunked path needs a
    /// `decodeMulti` override on the attention DecoderLayer slot; the
    /// Mamba 2 / MLP / nothing layer kinds keep the per-token default.
    /// Today this override is commit-count-batched only.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "NemotronHModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(tokenId: tok, position: position + i,
                             caches: caches, on: cmd, device: device)
        }
        return logits
    }

    // ─── VLM embedding-input path ────────────────────────────────────
    //
    // NemotronH is a VL-target text backbone (Nemotron-VLM wraps it).
    // The splice supplies a `[hidden]` row directly — a vision-encoder
    // token or a text-token embedding the VL model looked up. The
    // forward is identical to `forward(tokenId:...)` minus the embedding
    // gather; the whole pass queues onto the caller's `cmd` (NemotronH
    // has no command-buffer-committing layer kind — MoE is rejected).

    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "NemotronHModel.forward(inputEmbedding:): expected [\(hidden)], "
                     + "got \(inputEmbedding.shape)")
        var h = inputEmbedding.reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.decode(h, position: position, cache: caches[i],
                             cmd: cmd, device: device)
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

// ─── Load-time host helpers ──────────────────────────────────────────
//
// Small CPU-side derivations done once at load — the cost is in the
// noise. Mirror the `Mamba2` / `FalconH1` helpers.

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
private func readFloatsNH(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("NemotronH: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func writeFloatsNH(_ values: [Float], shape: [Int],
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
        fatalError("NemotronH: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// A_eff = -exp(A_log), per head, in the activation dtype.
private func computeAEffNH(aLog: Tensor, nHeads: Int,
                           dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsNH(aLog)
    precondition(floats.count == nHeads, "NemotronH: A_log expected [n_heads]")
    return writeFloatsNH(floats.map { -Foundation.exp($0) },
                         shape: [nHeads], dtype: dtype, device: device)
}

/// Cast a per-head / per-channel vector to the activation dtype.
private func castVectorNH(_ src: Tensor, count: Int,
                          dtype: DType, device: Device) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsNH(src)
    precondition(floats.count == count, "NemotronH: vector size mismatch")
    return writeFloatsNH(floats, shape: [count], dtype: dtype, device: device)
}

/// Tile `D[h]` across `head_dim` channels → `[n_heads * head_dim]`.
private func tileDNH(d: Tensor, nHeads: Int, headDim: Int,
                     dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsNH(d)
    precondition(floats.count == nHeads, "NemotronH: D expected [n_heads]")
    var tiled: [Float] = []
    tiled.reserveCapacity(nHeads * headDim)
    for h in 0..<nHeads {
        for _ in 0..<headDim { tiled.append(floats[h]) }
    }
    return writeFloatsNH(tiled, shape: [nHeads * headDim], dtype: dtype, device: device)
}

/// Transpose HF conv1d.weight `[C, 1, K]` → `[K, C]` for the metaltile
/// conv kernel.
private func transposeConv1dWeightNH(src: Tensor, kernel K: Int, channels C: Int,
                                     dtype: DType, device: Device) -> Tensor {
    let floats = readFloatsNH(src)
    precondition(floats.count == K * C, "NemotronH: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0..<C {
        for k in 0..<K { dst[k * C + c] = floats[c * K + k] }
    }
    return writeFloatsNH(dst, shape: [K, C], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVectorNH(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}
