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
// Jamba text — concrete variants + the hybrid decoder for AI21's
// Jamba family. The family enum (`enum Jamba`), variant protocol
// (`JambaVariant`), and error type (`JambaError`) live in
// `Models/Jamba.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `JambaHybrid` — `JambaVariant` conformance + the per-variant
//     `loadModel` entry,
//   • `JambaLayerKind`, `JambaMambaLayer`, `JambaAttentionLayer`,
//     `JambaModel` — the per-layer + full-model impl. Mamba mixer is
//     Mamba 1 (per-(channel, state) `A_log`), so the selective scan
//     runs on the CPU and the Mamba layer commits the command buffer.
//     Attention has no RoPE.

import Foundation
import Metal

// ─── Layer kind ──────────────────────────────────────────────────────

/// The two mixer kinds a `layers_block_type` entry can name.
enum JambaLayerKind: Equatable {
    case mamba  // "mamba"
    case attention  // "attention"

    init(from name: String) throws {
        switch name {
        case "mamba": self = .mamba
        case "attention": self = .attention
        default:
            throw JambaError.unsupportedConfig(
                "unknown layers_block_type entry '\(name)'")
        }
    }
}

// ─── JambaHybrid — the single variant ────────────────────────────────

public struct JambaHybrid: JambaVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Jamba ships both base + instruction-tuned checkpoints. Greedy by
    /// default keeps the integration suite deterministic.
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
    ) throws -> JambaModel {
        guard let hidden = config.hiddenSize,
            let vocab = config.vocabSize,
            let nHeads = config.numAttentionHeads,
            let nLayers = config.numLayers
        else {
            throw JambaError.missingConfig(
                "hidden_size / vocab_size / num_attention_heads / num_hidden_layers")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let eps = Float(config.rmsNormEps ?? 1e-6)
        let tieEmbed = config.tieWordEmbeddings

        // Quantized branch — Jamba 4-bit conversions (e.g.
        // mlx-community/AI21-Jamba-Reasoning-3B-4bit) ship per-projection
        // `weight + scales + biases` triplets for in_proj / out_proj /
        // x_proj on the mamba mixer, q/k/v/o on the attention mixer, and
        // gate/up/down on the feed-forward MLP. The conv1d, dt_proj,
        // A_log, D, and dt/B/C layernorm tensors remain raw fp16/bf16
        // (they're either tiny or have a custom kernel that wants raw
        // weights). The embedding table is also quantized; lm_head is
        // tied. This matches the FalconH1 quantized loader contract:
        // route every quantized projection through `loadLinear` /
        // `loadEmbedding`, leave the raw tensors alone.
        let quant = config.quantization
        let isQuantized = quant != nil

        // ── Mamba (Mamba 1) mixer geometry ────────────────────────────
        guard let dState = config.int("mamba_d_state")
        else { throw JambaError.missingConfig("mamba_d_state") }
        let convKernel = config.int("mamba_d_conv") ?? 4
        let expand = config.int("mamba_expand") ?? 2
        let useConvBias = config.bool("mamba_conv_bias") ?? true
        let useProjBias = config.bool("mamba_proj_bias") ?? false
        // d_inner = mamba_expand * hidden (the Mamba 1 expansion rule).
        let dInner = expand * hidden
        // dt_rank: the low-rank dimension of the dt projection. mlx-lm
        // defaults it to ceil(hidden / 16) when the config omits it.
        let dtRank =
            config.int("mamba_dt_rank")
            ?? Int((Double(hidden) / 16.0).rounded(.up))

        // ── Feed-forward geometry ─────────────────────────────────────
        let intermediate = config.intermediateSize ?? (4 * hidden)
        let numExperts = config.int("num_experts") ?? 1
        let numExpertsPerToken = config.int("num_experts_per_tok") ?? 1
        let useMoE = numExperts > 1

        // ── Hybrid layer schedule ─────────────────────────────────────
        // `layers_block_type` may be present explicitly; otherwise it is
        // derived from attn_layer_period / attn_layer_offset exactly the
        // way mlx-lm's JambaConfiguration post-init does.
        let kinds: [JambaLayerKind]
        if let names = config.raw["layers_block_type"] as? [String], !names.isEmpty {
            kinds = try names.map { try JambaLayerKind(from: $0) }
        } else {
            guard let attnPeriod = config.int("attn_layer_period"),
                let attnOffset = config.int("attn_layer_offset")
            else {
                throw JambaError.missingConfig(
                    "layers_block_type (or attn_layer_period + attn_layer_offset)")
            }
            kinds = (0 ..< nLayers).map { i in
                (i % attnPeriod == attnOffset) ? .attention : .mamba
            }
        }
        guard kinds.count == nLayers else {
            throw JambaError.unsupportedConfig(
                "layers_block_type has \(kinds.count) entries, "
                    + "num_hidden_layers is \(nLayers)")
        }

        // ── Activation dtype — taken from the embedding table ─────────
        // Quantized: derive activation dtype from `embed_tokens.scales`
        // (the packed `.weight` is u32 packed pairs/nibbles and isn't an
        // activation dtype). Raw: read it off `embed_tokens.weight`. The
        // unscaled raw embed tensor is kept aside for the tied-lm_head
        // raw path; nil under quantization (tied head reuses the
        // quantized embed triplet instead).
        let activationDtype: DType
        let embedWRawForTiedLmHead: Tensor?
        let embedTokens: AnyEmbedding
        if isQuantized, weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
            embedTokens = try loadEmbedding(
                base: "model.embed_tokens", in: weights,
                hidden: hidden, quantization: quant)
            embedWRawForTiedLmHead = nil
        } else {
            let embedW = try weights.tensor(named: "model.embed_tokens.weight")
            activationDtype = embedW.dtype
            embedTokens = AnyEmbedding(Embedding(weight: embedW))
            embedWRawForTiedLmHead = embedW
        }
        precondition(
            activationDtype == .f32 || activationDtype == .bf16 || activationDtype == .f16,
            "Jamba: unexpected activation dtype \(activationDtype)")

        // ── Per-layer construction ────────────────────────────────────
        var layers: [any DecoderLayer] = []
        layers.reserveCapacity(nLayers)
        for (i, kind) in kinds.enumerated() {
            let p = "model.layers.\(i)"
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: eps)
            let preFFNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).pre_ff_layernorm.weight"),
                eps: eps)

            // ── Feed-forward half — dense SwiGLU MLP or MoE ───────────
            // Quantized: each projection is a triplet, route through
            // loadLinear. Raw: plain Linear(weight: …) as before.
            let ffn: JambaFFN
            if useMoE {
                ffn = .moe(
                    try buildMoE(
                        prefix: p, weights: weights,
                        hidden: hidden, moeIntermediate: intermediate,
                        numExperts: numExperts, topK: numExpertsPerToken,
                        quantization: quant))
            } else {
                let gateProj: AnyLinear
                let upProj: AnyLinear
                let downProj: AnyLinear
                if isQuantized {
                    gateProj = try loadLinear(
                        base: "\(p).feed_forward.gate_proj",
                        in: weights, quantization: quant)
                    upProj = try loadLinear(
                        base: "\(p).feed_forward.up_proj",
                        in: weights, quantization: quant)
                    downProj = try loadLinear(
                        base: "\(p).feed_forward.down_proj",
                        in: weights, quantization: quant)
                } else {
                    gateProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).feed_forward.gate_proj.weight")))
                    upProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).feed_forward.up_proj.weight")))
                    downProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).feed_forward.down_proj.weight")))
                }
                ffn = .dense(
                    JambaDenseMLP(
                        gateProj: gateProj,
                        upProj: upProj,
                        downProj: downProj))
            }

            switch kind {
            case .mamba:
                let mixer = try buildMambaMixer(
                    prefix: "\(p).mamba", weights: weights,
                    hidden: hidden, dInner: dInner, dState: dState,
                    dtRank: dtRank, convKernel: convKernel,
                    useConvBias: useConvBias, useProjBias: useProjBias, eps: eps,
                    dtype: activationDtype, device: device,
                    quantization: quant)
                layers.append(
                    JambaMambaLayer(
                        inputNorm: inputNorm, preFFNorm: preFFNorm,
                        mixer: mixer, ffn: ffn, hidden: hidden))

            case .attention:
                let qProj: AnyLinear
                let kProj: AnyLinear
                let vProj: AnyLinear
                let oProj: AnyLinear
                if isQuantized {
                    qProj = try loadLinear(
                        base: "\(p).self_attn.q_proj",
                        in: weights, quantization: quant)
                    kProj = try loadLinear(
                        base: "\(p).self_attn.k_proj",
                        in: weights, quantization: quant)
                    vProj = try loadLinear(
                        base: "\(p).self_attn.v_proj",
                        in: weights, quantization: quant)
                    oProj = try loadLinear(
                        base: "\(p).self_attn.o_proj",
                        in: weights, quantization: quant)
                } else {
                    qProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).self_attn.q_proj.weight")))
                    kProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).self_attn.k_proj.weight")))
                    vProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).self_attn.v_proj.weight")))
                    oProj = AnyLinear(
                        Linear(
                            weight: try weights.tensor(
                                named: "\(p).self_attn.o_proj.weight")))
                }
                let mixer = JambaAttentionMixer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim)
                layers.append(
                    JambaAttentionLayer(
                        inputNorm: inputNorm, preFFNorm: preFFNorm,
                        mixer: mixer, ffn: ffn, hidden: hidden))
            }
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.final_layernorm.weight"), eps: eps)

        // lm_head — untied/quantized routes through loadLinear; tied/raw
        // reuses the unscaled embed tensor (the original Jamba contract).
        // Tied + quantized wraps the quantized embed triplet as a
        // QuantizedLinear so gemv dispatches identically to the per-row
        // attention/MLP projections above. Matches the FalconH1 tied-
        // lm_head pattern (and Qwen3.6 / GlmOcr before it).
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            if isQuantized, weights.isQuantized("lm_head") {
                lmHead = try loadLinear(
                    base: "lm_head", in: weights, quantization: quant)
            } else {
                lmHead = AnyLinear(
                    Linear(weight: try weights.tensor(named: "lm_head.weight")))
            }
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize))
        } else if let embedW = embedWRawForTiedLmHead {
            lmHead = AnyLinear(Linear(weight: embedW))
        } else {
            // Defensive: quantized config declared but neither
            // lm_head.weight nor a quantized embed triplet is on disk.
            throw JambaError.unsupportedConfig(
                "quantized Jamba checkpoint missing both an explicit "
                    + "lm_head.weight and a quantized embed_tokens triplet")
        }

        let maxSeq = config.int("max_position_embeddings") ?? 8192
        return JambaModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers,
            nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            dInner: dInner, dState: dState, dtRank: dtRank,
            convDim: dInner, convKernel: convKernel,
            vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)
    }

    /// Build one Mamba 1 mixer. Reads the projections, transposes the
    /// conv1d weight, and derives `A_eff = -exp(A_log)` as a full
    /// `[d_inner, d_state]` host array (the 2-D `A_log` does not
    /// collapse — see the file header).
    private static func buildMambaMixer(
        prefix p: String, weights: SafeTensorsBundle,
        hidden: Int, dInner: Int, dState: Int, dtRank: Int,
        convKernel: Int, useConvBias: Bool, useProjBias: Bool, eps: Float,
        dtype: DType, device: Device,
        quantization: ModelConfig.QuantizationConfig?
    ) throws -> JambaMambaMixer {
        // in_proj / out_proj / x_proj: quantized triplets on 4-bit
        // checkpoints; raw weights otherwise. dt_proj.weight stays raw
        // (dt_rank is small; quantizing the low-rank projection is not
        // worth it and dt_proj.bias is mandatory anyway).
        let inProj: AnyLinear
        let outProj: AnyLinear
        let xProj: AnyLinear
        if quantization != nil, weights.isQuantized("\(p).in_proj") {
            // in_proj: hidden → 2*d_inner  (split into x | z).
            inProj = try loadLinear(
                base: "\(p).in_proj", in: weights, quantization: quantization)
            // out_proj: d_inner → hidden.
            outProj = try loadLinear(
                base: "\(p).out_proj", in: weights, quantization: quantization)
            // x_proj: d_inner → dt_rank + 2*d_state  (split into dt | B | C).
            xProj = try loadLinear(
                base: "\(p).x_proj", in: weights, quantization: quantization)
        } else {
            let inProjW = try weights.tensor(named: "\(p).in_proj.weight")
            let inProjB = useProjBias ? try? weights.tensor(named: "\(p).in_proj.bias") : nil
            inProj = AnyLinear(Linear(weight: inProjW, bias: inProjB))
            let outProjW = try weights.tensor(named: "\(p).out_proj.weight")
            let outProjB = useProjBias ? try? weights.tensor(named: "\(p).out_proj.bias") : nil
            outProj = AnyLinear(Linear(weight: outProjW, bias: outProjB))
            xProj = AnyLinear(
                Linear(weight: try weights.tensor(named: "\(p).x_proj.weight")))
        }
        // dt_proj: dt_rank → d_inner, always biased and never quantized.
        let dtProj = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "\(p).dt_proj.weight"),
                bias: try weights.tensor(named: "\(p).dt_proj.bias")))

        // conv1d.weight ships [d_inner, 1, kernel]; the metaltile kernel
        // wants [kernel, d_inner].
        let convWSrc = try weights.tensor(named: "\(p).conv1d.weight")
        precondition(
            convWSrc.elementCount == dInner * convKernel,
            "Jamba: conv1d.weight count mismatch: \(convWSrc.shape)")
        let convW = transposeConv1dWeightJamba(
            src: convWSrc, kernel: convKernel, channels: dInner,
            dtype: dtype, device: device)
        let convB: Tensor = {
            if useConvBias, weights.has("\(p).conv1d.bias") {
                return castVectorJamba(
                    (try? weights.tensor(named: "\(p).conv1d.bias"))
                        ?? zeroVectorJamba(dInner, dtype: dtype, device: device),
                    count: dInner, dtype: dtype, device: device)
            }
            return zeroVectorJamba(dInner, dtype: dtype, device: device)
        }()

        // A_eff = -exp(A_log), full [d_inner, d_state] host array. D is
        // [d_inner] host array. Both feed the CPU selective scan.
        let aLogT = try weights.tensor(named: "\(p).A_log")
        precondition(
            aLogT.elementCount == dInner * dState,
            "Jamba: A_log expected [d_inner, d_state] = "
                + "[\(dInner), \(dState)], got \(aLogT.shape)")
        let aEff = readFloatsJamba(aLogT).map { -Foundation.exp($0) }
        let dHost = readFloatsJamba(try weights.tensor(named: "\(p).D"))
        precondition(dHost.count == dInner, "Jamba: D expected [d_inner]")

        // dt/B/C layernorms — small RMSNorm weights applied CPU-side in
        // the scan (dims dt_rank / d_state are not 128-aligned, so the
        // GPU rmsNorm kernel does not apply). Stored as host arrays.
        let dtNorm = readFloatsJamba(try weights.tensor(named: "\(p).dt_layernorm.weight"))
        let bNorm = readFloatsJamba(try weights.tensor(named: "\(p).b_layernorm.weight"))
        let cNorm = readFloatsJamba(try weights.tensor(named: "\(p).c_layernorm.weight"))

        return JambaMambaMixer(
            inProj: inProj, outProj: outProj, xProj: xProj, dtProj: dtProj,
            convW: convW, convB: convB,
            aEff: aEff, dHost: dHost,
            dtNorm: dtNorm, bNorm: bNorm, cNorm: cNorm,
            hidden: hidden, dInner: dInner, dState: dState, dtRank: dtRank,
            convKernel: convKernel, eps: eps, dtype: dtype)
    }

    /// Build the MoE feed-forward block. Jamba's experts ship as either
    /// per-expert tensors (`feed_forward.experts.<e>.{gate,up,down}_proj`)
    /// or — for mlx-community conversions — pre-stacked switch_mlp
    /// tensors. Both are non-quantized; quantized-MoE expert slicing is
    /// not yet wired into `MoELayer` (see the planning notes).
    private static func buildMoE(
        prefix p: String, weights: SafeTensorsBundle,
        hidden: Int, moeIntermediate: Int,
        numExperts: Int, topK: Int,
        quantization: ModelConfig.QuantizationConfig?
    ) throws -> MoELayer {
        // Router: hidden → numExperts logits. The router is small (one
        // weight per expert per channel) and never quantized in the
        // mlx-community conversions.
        if quantization != nil,
            weights.isQuantized("\(p).feed_forward.switch_mlp.gate_proj")
        {
            // Quantized MoE expert slicing isn't wired into MoELayer yet
            // (a quantized stacked switch_mlp triplet would need a
            // per-expert sliceRows on packed u32 weight + scales/biases,
            // the same pattern Qwen3.6 / GlmOcr use). Surface a clear
            // error rather than silently falling through to a raw load.
            throw JambaError.unsupportedConfig(
                "quantized Jamba MoE expert slicing not yet implemented "
                    + "— load a raw bf16/f16 Jamba MoE variant, or load "
                    + "the dense 3B-A0 quantized checkpoint")
        }
        let gate = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "\(p).feed_forward.router.weight")))

        var gateProj: [AnyLinear] = []
        var upProj: [AnyLinear] = []
        var downProj: [AnyLinear] = []
        gateProj.reserveCapacity(numExperts)
        upProj.reserveCapacity(numExperts)
        downProj.reserveCapacity(numExperts)

        // Stacked switch_mlp layout (mlx-community sanitize output):
        //   switch_mlp.{gate,up}_proj.weight : [numExperts, moeInter, hidden]
        //   switch_mlp.down_proj.weight      : [numExperts, hidden, moeInter]
        let stackedGateKey = "\(p).feed_forward.switch_mlp.gate_proj.weight"
        if weights.has(stackedGateKey) {
            let stackedGate = try weights.tensor(named: stackedGateKey)
            let stackedUp = try weights.tensor(
                named: "\(p).feed_forward.switch_mlp.up_proj.weight")
            let stackedDown = try weights.tensor(
                named: "\(p).feed_forward.switch_mlp.down_proj.weight")
            for e in 0 ..< numExperts {
                gateProj.append(
                    AnyLinear(
                        Linear(
                            weight:
                                stackedGate.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden]))))
                upProj.append(
                    AnyLinear(
                        Linear(
                            weight:
                                stackedUp.slicedRows(start: e, count: 1)
                                .reshaped(to: [moeIntermediate, hidden]))))
                downProj.append(
                    AnyLinear(
                        Linear(
                            weight:
                                stackedDown.slicedRows(start: e, count: 1)
                                .reshaped(to: [hidden, moeIntermediate]))))
            }
        } else {
            // Per-expert tensor layout.
            for e in 0 ..< numExperts {
                let ep = "\(p).feed_forward.experts.\(e)"
                gateProj.append(
                    AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(ep).gate_proj.weight"))))
                upProj.append(
                    AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(ep).up_proj.weight"))))
                downProj.append(
                    AnyLinear(
                        Linear(
                            weight: try weights.tensor(named: "\(ep).down_proj.weight"))))
            }
        }

        // Jamba routes top-K of the raw router logits then softmax over
        // just those K (mlx-lm's JambaSparseMoeBlock argPartition +
        // softmax) — `.topKThenSoftmax`, intrinsically normalised.
        let router = MoERouter(
            nExperts: numExperts, topK: topK, gatingMode: .topKThenSoftmax)
        return MoELayer(
            gate: gate,
            gateProj: gateProj, upProj: upProj, downProj: downProj,
            router: router, hidden: hidden)
    }
}

// ─── FFN sub-block enum ──────────────────────────────────────────────

/// The feed-forward half of a Jamba layer — a dense SwiGLU MLP
/// (`num_experts == 1`) or a block-sparse MoE block (commits the
/// command buffer).
enum JambaFFN {
    case dense(JambaDenseMLP)
    case moe(MoELayer)
}

// ─── JambaMambaMixer — Mamba 1 selective SSM ─────────────────────────
//
// The Mamba 1 mixer. GPU owns the projections (`in_proj`, `conv1d`,
// `x_proj`, `dt_proj`, `out_proj`); the host owns the dt/B/C layernorms
// and the per-(channel, state) selective scan. See the file header for
// why the scan is host-side.

public final class JambaMambaMixer: Module {
    let inProj, outProj, xProj, dtProj: AnyLinear
    let convW: Tensor  // [kernel, d_inner]
    let convB: Tensor  // [d_inner]
    let aEff: [Float]  // [d_inner * d_state]  = -exp(A_log), row-major
    let dHost: [Float]  // [d_inner]
    let dtNorm: [Float]  // [dt_rank]   dt_layernorm weight
    let bNorm: [Float]  // [d_state]   b_layernorm weight
    let cNorm: [Float]  // [d_state]   c_layernorm weight
    let hidden, dInner, dState, dtRank, convKernel: Int
    let eps: Float
    let dtype: DType

    init(
        inProj: AnyLinear, outProj: AnyLinear, xProj: AnyLinear, dtProj: AnyLinear,
        convW: Tensor, convB: Tensor,
        aEff: [Float], dHost: [Float],
        dtNorm: [Float], bNorm: [Float], cNorm: [Float],
        hidden: Int, dInner: Int, dState: Int, dtRank: Int,
        convKernel: Int, eps: Float, dtype: DType
    ) {
        self.inProj = inProj
        self.outProj = outProj
        self.xProj = xProj
        self.dtProj = dtProj
        self.convW = convW
        self.convB = convB
        self.aEff = aEff
        self.dHost = dHost
        self.dtNorm = dtNorm
        self.bNorm = bNorm
        self.cNorm = cNorm
        self.hidden = hidden
        self.dInner = dInner
        self.dState = dState
        self.dtRank = dtRank
        self.convKernel = convKernel
        self.eps = eps
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inProj.parameters() { out.append(("mamba.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("mamba.out_proj.\(k)", v)) }
        for (k, v) in xProj.parameters() { out.append(("mamba.x_proj.\(k)", v)) }
        for (k, v) in dtProj.parameters() { out.append(("mamba.dt_proj.\(k)", v)) }
        return out
    }

    /// Single-token mixer forward. `xNorm` is the already-normalized
    /// layer input. Commits `cmd` mid-way (the host scan needs the GPU
    /// projections on the CPU) and returns a fully-resident `[hidden]`
    /// tensor produced on a fresh command buffer.
    func forward(
        _ xNorm: Tensor, cache: ConvStateCache, ssmState: SSMScanState,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── GPU phase 1: in_proj → x | z, conv1d(x) + SiLU, x_proj ────
        let proj = inProj(xNorm, on: cmd)  // [2 * d_inner]
        let x = proj.slicedRows(start: 0, count: dInner)
        let z = proj.slicedRows(start: dInner, count: dInner)

        let convOut = Tensor.empty(shape: [dInner], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: x, w: convW, b: convB,
            state: cache.state, into: convOut,
            nChannels: dInner, kernelSize: convKernel, on: cmd)
        let convAct = Ops.silu(convOut, on: cmd)  // [d_inner]

        // x_proj(convAct) → [dt_rank + 2*d_state]; the host reads its
        // slices for the dt/B/C layernorms.
        let deltaBC = xProj(convAct, on: cmd)

        // Commit so the host can read convAct / z / deltaBC.
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Host phase: dt/B/C layernorms + dt_proj input prep ────────
        let convHost = convAct.toFloatArray()  // [d_inner]
        let zHost = z.toFloatArray()  // [d_inner]
        let deltaBCHost = deltaBC.toFloatArray()  // [dt_rank + 2*d_state]

        // Split dt_raw | B | C, then RMSNorm each (matches mlx-lm:
        // dt_layernorm / b_layernorm / c_layernorm are weighted RMSNorm).
        var dtRaw = Array(deltaBCHost[0 ..< dtRank])
        var bVec = Array(deltaBCHost[dtRank ..< (dtRank + dState)])
        var cVec = Array(deltaBCHost[(dtRank + dState) ..< (dtRank + 2 * dState)])
        dtRaw = Self.rmsNorm(dtRaw, weight: dtNorm, eps: eps)
        bVec = Self.rmsNorm(bVec, weight: bNorm, eps: eps)
        cVec = Self.rmsNorm(cVec, weight: cNorm, eps: eps)

        // ── GPU phase 2: dt_proj(dtRaw) on a fresh command buffer ─────
        let phase2 = device.makeCommandBuffer()
        let dtRawT = Tensor.empty(shape: [dtRank], dtype: dtype, device: device)
        writeFloatsJamba(dtRaw, into: dtRawT)
        let dtProjOut = dtProj(dtRawT, on: phase2)  // [d_inner]
        phase2.commit()
        phase2.waitUntilCompleted()

        // ── Host phase: softplus(dt) + selective scan ─────────────────
        let dtHost = dtProjOut.toFloatArray().map { softplusScalar($0) }  // [d_inner]
        // Selective scan: per channel c, per state n,
        //   state[c,n] = exp(A[c,n]·dt[c])·state[c,n] + dt[c]·x[c]·B[n]
        //   y[c]       = Σ_n C[n]·state[c,n]  +  D[c]·x[c]
        // then y *= silu(z).
        var yGated = [Float](repeating: 0, count: dInner)
        ssmState.h.withUnsafeMutableBufferPointer { state in
            for c in 0 ..< dInner {
                let xc = convHost[c]
                let dtc = dtHost[c]
                let dtx = dtc * xc
                let base = c * dState
                var yc: Float = 0
                for n in 0 ..< dState {
                    let decay = Foundation.exp(aEff[base + n] * dtc)
                    let s = decay * state[base + n] + dtx * bVec[n]
                    state[base + n] = s
                    yc += cVec[n] * s
                }
                yc += dHost[c] * xc
                yGated[c] = yc * siluScalar(zHost[c])
            }
        }

        // ── GPU phase 3: out_proj(yGated) on the phase-2 buffer's heir ─
        let phase3 = device.makeCommandBuffer()
        let yGatedT = Tensor.empty(shape: [dInner], dtype: dtype, device: device)
        writeFloatsJamba(yGated, into: yGatedT)
        let result = outProj(yGatedT, on: phase3)  // [hidden]
        phase3.commit()
        phase3.waitUntilCompleted()
        return result
    }

    /// Unweighted-then-weighted RMSNorm over a small host vector.
    /// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
    static func rmsNorm(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count, "Jamba.rmsNorm: size mismatch")
        var sumSq: Float = 0
        for v in x { sumSq += v * v }
        let inv = 1.0 / (sumSq / Float(x.count) + eps).squareRoot()
        return zip(x, weight).map { $0 * inv * $1 }
    }
}

// ─── JambaAttentionMixer — multi-head attention, no RoPE ─────────────

public final class JambaAttentionMixer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let nHeads, nKVHeads, headDim: Int
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        nHeads: Int, nKVHeads: Int, headDim: Int
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
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

    /// Single-token attention forward. Returns the post-o_proj
    /// contribution (residual add done by the enclosing layer).
    func forward(
        _ xNorm: Tensor, cache kv: KVCache,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // No RoPE — Jamba attention attends without positional rotation.
        kv.appendOnGPU(
            kFlat: k.reshaped(to: [nKVHeads, headDim]),
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

// ─── JambaDenseMLP — dense SwiGLU feed-forward ───────────────────────

public final class JambaDenseMLP: Module {
    let gateProj, upProj, downProj: AnyLinear

    init(gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.parameters() {
            out.append(("feed_forward.gate_proj.\(k)", v))
        }
        for (k, v) in upProj.parameters() { out.append(("feed_forward.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() {
            out.append(("feed_forward.down_proj.\(k)", v))
        }
        return out
    }

    /// down(silu(gate(x)) * up(x)).
    func forward(_ xNorm: Tensor, cmd: MTLCommandBuffer) -> Tensor {
        let g = gateProj(xNorm, on: cmd)
        let u = upProj(xNorm, on: cmd)
        // Fused silu(gate) * up — one kernel, intermediate silu(g)
        // stays in registers instead of round-tripping to DRAM.
        let inner = Ops.swiglu(gate: g, up: u, on: cmd)
        return downProj(inner, on: cmd)
    }
}

// ─── SSMScanState — host-side Mamba 1 recurrent state ────────────────
//
// The Mamba 1 selective scan keeps a `[d_inner, d_state]` state matrix
// that accumulates across decode steps. Because Jamba's scan runs on
// the CPU (see the file header) the state lives in a plain host array,
// not a GPU buffer. `SSMScanState` is the per-layer storage; it sits
// inside `JambaMambaLayerCache`.

public final class SSMScanState: @unchecked Sendable {
    /// `[d_inner * d_state]` row-major recurrent state, fp32.
    var h: [Float]
    let dInner, dState: Int

    init(dInner: Int, dState: Int) {
        self.dInner = dInner
        self.dState = dState
        self.h = [Float](repeating: 0, count: dInner * dState)
    }

    func reset() {
        for i in 0 ..< h.count { h[i] = 0 }
    }

    var bytesAllocated: Int { h.count * MemoryLayout<Float>.stride }
}

// ─── JambaMambaLayerCache — conv state + host SSM scan state ─────────
//
// A Jamba mamba layer needs two pieces of per-token state: the rolling
// conv1d window (`ConvStateCache`, GPU) and the Mamba 1 selective-scan
// matrix (`SSMScanState`, host). This bundles both behind
// `LayerCacheProtocol` so the heterogeneous decode loop can index it.

public final class JambaMambaLayerCache: LayerCacheProtocol, @unchecked Sendable {
    public let conv: ConvStateCache
    public let scan: SSMScanState

    public private(set) var length: Int = 0
    public let maxSeq: Int = .max

    public init(
        dInner: Int, dState: Int,
        convKernelSize: Int, dtype: DType, device: Device = .shared
    ) {
        self.conv = ConvStateCache(
            nChannels: dInner,
            kernelSize: convKernelSize,
            dtype: dtype, device: device)
        self.scan = SSMScanState(dInner: dInner, dState: dState)
    }

    public func reset() {
        conv.reset()
        scan.reset()
        length = 0
    }

    public func advance() { length += 1 }

    public var bytesAllocated: Int {
        conv.bytesAllocated + scan.bytesAllocated
    }

    public var bytesInUse: Int {
        length == 0 ? 0 : bytesAllocated
    }
}

// ─── JambaMambaLayer — "mamba" ───────────────────────────────────────
//
// One stack-interleaved hybrid layer with a Mamba 1 mixer. Conforms to
// `DecoderLayer`; its cache slot is a `JambaMambaLayerCache`.
//
// `commitsCommandBuffer` is ALWAYS true: the Mamba 1 scan runs host-side
// so `JambaMambaMixer.forward` commits the command buffer mid-decode.
// (An MoE FFN would also commit.) The host model refreshes `cmd` after
// this layer — see `JambaModel.forward`.

public final class JambaMambaLayer: Module, DecoderLayer {
    let inputNorm, preFFNorm: RMSNorm
    let mixer: JambaMambaMixer
    let ffn: JambaFFN
    let hidden: Int

    public let commitsCommandBuffer: Bool = true

    init(
        inputNorm: RMSNorm, preFFNorm: RMSNorm,
        mixer: JambaMambaMixer, ffn: JambaFFN, hidden: Int
    ) {
        self.inputNorm = inputNorm
        self.preFFNorm = preFFNorm
        self.mixer = mixer
        self.ffn = ffn
        self.hidden = hidden
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in preFFNorm.parameters() { out.append(("pre_ff_layernorm.\(k)", v)) }
        out.append(contentsOf: mixer.parameters())
        out.append(contentsOf: jambaFFNParameters(ffn))
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `JambaMambaLayerCache`.
    /// IMPORTANT: commits `cmd` (the host scan + an MoE FFN both need a
    /// CPU sync). The host model refreshes `cmd` afterwards.
    public func decode(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard let mc = cache as? JambaMambaLayerCache else {
            fatalError(
                "JambaMambaLayer: expected JambaMambaLayerCache, "
                    + "got \(type(of: cache))")
        }
        // ── Mixer half — pre-norm + Mamba 1 mixer + residual add ──────
        let xNorm = inputNorm(h, on: cmd)
        // mixer.forward commits `cmd` and returns a resident tensor.
        let mixerOut = mixer.forward(
            xNorm, cache: mc.conv, ssmState: mc.scan,
            cmd: cmd, device: device)
        mc.advance()

        // `h` was produced on the now-committed `cmd`; it is resident.
        // The residual add + FFN run on a fresh command buffer that this
        // layer owns and must commit (the host model swapped `cmd` away).
        let ffnCmd = device.makeCommandBuffer()
        let postMix = Ops.add(h, mixerOut, on: ffnCmd)
        return jambaApplyFFN(
            ffn, postMix: postMix, preFFNorm: preFFNorm,
            position: position, cmd: ffnCmd,
            commitCmd: true, device: device)
    }
}

// ─── JambaAttentionLayer — "attention" ───────────────────────────────
//
// One stack-interleaved hybrid layer with a multi-head attention mixer.
// Conforms to `DecoderLayer`; its cache slot is a `KVCache`.
//
// `commitsCommandBuffer` is true only when the FFN is an `MoELayer`
// (the attention mixer itself is pure GPU and does not commit).

public final class JambaAttentionLayer: Module, DecoderLayer {
    let inputNorm, preFFNorm: RMSNorm
    let mixer: JambaAttentionMixer
    let ffn: JambaFFN
    let hidden: Int

    public let commitsCommandBuffer: Bool

    init(
        inputNorm: RMSNorm, preFFNorm: RMSNorm,
        mixer: JambaAttentionMixer, ffn: JambaFFN, hidden: Int
    ) {
        self.inputNorm = inputNorm
        self.preFFNorm = preFFNorm
        self.mixer = mixer
        self.ffn = ffn
        self.hidden = hidden
        if case .moe = ffn {
            self.commitsCommandBuffer = true
        } else {
            self.commitsCommandBuffer = false
        }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in preFFNorm.parameters() { out.append(("pre_ff_layernorm.\(k)", v)) }
        out.append(contentsOf: mixer.parameters())
        out.append(contentsOf: jambaFFNParameters(ffn))
        return out
    }

    /// `DecoderLayer` conformance. Cache slot is a `KVCache`. Commits
    /// `cmd` only when the FFN is an `MoELayer`.
    public func decode(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        guard let kv = cache as? KVCache else {
            fatalError("JambaAttentionLayer: expected KVCache, got \(type(of: cache))")
        }
        // ── Mixer half — pre-norm + attention + residual add ──────────
        let xNorm = inputNorm(h, on: cmd)
        let mixerOut = mixer.forward(xNorm, cache: kv, cmd: cmd, device: device)

        // Fused residual add + pre-FF RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Jamba 1.5 Mini is hidden=4096 (fits);
        // wider variants fall through the validator gate.
        let postMix: Tensor
        let ffnNorm: Tensor?
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, mixerOut, weight: preFFNorm.weight, eps: preFFNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postMix = fused.residual
            ffnNorm = fused.normed
        } else {
            postMix = Ops.add(h, mixerOut, on: cmd)
            ffnNorm = nil
        }

        // ── Feed-forward half ─────────────────────────────────────────
        // `cmd` is the host model's `workCmd`; the model owns its commit
        // (or swaps it after an MoE FFN). This layer does not commit it.
        return jambaApplyFFN(
            ffn, postMix: postMix, preFFNorm: preFFNorm,
            position: position, cmd: cmd,
            commitCmd: false, device: device,
            preNormed: ffnNorm)
    }
}

// ─── Shared FFN helpers ──────────────────────────────────────────────

/// Re-key a flat `MoELayer` parameter name into Jamba's checkpoint
/// layout (`feed_forward.router.*` / `feed_forward.experts.<e>.*`).
private func jambaMoEKey(_ k: String) -> String {
    if k.hasPrefix("gate.") {
        return "feed_forward.router." + k.dropFirst("gate.".count)
    }
    return "feed_forward." + k
}

/// Collect the `(name, tensor)` parameters of a layer's FFN half.
private func jambaFFNParameters(_ ffn: JambaFFN) -> [(String, Tensor)] {
    switch ffn {
    case .dense(let mlp):
        return mlp.parameters()
    case .moe(let moe):
        return moe.parameters().map { (jambaMoEKey($0.0), $0.1) }
    }
}

/// Apply the feed-forward half of a Jamba layer: pre-FF norm, FFN, and
/// the residual add.
///
/// - `commitCmd`: when `true`, this function owns `cmd` and commits it
///   (the dense-FFN path on a mamba layer, which is handed a fresh
///   buffer). When `false`, the caller owns `cmd`'s commit (the
///   attention-layer path, where `cmd` is the host model's `workCmd`).
///   When the FFN is an `MoELayer`, `decode` commits `cmd` regardless;
///   the residual add then runs on a fresh, locally-committed buffer so
///   the returned tensor is resident either way.
private func jambaApplyFFN(
    _ ffn: JambaFFN, postMix: Tensor, preFFNorm: RMSNorm,
    position: Int, cmd: MTLCommandBuffer,
    commitCmd: Bool, device: Device,
    preNormed: Tensor? = nil
) -> Tensor {
    // `preNormed` lets the attention-layer caller supply the
    // fused-kernel output of `mt_add_rms_norm` (postMix already +
    // pre-FF norm in one dispatch). When nil, we run the separate norm.
    let ffnNorm = preNormed ?? preFFNorm(postMix, on: cmd)
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
        // MoELayer.decode commits `cmd`; run the residual add on a
        // fresh buffer so the returned tensor does not depend on a dead
        // command buffer.
        let ffnOut = moe.decode(
            ffnNorm, position: position,
            cache: StatelessLayerCache(),
            cmd: cmd, device: device)
        let addCmd = device.makeCommandBuffer()
        let result = Ops.add(postMix, ffnOut, on: addCmd)
        addCmd.commit()
        addCmd.waitUntilCompleted()
        return result
    }
}

// ─── JambaModel ──────────────────────────────────────────────────────

public final class JambaModel: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — each entry is a Mamba 1 or attention
    /// hybrid layer, ordered by `layers_block_type`.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    /// Mamba 1 mixer geometry.
    public let dInner, dState, dtRank, convDim, convKernel: Int
    public let dtype: DType

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [JambaLayerKind]
    /// True when any layer has an MoE FFN (purely informational —
    /// every mamba layer commits regardless).
    public let hasMoE: Bool

    init(
        embedTokens: AnyEmbedding, layers: [any DecoderLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        dInner: Int, dState: Int, dtRank: Int, convDim: Int, convKernel: Int,
        vocab: Int, maxSeq: Int, dtype: DType
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
        self.dInner = dInner
        self.dState = dState
        self.dtRank = dtRank
        self.convDim = convDim
        self.convKernel = convKernel
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.layerKinds = layers.map { layer in
            switch layer {
            case is JambaMambaLayer: return .mamba
            case is JambaAttentionLayer: return .attention
            default: return .mamba
            }
        }
        self.hasMoE = layers.contains { layer in
            if let m = layer as? JambaMambaLayer, case .moe = m.ffn { return true }
            if let a = layer as? JambaAttentionLayer, case .moe = a.ffn { return true }
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
            case let l as JambaMambaLayer: params = l.parameters()
            case let l as JambaAttentionLayer: params = l.parameters()
            default: params = []
            }
            for (k, v) in params { out.append(("model.layers.\(i).\(k)", v)) }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.final_layernorm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// One cache per layer index, matching the layer kind:
    ///   mamba → JambaMambaLayerCache, attention → KVCache.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return layerKinds.map { kind in
            switch kind {
            case .mamba:
                return JambaMambaLayerCache(
                    dInner: dInner, dState: dState,
                    convKernelSize: convKernel, dtype: dtype, device: device)
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
    /// CRITICAL — command-buffer contract. Every Jamba *mamba* layer
    /// commits the command buffer it is handed (the Mamba 1 selective
    /// scan runs host-side); an MoE FFN also commits. So the caller's
    /// `cmd` must NEVER be handed to a layer — if it were, the first
    /// mamba layer would commit it and the caller's later commit would
    /// double-commit. Instead the embedding + every layer run on
    /// internal `workCmd` buffers (committed by the layers themselves /
    /// refreshed after each committing layer), and ONLY the final
    /// `final_layernorm` + `lm_head` queue onto the caller's pristine
    /// `cmd`. The hidden state `h` handed to the final norm is already
    /// resident (the last layer committed its buffer), so the caller's
    /// single commit of `cmd` produces correct logits.
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
            // Refresh `workCmd` if the layer committed it.
            let committed: Bool
            switch layer {
            case let l as JambaMambaLayer: committed = l.commitsCommandBuffer
            case let l as JambaAttentionLayer: committed = l.commitsCommandBuffer
            default: committed = false
            }
            if committed { workCmd = device.makeCommandBuffer() }
        }

        // The last layer (index 27 on the published checkpoints) is a
        // mamba layer and committed its buffer, so `h` is resident and a
        // fresh `workCmd` is pending. If the stack ever ended on a
        // non-committing attention layer, `workCmd` would still carry
        // that layer's uncommitted work — commit it so `h` is resident
        // before the caller's `cmd` reads it.
        if let last = layers.last,
            !((last as? JambaMambaLayer)?.commitsCommandBuffer ?? false),
            !((last as? JambaAttentionLayer)?.commitsCommandBuffer ?? false)
        {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        // Final norm + lm_head queue onto the caller's pristine `cmd`.
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Multi-token forward — prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// Jamba interleaves Mamba 2 selective scan + attention + MoE FFN
    /// layers. The cmd-buffer-committing layers (MoE router, Mamba
    /// reduce) make a single-cmd chunked path non-trivial; the
    /// per-attention-layer `decodeMulti` follow-up will need to
    /// reproduce the per-layer cmd-refresh in `forward(tokenId:)`.
    /// Today this override is commit-count-batched only.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "JambaModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }
}

// ─── Load-time + host-scan helpers ───────────────────────────────────

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
private func readFloatsJamba(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("Jamba: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into an existing tensor, converting to its dtype.
private func writeFloatsJamba(_ values: [Float], into t: Tensor) {
    precondition(values.count == t.elementCount, "Jamba: writeFloats size mismatch")
    switch t.dtype {
    case .f32:
        t.copyIn(from: values)
    case .bf16:
        t.copyIn(
            from: values.map { v -> UInt16 in
                // Round-to-nearest before truncating the low 16 bits.
                let bits = v.bitPattern
                let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
                return UInt16(rounded >> 16)
            })
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    default:
        fatalError("Jamba: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Cast a per-channel vector to the activation dtype.
private func castVectorJamba(
    _ src: Tensor, count: Int,
    dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = readFloatsJamba(src)
    precondition(floats.count == count, "Jamba: vector size mismatch")
    let t = Tensor.empty(shape: [count], dtype: dtype, device: device)
    writeFloatsJamba(floats, into: t)
    return t
}

/// A zero-filled `[n]` vector in the requested dtype.
private func zeroVectorJamba(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}

/// Transpose HF conv1d.weight `[C, 1, K]` → `[K, C]` for the metaltile
/// conv kernel.
private func transposeConv1dWeightJamba(
    src: Tensor, kernel K: Int, channels C: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = readFloatsJamba(src)
    precondition(floats.count == K * C, "Jamba: conv1d.weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    for c in 0 ..< C {
        for k in 0 ..< K { dst[k * C + c] = floats[c * K + k] }
    }
    let t = Tensor.empty(shape: [K, C], dtype: dtype, device: device)
    writeFloatsJamba(dst, into: t)
    return t
}

/// Scalar SiLU: `x · sigmoid(x)`. Used in the host SSM gating.
private func siluScalar(_ x: Float) -> Float {
    return x / (1.0 + Foundation.exp(-x))
}

/// Scalar softplus: `log(1 + exp(x))`, numerically stable for large x.
private func softplusScalar(_ x: Float) -> Float {
    if x > 20 { return x }
    if x < -20 { return Foundation.exp(x) }
    return Foundation.log1p(Foundation.exp(x))
}
