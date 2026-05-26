// Copyright 2026 Eric Kryski (@ekryski)
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
// LFM2 text — concrete variants + the hybrid decoder for LiquidAI's
// LFM2 / LFM2.5 family. The family enum (`enum LFM2`), variant
// protocol (`LFM2Variant`), and error type (`LFM2Error`) live in
// `Models/LFM2.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `LFM2Dense` / `LFM2MoE` — `LFM2Variant` conformance + the per-
//     variant `loadModel` entries,
//   • `LFM2LayerKind`, `LFM2ConvCache`, `LFM2Model` — per-layer + full-
//     model impl. Stack-interleaved conv (double-gated short conv) +
//     attention (GQA + NeoX RoPE, host-side Q/K norm). MoE uses the
//     biased-router `.softmaxThenTopK` mode. `LFM2Model.forward`
//     follows the Granite4 command-buffer discipline.

import Foundation
import Metal

// ─── Layer kind + schedule ───────────────────────────────────────────

/// The two mixer kinds an LFM2 layer can carry.
public enum LFM2LayerKind: Equatable, Sendable {
    case conv
    case attention
}

/// Resolve the per-layer mixer schedule. LFM2 checkpoints carry EITHER a
/// `layer_types` string array (`"conv"` / `"full_attention"`) OR a
/// `full_attn_idxs` integer list naming the attention-layer indices —
/// older `LFM2-*` configs ship the latter, `LFM2.5-*` the former. This
/// resolver accepts both and is unit-tested directly.
func lfm2LayerKinds(
    layerTypes: [String]?, fullAttnIdxs: [Int]?,
    numLayers: Int
) throws -> [LFM2LayerKind] {
    precondition(numLayers > 0, "lfm2LayerKinds: numLayers must be positive")
    if let names = layerTypes, !names.isEmpty {
        guard names.count == numLayers else {
            throw LFM2Error.unsupportedConfig(
                "layer_types has \(names.count) entries, "
                    + "num_hidden_layers is \(numLayers)")
        }
        return try names.map { name in
            switch name {
            case "conv": return .conv
            case "full_attention": return .attention
            default:
                throw LFM2Error.unsupportedConfig(
                    "unknown layer_types entry '\(name)'")
            }
        }
    }
    if let idxs = fullAttnIdxs {
        let attn = Set(idxs)
        return (0 ..< numLayers).map { attn.contains($0) ? .attention : .conv }
    }
    throw LFM2Error.missingConfig("layer_types / full_attn_idxs")
}

// ─── LFM2ConvCache — per-layer rolling conv state ────────────────────

/// `LayerCacheProtocol` wrapper around a `ConvStateCache` for an LFM2
/// conv layer. The conv mixer holds no other per-token state.
public final class LFM2ConvCache: LayerCacheProtocol, @unchecked Sendable {
    public let conv: ConvStateCache
    public private(set) var length: Int = 0
    public let maxSeq: Int = .max

    public init(
        channels: Int, kernelSize: Int, dtype: DType,
        device: Device = .shared
    ) {
        self.conv = ConvStateCache(
            nChannels: channels, kernelSize: kernelSize,
            dtype: dtype, device: device)
    }

    public func reset() {
        conv.reset()
        length = 0
    }
    /// Advance the step counter (storage is constant-size).
    public func advance() { length += 1 }

    public var bytesAllocated: Int { conv.bytesAllocated }
    public var bytesInUse: Int { length == 0 ? 0 : bytesAllocated }
}

// ─── Variants ────────────────────────────────────────────────────────

/// LFM2 / LFM2.5 greedy defaults — keeps the integration suite
/// deterministic. Shared by both the dense and MoE variants.
private let lfm2DefaultGenerationParameters = GenerationParameters(
    maxTokens: 256,
    prefillStepSize: 1024,
    temperature: 0.0,
    topP: 1.0,
    topK: 0,
    minP: 0.0,
    repetitionPenalty: 1.0
)

/// `Lfm2ForCausalLM` — dense SwiGLU FFN on every layer. Serves both the
/// LFM2 and LFM2.5 collections.
public struct LFM2Dense: LFM2Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]
    public static let defaultGenerationParameters = lfm2DefaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options _: LoadOptions, device: Device
    ) throws -> LFM2Model {
        try lfm2LoadModel(
            config: config, weights: weights,
            moe: false, device: device)
    }
}

/// `Lfm2MoeForCausalLM` — a block-sparse MoE FFN on every layer at index
/// ≥ `num_dense_layers` (LFM2-8B-A1B / LFM2-24B-A2B).
public struct LFM2MoE: LFM2Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]
    public static let defaultGenerationParameters = lfm2DefaultGenerationParameters

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options _: LoadOptions, device: Device
    ) throws -> LFM2Model {
        try lfm2LoadModel(
            config: config, weights: weights,
            moe: true, device: device)
    }
}

// ─── Shared loader ───────────────────────────────────────────────────

/// Build an `LFM2Model` from a checkpoint. `moe` selects the
/// feed-forward shape: `false` → a dense SwiGLU MLP on every layer
/// (`lfm2`); `true` → an MoE block on every layer at index ≥
/// `num_dense_layers`, dense before that (`lfm2_moe`).
func lfm2LoadModel(
    config: ModelConfig, weights: SafeTensorsBundle,
    moe: Bool, device: Device
) throws -> LFM2Model {
    guard let hidden = config.hiddenSize,
        let nLayers = config.numLayers,
        let nHeads = config.numAttentionHeads,
        let vocab = config.vocabSize
    else {
        throw LFM2Error.missingConfig(
            "hidden_size / num_hidden_layers / num_attention_heads / vocab_size")
    }
    let nKVHeads = config.numKeyValueHeads ?? nHeads
    let headDim = config.headDim ?? (hidden / nHeads)
    // LFM2 uses `norm_eps` (not `rms_norm_eps`).
    let eps = Float(config.float("norm_eps") ?? config.rmsNormEps ?? 1e-5)
    // rope_theta is flat on LFM2, nested under rope_parameters on LFM2.5.
    let theta = Float(
        config.ropeTheta
            ?? (config.nested("rope_parameters")?["rope_theta"] as? Double)
            ?? 1_000_000.0)
    let maxSeq = config.int("max_position_embeddings") ?? 128_000
    let convKernel = config.int("conv_L_cache") ?? 3
    let convBias = config.bool("conv_bias") ?? false

    guard convKernel >= 2 else {
        throw LFM2Error.unsupportedConfig(
            "conv_L_cache (\(convKernel)) must be ≥ 2")
    }
    guard
        headDim == 64 || headDim == 128
            || headDim == 256 || headDim == 512
    else {
        throw LFM2Error.unsupportedConfig(
            "head_dim \(headDim) — Ops.sdpaDecode supports {64,128,256,512}")
    }
    guard config.quantization == nil else {
        throw LFM2Error.unsupportedConfig(
            "quantized LFM2 checkpoints not yet supported — "
                + "load a raw bf16/f16 variant")
    }

    // ── MoE geometry ──────────────────────────────────────────────────
    let numExperts = config.int("num_experts") ?? 0
    let numExpertsPerTok = config.int("num_experts_per_tok") ?? 0
    let moeIntermediate = config.int("moe_intermediate_size") ?? 0
    let numDenseLayers = config.int("num_dense_layers") ?? 0
    let normTopKProb = config.bool("norm_topk_prob") ?? true
    let useExpertBias = config.bool("use_expert_bias") ?? false
    if moe {
        guard numExperts > 0, numExpertsPerTok > 0, moeIntermediate > 0 else {
            throw LFM2Error.missingConfig(
                "num_experts / num_experts_per_tok / moe_intermediate_size")
        }
    }

    // ── Layer schedule ────────────────────────────────────────────────
    let kinds = try lfm2LayerKinds(
        layerTypes: config.raw["layer_types"] as? [String],
        fullAttnIdxs: config.intArray("full_attn_idxs"),
        numLayers: nLayers)

    // ── Embedding ─────────────────────────────────────────────────────
    let embedW = try weights.tensor(named: "model.embed_tokens.weight")
    let activationDtype = embedW.dtype
    precondition(
        activationDtype == .f32 || activationDtype == .bf16
            || activationDtype == .f16,
        "LFM2: unexpected activation dtype \(activationDtype)")
    let embedTokens = AnyEmbedding(Embedding(weight: embedW))

    // ── Per-layer construction ────────────────────────────────────────
    var layers: [any DecoderLayer] = []
    layers.reserveCapacity(nLayers)
    for (i, kind) in kinds.enumerated() {
        let p = "model.layers.\(i)"
        let operatorNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).operator_norm.weight"),
            eps: eps)
        let ffnNorm = RMSNorm(
            weight: try weights.tensor(named: "\(p).ffn_norm.weight"),
            eps: eps)

        let mixer: LFM2Mixer
        switch kind {
        case .conv:
            let inProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).conv.in_proj.weight")))
            let outProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).conv.out_proj.weight")))
            // HF Conv1d weight ships [hidden, 1, kernel]; the metaltile
            // conv1d kernel wants [kernel, hidden].
            let convWSrc = try weights.tensor(named: "\(p).conv.conv.weight")
            precondition(
                convWSrc.elementCount == hidden * convKernel,
                "LFM2: conv.conv.weight count \(convWSrc.elementCount) "
                    + "≠ hidden·kernel \(hidden * convKernel)")
            let convW = lfm2TransposeConvWeight(
                convWSrc, kernel: convKernel, channels: hidden,
                dtype: activationDtype, device: device)
            let convB: Tensor
            if convBias, weights.has("\(p).conv.conv.bias") {
                convB = lfm2CastVector(
                    try weights.tensor(named: "\(p).conv.conv.bias"),
                    count: hidden, dtype: activationDtype, device: device)
            } else {
                convB = lfm2ZeroVector(
                    hidden, dtype: activationDtype,
                    device: device)
            }
            mixer = .conv(
                LFM2ConvMixer(
                    inProj: inProj, outProj: outProj,
                    convW: convW, convB: convB,
                    hidden: hidden, kernel: convKernel, dtype: activationDtype))
        case .attention:
            let qProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.q_proj.weight")))
            let kProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.k_proj.weight")))
            let vProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.v_proj.weight")))
            let outProj = AnyLinear(
                Linear(
                    weight: try weights.tensor(named: "\(p).self_attn.out_proj.weight")))
            let qNormW = lfm2CastVector(
                try weights.tensor(named: "\(p).self_attn.q_layernorm.weight"),
                count: headDim, dtype: activationDtype, device: device)
            let kNormW = lfm2CastVector(
                try weights.tensor(named: "\(p).self_attn.k_layernorm.weight"),
                count: headDim, dtype: activationDtype, device: device)
            mixer = .attention(
                LFM2AttentionMixer(
                    qProj: qProj, kProj: kProj, vProj: vProj, outProj: outProj,
                    qNormW: qNormW, kNormW: kNormW, normEps: eps,
                    nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
                    ropeTheta: theta))
        }

        // ── Feed-forward half ──────────────────────────────────────────
        let ffn: LFM2FFN
        if moe, i >= numDenseLayers {
            ffn = .moe(
                try buildLFM2MoE(
                    prefix: p, weights: weights,
                    hidden: hidden, moeIntermediate: moeIntermediate,
                    numExperts: numExperts, topK: numExpertsPerTok,
                    normTopKProb: normTopKProb, useExpertBias: useExpertBias))
        } else {
            // Dense SwiGLU — LFM2 names the projections w1 (gate) / w3
            // (up) / w2 (down).
            ffn = .dense(
                LFM2MLP(
                    w1: AnyLinear(
                        Linear(
                            weight: try lfm2MLPWeight(
                                "\(p).feed_forward", "gate", in: weights))),
                    w3: AnyLinear(
                        Linear(
                            weight: try lfm2MLPWeight(
                                "\(p).feed_forward", "up", in: weights))),
                    w2: AnyLinear(
                        Linear(
                            weight: try lfm2MLPWeight(
                                "\(p).feed_forward", "down", in: weights)))))
        }

        layers.append(
            LFM2Layer(
                operatorNorm: operatorNorm, ffnNorm: ffnNorm,
                mixer: mixer, ffn: ffn, hidden: hidden))
    }

    // LFM2's final norm is `model.embedding_norm` (not `model.norm`).
    let finalNorm = RMSNorm(
        weight: try weights.tensor(named: "model.embedding_norm.weight"),
        eps: eps)

    // LFM2 ties the LM head to the embedding table (the reference
    // projects with `embed_tokens.as_linear`). A standalone
    // `lm_head.weight` is honoured if a checkpoint ever ships one.
    let lmHead: AnyLinear
    if weights.has("lm_head.weight") {
        lmHead = AnyLinear(
            Linear(
                weight: try weights.tensor(named: "lm_head.weight")))
    } else {
        lmHead = AnyLinear(Linear(weight: embedW))
    }

    return LFM2Model(
        embedTokens: embedTokens, layers: layers,
        finalNorm: finalNorm, lmHead: lmHead,
        hidden: hidden, nLayers: nLayers,
        nHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
        convDim: hidden, convKernel: convKernel,
        vocab: vocab, maxSeq: maxSeq, dtype: activationDtype)
}

/// Build one LFM2-MoE feed-forward block: a router + `num_experts`
/// SwiGLU experts (no shared expert). Routing is softmax → add
/// `expert_bias` → top-K → optional re-normalisation, wired through
/// `MoERouter`'s `expertBias` parameter.
private func buildLFM2MoE(
    prefix p: String, weights: SafeTensorsBundle,
    hidden: Int, moeIntermediate: Int,
    numExperts: Int, topK: Int,
    normTopKProb: Bool, useExpertBias: Bool
) throws -> MoELayer {
    let gate = AnyLinear(
        Linear(
            weight: try weights.tensor(named: "\(p).feed_forward.gate.weight")))

    var gateProj: [AnyLinear] = []
    var upProj: [AnyLinear] = []
    var downProj: [AnyLinear] = []
    gateProj.reserveCapacity(numExperts)
    upProj.reserveCapacity(numExperts)
    downProj.reserveCapacity(numExperts)
    for e in 0 ..< numExperts {
        let eb = "\(p).feed_forward.experts.\(e)"
        gateProj.append(
            AnyLinear(
                Linear(
                    weight: try lfm2MLPWeight(eb, "gate", in: weights))))
        upProj.append(
            AnyLinear(
                Linear(
                    weight: try lfm2MLPWeight(eb, "up", in: weights))))
        downProj.append(
            AnyLinear(
                Linear(
                    weight: try lfm2MLPWeight(eb, "down", in: weights))))
    }

    // Per-expert load-balancing bias — added to the post-softmax gate
    // values before top-K (and used as the combine weight).
    var expertBias: [Float]? = nil
    if useExpertBias, weights.has("\(p).feed_forward.expert_bias") {
        let bias = lfm2ReadFloats(
            try weights.tensor(named: "\(p).feed_forward.expert_bias"))
        precondition(
            bias.count == numExperts,
            "LFM2: expert_bias has \(bias.count) entries, "
                + "num_experts is \(numExperts)")
        expertBias = bias
    }

    let router = MoERouter(
        nExperts: numExperts, topK: topK,
        gatingMode: .softmaxThenTopK,
        normTopKProb: normTopKProb, expertBias: expertBias)
    return MoELayer(
        gate: gate,
        gateProj: gateProj, upProj: upProj, downProj: downProj,
        router: router, hidden: hidden)
}

/// Read an LFM2 SwiGLU projection weight. LFM2 names them `w1` (gate) /
/// `w3` (up) / `w2` (down); some conversions rename them to
/// `gate_proj` / `up_proj` / `down_proj`. Accept either.
/// `which` ∈ {`gate`, `up`, `down`}.
private func lfm2MLPWeight(
    _ base: String, _ which: String,
    in weights: SafeTensorsBundle
) throws -> Tensor {
    let wName: String
    let projName: String
    switch which {
    case "gate":
        wName = "w1"
        projName = "gate_proj"
    case "up":
        wName = "w3"
        projName = "up_proj"
    case "down":
        wName = "w2"
        projName = "down_proj"
    default: fatalError("lfm2MLPWeight: unknown projection '\(which)'")
    }
    if weights.has("\(base).\(wName).weight") {
        return try weights.tensor(named: "\(base).\(wName).weight")
    }
    return try weights.tensor(named: "\(base).\(projName).weight")
}

// ─── LFM2Mixer / LFM2FFN — the two halves of a layer ─────────────────

enum LFM2Mixer {
    case conv(LFM2ConvMixer)
    case attention(LFM2AttentionMixer)
}

/// The feed-forward half — a dense SwiGLU MLP or a block-sparse MoE
/// block (`MoELayer`, which commits the command buffer).
enum LFM2FFN {
    case dense(LFM2MLP)
    case moe(MoELayer)
}

// ─── LFM2ConvMixer — double-gated short convolution ──────────────────

public final class LFM2ConvMixer: Module {
    let inProj, outProj: AnyLinear
    let convW: Tensor  // [kernel, hidden]
    let convB: Tensor  // [hidden]
    let hidden, kernel: Int
    let dtype: DType

    init(
        inProj: AnyLinear, outProj: AnyLinear,
        convW: Tensor, convB: Tensor,
        hidden: Int, kernel: Int, dtype: DType
    ) {
        self.inProj = inProj
        self.outProj = outProj
        self.convW = convW
        self.convB = convB
        self.hidden = hidden
        self.kernel = kernel
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in inProj.parameters() { out.append(("conv.in_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("conv.out_proj.\(k)", v)) }
        return out
    }

    /// Single-token mixer forward. `xNorm` is the already-normalized
    /// layer input. Returns the post-`out_proj` contribution (the
    /// residual add is the enclosing layer's job). All work queues onto
    /// `cmd` — no commit (conv layers carry no CPU sync point).
    func forward(
        _ xNorm: Tensor, cache: LFM2ConvCache,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // in_proj → [3·hidden], split into B / C / x.
        let bcx = inProj(xNorm, on: cmd)
        let b = bcx.slicedRows(start: 0, count: hidden)
        let c = bcx.slicedRows(start: hidden, count: hidden)
        let x = bcx.slicedRows(start: 2 * hidden, count: hidden)

        // Input gate, then depthwise causal conv1d (rolling state).
        let bx = Ops.mul(b, x, on: cmd)
        let convOut = Tensor.empty(shape: [hidden], dtype: dtype, device: device)
        Ops.conv1dCausalStep(
            x: bx, w: convW, b: convB,
            state: cache.conv.state, into: convOut,
            nChannels: hidden, kernelSize: kernel, on: cmd)

        // Output gate, then out_proj. No activation between conv and
        // gate (LFM2 differs from Mamba's post-conv SiLU here).
        let y = Ops.mul(c, convOut, on: cmd)
        return outProj(y, on: cmd)
    }
}

// ─── LFM2AttentionMixer — GQA + RoPE, host-side per-head Q/K norm ─────

public final class LFM2AttentionMixer: Module {
    let qProj, kProj, vProj, outProj: AnyLinear
    let qNormW, kNormW: Tensor  // [headDim]
    let normEps: Float
    let nHeads, nKVHeads, headDim: Int
    let ropeTheta: Float
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, outProj: AnyLinear,
        qNormW: Tensor, kNormW: Tensor, normEps: Float,
        nHeads: Int, nKVHeads: Int, headDim: Int, ropeTheta: Float
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.outProj = outProj
        self.qNormW = qNormW
        self.kNormW = kNormW
        self.normEps = normEps
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.scale = 1.0 / Float(Double(headDim).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in outProj.parameters() { out.append(("self_attn.out_proj.\(k)", v)) }
        out.append(("self_attn.q_layernorm.weight", qNormW))
        out.append(("self_attn.k_layernorm.weight", kNormW))
        return out
    }

    /// Single-token attention forward. Queues Q/K/V projections onto
    /// `cmd`, then COMMITS `cmd` to read Q/K back for the host-side
    /// per-head RMSNorm (head_dim 64 is not 128-aligned — the GPU
    /// reduction kernel cannot run it). Returns the post-`out_proj`
    /// contribution on a fresh, locally-committed buffer.
    func forward(
        _ xNorm: Tensor, position: Int, cache kv: KVCache,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Commit so the host can read Q/K for the per-head norm.
        cmd.commit()
        cmd.waitUntilCompleted()

        let qNormed = lfm2HostPerHeadRMSNorm(
            q, weight: qNormW, eps: normEps,
            nHeads: nHeads, headDim: headDim, device: device)
        let kNormed = lfm2HostPerHeadRMSNorm(
            k, weight: kNormW, eps: normEps,
            nHeads: nKVHeads, headDim: headDim, device: device)

        // GPU phase 2: RoPE → KV append → SDPA → out_proj.
        let c2 = device.makeCommandBuffer()
        let qRot = Ops.rope(
            qNormed.reshaped(to: [nHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, on: c2)
        let kRot = Ops.rope(
            kNormed.reshaped(to: [nKVHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, on: c2)
        kv.appendOnGPU(
            kFlat: kRot,
            vFlat: v.reshaped(to: [nKVHeads, headDim]), on: c2)
        let (cacheK, cacheV) = kv.prepareForAttention(on: c2)
        let attnOut = Ops.sdpaDecode(
            q: qRot, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: kv.length, kvStride: kv.maxSeq, scale: scale, on: c2)
        let result = outProj(attnOut.reshaped(to: [nHeads * headDim]), on: c2)
        c2.commit()
        c2.waitUntilCompleted()
        return result
    }
}

// ─── LFM2MLP — SwiGLU feed-forward ───────────────────────────────────

public final class LFM2MLP: Module {
    let w1, w3, w2: AnyLinear

    init(w1: AnyLinear, w3: AnyLinear, w2: AnyLinear) {
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in w1.parameters() { out.append(("feed_forward.w1.\(k)", v)) }
        for (k, v) in w3.parameters() { out.append(("feed_forward.w3.\(k)", v)) }
        for (k, v) in w2.parameters() { out.append(("feed_forward.w2.\(k)", v)) }
        return out
    }

    /// w2(silu(w1(x)) · w3(x)).
    func forward(_ x: Tensor, on cmd: MTLCommandBuffer) -> Tensor {
        let gate = w1(x, on: cmd)
        let up = w3(x, on: cmd)
        let inner = Ops.mul(Ops.silu(gate, on: cmd), up, on: cmd)
        return w2(inner, on: cmd)
    }
}

/// Re-key a flat `MoELayer` parameter name into LFM2-MoE's checkpoint
/// layout. `MoELayer` emits `gate.*` / `experts.<e>.*`; LFM2-MoE stores
/// them under `feed_forward.*`.
private func lfm2MoEKey(_ k: String) -> String { "feed_forward." + k }

// ─── LFM2Layer — one stack-interleaved hybrid layer ──────────────────

public final class LFM2Layer: Module, DecoderLayer {
    let operatorNorm, ffnNorm: RMSNorm
    let mixer: LFM2Mixer
    let ffn: LFM2FFN
    let hidden: Int

    /// True when this layer commits the command buffer it is handed —
    /// an attention layer (host-side Q/K norm) OR a MoE FFN
    /// (`MoELayer.decode`'s router CPU readback). The host decode loop
    /// refreshes `cmd` after any such layer.
    public let commitsCommandBuffer: Bool

    init(
        operatorNorm: RMSNorm, ffnNorm: RMSNorm,
        mixer: LFM2Mixer, ffn: LFM2FFN, hidden: Int
    ) {
        self.operatorNorm = operatorNorm
        self.ffnNorm = ffnNorm
        self.mixer = mixer
        self.ffn = ffn
        self.hidden = hidden
        let attentionMixer: Bool
        if case .attention = mixer { attentionMixer = true } else { attentionMixer = false }
        let moeFFN: Bool
        if case .moe = ffn { moeFFN = true } else { moeFFN = false }
        self.commitsCommandBuffer = attentionMixer || moeFFN
    }

    var kind: LFM2LayerKind {
        switch mixer {
        case .conv: return .conv
        case .attention: return .attention
        }
    }

    /// True when this layer's feed-forward half is a block-sparse MoE.
    var isMoELayer: Bool {
        if case .moe = ffn { return true }
        return false
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in operatorNorm.parameters() {
            out.append(("operator_norm.\(k)", v))
        }
        for (k, v) in ffnNorm.parameters() { out.append(("ffn_norm.\(k)", v)) }
        switch mixer {
        case .conv(let m): out.append(contentsOf: m.parameters())
        case .attention(let a): out.append(contentsOf: a.parameters())
        }
        switch ffn {
        case .dense(let mlp):
            out.append(contentsOf: mlp.parameters())
        case .moe(let moe):
            for (k, v) in moe.parameters() { out.append((lfm2MoEKey(k), v)) }
        }
        return out
    }

    /// `DecoderLayer` conformance — layer-local single-token decode.
    ///
    /// `operator_norm` → mixer → residual; `ffn_norm` → FFN → residual.
    /// An attention mixer and a MoE FFN each commit the command buffer
    /// they are handed; the work after a committing sub-block runs on a
    /// fresh, locally-committed buffer. A conv layer with a dense FFN
    /// queues everything onto `cmd` and never commits.
    public func decode(
        _ h: Tensor, position: Int,
        cache: any LayerCacheProtocol,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let xNorm = operatorNorm(h, on: cmd)

        // ── Mixer half ────────────────────────────────────────────────
        // The attention-mixer branch fuses (h + mixerOut) → ffnNorm via
        // mt_add_rms_norm (hidden ≤ 4096). The conv-mixer branch is the
        // SSM recurrence path and per the task carve-out keeps the
        // separate add + norm sequence.
        var workCmd = cmd
        let postMix: Tensor
        var preFusedFfnInput: Tensor? = nil
        switch mixer {
        case .conv(let m):
            guard let cc = cache as? LFM2ConvCache else {
                fatalError(
                    "LFM2Layer: conv layer expected LFM2ConvCache, "
                        + "got \(type(of: cache))")
            }
            let mixerOut = m.forward(xNorm, cache: cc, cmd: workCmd, device: device)
            cc.advance()
            postMix = Ops.add(h, mixerOut, on: workCmd)
        case .attention(let a):
            guard let kv = cache as? KVCache else {
                fatalError(
                    "LFM2Layer: attention layer expected KVCache, "
                        + "got \(type(of: cache))")
            }
            // The mixer commits `cmd` and returns a resident tensor —
            // continue the layer on a fresh buffer.
            let mixerOut = a.forward(
                xNorm, position: position,
                cache: kv, cmd: workCmd, device: device)
            workCmd = device.makeCommandBuffer()
            if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
                let fused = Ops.addAndRmsNorm(
                    h, mixerOut, weight: ffnNorm.weight, eps: ffnNorm.eps,
                    nRows: 1, rowSize: hidden, on: workCmd)
                postMix = fused.residual
                preFusedFfnInput = fused.normed
            } else {
                postMix = Ops.add(h, mixerOut, on: workCmd)
            }
        }

        // ── Feed-forward half ─────────────────────────────────────────
        let ffnInput = preFusedFfnInput ?? ffnNorm(postMix, on: workCmd)
        switch ffn {
        case .dense(let mlp):
            let ffnOut = mlp.forward(ffnInput, on: workCmd)
            let result = Ops.add(postMix, ffnOut, on: workCmd)
            // A conv mixer leaves `workCmd == cmd` uncommitted — the host
            // loop commits it. An attention mixer made `workCmd` a fresh
            // private buffer — commit it so `result` is resident.
            if case .attention = mixer {
                workCmd.commit()
                workCmd.waitUntilCompleted()
            }
            return result
        case .moe(let moe):
            // `MoELayer.decode` commits `workCmd`, runs the experts on
            // its own private buffer, and returns a fully-resident
            // tensor. `postMix` is resident afterwards (workCmd was
            // committed + waited). The residual add runs on a fresh
            // buffer so the result does not depend on the dead workCmd.
            let ffnOut = moe.decode(
                ffnInput, position: position,
                cache: StatelessLayerCache(),
                cmd: workCmd, device: device)
            let addCmd = device.makeCommandBuffer()
            let result = Ops.add(postMix, ffnOut, on: addCmd)
            addCmd.commit()
            addCmd.waitUntilCompleted()
            return result
        }
    }
}

// ─── LFM2Model ───────────────────────────────────────────────────────

public final class LFM2Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    /// Heterogeneous layer stack — conv or attention, ordered by the
    /// `layer_types` / `full_attn_idxs` schedule.
    public let layers: [any DecoderLayer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let convDim, convKernel: Int
    public let dtype: DType

    /// Layer kinds, index-aligned with `layers` — drives `makeLayerCaches`.
    let layerKinds: [LFM2LayerKind]
    /// True when any layer carries a MoE feed-forward block.
    public let hasMoE: Bool

    init(
        embedTokens: AnyEmbedding, layers: [any DecoderLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        convDim: Int, convKernel: Int,
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
        self.convDim = convDim
        self.convKernel = convKernel
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.dtype = dtype
        self.layerKinds = layers.map { ($0 as? LFM2Layer)?.kind ?? .conv }
        self.hasMoE = layers.contains { ($0 as? LFM2Layer)?.isMoELayer == true }
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() {
            out.append(("model.embed_tokens.\(k)", v))
        }
        for (i, layer) in layers.enumerated() {
            if let l = layer as? LFM2Layer {
                for (k, v) in l.parameters() {
                    out.append(("model.layers.\(i).\(k)", v))
                }
            }
        }
        for (k, v) in finalNorm.parameters() {
            out.append(("model.embedding_norm.\(k)", v))
        }
        return out
    }

    /// One cache per layer index, matching the layer kind: conv →
    /// `LFM2ConvCache`, attention → `KVCache`.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return layerKinds.map { kind in
            switch kind {
            case .conv:
                return LFM2ConvCache(
                    channels: convDim, kernelSize: convKernel,
                    dtype: dtype, device: device)
            case .attention:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim,
                    maxSeq: cap, dtype: dtype, device: device)
            }
        }
    }

    /// Queue a single-token forward pass. **Does not commit `cmd`** —
    /// the final norm + lm_head queue onto the caller's pristine `cmd`
    /// so the default `forwardSample` extensions compose cleanly.
    ///
    /// Command-buffer discipline (Granite4). Attention layers
    /// and MoE FFNs commit the buffer they are handed (host-side Q/K
    /// norm / router CPU sync). The caller's `cmd` is therefore never
    /// handed to a layer: the embedding + every layer run on internal
    /// `workCmd` buffers, refreshed after each committing layer; only
    /// the final norm + lm_head queue onto `cmd`.
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var workCmd = device.makeCommandBuffer()
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.decode(
                h, position: position, cache: caches[i],
                cmd: workCmd, device: device)
            if let l = layer as? LFM2Layer, l.commitsCommandBuffer {
                workCmd = device.makeCommandBuffer()
            }
        }

        // After a committing layer `workCmd` is fresh + empty and `h` is
        // resident. After a non-committing (conv + dense) layer `workCmd`
        // still carries uncommitted work — commit it so `h` is resident
        // before the caller's `cmd` reads it.
        let lastCommitted =
            (layers.last as? LFM2Layer)?
            .commitsCommandBuffer ?? false
        if !lastCommitted {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Embedding-input forward — the VLM splice path. Identical to
    /// `forward(tokenId:...)` minus the embedding gather: the `[hidden]`
    /// row is supplied directly (a vision-encoder token, or a text-token
    /// embedding the VL model looked up itself). LFM2-VL routes its text
    /// backbone through this primitive.
    ///
    /// Mirrors the command-buffer discipline of `forward(tokenId:)`:
    /// attention + MoE layers may commit the buffer they are handed, so
    /// the layer stack runs on internal `workCmd` buffers (refreshed
    /// after each committing layer) and only the final norm + lm_head
    /// queue onto the caller's pristine `cmd`.
    public var supportsEmbeddingInput: Bool { true }

    public func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            inputEmbedding.elementCount == hidden,
            "LFM2Model.forward(inputEmbedding:): expected [\(hidden)], "
                + "got \(inputEmbedding.shape)")
        var workCmd = device.makeCommandBuffer()
        var h = inputEmbedding.reshaped(to: [hidden])

        for (i, layer) in layers.enumerated() {
            h = layer.decode(
                h, position: position, cache: caches[i],
                cmd: workCmd, device: device)
            if let l = layer as? LFM2Layer, l.commitsCommandBuffer {
                workCmd = device.makeCommandBuffer()
            }
        }

        // After a committing layer `workCmd` is fresh + empty and `h` is
        // resident. After a non-committing (conv + dense) layer `workCmd`
        // still carries uncommitted work — commit it so `h` is resident
        // before the caller's `cmd` reads it.
        let lastCommitted =
            (layers.last as? LFM2Layer)?
            .commitsCommandBuffer ?? false
        if !lastCommitted {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Raw embedding-table lookup for one text token — the text-token
    /// half of the VLM splice stream.
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

    /// Multi-token forward — prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// LFM2 / LFM2.5 alternates short-conv blocks with GQA attention
    /// blocks; the dense + MoE variants share the layer-kind
    /// dispatch. A `decodeMulti` override on the attention-layer slot
    /// would adopt the Llama chunked path; the conv-layer kind keeps
    /// the per-token default (the causal conv1d is single-position).
    /// Today this override is commit-count-batched only.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "LFM2Model.forwardMulti: tokenIds must be non-empty")
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

/// Read an f32 / bf16 / f16 tensor into `[Float]`.
func lfm2ReadFloats(_ t: Tensor) -> [Float] {
    switch t.dtype {
    case .f32:
        return t.toArray(as: Float.self)
    case .bf16:
        return t.toArray(as: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
    case .f16:
        return t.toArray(as: Float16.self).map { Float($0) }
    default:
        fatalError("LFM2: unsupported dtype for host conversion: \(t.dtype)")
    }
}

/// Write a `[Float]` into a fresh tensor of the requested dtype.
private func lfm2WriteFloats(
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
        fatalError("LFM2: unsupported dtype for host conversion: \(dtype)")
    }
    return t
}

/// Transpose an HF Conv1d weight `[channels, 1, kernel]` → `[kernel,
/// channels]` for the metaltile `conv1d_causal_step` kernel.
private func lfm2TransposeConvWeight(
    _ src: Tensor, kernel K: Int,
    channels C: Int,
    dtype: DType, device: Device
) -> Tensor {
    let floats = lfm2ReadFloats(src)
    precondition(floats.count == K * C, "LFM2: conv weight count mismatch")
    var dst = [Float](repeating: 0, count: K * C)
    // Source flattens [C, 1, K] as floats[c*K + k]; destination is
    // [K, C] as dst[k*C + c].
    for c in 0 ..< C {
        for k in 0 ..< K { dst[k * C + c] = floats[c * K + k] }
    }
    return lfm2WriteFloats(dst, shape: [K, C], dtype: dtype, device: device)
}

/// Cast a per-channel / per-head vector to the activation dtype.
private func lfm2CastVector(
    _ src: Tensor, count: Int,
    dtype: DType, device: Device
) -> Tensor {
    if src.dtype == dtype { return src }
    let floats = lfm2ReadFloats(src)
    precondition(floats.count == count, "LFM2: vector size mismatch")
    return lfm2WriteFloats(floats, shape: [count], dtype: dtype, device: device)
}

/// A zero-filled `[n]` vector in the requested dtype.
private func lfm2ZeroVector(_ n: Int, dtype: DType, device: Device) -> Tensor {
    let t = Tensor.empty(shape: [n], dtype: dtype, device: device)
    t.zero()
    return t
}

/// Host-side per-head RMSNorm of a resident `[nHeads · headDim]` tensor.
/// LFM2's head_dim (64) is not 128-aligned, so the GPU `rmsNormRows`
/// kernel cannot run; a per-head norm over one decode token is trivial
/// on the CPU. `weight` is `[headDim]`, shared across heads. Returns a
/// fresh resident `[nHeads · headDim]` tensor in `x`'s dtype.
func lfm2HostPerHeadRMSNorm(
    _ x: Tensor, weight: Tensor, eps: Float,
    nHeads: Int, headDim: Int,
    device: Device
) -> Tensor {
    let xs = lfm2ReadFloats(x)
    let ws = lfm2ReadFloats(weight)
    precondition(
        xs.count == nHeads * headDim,
        "LFM2 perHeadRMSNorm: x has \(xs.count) elements, "
            + "expected \(nHeads * headDim)")
    precondition(
        ws.count == headDim,
        "LFM2 perHeadRMSNorm: weight has \(ws.count) elements, "
            + "expected headDim \(headDim)")
    var out = [Float](repeating: 0, count: xs.count)
    for h in 0 ..< nHeads {
        let base = h * headDim
        var sumSq: Float = 0
        for d in 0 ..< headDim {
            let v = xs[base + d]
            sumSq += v * v
        }
        let inv = 1.0 / (sumSq / Float(headDim) + eps).squareRoot()
        for d in 0 ..< headDim {
            out[base + d] = xs[base + d] * inv * ws[d]
        }
    }
    return lfm2WriteFloats(out, shape: x.shape, dtype: x.dtype, device: device)
}
