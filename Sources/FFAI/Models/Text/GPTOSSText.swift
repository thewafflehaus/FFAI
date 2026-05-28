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
// GPT-OSS text — concrete variants + the MoE decoder for OpenAI's
// GPT-OSS family. The family enum (`enum GPTOSS`), variant protocol
// (`GPTOSSVariant`), and error type (`GPTOSSError`) live in
// `Models/GPTOSS.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `GPTOSSMoEVariant` — `GPTOSSVariant` conformance + the per-
//     variant `loadModel` entry,
//   • `GPTOSSAttentionKind`, `GPTOSSAttention`, `GPTOSSExpert`,
//     `GPTOSSMoELayer`, `buildGPTOSSMoE` — the per-layer impl,
//     MXFP4-to-affine-int4 codec, and biased-router MoE FFN,
//   • `GPTOSSModel` — the full LanguageModel decoder, with the Jamba-
//     style command-buffer discipline (every layer commits, so per-
//     layer work runs on internal `workCmd` buffers).

import Foundation
import Metal

// ─── Attention kind ──────────────────────────────────────────────────

/// The two attention kinds a `layer_types` entry can name.
public enum GPTOSSAttentionKind: Equatable, Sendable {
    case sliding  // "sliding_attention" — capped at `sliding_window`
    case full  // "full_attention"    — attends the whole context

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

public struct GPTOSSMoEVariant: GPTOSSVariant, ReasoningCapable {
    /// GPT-OSS-20B advertises text + reasoning-level control. The
    /// Harmony template renders a `reasoning_effort` variable; FFAI's
    /// public `ReasoningLevel` dial maps onto it via `clamped(to:)`
    /// against `supportedReasoningLevels` below.
    public static let availableCapabilities: Set<Capability> = [
        .textIn, .textOut, .reasoningLevel,
    ]

    /// GPT-OSS-20B's Harmony template recognises low / medium / high
    /// for the `reasoning_effort` variable. User requests of
    /// `.extraHigh` or `.max` clamp to `.high` (the highest native
    /// value); `.none` always disables reasoning regardless.
    public static let supportedReasoningLevels: Set<ReasoningLevel> = [
        .low, .medium, .high,
    ]

    /// GPT-OSS-20B greedy defaults — keeps the integration suite
    /// deterministic. 2048-token prefill chunk matches mlx-swift-lm's
    /// audited optimum (pure-attention model, no SSM bottleneck).
    /// Reasoning defaults to `.none` (disabled until the caller
    /// explicitly opts in) per the FFAI convention.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 2048,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        minP: 0.0,
        repetitionPenalty: 1.0,
        // `reasoningLevel: nil` — the field is `Optional<ReasoningLevel>`,
        // so the bare `.none` literal previously inferred Optional's
        // nil. The mechanical Swift 5.10 warning sweep wrongly qualified
        // it as `ReasoningLevel.none` (the enum case, NOT nil), which
        // wraps to `Optional.some(.none)` and breaks the
        // `defaultGenerationParameters.reasoningLevel == .none` test.
        // Write `nil` explicitly so the intent is unambiguous and the
        // older toolchain doesn't surface its "assuming Optional.none"
        // ambiguity warning.
        reasoningLevel: nil
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
            ((rs["rope_type"] as? String) ?? (rs["type"] as? String)) == "yarn"
        {
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
                named: "model.embed_tokens.scales"
            ).dtype
        } else {
            activationDtype = try weights.tensor(
                named: "model.embed_tokens.weight"
            ).dtype
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
            precondition(
                sinkTensor.elementCount == nHeads,
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

            layers.append(
                GPTOSSLayer(
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
            let q = quantMap.config(for: "model.embed_tokens")
        {
            // Tied + quantized — reuse the embedding triplet as lm_head.
            let t = try weights.quantizedTriplet("model.embed_tokens")
            lmHead = AnyLinear(
                QuantizedLinear(
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
            maxContextWindow: maxSeq, slidingWindow: slidingWindow,
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
        inner = AnyLinear(
            QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: q.bits, groupSize: q.groupSize))
    } else {
        inner = AnyLinear(
            Linear(
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

    init(
        qProj: GPTOSSBiasedLinear, kProj: GPTOSSBiasedLinear,
        vProj: GPTOSSBiasedLinear, oProj: GPTOSSBiasedLinear,
        sinks: [Float], kind: GPTOSSAttentionKind,
        nHeads: Int, nKVHeads: Int, headDim: Int,
        ropeTheta: Float, yarn: Ops.RoPEYaRN
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.sinks = sinks
        self.kind = kind
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.yarn = yarn
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
    func forward(
        _ xNorm: Tensor, position: Int, cache: KVCache,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── GPU phase: project, RoPE, append, plain SDPA ──────────────
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        let qRot = Ops.ropeYaRN(
            q.reshaped(to: [nHeads * headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, yarn: yarn, on: cmd)
        let kRot = Ops.ropeYaRN(
            k.reshaped(to: [nKVHeads * headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, yarn: yarn, on: cmd)

        cache.appendOnGPU(
            kFlat: kRot.reshaped(to: [nKVHeads, headDim]),
            vFlat: v.reshaped(to: [nKVHeads, headDim]), on: cmd)

        let nKV = cache.length
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRot.reshaped(to: [nHeads, headDim]), k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: nKV, kvStride: cache.capacity,
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
        let qHost = readGPTOSSFloats(qRot)  // [nHeads*headDim]
        let headsPerGroup = nHeads / nKVHeads
        // Per-kv-head live K slice [nKV, headDim], read once and reused
        // across the headsPerGroup query heads that share it.
        // `nonisolated(unsafe)`: read-only inside the per-head concurrent
        // closure below; populated serially here first.
        nonisolated(unsafe) var kLive = [[Float]](repeating: [], count: nKVHeads)
        if nKV > 0 {
            for kvHead in 0 ..< nKVHeads {
                // kBuffer is [nKVHeads, maxSeq, headDim]; slice head
                // kvHead, then its first nKV timesteps.
                let headSlab = cache.kBuffer
                    .slicedRows(start: kvHead, count: 1)
                    .reshaped(to: [cache.capacity, headDim])
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
        // Bind every self-borrowed value to a local before crossing the
        // `@Sendable` boundary — otherwise the closure captures `self`
        // (an actor-isolated GPTOSSAttention) just to read scalar
        // properties like `headDim` / `scale` / `sinks`. The bindings
        // themselves are Sendable (Int, Float, [Float]) so no
        // `nonisolated(unsafe)` qualifier is needed; only the raw
        // `fPtr` pointer write needs it.
        let sinksLocal = sinks
        let headDimLocal = headDim
        let scaleLocal = scale
        factors.withUnsafeMutableBufferPointer { fBuf in
            nonisolated(unsafe) let fPtr = fBuf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nHeads) { h in
                let kvHead = h / headsPerGroup
                let qBase = h * headDimLocal
                let kHead = kLive[kvHead]
                var m = sinksLocal[h]
                var z: Float = 0
                for t in 0 ..< nKV {
                    let kBase = t * headDimLocal
                    var dot: Float = 0
                    for d in 0 ..< headDimLocal {
                        dot += qHost[qBase + d] * kHead[kBase + d]
                    }
                    let s = dot * scaleLocal
                    if s > m {
                        // Renormalise the running sum to the new max.
                        z = z * Foundation.exp(m - s)
                        m = s
                    }
                    z += Foundation.exp(s - m)
                }
                let sinkTerm = Foundation.exp(sinksLocal[h] - m)
                // O' = O · Z / (Z + exp(sink - M)). Z > 0 whenever nKV > 0.
                fPtr[h] = z / (z + sinkTerm)
            }
        }

        // ── GPU phase 2: rescale O per-head, then o_proj ──────────────
        let phase2 = device.makeCommandBuffer()
        // Broadcast the per-head scalar across each head's headDim slice.
        let factorTensor = Tensor.empty(
            shape: [nHeads * headDim],
            dtype: attnOut.dtype, device: device)
        var factorFlat = [Float](repeating: 0, count: nHeads * headDim)
        for h in 0 ..< nHeads {
            for d in 0 ..< headDim { factorFlat[h * headDim + d] = factors[h] }
        }
        writeGPTOSSFloats(factorFlat, into: factorTensor)
        let corrected = Ops.mul(
            attnOut.reshaped(to: [nHeads * headDim]),
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
    precondition(
        values.count == t.elementCount,
        "GPT-OSS: writeFloats size mismatch")
    switch t.dtype {
    case .f32:
        t.copyIn(from: values)
    case .f16:
        t.copyIn(from: values.map { Float16($0) })
    case .bf16:
        t.copyIn(
            from: values.map { v -> UInt16 in
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

    init(
        attention: GPTOSSAttention, moe: GPTOSSMoELayer,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm, hidden: Int
    ) {
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
    func decode(
        _ h: Tensor, position: Int, cache: KVCache,
        cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // ── Attention half ────────────────────────────────────────────
        // `h` is resident on entry (the previous layer committed). The
        // input norm runs on the host and writes `xNorm` directly.
        let xNorm = gptOSSHostRMSNorm(
            h, weight: inputNorm.weight,
            eps: inputNorm.eps, device: device)
        // attention.forward commits `cmd`; the returned tensor is
        // resident on a fresh buffer it owns.
        let attnOut = attention.forward(
            xNorm, position: position,
            cache: cache, cmd: cmd, device: device)

        // The residual add + post-attention RMSNorm run as ONE fused
        // GPU dispatch via mt_add_rms_norm — replaces the previous
        // (Ops.add on GPU + gptOSSHostRMSNorm on CPU) pair. GPT-OSS's
        // hidden=2880 is a multiple of 4 and ≤ 4096, so it fits the
        // fused kernel's row-width invariants even though it does NOT
        // satisfy the legacy `mt_rms_norm`'s 128-element invariant
        // (which is why the standalone norm was on the host).
        precondition(
            OpsValidation.validateAddRmsNorm(n: hidden) == nil,
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
func gptOSSHostRMSNorm(
    _ x: Tensor, weight: Tensor, eps: Float,
    device: Device
) -> Tensor {
    let xs = readGPTOSSFloats(x)
    let ws = readGPTOSSFloats(weight)
    precondition(
        xs.count == ws.count,
        "GPT-OSS RMSNorm: x (\(xs.count)) / weight (\(ws.count)) "
            + "size mismatch")
    var sumSq: Float = 0
    for v in xs { sumSq += v * v }
    let inv = 1.0 / (sumSq / Float(xs.count) + eps).squareRoot()
    var out = [Float](repeating: 0, count: xs.count)
    for i in 0 ..< xs.count { out[i] = xs[i] * inv * ws[i] }
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

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxContextWindow: Int
    /// Sliding-window size for sliding-attention layers.
    public let slidingWindow: Int
    public let dtype: DType
    /// Attention kind per layer index — drives `makeLayerCaches`
    /// (sliding layers get a `.window` eviction cache).
    public let attnKinds: [GPTOSSAttentionKind]

    init(
        embedTokens: AnyEmbedding, layers: [GPTOSSLayer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        attnKinds: [GPTOSSAttentionKind],
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxContextWindow: Int, slidingWindow: Int, dtype: DType
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.attnKinds = attnKinds
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxContextWindow = maxContextWindow
        self.slidingWindow = slidingWindow
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
        let cap = maxSeq ?? self.maxContextWindow
        return attnKinds.map { kind in
            switch kind {
            case .sliding:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                    dtype: dtype,
                    eviction: .window(maxSize: min(slidingWindow, cap), keep: 0),
                    device: device)
            case .full:
                return KVCache(
                    nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
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
    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
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
                fatalError(
                    "GPTOSSModel: expected KVCache at layer \(i), "
                        + "got \(type(of: caches[i]))")
            }
            // Each layer runs on its own internal buffers (the attention
            // sink correction + the MoE router both commit), so it gets
            // a fresh buffer and returns a fully-resident `h`.
            let layerCmd = device.makeCommandBuffer()
            h = layer.decode(
                h, position: position, cache: kv,
                cmd: layerCmd, device: device)
        }

        // `h` is resident (the last layer committed). The final RMSNorm
        // runs host-side (hidden=2880 is not 128-aligned); lm_head
        // queues onto the caller's pristine `cmd`.
        let normed = gptOSSHostRMSNorm(
            h, weight: finalNorm.weight,
            eps: finalNorm.eps, device: device)
        return lmHead(normed, on: cmd)
    }

    /// Multi-token forward — prefill fast path. Loops
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
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "GPTOSSModel.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }
}

// ─── MoE FFN — MXFP4 codec + layer implementation ────────────────────
//
// Everything below was previously in Models/Text/GPTOSSMoEText.swift.
// It is kept here so the GPT-OSS family lives in a single text file
// (matching every other family-with-MoE), since the MoE block diverges
// enough from the reusable Models/MoELayer.swift (MXFP4-sourced
// experts, biased router, clipped α-swish + (up+1) activation,
// per-projection biases) that sharing the common implementation
// would only bloat MoELayer.swift's surface.

// ─── MXFP4 constants ─────────────────────────────────────────────────

private enum MXFP4 {
    /// MXFP4 group size — one e8m0 scale byte per 32 fp4 codes.
    static let groupSize = 4 << 3
    /// fp4 codes packed per uint32 word.
    static let codesPerWord = 8
    /// The 16-entry MXFP4 value lookup table. Code → fp value; the
    /// top bit is sign. Matches mlx-lm's `gpt_oss` dequant table.
    static let lut: [Float] = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    /// e8m0 exponent bias.
    static let e8m0Bias = 127
}

/// The α coefficient of GPT-OSS's swish gate (`gate · sigmoid(α·gate)`).
private let gptOSSSwishAlpha: Float = 1.702

// ─── buildGPTOSSMoE — load + transcode the MoE block ─────────────────

/// Build one layer's GPT-OSS MoE feed-forward block. Reads the biased
/// router and the three MXFP4-packed expert projections, transcodes
/// each expert to affine int4, and wires the `GPTOSSExpert` list.
func buildGPTOSSMoE(
    prefix p: String, weights: SafeTensorsBundle, quantMap: GPTOSSQuantMap,
    hidden: Int, intermediate: Int,
    numExperts: Int, topK: Int, swigluLimit: Float,
    dtype: DType, device: Device
) throws -> GPTOSSMoELayer {
    // Router: hidden → numExperts logits. Carries a `.bias` projection
    // bias and is affine-quantized on the published checkpoints (the
    // experts are the only MXFP4 tensors).
    let router: GPTOSSBiasedLinear = {
        let inner: AnyLinear
        if let q = quantMap.config(for: "\(p).router"),
            weights.isQuantized("\(p).router"),
            let t = try? weights.quantizedTriplet("\(p).router")
        {
            inner = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: q.bits, groupSize: q.groupSize))
        } else {
            inner = AnyLinear(
                Linear(
                    weight: (try? weights.tensor(named: "\(p).router.weight"))!))
        }
        let bias =
            weights.has("\(p).router.bias")
            ? castGPTOSSTensor(
                try! weights.tensor(named: "\(p).router.bias"),
                to: dtype, device: device)
            : nil
        return GPTOSSBiasedLinear(linear: inner, bias: bias)
    }()

    // Expert projections. The checkpoint ships them stacked over the
    // expert axis: gate_proj / up_proj are [E, intermediate, hidden],
    // down_proj is [E, hidden, intermediate].
    let gate = try transcodeStackedExperts(
        base: "\(p).experts.gate_proj", in: weights,
        numExperts: numExperts, outDim: intermediate, inDim: hidden,
        dtype: dtype, device: device)
    let up = try transcodeStackedExperts(
        base: "\(p).experts.up_proj", in: weights,
        numExperts: numExperts, outDim: intermediate, inDim: hidden,
        dtype: dtype, device: device)
    let down = try transcodeStackedExperts(
        base: "\(p).experts.down_proj", in: weights,
        numExperts: numExperts, outDim: hidden, inDim: intermediate,
        dtype: dtype, device: device)

    var experts: [GPTOSSExpert] = []
    experts.reserveCapacity(numExperts)
    for e in 0 ..< numExperts {
        experts.append(
            GPTOSSExpert(
                gateProj: gate[e], upProj: up[e], downProj: down[e]))
    }

    return GPTOSSMoELayer(
        router: router, experts: experts,
        topK: topK, hidden: hidden, swigluLimit: swigluLimit,
        dtype: dtype)
}

// ─── transcodeStackedExperts — MXFP4 → affine int4 ───────────────────
//
// The transcode fits a PER-GROUP affine grid: for each 32-value MXFP4
// group it finds the group's min / max dequantized value and lays a
// 16-level uniform grid across exactly that range —
//
//   affine scale = (groupMax − groupMin) / 15
//   affine bias  = groupMin
//   affine code  = round((LUT[code]·mxScale − bias) / scale)
//
// A per-group fit is materially more accurate than a fixed full-range
// (±6) grid: most trained-weight groups occupy a small sub-range, and
// the fixed grid would waste most of its 16 levels on the unused
// extremes — coarse near-zero quantization that flattens the MoE
// output enough to push greedy decode into repetition loops.
//
// The per-group fit is kept fast (the transcode covers ~20B codes) by
// scanning each group's 16 packed bytes through the 16-entry MXFP4 LUT
// with `while` loops over raw pointers — a debug build does NOT inline
// `Range`'s `formIndex` iterator witness, which otherwise dominates.
// The e8m0 microscale is looked up from a 256-entry `2^(e8m0−127)`
// table so no `exp2` runs in the hot loop.

/// `2^(e8m0 − 127)` for every possible e8m0 byte — the MXFP4 group
/// microscale. Precomputed so the transcode never calls `exp2`.
private let mxfp4MicroScaleForByte: [Float] = {
    (0 ..< 256).map { e8m0 in
        Float(Foundation.exp2(Double(e8m0 - MXFP4.e8m0Bias)))
    }
}()

/// A transcoded affine-int4 expert projection: the `QuantizedLinear`
/// triplet plus the per-output-row bias read straight from the
/// checkpoint.
struct GPTOSSExpertProjection {
    let linear: QuantizedLinear
    /// Per-output-row bias `[outDim]`, in the activation dtype.
    let bias: Tensor
}

/// Transcode a stacked `[E, outDim, inDim]` MXFP4 expert tensor into
/// `E` per-expert affine-int4 `QuantizedLinear`s (+ their biases).
private func transcodeStackedExperts(
    base: String, in weights: SafeTensorsBundle,
    numExperts: Int, outDim: Int, inDim: Int,
    dtype: DType, device: Device
) throws -> [GPTOSSExpertProjection] {
    let packed = try weights.tensor(named: "\(base).weight")  // u32
    let scales = try weights.tensor(named: "\(base).scales")  // u8
    let biases = try weights.tensor(named: "\(base).bias")  // f16

    precondition(
        packed.dtype == .u32,
        "GPT-OSS MoE: \(base).weight expected u32, got \(packed.dtype)")
    precondition(
        scales.dtype == .u8,
        "GPT-OSS MoE: \(base).scales expected u8, got \(scales.dtype)")
    precondition(
        inDim % MXFP4.groupSize == 0,
        "GPT-OSS MoE: inDim \(inDim) must be a multiple of "
            + "MXFP4 group size \(MXFP4.groupSize)")

    let wordsPerRow = inDim / MXFP4.codesPerWord  // packed u32 / output row
    let groupsPerRow = inDim / MXFP4.groupSize  // scale bytes / output row
    let bytesPerRow = wordsPerRow * 4  // packed bytes / output row
    precondition(
        packed.elementCount == numExperts * outDim * wordsPerRow,
        "GPT-OSS MoE: \(base).weight count mismatch — got \(packed.shape)")
    precondition(
        scales.elementCount == numExperts * outDim * groupsPerRow,
        "GPT-OSS MoE: \(base).scales count mismatch — got \(scales.shape)")

    let biasFloats = readGPTOSSFloats(biases)
    precondition(
        biasFloats.count == numExperts * outDim,
        "GPT-OSS MoE: \(base).bias count mismatch")

    let rowWords = wordsPerRow
    let rowGroups = groupsPerRow
    // The affine group size equals the MXFP4 group size (32) and both
    // pack 8 codes per u32, so the dst word/byte layout matches the src
    // 1:1 — the transcode is a pure nibble remap, no repacking.

    // Pre-allocate every expert's device-resident affine triplet so the
    // transcode writes straight into GPU memory. Experts are
    // independent — the outer loop parallelizes across cores.
    let weightTs = (0 ..< numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowWords], dtype: .u32, device: device)
    }
    let scaleTs = (0 ..< numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowGroups], dtype: dtype, device: device)
    }
    let biasTs = (0 ..< numExperts).map { _ in
        Tensor.empty(shape: [outDim, rowGroups], dtype: dtype, device: device)
    }

    // Bytes per MXFP4 group: 32 codes / 2 codes-per-byte = 16.
    let bytesPerGroup = MXFP4.groupSize / 2

    DispatchQueue.concurrentPerform(iterations: numExperts) { e in
        // Pointer views derived inside the closure — `Tensor` is
        // `@unchecked Sendable`, raw pointers are not. The packed
        // source is read-only; each expert's destinations are disjoint.
        // The src/dst weight word layouts are identical (the remap is
        // nibble-for-nibble in place). The hot loops are `while` loops
        // over raw pointers: a debug build does NOT inline `Range`'s
        // `formIndex` iterator witness, which dominates a `for i in
        // 0..<n` over a ~20-element-billion transcode.
        let weightBytesPerExpert = outDim * bytesPerRow
        let groupsPerExpert = outDim * rowGroups
        let packedBytes = packed.buffer.contents()
            .advanced(by: packed.offset + e * weightBytesPerExpert)
            .assumingMemoryBound(to: UInt8.self)
        let scaleHost = scales.buffer.contents()
            .advanced(by: scales.offset + e * groupsPerExpert)
            .assumingMemoryBound(to: UInt8.self)
        let wDstBytes = weightTs[e].buffer.contents()
            .advanced(by: weightTs[e].offset)
            .assumingMemoryBound(to: UInt8.self)
        let sDst = scaleTs[e].buffer.contents().advanced(by: scaleTs[e].offset)
        let bDst = biasTs[e].buffer.contents().advanced(by: biasTs[e].offset)

        MXFP4.lut.withUnsafeBufferPointer { lut in
            mxfp4MicroScaleForByte.withUnsafeBufferPointer { microTbl in
                // Per-group affine fit. `g` indexes groups; each group is
                // `bytesPerGroup` packed bytes (= 32 codes) and one e8m0
                // scale byte. Output scale/bias dtype branches once.
                let dt = dtype
                var g = 0
                while g < groupsPerExpert {
                    let mxScale = microTbl[Int(scaleHost[g])]
                    let byteBase = g * bytesPerGroup

                    // Pass 1: min / max of the group's LUT values.
                    var lo: Float = 1e30
                    var hi: Float = -1e30
                    var b = 0
                    while b < bytesPerGroup {
                        let byte = packedBytes[byteBase + b]
                        let v0 = lut[Int(byte & 0x0F)]
                        let v1 = lut[Int(byte >> 4)]
                        if v0 < lo { lo = v0 }
                        if v0 > hi { hi = v0 }
                        if v1 < lo { lo = v1 }
                        if v1 > hi { hi = v1 }
                        b &+= 1
                    }
                    // Affine grid over [lo, hi]·mxScale, 16 levels.
                    let affBias = lo * mxScale
                    let span = (hi - lo) * mxScale
                    let affScale = span > 0 ? span / 15.0 : 0
                    let invScale: Float = affScale > 0 ? 1.0 / affScale : 0

                    // Write scale / bias in the activation dtype.
                    switch dt {
                    case .f16:
                        sDst.assumingMemoryBound(to: Float16.self)[g] =
                            Float16(affScale)
                        bDst.assumingMemoryBound(to: Float16.self)[g] =
                            Float16(affBias)
                    case .bf16:
                        sDst.assumingMemoryBound(to: UInt16.self)[g] =
                            bf16Bits(affScale)
                        bDst.assumingMemoryBound(to: UInt16.self)[g] =
                            bf16Bits(affBias)
                    case .f32:
                        sDst.assumingMemoryBound(to: Float.self)[g] = affScale
                        bDst.assumingMemoryBound(to: Float.self)[g] = affBias
                    default:
                        fatalError("GPT-OSS MoE: unsupported dtype \(dt)")
                    }

                    // Pass 2: remap each code to its affine int4 quantum.
                    // affine code  q = round((LUT[code]·mxScale − affBias)
                    //                         / affScale)
                    //                = round((LUT[code] − lo)·mxScale·invScale)
                    let codeScale = mxScale * invScale
                    b = 0
                    while b < bytesPerGroup {
                        let byte = packedBytes[byteBase + b]
                        var q0 = Int(
                            ((lut[Int(byte & 0x0F)] - lo) * codeScale)
                                .rounded())
                        var q1 = Int(
                            ((lut[Int(byte >> 4)] - lo) * codeScale)
                                .rounded())
                        if q0 < 0 { q0 = 0 }
                        if q0 > 15 { q0 = 15 }
                        if q1 < 0 { q1 = 0 }
                        if q1 > 15 { q1 = 15 }
                        wDstBytes[byteBase + b] = UInt8(q0 | (q1 << 4))
                        b &+= 1
                    }
                    g &+= 1
                }
            }
        }
    }

    var out: [GPTOSSExpertProjection] = []
    out.reserveCapacity(numExperts)
    for e in 0 ..< numExperts {
        let qlinear = QuantizedLinear(
            weight: weightTs[e], scales: scaleTs[e], biases: biasTs[e],
            bits: 4, groupSize: MXFP4.groupSize)

        // Per-output-row expert bias, in the activation dtype.
        let rowBias = Tensor.empty(shape: [outDim], dtype: dtype, device: device)
        writeGPTOSSFloats(
            Array(biasFloats[(e * outDim) ..< ((e + 1) * outDim)]),
            into: rowBias)

        out.append(GPTOSSExpertProjection(linear: qlinear, bias: rowBias))
    }
    return out
}

/// Round a `Float` to its bf16 bit pattern (round-to-nearest-even on
/// the truncated low 16 bits). Used by the transcode's direct-to-tensor
/// scale/bias writes.
@inline(__always)
private func bf16Bits(_ v: Float) -> UInt16 {
    let bits = v.bitPattern
    let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
    return UInt16(rounded >> 16)
}

// ─── GPTOSSExpert — one clipped-SwiGLU expert ────────────────────────

/// A single MoE expert: the three affine-int4 projections + their
/// per-output-row biases. The clipped-SwiGLU activation runs host-side
/// (see the file header) so the expert exposes the GEMVs separately.
public final class GPTOSSExpert: Module {
    let gateProj, upProj, downProj: GPTOSSExpertProjection

    init(
        gateProj: GPTOSSExpertProjection,
        upProj: GPTOSSExpertProjection,
        downProj: GPTOSSExpertProjection
    ) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in gateProj.linear.parameters() {
            out.append(("gate_proj.\(k)", v))
        }
        out.append(("gate_proj.bias", gateProj.bias))
        for (k, v) in upProj.linear.parameters() {
            out.append(("up_proj.\(k)", v))
        }
        out.append(("up_proj.bias", upProj.bias))
        for (k, v) in downProj.linear.parameters() {
            out.append(("down_proj.\(k)", v))
        }
        out.append(("down_proj.bias", downProj.bias))
        return out
    }
}

// ─── GPTOSSMoELayer — the block-sparse MoE feed-forward layer ────────
//
// `decode` runs: router GEMV → CPU top-K + softmax → per selected
// expert {gate/up GEMV → host clipped-SwiGLU → down GEMV} → combine.
//
// IMPORTANT — command-buffer contract. `decode` commits the passed
// `cmd` (the router needs the logits on the CPU, and the per-expert
// activation is host-side). The enclosing `GPTOSSLayer` obtains a fresh
// buffer afterwards. Mirrors `MoELayer`'s contract.

public final class GPTOSSMoELayer: Module {
    public let router: GPTOSSBiasedLinear
    public let experts: [GPTOSSExpert]
    public let topK: Int
    public let hidden: Int
    public let swigluLimit: Float
    public let dtype: DType

    init(
        router: GPTOSSBiasedLinear, experts: [GPTOSSExpert],
        topK: Int, hidden: Int, swigluLimit: Float, dtype: DType
    ) {
        precondition(
            topK > 0 && topK <= experts.count,
            "GPTOSSMoELayer: topK \(topK) out of range "
                + "1…\(experts.count)")
        self.router = router
        self.experts = experts
        self.topK = topK
        self.hidden = hidden
        self.swigluLimit = swigluLimit
        self.dtype = dtype
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in router.parameters() { out.append(("router.\(k)", v)) }
        for (e, expert) in experts.enumerated() {
            for (k, v) in expert.parameters() {
                out.append(("experts.\(e).\(k)", v))
            }
        }
        return out
    }

    /// Single-token MoE forward. Commits `cmd` (router CPU readback)
    /// and returns a fully-resident `[hidden]` tensor produced on a
    /// fresh, locally-committed buffer.
    func decode(_ x: Tensor, cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(
            x.elementCount == hidden,
            "GPTOSSMoELayer.decode: input has \(x.elementCount) "
                + "elements, expected hidden \(hidden)")

        // ── Router GEMV on the caller's buffer, then CPU sync ─────────
        let logitsTensor = router(x, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── CPU routing — top-K of raw logits, softmax over the K ─────
        // GPT-OSS routes top-K of the raw router logits then softmaxes
        // just those K (mlx-lm's `topK` then `softmax`).
        let logits = logitsTensor.toFloatArray()
        let order = (0 ..< logits.count).sorted { a, b in
            if logits[a] != logits[b] { return logits[a] > logits[b] }
            return a < b
        }
        let idx = Array(order.prefix(topK))
        let pickedLogits = idx.map { logits[$0] }
        let combineWeights = softmaxSmall(pickedLogits)

        // ── Per-expert clipped-SwiGLU ─────────────────────────────────
        // Each `runExpert` runs on its own command buffers (it commits
        // mid-way for the host-side activation) and returns a resident
        // tensor. The weighted combine then runs on one fresh buffer.
        var expertOuts: [Tensor] = []
        expertOuts.reserveCapacity(idx.count)
        for expertId in idx {
            expertOuts.append(
                runExpert(
                    experts[expertId], x: x,
                    device: device))
        }

        let work = device.makeCommandBuffer()
        var accumulator: Tensor?
        for (slot, expertOut) in expertOuts.enumerated() {
            let weightTensor = Tensor.filled(
                combineWeights[slot],
                shape: [hidden], dtype: dtype,
                device: device)
            let scaled = Ops.mul(expertOut, weightTensor, on: work)
            accumulator =
                accumulator.map { Ops.add($0, scaled, on: work) }
                ?? scaled
        }
        let result = accumulator!  // topK ≥ 1
        work.commit()
        work.waitUntilCompleted()
        return result
    }

    /// Run one expert: gate/up GEMVs on the GPU, the clipped-SwiGLU
    /// activation on the host, the down GEMV on the GPU. The clip has
    /// no GPU op in FFAI and the activation vector is small, so the
    /// host fold is the simplest correct path (see the file header).
    /// Owns its command buffers; returns a fully-resident tensor.
    private func runExpert(
        _ expert: GPTOSSExpert, x: Tensor,
        device: Device
    ) -> Tensor {
        // ── GPU phase 1: gate / up GEMVs (+ per-row expert bias) ──────
        let cmd = device.makeCommandBuffer()
        let gate = Ops.add(
            expert.gateProj.linear(x, on: cmd),
            expert.gateProj.bias, on: cmd)
        let up = Ops.add(
            expert.upProj.linear(x, on: cmd),
            expert.upProj.bias, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()

        // ── Host phase: clipped-SwiGLU ────────────────────────────────
        //   gate = clip(gate, max: limit)
        //   up   = clip(up, -limit … limit)
        //   glu  = gate · sigmoid(α·gate)
        //   act  = glu · (up + 1)
        let gateHost = readGPTOSSFloats(gate)
        let upHost = readGPTOSSFloats(up)
        let limit = swigluLimit
        var act = [Float](repeating: 0, count: gateHost.count)
        for i in 0 ..< gateHost.count {
            var g = gateHost[i]
            var u = upHost[i]
            if g > limit { g = limit }
            if u > limit { u = limit }
            if u < -limit { u = -limit }
            let glu = g / (1.0 + Foundation.exp(-gptOSSSwishAlpha * g))
            act[i] = glu * (u + 1.0)
        }

        // ── GPU phase 2: down GEMV (+ per-row bias) on a fresh buffer ─
        let phase2 = device.makeCommandBuffer()
        let actTensor = Tensor.empty(
            shape: [act.count], dtype: dtype,
            device: device)
        writeGPTOSSFloats(act, into: actTensor)
        let down = Ops.add(
            expert.downProj.linear(actTensor, on: phase2),
            expert.downProj.bias, on: phase2)
        phase2.commit()
        phase2.waitUntilCompleted()
        return down
    }
}

/// Numerically-stable softmax over a small host vector.
private func softmaxSmall(_ x: [Float]) -> [Float] {
    guard let maxV = x.max() else { return [] }
    let exps = x.map { Foundation.exp($0 - maxV) }
    let sum = exps.reduce(0, +)
    guard sum > 0 else {
        return [Float](repeating: 1 / Float(x.count), count: x.count)
    }
    return exps.map { $0 / sum }
}
