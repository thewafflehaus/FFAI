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
// Gemma 3 text — concrete variants + the dense decoder for the Gemma 3
// family. The family enum (`enum Gemma3`), variant protocol
// (`Gemma3Variant`), and error type (`Gemma3Error`) live in
// `Models/Gemma3.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `Gemma3Dense` — `Gemma3Variant` conformance + the per-variant
//     `loadModel` entry,
//   • `Gemma3Model` — the full LanguageModel decoder.
//
// The smallest Gemma 3 ships as a 1B variant; 4B/12B/27B exist for the
// VL track.
//
// Architectural differences from the 3-series Llama backbone we
// already ship:
//
//   1. **GemmaRMSNorm** — the norm computation is
//      `rmsNorm(x, weight: 1 + weight, eps)`. We fold the `+1.0` into
//      the loaded weight at construction time so the existing
//      `RMSNorm` / `Ops.rmsNorm{Rows,}` kernels apply unchanged.
//
//   2. **Four per-block norms** (vs Llama's two): input → attn →
//      post_attention_norm → (+residual) → pre_feedforward_norm → MLP
//      → post_feedforward_norm → (+residual).
//
//   3. **Per-head q_norm / k_norm** (head-dim RMSNorm) sit between the
//      Q/K projection and RoPE — separate from input_layernorm.
//
//   4. **Alternating RoPE base per layer.** Every
//      `sliding_window_pattern`-th layer (the 0-indexed `i` such that
//      `(i + 1) % pattern == 0`) is a global-attention layer with
//      `rope_theta = 1e6`; the others are sliding-window layers with
//      `rope_local_base_freq = 10_000`.
//
//   5. **Per-layer sliding-window KV cache.** Sliding layers cap the
//      KV at `sliding_window` positions (FIFO eviction); global layers
//      stay unbounded.
//
//   6. **queryPreAttnScalar.** Q is scaled by
//      `1 / sqrt(query_pre_attn_scalar)` (default 256) instead of the
//      usual `1 / sqrt(head_dim)`. We pass the scale through to
//      `Ops.sdpaDecode`.
//
//   7. **GELU MLP** (geluApproximate(gate) * up → down), not SwiGLU.
//
//   8. **Embedding scale.** The embedded token row is multiplied by
//      `sqrt(hidden_size)` at the start of each forward pass —
//      Google's original Gemma normalization.
//
//   9. **Optional final-logit soft-cap.** Some 1B revisions ship with
//      `final_logit_softcapping = 30.0`; logits are passed through
//      `softcap · tanh(logits / softcap)`. Deferred — the 1B-it bf16
//      checkpoint we test against has it absent.
//
// First-light scope: text-only generation. No VL, no PLE (Gemma 4
// territory), no fused attention-sinks. Sliding window for the
// sliding layers uses the same `.window(maxSize:)` eviction we ship
// for every other cache type; global layers use `.unbounded`.

import Foundation
import Metal

// MARK: - Gemma3Dense — 1B text decoder

public struct Gemma3Dense: Gemma3Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256, prefillStepSize: 1024,
        temperature: 1.0, topP: 0.95, topK: 64,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma3Model {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let intermediate = config.intermediateSize,
            let eps = config.rmsNormEps,
            let vocab = config.vocabSize
        else {
            throw Gemma3Error.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // Gemma 3 declares head_dim explicitly (often 256 for 1B+).
        let headDim = config.headDim ?? (hidden / nHeads)
        let maxSeq = config.int("max_position_embeddings") ?? 32768
        let slidingWindow = config.int("sliding_window") ?? 512
        let slidingWindowPattern = config.int("sliding_window_pattern") ?? 6
        let queryPreAttnScalar = Float(config.float("query_pre_attn_scalar") ?? Double(headDim))
        let ropeTheta = Float(config.ropeTheta ?? 1_000_000)
        let ropeLocalBaseFreq = Float(config.float("rope_local_base_freq") ?? 10_000.0)
        let tieEmbed = config.tieWordEmbeddings
        let quant = config.quantization

        // Load the embedding (Gemma 3 1B has vocab=262144, hidden=1152;
        // the standard quant-aware loader handles both raw + 4-bit).
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        var layers: [Gemma3Layer] = []
        layers.reserveCapacity(nLayers)

        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            // Layer index (i+1) % pattern == 0 is a global-attention
            // layer; the others are sliding (per mlx-swift-lm + the
            // HF reference impl).
            let isSliding = (i + 1) % slidingWindowPattern != 0
            let layerRopeTheta = isSliding ? ropeLocalBaseFreq : ropeTheta

            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj",
                in: weights, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj",
                in: weights, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj",
                in: weights, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj",
                in: weights, quantization: quant)
            let gateProj = try loadLinear(
                base: "\(p).mlp.gate_proj",
                in: weights, quantization: quant)
            let upProj = try loadLinear(
                base: "\(p).mlp.up_proj",
                in: weights, quantization: quant)
            let downProj = try loadLinear(
                base: "\(p).mlp.down_proj",
                in: weights, quantization: quant)

            let inputNorm = try loadGemmaRMSNorm(
                base: "\(p).input_layernorm.weight", in: weights, eps: eps)
            let postAttnNorm = try loadGemmaRMSNorm(
                base: "\(p).post_attention_layernorm.weight", in: weights, eps: eps)
            let preFFNorm = try loadGemmaRMSNorm(
                base: "\(p).pre_feedforward_layernorm.weight", in: weights, eps: eps)
            let postFFNorm = try loadGemmaRMSNorm(
                base: "\(p).post_feedforward_layernorm.weight", in: weights, eps: eps)
            let qNorm = try loadGemmaRMSNorm(
                base: "\(p).self_attn.q_norm.weight", in: weights, eps: eps)
            let kNorm = try loadGemmaRMSNorm(
                base: "\(p).self_attn.k_norm.weight", in: weights, eps: eps)

            layers.append(
                Gemma3Layer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    preFFNorm: preFFNorm, postFFNorm: postFFNorm,
                    qNorm: qNorm, kNorm: kNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: layerRopeTheta,
                    queryPreAttnScalar: queryPreAttnScalar,
                    isSliding: isSliding
                ))
        }

        let finalNorm = try loadGemmaRMSNorm(
            base: "model.norm.weight", in: weights, eps: eps)

        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: bits, groupSize: q.groupSize
                ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        // Pre-build the [hidden]-wide sqrt(hidden) scale tensor.
        // Multiplying the embedded row by this tensor at the start of
        // every forward is the original Gemma normalization. Sized at
        // the activation dtype so Ops.mul shape/dtype-checks pass.
        let embedScale = Tensor.empty(shape: [hidden], dtype: activationDtype, device: device)
        fillScalar(embedScale, scalar: Float(Double(hidden).squareRoot()), dtype: activationDtype)

        return Gemma3Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            embedScale: embedScale,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxContextWindow: maxSeq, dtype: activationDtype,
            slidingWindow: slidingWindow,
            slidingWindowPattern: slidingWindowPattern,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction
        )
    }
}

/// Load a Gemma RMSNorm weight by folding the GemmaRMSNorm `+ 1.0`
/// offset into the loaded tensor up front, so the existing RMSNorm
/// kernel (which computes `x / rms(x) * weight`) is equivalent to
/// `rmsNorm(x, weight: 1 + raw_weight)` without any kernel change.
///
/// MTLBuffers in FFAI are allocated `storageModeShared`, so we read
/// the raw weight bytes into a fresh shared buffer with +1 baked in.
/// The original weight buffer stays in place but is no longer
/// referenced by this RMSNorm — its retain count drops to zero with
/// the SafeTensors file's `entries` map.
internal func loadGemmaRMSNorm(
    base: String, in weights: SafeTensorsBundle, eps: Double
) throws -> RMSNorm {
    let raw = try weights.tensor(named: base)
    precondition(raw.shape.count == 1, "Gemma3 RMSNorm weight must be 1D, got \(raw.shape)")
    let n = raw.elementCount

    let foldedBuf = Device.shared.makeBuffer(length: raw.byteCount)
    let folded = Tensor(buffer: foldedBuf, offset: 0, shape: raw.shape, dtype: raw.dtype)

    let dstPtr = foldedBuf.contents()
    let srcPtr = raw.buffer.contents().advanced(by: raw.offset)

    switch raw.dtype {
    case .f32:
        let src = srcPtr.bindMemory(to: Float.self, capacity: n)
        let dst = dstPtr.bindMemory(to: Float.self, capacity: n)
        for i in 0 ..< n { dst[i] = src[i] + 1.0 }
    case .f16:
        // f16 add via a single fp32 round-trip per element. n is the
        // hidden size (a few thousand), so this is one-shot cost.
        let src = srcPtr.bindMemory(to: UInt16.self, capacity: n)
        let dst = dstPtr.bindMemory(to: UInt16.self, capacity: n)
        for i in 0 ..< n {
            let f = halfBitsToFloat(src[i]) + 1.0
            dst[i] = floatToHalfBits(f)
        }
    case .bf16:
        let src = srcPtr.bindMemory(to: UInt16.self, capacity: n)
        let dst = dstPtr.bindMemory(to: UInt16.self, capacity: n)
        for i in 0 ..< n {
            let f = bf16BitsToFloat(src[i]) + 1.0
            dst[i] = floatToBf16Bits(f)
        }
    default:
        fatalError("Gemma3 RMSNorm: unsupported weight dtype \(raw.dtype)")
    }

    return RMSNorm(weight: folded, eps: Float(eps))
}

/// Fill a flat [n] tensor with a single scalar value at runtime.
/// Used once at load to materialize the sqrt(hidden) embedding scale.
internal func fillScalar(_ t: Tensor, scalar: Float, dtype: DType) {
    let n = t.elementCount
    let ptr = t.buffer.contents().advanced(by: t.offset)
    switch dtype {
    case .f32:
        let p = ptr.bindMemory(to: Float.self, capacity: n)
        for i in 0 ..< n { p[i] = scalar }
    case .f16:
        let p = ptr.bindMemory(to: UInt16.self, capacity: n)
        let h = floatToHalfBits(scalar)
        for i in 0 ..< n { p[i] = h }
    case .bf16:
        let p = ptr.bindMemory(to: UInt16.self, capacity: n)
        let h = floatToBf16Bits(scalar)
        for i in 0 ..< n { p[i] = h }
    default:
        fatalError("Gemma3: unsupported fillScalar dtype \(dtype)")
    }
}

// IEEE-754 half / bf16 round-trip helpers — minimal, no SIMD, just
// enough to fold the Gemma RMSNorm +1.0 once at load time.
//
// half (f16): 1 sign, 5 exponent, 10 mantissa.
// bf16:       1 sign, 8 exponent, 7  mantissa (truncated f32).

private func halfBitsToFloat(_ bits: UInt16) -> Float {
    let sign = UInt32(bits >> 15) << 31
    let exp16 = UInt32((bits >> 10) & 0x1F)
    let frac16 = UInt32(bits & 0x3FF)
    if exp16 == 0 {
        if frac16 == 0 {
            return Float(bitPattern: sign)
        }
        // Subnormal — renormalize.
        var e: UInt32 = 0
        var m = frac16
        while (m & 0x400) == 0 {
            m <<= 1
            e += 1
        }
        m &= 0x3FF
        let exp32 = 127 - 15 - e + 1
        return Float(bitPattern: sign | (exp32 << 23) | (m << 13))
    }
    if exp16 == 0x1F {
        return Float(bitPattern: sign | 0x7F80_0000 | (frac16 << 13))
    }
    let exp32 = exp16 + (127 - 15)
    return Float(bitPattern: sign | (exp32 << 23) | (frac16 << 13))
}

private func floatToHalfBits(_ f: Float) -> UInt16 {
    let b = f.bitPattern
    let sign = UInt16((b >> 16) & 0x8000)
    let exp32 = Int((b >> 23) & 0xFF) - 127
    let frac = b & 0x7FFFFF
    if exp32 > 15 {
        return sign | 0x7C00  // inf
    }
    if exp32 < -14 {
        // Subnormal or zero — flush to zero (rare in production weights).
        return sign
    }
    let exp16 = UInt16(exp32 + 15) << 10
    let frac16 = UInt16(frac >> 13)
    return sign | exp16 | frac16
}

private func bf16BitsToFloat(_ bits: UInt16) -> Float {
    return Float(bitPattern: UInt32(bits) << 16)
}

private func floatToBf16Bits(_ f: Float) -> UInt16 {
    // Round-to-nearest-even.
    let b = f.bitPattern
    let rounded = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return UInt16(truncatingIfNeeded: rounded)
}

// ─── Test-only re-exports ────────────────────────────────────────────
//
// The bf16 / f16 conversion helpers above are file-private, but the
// Gemma 3 fold + embed-scale path is the most likely first-light bug
// site. Expose them via `…ForTest` shims so Tests/ can pin the
// round-trip math without forcing the implementation into the public
// FFAI surface.
//
// Keep these strictly testing-related — they're not part of the
// library's exported API and may move when the proper f16 / bf16
// dtype helpers consolidate.

public func bf16BitsToFloatForTest(_ bits: UInt16) -> Float {
    bf16BitsToFloat(bits)
}
public func floatToBf16BitsForTest(_ f: Float) -> UInt16 {
    floatToBf16Bits(f)
}
public func halfBitsToFloatForTest(_ bits: UInt16) -> Float {
    halfBitsToFloat(bits)
}
public func floatToHalfBitsForTest(_ f: Float) -> UInt16 {
    floatToHalfBits(f)
}
public func fillScalarForTest(_ t: Tensor, scalar: Float, dtype: DType) {
    fillScalar(t, scalar: scalar, dtype: dtype)
}
/// Mirrors the dtype-dispatch loop inside `loadGemmaRMSNorm` so tests
/// can verify the +1 fold round-trip against a synthetic input buffer
/// without going through SafeTensors. Returns a fresh shared-storage
/// MTLBuffer holding the folded values.
public func gemmaFoldRMSNormForTest(
    inputBuf: MTLBuffer, count n: Int, dtype: DType, device: Device
) -> MTLBuffer {
    let bytes = n * dtype.byteSize
    let outBuf = device.makeBuffer(length: bytes)
    let srcPtr = inputBuf.contents()
    let dstPtr = outBuf.contents()
    switch dtype {
    case .f32:
        let src = srcPtr.bindMemory(to: Float.self, capacity: n)
        let dst = dstPtr.bindMemory(to: Float.self, capacity: n)
        for i in 0 ..< n { dst[i] = src[i] + 1.0 }
    case .f16:
        let src = srcPtr.bindMemory(to: UInt16.self, capacity: n)
        let dst = dstPtr.bindMemory(to: UInt16.self, capacity: n)
        for i in 0 ..< n {
            dst[i] = floatToHalfBits(halfBitsToFloat(src[i]) + 1.0)
        }
    case .bf16:
        let src = srcPtr.bindMemory(to: UInt16.self, capacity: n)
        let dst = dstPtr.bindMemory(to: UInt16.self, capacity: n)
        for i in 0 ..< n {
            dst[i] = floatToBf16Bits(bf16BitsToFloat(src[i]) + 1.0)
        }
    default:
        fatalError("gemmaFoldRMSNormForTest: unsupported dtype \(dtype)")
    }
    return outBuf
}

// MARK: - Gemma3Layer

public final class Gemma3Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm, preFFNorm, postFFNorm: RMSNorm
    let qNorm, kNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    /// SDPA scale, derived once at construction from queryPreAttnScalar.
    let scale: Float
    /// Sliding vs full attention. Drives per-layer KV cache choice in
    /// Gemma3Model.makeLayerCaches.
    public let isSliding: Bool

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        preFFNorm: RMSNorm, postFFNorm: RMSNorm,
        qNorm: RMSNorm, kNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        intermediate: Int, ropeTheta: Float,
        queryPreAttnScalar: Float, isSliding: Bool
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
        self.preFFNorm = preFFNorm
        self.postFFNorm = postFFNorm
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.isSliding = isSliding
        // mlx-swift-lm: scale = pow(queryPreAttnScalar, -0.5) = 1/sqrt(scalar)
        self.scale = 1.0 / Float(Double(queryPreAttnScalar).squareRoot())
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in qProj.parameters() { out.append(("self_attn.q_proj.\(k)", v)) }
        for (k, v) in kProj.parameters() { out.append(("self_attn.k_proj.\(k)", v)) }
        for (k, v) in vProj.parameters() { out.append(("self_attn.v_proj.\(k)", v)) }
        for (k, v) in oProj.parameters() { out.append(("self_attn.o_proj.\(k)", v)) }
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        for (k, v) in preFFNorm.parameters() { out.append(("pre_feedforward_layernorm.\(k)", v)) }
        for (k, v) in postFFNorm.parameters() { out.append(("post_feedforward_layernorm.\(k)", v)) }
        for (k, v) in qNorm.parameters() { out.append(("self_attn.q_norm.\(k)", v)) }
        for (k, v) in kNorm.parameters() { out.append(("self_attn.k_norm.\(k)", v)) }
        return out
    }

    /// Single-token forward. `position` is the absolute sequence
    /// index of this token (used for RoPE).
    ///
    /// Debug-time intermediate-value dumps live at layer
    /// *boundaries* (h_in, h_out per layer) and are wired in
    /// `Gemma3Model.forward(...)` via `InspectTap`, NOT here. The
    /// layer's hot path stays bespoke-debug-free; first-light
    /// triage happens at `ffai inspect --layer-trace`.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Pre-attn norm.
        let xNorm = inputNorm(h, on: cmd)

        // QKV projections.
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head q_norm / k_norm sit between projection and RoPE.
        // q has shape [nHeads * headDim], rmsNormRows applies the
        // [headDim]-wide weight per row.
        let qNorm2D = Ops.rmsNormRows(
            q, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nHeads, headDim])
        let kNorm2D = Ops.rmsNormRows(
            k, weight: kNorm.weight, eps: kNorm.eps,
            nRows: nKVHeads, rowSize: headDim, on: cmd
        ).reshaped(to: [nKVHeads, headDim])

        // RoPE on the normed q and k.
        let qRotated = Ops.rope(
            qNorm2D,
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(
            kNorm2D,
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)

        // Append + SDPA.
        cache.appendOnGPU(
            kFlat: kRotated,
            vFlat: v.reshaped(to: [nKVHeads, headDim]),
            on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        // Sliding layers run a `.window` KV cache; the helper derives
        // the kernel's sink/window fast-path bounds from the eviction
        // policy (FFAI's ring buffer keeps live data contiguous, so
        // this is (0, 0) today — see KVCacheProtocol.sdpaSinkWindow).
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd,
            sinkEnd: sinkEnd, windowStart: windowStart)

        // o_proj → post_attention_layernorm → +residual.
        // Gemma 3 normalises oOut FIRST, then adds to the residual
        // (Gemma's "post-norm" placement — not the standard pre-norm
        // add+norm), so the residual add itself cannot be fused with
        // postAttnNorm. The downstream `h + normedAttn → preFFNorm`
        // pair IS the standard add+rmsNorm pattern, and we fuse that.
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let normedAttn = postAttnNorm(oOut, on: cmd)

        // Fused residual add + pre-FFN RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Gemma 3 27B (hidden 5376) falls through the
        // validator gate to the unfused path.
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, normedAttn, weight: preFFNorm.weight, eps: preFFNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postAttn = fused.residual
            mlpNorm = fused.normed
        } else {
            postAttn = Ops.add(h, normedAttn, on: cmd)
            mlpNorm = preFFNorm(postAttn, on: cmd)
        }

        // MLP path: pre_feedforward_layernorm → MLP → post_feedforward_layernorm → +residual.
        // Gemma 3 uses gelu(gate) * up → down (vs Llama's silu).
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let geluGate = Ops.gelu(gate, on: cmd)
        let mlpInner = Ops.mul(geluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        let normedMLP = postFFNorm(mlpOut, on: cmd)
        return Ops.add(postAttn, normedMLP, on: cmd)
    }
}

// MARK: - Gemma3Model

public final class Gemma3Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Gemma3Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    /// Pre-baked [hidden] tensor filled with sqrt(hidden_size). Multiplied
    /// into the embedded token row at the start of each forward pass
    /// (the original Gemma "embed_scale" normalization).
    public let embedScale: Tensor

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxContextWindow: Int
    public let dtype: DType

    /// Window size for sliding-attention layers (every layer that
    /// isn't a multiple of `slidingWindowPattern`).
    public let slidingWindow: Int
    public let slidingWindowPattern: Int

    public let kvCacheKind: KVCacheKind
    /// User-requested baseline eviction (if `.window`, it overrides our
    /// sliding-window heuristic and applies to every layer). The
    /// default `.unbounded` means: sliding layers get
    /// `.window(slidingWindow)`, global layers stay `.unbounded`.
    public let kvEviction: KVEviction

    init(
        embedTokens: AnyEmbedding, layers: [Gemma3Layer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        embedScale: Tensor,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxContextWindow: Int, dtype: DType,
        slidingWindow: Int, slidingWindowPattern: Int,
        kvCacheKind: KVCacheKind, kvEviction: KVEviction
    ) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.embedScale = embedScale
        self.hidden = hidden
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.vocab = vocab
        self.maxContextWindow = maxContextWindow
        self.dtype = dtype
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
    }

    public func parameters() -> [(String, Tensor)] {
        var out: [(String, Tensor)] = []
        for (k, v) in embedTokens.parameters() { out.append(("model.embed_tokens.\(k)", v)) }
        for (i, layer) in layers.enumerated() {
            for (k, v) in layer.parameters() {
                out.append(("model.layers.\(i).\(k)", v))
            }
        }
        for (k, v) in finalNorm.parameters() { out.append(("model.norm.\(k)", v)) }
        for (k, v) in lmHead.parameters() { out.append(("lm_head.\(k)", v)) }
        return out
    }

    /// Per-layer KV cache. Sliding layers use a windowed cache
    /// (`.window(slidingWindow)`), global layers stay unbounded.
    /// If the user explicitly passed `.window` via LoadOptions, that
    /// applies uniformly and overrides the heuristic.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxContextWindow
        var caches: [any LayerCacheProtocol] = []
        caches.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let layerEviction: KVEviction
            switch kvEviction {
            case .window:
                // User override — apply uniformly.
                layerEviction = kvEviction
            case .unbounded:
                // Heuristic: sliding layers cap at slidingWindow.
                layerEviction =
                    layers[i].isSliding
                    ? .window(maxSize: min(slidingWindow, cap), keep: 0)
                    : .unbounded
            }
            switch kvCacheKind {
            case .raw:
                caches.append(
                    KVCache(
                        nKVHeads: nKVHeads, headDim: headDim, contextLength: cap,
                        dtype: dtype, eviction: layerEviction, device: device
                    ))
            default:
                // First-light: only raw KV supported on Gemma 3. The
                // affine + AURA paths require a per-layer working
                // buffer wired through Gemma3Model's storage, which
                // we haven't built yet.
                preconditionFailure(
                    "Gemma3: only .raw KV cache supported today; got \(kvCacheKind)")
            }
        }
        return caches
    }

    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Debug-tap mode forces a *private* cmdbuf so per-op
        // commit+wait sync points don't double-commit the caller's
        // cmd. In production / non-debug mode, queue everything on
        // the caller's cmd as the protocol expects.
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        // Embed lookup + sqrt(hidden) scale. The scale buffer is
        // pre-baked at load time (`Gemma3Dense.loadModel`) and tied
        // to the activation dtype.
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let h0 = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        var h = Ops.mul(h0, embedScale, on: workCmd)
        workCmd = tap.dumpLayerBoundary(
            h, label: "embed*scale", layer: -1,
            cmd: workCmd, device: device)

        // Per-layer forward. Tap fires at the OUTPUT of each layer.
        // The first layer's input is the embed dump above; every
        // subsequent layer's input is the prior layer's output, so
        // we only need one dump per boundary.
        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                on: workCmd, device: device)
            workCmd = tap.dumpLayerBoundary(
                h, label: "layer_out", layer: i,
                cmd: workCmd, device: device)
        }
        h = finalNorm(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(
            h, label: "final_norm", layer: -1,
            cmd: workCmd, device: device)
        let logits = lmHead(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(
            logits, label: "logits", layer: -1,
            cmd: workCmd, device: device)

        // In tap-active mode the last tap committed + waited;
        // logits buffer holds valid data. The caller's cmd has no
        // work queued, so their commit() is a fast no-op.
        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    /// Multi-token forward — prefill fast path. Loops
    /// `forward(tokenId:)` per row on the supplied `cmd`.
    ///
    /// Gemma 3's chunked SDPA-collapse follow-up is more involved than
    /// Llama's because every layer mixes sliding-window vs full
    /// attention (the `isSliding` flag picks the per-layer
    /// `KVEviction.window` cache). Collapsing N sdpaDecode dispatches
    /// to one `sdpaMulti(causal: true)` per layer is straightforward
    /// for full-attention layers; sliding-window layers need their own
    /// chunk-aware path (the rotated K/V buffer's stale slots and
    /// `sdpaSinkWindow` bounds make a naïve sdpaMulti read past the
    /// window). Pending that per-layer split, this override is
    /// commit-count-batched only — same correctness as the protocol
    /// default, no new SDPA dispatch savings yet.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "Gemma3Model.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }

    /// Embedding-input forward — the VLM splice path. Identical to
    /// `forward(tokenId:...)` except the `[hidden]` embedding row is
    /// supplied directly (a vision-encoder token, or a text-token
    /// embedding the VL model looked up itself). The Gemma embed-scale
    /// is still applied: image tokens in Gemma 3 are scaled the same
    /// way as text tokens.
    public var supportsEmbeddingInput: Bool { true }

    public func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            inputEmbedding.elementCount == hidden,
            "Gemma3Model.forward(inputEmbedding:): expected [\(hidden)], "
                + "got \(inputEmbedding.shape)")
        var h = Ops.mul(inputEmbedding.reshaped(to: [hidden]), embedScale, on: cmd)
        for (i, layer) in layers.enumerated() {
            h = layer.forward(
                h, position: position,
                cache: caches[i] as! any KVCacheProtocol,
                on: cmd, device: device)
        }
        h = finalNorm(h, on: cmd)
        return lmHead(h, on: cmd)
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
