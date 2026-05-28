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
// Gemma 2 text — concrete variants + the dense decoder for the Gemma 2
// family. The family enum (`enum Gemma2`), variant protocol
// (`Gemma2Variant`), and error type (`Gemma2Error`) live in
// `Models/Gemma2.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `Gemma2Dense` — `Gemma2Variant` conformance + the per-variant
//     `loadModel` entry,
//   • `Gemma2Layer`, `Gemma2Model` — the per-layer + full-model impl
//     (alternating sliding-window / full attention, GemmaRMSNorm
//     `+1.0` fold, tied LM head, GELU-tanh MLP).

import Foundation
import Metal

// MARK: - Gemma2Dense — 2B / 9B / 27B text decoder

public struct Gemma2Dense: Gemma2Variant {
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
    ) throws -> Gemma2Model {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let intermediate = config.intermediateSize,
            let eps = config.rmsNormEps,
            let vocab = config.vocabSize
        else {
            throw Gemma2Error.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // Gemma 2 declares head_dim explicitly (256 for every published
        // variant). Same fallback as Llama / Gemma 3 for safety.
        let headDim = config.headDim ?? (hidden / nHeads)
        let maxSeq = config.int("max_position_embeddings") ?? 8192
        let slidingWindow = config.int("sliding_window") ?? 4096
        // Pattern 2 = alternating sliding / full each layer. HF's
        // `Gemma2Config` defaults to 2 and never overrides it; the
        // explicit fallback documents the intent.
        let slidingWindowPattern = config.int("sliding_window_pattern") ?? 2
        // Gemma 2's Q scale uses `query_pre_attn_scalar` (typically =
        // head_dim, but expressible separately) — same convention as
        // Gemma 3. `Ops.sdpaDecode` takes the final `1 / sqrt(scalar)`.
        let queryPreAttnScalar = Float(config.float("query_pre_attn_scalar") ?? Double(headDim))
        let ropeTheta = Float(config.ropeTheta ?? 10_000)
        // Soft-cap configuration is loaded for completeness even though
        // first-light skips both — documenting the gap is the
        // quality-conscious thing to do. Future kernel work can read
        // them off `Gemma2Model` directly.
        let attnLogitSoftcap = Float(config.float("attn_logit_softcapping") ?? 0)
        let finalLogitSoftcap = Float(config.float("final_logit_softcapping") ?? 0)
        let tieEmbed = config.tieWordEmbeddings
        let quant = config.quantization

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        var layers: [Gemma2Layer] = []
        layers.reserveCapacity(nLayers)

        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            // Same formula as Gemma 3 — see file header for why
            // pattern=2 reproduces HF's `not bool(layer_idx % 2)`.
            let isSliding = (i + 1) % slidingWindowPattern != 0

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

            layers.append(
                Gemma2Layer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    preFFNorm: preFFNorm, postFFNorm: postFFNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: ropeTheta,
                    queryPreAttnScalar: queryPreAttnScalar,
                    isSliding: isSliding
                ))
        }

        let finalNorm = try loadGemmaRMSNorm(
            base: "model.norm.weight", in: weights, eps: eps)

        // Gemma 2 ties the LM head with embed_tokens by default. The
        // raw bf16 conversion ships no `lm_head.weight`; mlx-community
        // 4-bit conversions also tie (no `lm_head` in the safetensors).
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

        // sqrt(hidden) embed-scale — original Gemma normalization.
        let embedScale = Tensor.empty(shape: [hidden], dtype: activationDtype, device: device)
        fillScalar(embedScale, scalar: Float(Double(hidden).squareRoot()), dtype: activationDtype)

        return Gemma2Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            embedScale: embedScale,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxContextWindow: maxSeq, dtype: activationDtype,
            slidingWindow: slidingWindow,
            slidingWindowPattern: slidingWindowPattern,
            attnLogitSoftcap: attnLogitSoftcap,
            finalLogitSoftcap: finalLogitSoftcap,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction
        )
    }
}

// MARK: - Gemma2Layer

public final class Gemma2Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm, preFFNorm, postFFNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    /// SDPA scale, `1 / sqrt(queryPreAttnScalar)`. Same convention as
    /// Gemma 3 — the scalar is usually `head_dim` but the config
    /// expresses it separately so unusual checkpoints can override.
    let scale: Float
    /// Sliding vs full attention. Drives per-layer KV cache choice in
    /// `Gemma2Model.makeLayerCaches`.
    public let isSliding: Bool

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        inputNorm: RMSNorm, postAttnNorm: RMSNorm,
        preFFNorm: RMSNorm, postFFNorm: RMSNorm,
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
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.isSliding = isSliding
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
        return out
    }

    /// Single-token forward. Same block schematic as Gemma 3 (input →
    /// QKV → RoPE → SDPA → o_proj → post_attn norm → +residual →
    /// pre_ff norm → gate*up → down → post_ff norm → +residual) MINUS
    /// the per-head q_norm / k_norm step Gemma 3 adds between
    /// projection and RoPE.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        // Pre-attn norm + QKV.
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // RoPE — straight from projection (no q_norm / k_norm
        // intermediate, unlike Gemma 3). Reshape into the
        // `[nHeads, headDim]` layout the kernel expects.
        let qHeads = q.reshaped(to: [nHeads, headDim])
        let kHeads = k.reshaped(to: [nKVHeads, headDim])
        let qRotated = Ops.rope(
            qHeads,
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(
            kHeads,
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)

        // Append + SDPA.
        cache.appendOnGPU(
            kFlat: kRotated,
            vFlat: v.reshaped(to: [nKVHeads, headDim]),
            on: cmd)
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        // `attn_logit_softcapping` is intentionally NOT applied here —
        // see file header. `sdpaSinkWindow` returns the sliding-window
        // bounds the kernel needs to skip stale slots; for the FFAI
        // ring-buffer cache this is `(0, 0)` and is a no-op fast path.
        let (sinkEnd, windowStart) = cache.sdpaSinkWindow(nKV: cache.length)
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd,
            sinkEnd: sinkEnd, windowStart: windowStart)

        // Gemma 2 keeps the same post-attn norm placement as Gemma 3
        // (o_proj → post_attn_norm → +residual). The normed-attn add
        // can't be fused with post_attn_norm; the downstream
        // `h + normedAttn → preFFNorm` pair is the standard
        // add+rmsNorm pattern and is fused below.
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)
        let normedAttn = postAttnNorm(oOut, on: cmd)

        // Fused residual + pre-FFN RMSNorm (validator gates by row size
        // — Gemma 2 27B has hidden=4608, well within the 4096 fused
        // path's cap, so this branch is hit for every published Gemma 2
        // size. Gemma 2 9B has hidden=3584 too — both fused).
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

        // MLP — same GELU(gate) * up → down pattern as Gemma 3.
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let geluGate = Ops.gelu(gate, on: cmd)
        let mlpInner = Ops.mul(geluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        let normedMLP = postFFNorm(mlpOut, on: cmd)
        return Ops.add(postAttn, normedMLP, on: cmd)
    }
}

// MARK: - Gemma2Model

public final class Gemma2Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Gemma2Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    /// Pre-baked [hidden] tensor filled with sqrt(hidden_size).
    /// Multiplied into the embedded row at the start of every forward
    /// (original Gemma "embed_scale" normalization).
    public let embedScale: Tensor

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxContextWindow: Int
    public let dtype: DType

    /// Sliding window size (4096 for every published Gemma 2 variant).
    public let slidingWindow: Int
    public let slidingWindowPattern: Int

    /// Loaded for completeness — see file header for why first-light
    /// skips both. Future kernel work can read these to enable soft-cap.
    public let attnLogitSoftcap: Float
    public let finalLogitSoftcap: Float

    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction

    init(
        embedTokens: AnyEmbedding, layers: [Gemma2Layer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        embedScale: Tensor,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxContextWindow: Int, dtype: DType,
        slidingWindow: Int, slidingWindowPattern: Int,
        attnLogitSoftcap: Float, finalLogitSoftcap: Float,
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
        self.attnLogitSoftcap = attnLogitSoftcap
        self.finalLogitSoftcap = finalLogitSoftcap
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

    /// Per-layer KV cache. Sliding layers cap at `slidingWindow` with
    /// FIFO eviction; full-attention layers stay unbounded. If the user
    /// passed `.window(...)` via `LoadOptions.kvEviction`, that applies
    /// uniformly and overrides the per-layer heuristic — matches Gemma 3.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxContextWindow
        var caches: [any LayerCacheProtocol] = []
        caches.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let layerEviction: KVEviction
            switch kvEviction {
            case .window:
                layerEviction = kvEviction
            case .unbounded:
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
                preconditionFailure(
                    "Gemma2: only .raw KV cache supported today; got \(kvCacheKind)")
            }
        }
        return caches
    }

    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        let h0 = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        var h = Ops.mul(h0, embedScale, on: workCmd)
        workCmd = tap.dumpLayerBoundary(
            h, label: "embed*scale", layer: -1,
            cmd: workCmd, device: device)

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
        // `final_logit_softcapping` is loaded onto Gemma2Model but
        // intentionally NOT applied here — it has no effect on greedy
        // (argmax) decode since tanh is monotonic. Sampling paths that
        // care should consult `finalLogitSoftcap` before token choice.
        let logits = lmHead(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(
            logits, label: "logits", layer: -1,
            cmd: workCmd, device: device)

        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    /// Multi-token forward — same per-token loop as Gemma 3. A
    /// chunked SDPA collapse for the full-attention layers is plausible
    /// future work (see Gemma 3's matching doc comment) but the
    /// sliding-window layers' ring-buffer state makes naive sdpaMulti
    /// unsafe, so first-light keeps it commit-batched only.
    public func forwardMulti(
        tokenIds: [Int], startingAt position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            !tokenIds.isEmpty,
            "Gemma2Model.forwardMulti: tokenIds must be non-empty")
        var logits: Tensor!
        for (i, tok) in tokenIds.enumerated() {
            logits = forward(
                tokenId: tok, position: position + i,
                caches: caches, on: cmd, device: device)
        }
        return logits
    }

    /// Gemma 2 supports the VLM splice protocol so any future VLM
    /// built on this backbone (PaliGemma 2 follows this path) can hand
    /// in vision-encoder embeddings directly.
    public var supportsEmbeddingInput: Bool { true }

    public func forward(
        inputEmbedding: Tensor, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        precondition(
            inputEmbedding.elementCount == hidden,
            "Gemma2Model.forward(inputEmbedding:): expected [\(hidden)], "
                + "got \(inputEmbedding.shape)")
        // Apply the same sqrt(hidden) embed-scale Gemma 3 does for VLM
        // splice — image tokens and text tokens get the same scaling.
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
