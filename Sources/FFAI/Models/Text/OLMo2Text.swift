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
// OLMo 2 text — Allen AI's OLMo-2 dense decoder (Olmo2ForCausalLM). The
// family metadata enum (`enum OLMo`) lives in `Models/OLMo.swift`; this
// file holds the OLMo-2 impl. OLMo 1 (`OlmoForCausalLM`) stays on the
// Llama dense path — only OLMo 2 needs this loader.
//
// OLMo 2 is RMSNorm + RoPE + SwiGLU like Llama 3, but with two
// architectural deltas the Llama layer can't express:
//
//   1. **Post-norm.** There is NO `input_layernorm`. The norm is
//      applied to each sublayer's OUTPUT, then added to the residual:
//        h = h + post_attention_layernorm(attn(h))
//        h = h + post_feedforward_layernorm(mlp(h))
//      (Llama is pre-norm: norm(h) feeds the sublayer.)
//   2. **q/k-norm over the full projection.** RMSNorm is applied to the
//      entire q (`[n_heads · head_dim]`) and k (`[n_kv_heads · head_dim]`)
//      projection — one norm across all heads, NOT Qwen3's per-head
//      `[head_dim]` norm — before reshape + RoPE.
//
// Tensor names are otherwise standard (embed_tokens, q/k/v/o_proj,
// mlp.gate/up/down_proj, lm_head, model.norm), so no remapping is
// needed.

import Foundation
import Metal

// ─── OLMo2Dense — single dense variant ────────────────────────────────

public enum OLMo2Dense {

    /// OLMo 2 dense defaults — Llama-shaped, greedy-friendly baseline.
    public static let defaultGenerationParameters = LlamaDense.defaultGenerationParameters

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device _: Device
    ) throws -> OLMo2Model {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let vocab = config.vocabSize,
            let intermediate = config.intermediateSize,
            let eps = config.rmsNormEps
        else {
            throw LlamaError.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let headDim = config.headDim ?? (hidden / nHeads)
        let theta = Float(config.ropeTheta ?? 500_000)
        let maxSeq = config.int("max_position_embeddings") ?? 4096
        let tieEmbed = config.tieWordEmbeddings
        let quant = config.quantization

        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant)

        var layers: [OLMo2Layer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            // q/k-norm over the full projection (weight is
            // [n_heads·head_dim] / [n_kv_heads·head_dim]).
            let qNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                eps: Float(eps))
            let kNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                eps: Float(eps))

            let gateProj = try loadLinear(
                base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(
                base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(
                base: "\(p).mlp.down_proj", in: weights, quantization: quant)

            // Post-norm: norm is applied to the sublayer OUTPUT (see file
            // header). No `input_layernorm` tensor exists.
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))
            let postFFNNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_feedforward_layernorm.weight"),
                eps: Float(eps))

            layers.append(
                OLMo2Layer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    qNorm: qNorm, kNorm: kNorm,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    postAttnNorm: postAttnNorm, postFFNNorm: postFFNNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate, ropeTheta: theta))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

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
                    bits: bits, groupSize: q.groupSize))
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

        return OLMo2Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxContextWindow: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache, kvEviction: options.kvEviction,
            auraDecodePath: options.auraDecodePath)
    }
}

// ─── OLMo2Layer (post-norm attention + SwiGLU MLP) ────────────────────

public final class OLMo2Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let gateProj, upProj, downProj: AnyLinear
    let postAttnNorm, postFFNNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        qNorm: RMSNorm, kNorm: RMSNorm,
        gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
        postAttnNorm: RMSNorm, postFFNNorm: RMSNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        intermediate: Int, ropeTheta: Float
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.postAttnNorm = postAttnNorm
        self.postFFNNorm = postFFNNorm
        self.hidden = hidden
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.headDim = headDim
        self.intermediate = intermediate
        self.ropeTheta = ropeTheta
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
        for (k, v) in gateProj.parameters() { out.append(("mlp.gate_proj.\(k)", v)) }
        for (k, v) in upProj.parameters() { out.append(("mlp.up_proj.\(k)", v)) }
        for (k, v) in downProj.parameters() { out.append(("mlp.down_proj.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        for (k, v) in postFFNNorm.parameters() {
            out.append(("post_feedforward_layernorm.\(k)", v))
        }
        return out
    }

    /// Single-token forward pass. `h` is the residual stream [hidden].
    /// Post-norm: each sublayer runs on the RAW residual stream, its
    /// output is normed, then added back. All work queued on `cmd`.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        // Attention on the raw residual (no input norm). q/k-norm over
        // the full projection, then reshape + RoPE.
        let q = qNorm(qProj(h, on: cmd), on: cmd)  // [n_heads * head_dim]
        let k = kNorm(kProj(h, on: cmd), on: cmd)  // [n_kv_heads * head_dim]
        let v = vProj(h, on: cmd)

        let qRotated = Ops.rope(
            q.reshaped(to: [nHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)
        let kRotated = Ops.rope(
            k.reshaped(to: [nKVHeads, headDim]),
            position: position, headDim: headDim,
            thetaBase: ropeTheta, scaling: .none, on: cmd)

        cache.appendOnGPU(
            kFlat: kRotated,
            vFlat: v.reshaped(to: [nKVHeads, headDim]),
            on: cmd)

        // AURA caches store K/V in Π-rotated space: rotate Q before SDPA
        // and un-rotate the output before o_proj so the residual stream
        // stays in the original activation space. Raw / affine skip both.
        let qForSdpa: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            qForSdpa = Ops.auraRotatePerHead(
                qRotated.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtype,
                nHeads: nHeads, headDim: headDim, on: cmd
            ).reshaped(to: [nHeads, headDim])
        } else {
            qForSdpa = qRotated
        }

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.capacity,
            scale: scale, on: cmd)

        let attnReady: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            attnReady = Ops.auraRotatePerHead(
                attnOut.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtypeT,
                nHeads: nHeads, headDim: headDim, on: cmd)
        } else {
            attnReady = attnOut.reshaped(to: [nHeads * headDim])
        }
        let oOut = oProj(attnReady, on: cmd)

        // Post-norm residual: h + post_attention_layernorm(attn_out).
        let h1 = Ops.add(h, postAttnNorm(oOut, on: cmd), on: cmd)

        // MLP (SwiGLU) on the post-attention stream, post-normed.
        let gate = gateProj(h1, on: cmd)
        let up = upProj(h1, on: cmd)
        let mlpInner = Ops.swiglu(gate: gate, up: up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(h1, postFFNNorm(mlpOut, on: cmd), on: cmd)
    }
}

// ─── OLMo2Model — whole decoder ───────────────────────────────────────

public final class OLMo2Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [OLMo2Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxContextWindow: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction
    public let auraDecodePath: AURADecodePath

    init(
        embedTokens: AnyEmbedding, layers: [OLMo2Layer],
        finalNorm: RMSNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxContextWindow: Int, ropeTheta: Float, dtype: DType,
        kvCacheKind: KVCacheKind = .raw,
        kvEviction: KVEviction = .unbounded,
        auraDecodePath: AURADecodePath = .compressed
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
        self.vocab = vocab
        self.maxContextWindow = maxContextWindow
        self.ropeTheta = ropeTheta
        self.dtype = dtype
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
        self.auraDecodePath = auraDecodePath
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

    /// Per-layer caches honoring `LoadOptions.kvCache` (raw / affine /
    /// AURA) via the shared factory.
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        makeAttentionCaches(
            kind: kvCacheKind, count: nLayers,
            nKVHeads: nKVHeads, headDim: headDim,
            contextLength: maxSeq ?? maxContextWindow,
            dtype: dtype, eviction: kvEviction,
            auraDecodePath: auraDecodePath, device: device)
    }

    public func forward(
        tokenId: Int, position: Int,
        caches: [any LayerCacheProtocol],
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)

        var h = embedTokens(tokenTensor, on: cmd).reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            guard let kv = caches[i] as? any KVCacheProtocol else {
                fatalError(
                    "OLMo2Model: expected a KV cache at layer \(i), got \(type(of: caches[i]))")
            }
            h = layer.forward(h, position: position, cache: kv, cmd: cmd, device: device)
        }
        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }
}
