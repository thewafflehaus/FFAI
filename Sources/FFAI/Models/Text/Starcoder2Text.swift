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
// Starcoder 2 text — concrete variants + the dense decoder for
// BigCode's Starcoder2 family. The family enum (`enum Starcoder2`)
// + variant protocol (`Starcoder2Variant`) + error type
// (`Starcoder2Error`) live in `Models/Starcoder2.swift` (the family
// root). This file holds the text-only impl:
//
//   • `Starcoder2Dense` — `Starcoder2Variant` conformance + the
//     per-variant `loadModel` entry,
//   • `Starcoder2Layer` — one attention + GELU MLP block,
//   • `Starcoder2Model` — the full LanguageModel decoder.
//
// Architecture deltas vs Llama 3:
//
//   - LayerNorm with `.bias` (NOT RMSNorm). Two per layer
//     (`input_layernorm` + `post_attention_layernorm`) plus a final
//     `model.norm` — each carries `.weight` AND `.bias`.
//   - Single-projection GELU-tanh MLP with `c_fc` (up, hidden → 4·hidden)
//     + `c_proj` (down, 4·hidden → hidden) names, NOT the SwiGLU
//     `gate_proj` + `up_proj` + `down_proj` triad. Activation is
//     `gelu_pytorch_tanh` (matches `Ops.gelu`).
//   - Attention biases on all four q/k/v/o projections
//     (`use_bias: true` in config). `loadLinear`'s auto-detection
//     handles the `.bias` companion tensor transparently — same path
//     Qwen 2.x QKV biases ride.
//   - Config field is `norm_epsilon` (NOT `rms_norm_eps`).
//   - Sliding-window flag in config is informational only here; we
//     decode against a full KVCache and rely on natural prompt
//     truncation. (`sliding_window: 4096` on the published
//     Starcoder2-3B / 7B / 15B-instruct checkpoints; the bench /
//     KV-cache-matrix coverage tracks the actual windowed path.)

import Foundation
import Metal

// ─── Starcoder2Dense — single dense variant ───────────────────────────

public struct Starcoder2Dense: Starcoder2Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Starcoder2 ships as a code completion model — default temperature
    /// kept at 0.6 (mlx-lm convention) so the test surface lines up with
    /// the rest of the dense Llama-family variants. Prefill step matches
    /// the Llama dense baseline.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options _: LoadOptions,
        device _: Device
    ) throws -> Starcoder2Model {
        guard let hidden = config.hiddenSize,
            let nLayers = config.numLayers,
            let nHeads = config.numAttentionHeads,
            let vocab = config.vocabSize,
            let intermediate = config.intermediateSize
        else {
            throw Starcoder2Error.missingConfig(
                "hidden_size / num_hidden_layers / num_attention_heads / "
                    + "vocab_size / intermediate_size")
        }
        // Starcoder2 uses `norm_epsilon` (LayerNorm) — NOT `rms_norm_eps`.
        guard let eps = config.float("norm_epsilon")
        else {
            throw Starcoder2Error.missingConfig("norm_epsilon")
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        // head_dim is derived (config field absent on the published
        // Starcoder2 configs — hidden_size / num_attention_heads).
        let headDim = config.headDim ?? (hidden / nHeads)
        let theta = Float(config.ropeTheta ?? 999_999.4420358813)
        let maxSeq = config.int("max_position_embeddings") ?? 16_384
        let tieEmbed = config.tieWordEmbeddings

        let quant = config.quantization

        // Embedding — quantized if the bundle has matching scales/biases
        // (mlx-community starcoder2-3b-4bit quantizes everything).
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant)

        // Layers
        var layers: [Starcoder2Layer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"

            // q/k/v/o all carry `.bias` (Starcoder2 `use_bias: true`).
            // `loadLinear` auto-detects the bias companion — same path
            // Qwen 2.x QKV biases ride.
            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            // Single-projection MLP: c_fc (up) + c_proj (down). Both
            // biased; GELU-tanh activation lives between them.
            let cFc = try loadLinear(
                base: "\(p).mlp.c_fc", in: weights, quantization: quant)
            let cProj = try loadLinear(
                base: "\(p).mlp.c_proj", in: weights, quantization: quant)

            // LayerNorm — NOT RMSNorm. Two per layer.
            let inputNorm = LayerNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                bias: try weights.tensor(named: "\(p).input_layernorm.bias"),
                eps: Float(eps))
            let postAttnNorm = LayerNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                bias: try weights.tensor(named: "\(p).post_attention_layernorm.bias"),
                eps: Float(eps))

            layers.append(
                Starcoder2Layer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    cFc: cFc, cProj: cProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermediate,
                    ropeTheta: theta))
        }

        // Final LayerNorm — also carries `.bias`.
        let finalNorm = LayerNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            bias: try weights.tensor(named: "model.norm.bias"),
            eps: Float(eps))

        // LM head. Starcoder2-3B sets `tie_word_embeddings: true` so the
        // head reuses the embedding triplet. Larger Starcoder2 variants
        // may ship an explicit `lm_head.weight` — honour it if present.
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

        // Activation dtype — prefer scales dtype for quantized models
        // (the f16/bf16 the model actually computes in), fall back to
        // the embedding weight dtype otherwise.
        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
            let scales = try? weights.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return Starcoder2Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype)
    }
}

// ─── Starcoder2Layer (attention + GELU MLP) ───────────────────────────

public final class Starcoder2Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let cFc, cProj: AnyLinear
    let inputNorm, postAttnNorm: LayerNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let scale: Float

    init(
        qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
        cFc: AnyLinear, cProj: AnyLinear,
        inputNorm: LayerNorm, postAttnNorm: LayerNorm,
        hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        intermediate: Int, ropeTheta: Float
    ) {
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.cFc = cFc
        self.cProj = cProj
        self.inputNorm = inputNorm
        self.postAttnNorm = postAttnNorm
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
        for (k, v) in cFc.parameters() { out.append(("mlp.c_fc.\(k)", v)) }
        for (k, v) in cProj.parameters() { out.append(("mlp.c_proj.\(k)", v)) }
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() {
            out.append(("post_attention_layernorm.\(k)", v))
        }
        return out
    }

    /// Single-token forward pass. `h` is the residual stream [hidden].
    /// `position` is the absolute sequence index of this token.
    /// Returns the updated residual stream. All work queued on `cmd`;
    /// caller commits at end-of-token.
    func forward(
        _ h: Tensor, position: Int, cache: any KVCacheProtocol,
        cmd: MTLCommandBuffer, device _: Device
    ) -> Tensor {
        // Attention — pre-LayerNorm.
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)  // [n_heads * head_dim]
        let k = kProj(xNorm, on: cmd)  // [n_kv_heads * head_dim]
        let v = vProj(xNorm, on: cmd)  // [n_kv_heads * head_dim]

        // RoPE on q and k. Starcoder2 uses plain rotary (no llama3 /
        // longrope scaling); pass `.none`.
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

        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaDecode(
            q: qRotated, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)
        let oOut = oProj(attnOut.reshaped(to: [nHeads * headDim]), on: cmd)

        // Residual add + post-attention LayerNorm. LayerNorm has a
        // bias the fused `mt_add_rms_norm` kernel doesn't model — so
        // we just stack `Ops.add` + `Ops.layerNorm` here. (The
        // fused-add-rms-norm fast path is RMSNorm-only.)
        let postAttn = Ops.add(h, oOut, on: cmd)
        let mlpNorm = postAttnNorm(postAttn, on: cmd)

        // MLP — single-projection GELU-tanh. c_fc lifts hidden →
        // intermediate (= 4 · hidden); GELU element-wise; c_proj
        // contracts back to hidden.
        let pre = cFc(mlpNorm, on: cmd)
        let act = Ops.gelu(pre, on: cmd)
        let mlpOut = cProj(act, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }
}

// ─── Starcoder2Model — whole decoder ──────────────────────────────────

public final class Starcoder2Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Starcoder2Layer]
    public let finalNorm: LayerNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType

    init(
        embedTokens: AnyEmbedding, layers: [Starcoder2Layer],
        finalNorm: LayerNorm, lmHead: AnyLinear,
        hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
        vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType
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
        self.maxSeq = maxSeq
        self.ropeTheta = ropeTheta
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

    /// Per-layer raw `KVCache` — Starcoder2 doesn't use any of the
    /// affine-quantized / AURA cache modes the Llama dense path
    /// exposes. The sliding-window declared in config is honored at
    /// the prompt level rather than the cache level (truncation is the
    /// caller's job).
    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        return (0 ..< nLayers).map { _ in
            KVCache(
                nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                dtype: dtype, device: device)
        }
    }

    /// Queue a single-token forward pass on `cmd`. Does NOT commit —
    /// the default `forwardSample` / `forwardSampleCategorical`
    /// extensions compose the output kernels onto the same `cmd` and
    /// commit once, matching every other family.
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
            guard let kv = caches[i] as? KVCache else {
                fatalError(
                    "Starcoder2Model: expected KVCache at layer \(i), got "
                        + "\(type(of: caches[i]))")
            }
            h = layer.forward(h, position: position, cache: kv, cmd: cmd, device: device)
        }

        let normed = finalNorm(h, on: cmd)
        return lmHead(normed, on: cmd)
    }
}
