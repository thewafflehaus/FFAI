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
// Qwen3 text — concrete variants + the dense decoder for the Qwen 3
// family. The family enum (`enum Qwen3`), variant protocol
// (`Qwen3Variant`), and error type (`Qwen3Error`) live in
// `Models/Qwen3.swift` (the family root / main interface). This file
// holds the text-only impl:
//
//   • `Qwen3Dense` — `Qwen3Variant` conformance + the per-variant
//     `loadModel` entry,
//   • `Qwen3Layer` — one attention + MLP block,
//   • `Qwen3Model` — the full LanguageModel decoder.

import Foundation
import Metal

// ─── Qwen3Dense — standard dense transformer with q_norm / k_norm ─────

public struct Qwen3Dense: Qwen3Variant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut]

    /// Qwen 3 dense defaults. Tracks mlx-swift-lm's Qwen 3 family
    /// values (temp 0.6, top-p 0.95, top-k 20, min-p 0.0,
    /// rep-penalty 1.0) and a 1024-token prefill chunk for dense
    /// attention. Qwen 3.5 hybrid / MoE will declare their own when
    /// those variants land in planned.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.6,
        topP: 0.95,
        topK: 20,
        minP: 0.0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen3Model {
        guard let hidden = config.hiddenSize,
              let nLayers = config.numLayers,
              let nHeads = config.numAttentionHeads,
              let headDim = config.headDim,
              let vocab = config.vocabSize,
              let intermediate = config.intermediateSize,
              let eps = config.rmsNormEps
        else {
            throw Qwen3Error.missingConfig
        }
        let nKVHeads = config.numKeyValueHeads ?? nHeads
        let theta = Float(config.ropeTheta ?? 1_000_000)
        let maxSeq = config.int("max_position_embeddings") ?? 32_768
        let tieEmbed = config.tieWordEmbeddings

        // Optional linear RoPE scaling (Qwen3 dense uses rope_scaling
        // {type: "linear", factor: F} which uniformly divides every
        // inv_freq by F. We model this with the Llama-3 scaling shape
        // by setting scaleFactor = F and pinning everything else so the
        // wavelength branches always go down the "low_freq → divide"
        // path, equivalent to a uniform scale.)
        var ropeScaling = Ops.RoPEScaling.none
        if let rs = config.nested("rope_scaling") {
            if let typeStr = (rs["type"] as? String) ?? (rs["rope_type"] as? String),
               typeStr == "linear",
               let factor = rs["factor"] as? Double
            {
                ropeScaling = Ops.RoPEScaling(
                    scaleFactor: Float(factor),
                    lowFreqFactor: 1, highFreqFactor: 1,
                    originalMaxPosition: 1  // forces low_freq_wavelen = 1, so all freqs are "low"
                )
            } else if (rs["rope_type"] as? String) == "llama3" {
                // Some Qwen3 variants ship Llama-3 style scaling; reuse
                // the same parameters.
                ropeScaling = Ops.RoPEScaling(
                    scaleFactor: Float((rs["factor"] as? Double) ?? 1),
                    lowFreqFactor: Float((rs["low_freq_factor"] as? Double) ?? 1),
                    highFreqFactor: Float((rs["high_freq_factor"] as? Double) ?? 4),
                    originalMaxPosition: Float((rs["original_max_position_embeddings"] as? Int) ?? 8192)
                )
            }
        }

        let quant = config.quantization

        // Embedding — quantized if the bundle has matching scales/biases.
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: weights,
            hidden: hidden, quantization: quant
        )

        // Layers
        var layers: [Qwen3Layer] = []
        layers.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            let p = "model.layers.\(i)"

            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: weights, quantization: quant)

            // Per-head Q/K RMSNorm — the structural delta vs Llama.
            let qNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.q_norm.weight"),
                eps: Float(eps))
            let kNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).self_attn.k_norm.weight"),
                eps: Float(eps))

            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: weights, quantization: quant)

            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))

            layers.append(Qwen3Layer(
                qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                qNorm: qNorm, kNorm: kNorm,
                gateProj: gateProj, upProj: upProj, downProj: downProj,
                inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                headDim: headDim, intermediate: intermediate,
                ropeTheta: theta, ropeScaling: ropeScaling
            ))
        }

        // Final norm
        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        // LM head. Tied / quantized variants — same as Llama.
        let lmHead: AnyLinear
        if !tieEmbed, weights.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("model.embed_tokens") {
            let t = try weights.quantizedTriplet("model.embed_tokens")
            let bits = deriveAffineQuantBits(
                weightPackedCols: t.weight.shape[t.weight.shape.count - 1],
                scaleCols: t.scales.shape[t.scales.shape.count - 1],
                groupSize: q.groupSize)
            lmHead = AnyLinear(QuantizedLinear(
                weight: t.weight, scales: t.scales, biases: t.biases,
                bits: bits, groupSize: q.groupSize
            ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation/inference dtype: prefer scales for quantized models.
        let activationDtype: DType
        if weights.isQuantized("model.embed_tokens"),
           let scales = try? weights.tensor(named: "model.embed_tokens.scales") {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        return Qwen3Model(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: theta, dtype: activationDtype,
            kvCacheKind: options.kvCache,
            kvEviction: options.kvEviction,
            auraDecodePath: options.auraDecodePath
        )
    }
}

// ─── Qwen3Layer ──────────────────────────────────────────────────────

public final class Qwen3Layer: Module {
    let qProj, kProj, vProj, oProj: AnyLinear
    let qNorm, kNorm: RMSNorm
    let gateProj, upProj, downProj: AnyLinear
    let inputNorm, postAttnNorm: RMSNorm
    let hidden, nHeads, nKVHeads, headDim, intermediate: Int
    let ropeTheta: Float
    let ropeScaling: Ops.RoPEScaling
    let scale: Float

    init(qProj: AnyLinear, kProj: AnyLinear, vProj: AnyLinear, oProj: AnyLinear,
         qNorm: RMSNorm, kNorm: RMSNorm,
         gateProj: AnyLinear, upProj: AnyLinear, downProj: AnyLinear,
         inputNorm: RMSNorm, postAttnNorm: RMSNorm,
         hidden: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         intermediate: Int, ropeTheta: Float,
         ropeScaling: Ops.RoPEScaling) {
        self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
        self.qNorm = qNorm; self.kNorm = kNorm
        self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
        self.hidden = hidden; self.nHeads = nHeads; self.nKVHeads = nKVHeads
        self.headDim = headDim; self.intermediate = intermediate
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
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
        for (k, v) in inputNorm.parameters() { out.append(("input_layernorm.\(k)", v)) }
        for (k, v) in postAttnNorm.parameters() { out.append(("post_attention_layernorm.\(k)", v)) }
        return out
    }

    /// Single-token forward pass. Same shape as LlamaLayer.forward but
    /// applies q_norm / k_norm to each head's [head_dim] vector before
    /// RoPE. All work queued on `cmd`, no commit/wait inside.
    func forward(_ h: Tensor, position: Int, cache: any KVCacheProtocol,
                 cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // Attention
        let xNorm = inputNorm(h, on: cmd)
        let q = qProj(xNorm, on: cmd)
        let k = kProj(xNorm, on: cmd)
        let v = vProj(xNorm, on: cmd)

        // Per-head q_norm / k_norm
        let qNormed = applyPerHeadRMSNorm(q, weight: qNorm.weight, eps: qNorm.eps,
                                          nHeads: nHeads, headDim: headDim,
                                          on: cmd, device: device)
        let kNormed = applyPerHeadRMSNorm(k, weight: kNorm.weight, eps: kNorm.eps,
                                          nHeads: nKVHeads, headDim: headDim,
                                          on: cmd, device: device)

        // RoPE
        let qRotated = Ops.rope(qNormed.reshaped(to: [nHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)
        let kRotated = Ops.rope(kNormed.reshaped(to: [nKVHeads, headDim]),
                                position: position, headDim: headDim,
                                thetaBase: ropeTheta, scaling: ropeScaling, on: cmd)

        // GPU KV cache update — no CPU sync.
        cache.appendOnGPU(kFlat: kRotated,
                          vFlat: v.reshaped(to: [nKVHeads, headDim]),
                          on: cmd)

        // AURA cache stores K and V in Π-rotated space, so for the
        // scores to cancel out we apply Π to Q before SDPA and Π^T to
        // the attention output before oProj. RoPE doesn't commute with
        // arbitrary orthogonal rotations, so the per-token order is
        // strictly project → RMSNorm → RoPE → Π·. Raw / affine caches
        // skip this entirely.
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
            nKV: cache.length, kvStride: cache.maxSeq,
            scale: scale, on: cmd)

        let attnReadyForOProj: Tensor
        if let auraCache = cache as? AURAQuantizedKVCache {
            // attnOut is in Π-rotated space; Π^T puts it back in the
            // original activation space oProj expects.
            attnReadyForOProj = Ops.auraRotatePerHead(
                attnOut.reshaped(to: [nHeads * headDim]),
                rotation: auraCache.rotationDtypeT,
                nHeads: nHeads, headDim: headDim, on: cmd)
        } else {
            attnReadyForOProj = attnOut.reshaped(to: [nHeads * headDim])
        }
        let oOut = oProj(attnReadyForOProj, on: cmd)

        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Validator gate falls through for Qwen 3 14B/32B
        // (hidden 5120) and similarly wide variants.
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h, oOut, weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: 1, rowSize: hidden, on: cmd)
            postAttn = fused.residual
            mlpNorm = fused.normed
        } else {
            postAttn = Ops.add(h, oOut, on: cmd)
            mlpNorm = postAttnNorm(postAttn, on: cmd)
        }

        // MLP — SwiGLU
        let gate = gateProj(mlpNorm, on: cmd)
        let up = upProj(mlpNorm, on: cmd)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj(mlpInner, on: cmd)
        return Ops.add(postAttn, mlpOut, on: cmd)
    }

    /// Apply RMSNorm independently to each head's [head_dim] slice of a
    /// flat [nHeads * headDim] tensor via a single multi-row dispatch
    /// (Ops.rmsNormRows). collapse from one-launch-per-head.
    private func applyPerHeadRMSNorm(
        _ x: Tensor, weight: Tensor, eps: Float,
        nHeads: Int, headDim: Int,
        on cmd: MTLCommandBuffer, device: Device
    ) -> Tensor {
        Ops.rmsNormRows(x, weight: weight, eps: eps,
                        nRows: nHeads, rowSize: headDim, on: cmd)
    }

    /// Chunked forward — process `nRows` tokens at once. See
    /// `LlamaLayer.forwardMulti` for the design + perf notes; the only
    /// shape difference vs Llama is the per-head q_norm / k_norm step
    /// that Qwen 3 applies between projection and RoPE.
    ///
    /// **Cache compatibility.** AURA caches need Q π-rotation; the
    /// fast path skips it. `Qwen3Model.forwardMulti` falls back to the
    /// per-token loop when any layer's cache is `AURAQuantizedKVCache`.
    func forwardMulti(_ h: Tensor, startingAt position: Int,
                      cache: any KVCacheProtocol,
                      cmd: MTLCommandBuffer, device: Device) -> Tensor {
        let nRows = h.shape[0]
        precondition(h.shape == [nRows, hidden],
                     "Qwen3Layer.forwardMulti: h shape \(h.shape) ≠ [nRows, hidden]")

        // ── Attention ───────────────────────────────────────────────────
        let xNorm = Ops.rmsNormRows(
            h.reshaped(to: [nRows * hidden]),
            weight: inputNorm.weight, eps: inputNorm.eps,
            nRows: nRows, rowSize: hidden, on: cmd
        ).reshaped(to: [nRows, hidden])

        let q = qProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nHeads*headDim]
        let k = kProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nKVHeads*headDim]
        let v = vProj.callMany(xNorm, t: nRows, on: cmd, device: device)   // [N, nKVHeads*headDim]

        // Per-head q_norm / k_norm — flatten to one big [nRows*nHeads,
        // headDim] (resp. [nRows*nKVHeads, headDim]) row stack so the
        // single rmsNormRows dispatch normalises every (token, head)
        // pair at once.
        let qFlat = q.reshaped(to: [nRows * nHeads * headDim])
        let qNormedFlat = Ops.rmsNormRows(
            qFlat, weight: qNorm.weight, eps: qNorm.eps,
            nRows: nRows * nHeads, rowSize: headDim, on: cmd
        )
        let qNormed = qNormedFlat.reshaped(to: [nRows, nHeads * headDim])

        let kFlat = k.reshaped(to: [nRows * nKVHeads * headDim])
        let kNormedFlat = Ops.rmsNormRows(
            kFlat, weight: kNorm.weight, eps: kNorm.eps,
            nRows: nRows * nKVHeads, rowSize: headDim, on: cmd
        )
        let kNormed = kNormedFlat.reshaped(to: [nRows, nKVHeads * headDim])

        // RoPE — single-position kernel looped per row on the same cmd.
        let qRot = Tensor.empty(shape: [nRows, nHeads * headDim],
                                dtype: q.dtype, device: device)
        let kRot = Tensor.empty(shape: [nRows, nKVHeads * headDim],
                                dtype: k.dtype, device: device)
        for i in 0..<nRows {
            let qRow = qNormed.slicedRows(start: i, count: 1).reshaped(to: [nHeads * headDim])
            let qOut = qRot.slicedRows(start: i, count: 1).reshaped(to: [nHeads * headDim])
            _ = Ops.rope(qRow, position: position + i, headDim: headDim,
                         thetaBase: ropeTheta, scaling: ropeScaling,
                         on: cmd, into: qOut)
            let kRow = kNormed.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads * headDim])
            let kOut = kRot.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads * headDim])
            _ = Ops.rope(kRow, position: position + i, headDim: headDim,
                         thetaBase: ropeTheta, scaling: ropeScaling,
                         on: cmd, into: kOut)
            cache.appendOnGPU(
                kFlat: kOut.reshaped(to: [nKVHeads, headDim]),
                vFlat: v.slicedRows(start: i, count: 1).reshaped(to: [nKVHeads, headDim]),
                on: cmd
            )
        }

        // SDPA — ONE dispatch over the chunk with causal mask.
        let qForSdpa = qRot.reshaped(to: [nRows, nHeads, headDim])
        let (cacheK, cacheV) = cache.prepareForAttention(on: cmd)
        let attnOut = Ops.sdpaMulti(
            q: qForSdpa, k: cacheK, v: cacheV,
            nQHeads: nHeads, nKVHeads: nKVHeads, headDim: headDim,
            baseKV: position, nQuery: nRows, kvStride: cache.maxSeq,
            causal: true, scale: scale, on: cmd
        )

        let attnFlat = attnOut.reshaped(to: [nRows, nHeads * headDim])
        let oOut = oProj.callMany(attnFlat, t: nRows, on: cmd, device: device)

        // Fused residual add + post-attn RMSNorm via mt_add_rms_norm
        // (hidden ≤ 4096). Validator gate covers wider Qwen 3 variants.
        let postAttn: Tensor
        let mlpNorm: Tensor
        if OpsValidation.validateAddRmsNorm(n: hidden) == nil {
            let fused = Ops.addAndRmsNorm(
                h.reshaped(to: [nRows * hidden]),
                oOut.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: cmd)
            postAttn = fused.residual.reshaped(to: [nRows, hidden])
            mlpNorm = fused.normed.reshaped(to: [nRows, hidden])
        } else {
            postAttn = Ops.add(
                h.reshaped(to: [nRows * hidden]),
                oOut.reshaped(to: [nRows * hidden]),
                on: cmd
            ).reshaped(to: [nRows, hidden])
            mlpNorm = Ops.rmsNormRows(
                postAttn.reshaped(to: [nRows * hidden]),
                weight: postAttnNorm.weight, eps: postAttnNorm.eps,
                nRows: nRows, rowSize: hidden, on: cmd
            ).reshaped(to: [nRows, hidden])
        }

        // ── MLP — SwiGLU, batched ───────────────────────────────────────
        let gate = gateProj.callMany(mlpNorm, t: nRows, on: cmd, device: device)
        let up = upProj.callMany(mlpNorm, t: nRows, on: cmd, device: device)
        let siluGate = Ops.silu(gate, on: cmd)
        let mlpInner = Ops.mul(siluGate, up, on: cmd)
        let mlpOut = downProj.callMany(mlpInner, t: nRows, on: cmd, device: device)

        return Ops.add(
            postAttn.reshaped(to: [nRows * hidden]),
            mlpOut.reshaped(to: [nRows * hidden]),
            on: cmd
        ).reshaped(to: [nRows, hidden])
    }
}

// ─── Qwen3Model ──────────────────────────────────────────────────────

public final class Qwen3Model: LanguageModel {
    public let embedTokens: AnyEmbedding
    public let layers: [Qwen3Layer]
    public let finalNorm: RMSNorm
    public let lmHead: AnyLinear

    public let hidden, nLayers, nHeads, nKVHeads, headDim, vocab, maxSeq: Int
    public let ropeTheta: Float
    public let dtype: DType
    public let kvCacheKind: KVCacheKind
    public let kvEviction: KVEviction
    /// AURA decode-time attention path. Forwarded into every
    /// `AURAQuantizedKVCache` instantiated by `makeLayerCaches`. See
    /// `AURADecodePath` for the path semantics.
    public let auraDecodePath: AURADecodePath

    init(embedTokens: AnyEmbedding, layers: [Qwen3Layer],
         finalNorm: RMSNorm, lmHead: AnyLinear,
         hidden: Int, nLayers: Int, nHeads: Int, nKVHeads: Int, headDim: Int,
         vocab: Int, maxSeq: Int, ropeTheta: Float, dtype: DType,
         kvCacheKind: KVCacheKind = .raw,
         kvEviction: KVEviction = .unbounded,
         auraDecodePath: AURADecodePath = .compressed) {
        self.embedTokens = embedTokens
        self.layers = layers
        self.finalNorm = finalNorm
        self.lmHead = lmHead
        self.hidden = hidden; self.nLayers = nLayers; self.nHeads = nHeads
        self.nKVHeads = nKVHeads; self.headDim = headDim; self.vocab = vocab
        self.maxSeq = maxSeq; self.ropeTheta = ropeTheta; self.dtype = dtype
        self.kvCacheKind = kvCacheKind
        self.kvEviction = kvEviction
        self.auraDecodePath = auraDecodePath
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

    public func makeLayerCaches(maxSeq: Int?, device: Device) -> [any LayerCacheProtocol] {
        let cap = maxSeq ?? self.maxSeq
        let eviction = kvEviction
        switch kvCacheKind {
        case .raw:
            return (0..<nLayers).map { _ in
                KVCache(nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                        dtype: dtype, eviction: eviction, device: device)
            }
        case .affineQuantized(let bits, let groupSize):
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { _ in
                AffineQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, bits: bits, groupSize: groupSize,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    eviction: eviction,
                    device: device
                )
            }
        case .auraQuantized(let scheme):
            // Codebooks are shared across layers (Lloyd-Max levels are
            // dim-only — no per-layer statistics baked in yet). Rotations
            // are per-layer: each Π_l is an SRHT matrix seeded by the
            // layer index, matching the AURA paper's "fresh rotation per
            // tensor" recipe for de-correlating activation statistics.
            let kCodebookData = AURACodebook.centroids(dim: headDim, bits: scheme.keyBits)
            let kBoundariesData = AURACodebook.boundaries(dim: headDim, bits: scheme.keyBits)
            let vCodebookData = AURACodebook.centroids(dim: headDim, bits: scheme.valueBits)
            let vBoundariesData = AURACodebook.boundaries(dim: headDim, bits: scheme.valueBits)

            let kCodebook = Tensor.empty(shape: [kCodebookData.count], dtype: .f32, device: device)
            kCodebook.copyIn(from: kCodebookData)
            let kBoundaries = Tensor.empty(shape: [kBoundariesData.count], dtype: .f32, device: device)
            kBoundaries.copyIn(from: kBoundariesData)
            let vCodebook = Tensor.empty(shape: [vCodebookData.count], dtype: .f32, device: device)
            vCodebook.copyIn(from: vCodebookData)
            let vBoundaries = Tensor.empty(shape: [vBoundariesData.count], dtype: .f32, device: device)
            vBoundaries.copyIn(from: vBoundariesData)

            // Shared working buffers — same pattern as affineQuantized:
            // bulk-dequant target shared across all layers.
            let sharedK = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            let sharedV = Tensor.empty(shape: [nKVHeads, cap, headDim],
                                       dtype: dtype, device: device)
            return (0..<nLayers).map { i in
                let rot = AURAQuantizedKVCacheRotations.build(
                    headDim: headDim, layerIndex: i,
                    activationDtype: dtype, device: device)
                return AURAQuantizedKVCache(
                    nKVHeads: nKVHeads, headDim: headDim, maxSeq: cap,
                    dtype: dtype, scheme: scheme,
                    rotation: rot.rotation, rotationT: rot.rotationT,
                    rotationDtype: rot.rotationDtype, rotationDtypeT: rot.rotationDtypeT,
                    kCodebook: kCodebook, kBoundaries: kBoundaries,
                    vCodebook: vCodebook, vBoundaries: vBoundaries,
                    sharedWorkingK: sharedK, sharedWorkingV: sharedV,
                    eviction: eviction,
                    decodePath: auraDecodePath,
                    device: device
                )
            }
        }
    }

    /// Primitive: queue a single-token forward pass onto the caller's
    /// command buffer. No commit. The `LanguageModel` default
    /// extension composes this with the appropriate output kernel
    /// (`argmax` for forwardSample, `softmax_categorical_sample` for
    /// forwardSampleCategorical) on the same cmdbuf.
    public func forward(tokenId: Int, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        // See Sources/FFAI/Inspect/InspectTap.swift — no-op when
        // FFAI_INSPECT isn't set.
        let tap = InspectTap.fromEnvironment
        var workCmd = tap.makeWorkCmd(from: cmd, device: device)

        let tokenBuf = device.makeBuffer(length: 4)
        var tid = UInt32(tokenId)
        memcpy(tokenBuf.contents(), &tid, 4)
        let tokenTensor = Tensor(buffer: tokenBuf, offset: 0, shape: [1], dtype: .u32)
        var h = embedTokens(tokenTensor, on: workCmd).reshaped(to: [hidden])
        workCmd = tap.dumpLayerBoundary(h, label: "embed", layer: -1,
                                        cmd: workCmd, device: device)

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
                              cmd: workCmd, device: device)
            workCmd = tap.dumpLayerBoundary(h, label: "layer_out", layer: i,
                                            cmd: workCmd, device: device)
        }

        let normed = finalNorm(h, on: workCmd)
        workCmd = tap.dumpLayerBoundary(normed, label: "final_norm", layer: -1,
                                        cmd: workCmd, device: device)
        let logits = lmHead(normed, on: workCmd)
        workCmd = tap.dumpLayerBoundary(logits, label: "logits", layer: -1,
                                        cmd: workCmd, device: device)

        if tap.active {
            workCmd.commit()
            workCmd.waitUntilCompleted()
        }
        return logits
    }

    /// Chunked multi-token forward — prefill fast path. See
    /// `LlamaModel.forwardMulti` for the design notes; Qwen3's layer
    /// fast path differs only in the per-head q_norm / k_norm step.
    public func forwardMulti(tokenIds: [Int], startingAt position: Int,
                             caches: [any LayerCacheProtocol],
                             on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(!tokenIds.isEmpty,
                     "Qwen3Model.forwardMulti: tokenIds must be non-empty")

        // Fallback to per-token loop when (a) AURA caches require
        // π-rotated K/V layout not implemented by the chunked path,
        // or (b) head_dim != 128 — `ffai_sdpa_multi` is hardcoded to
        // d128 (Qwen 3 0.5B / 1.7B have d=64 vs d=128 on 4B+).
        let hasAura = caches.contains { $0 is AURAQuantizedKVCache }
        let headDimUnsupported = headDim != 128
        if hasAura || headDimUnsupported {
            var logits: Tensor!
            for (i, tok) in tokenIds.enumerated() {
                logits = forward(tokenId: tok, position: position + i,
                                 caches: caches, on: cmd, device: device)
            }
            return logits
        }

        let n = tokenIds.count
        let idsBuf = device.makeBuffer(length: n * 4)
        idsBuf.contents().withMemoryRebound(to: UInt32.self, capacity: n) { p in
            for (i, t) in tokenIds.enumerated() { p[i] = UInt32(t) }
        }
        let idsTensor = Tensor(buffer: idsBuf, offset: 0, shape: [n], dtype: .u32)
        var h = embedTokens(idsTensor, on: cmd)
        precondition(h.shape == [n, hidden],
                     "Qwen3Model.forwardMulti: embedding shape \(h.shape) ≠ [n, hidden]")

        for (i, layer) in layers.enumerated() {
            h = layer.forwardMulti(
                h, startingAt: position,
                cache: caches[i] as! any KVCacheProtocol,
                cmd: cmd, device: device
            )
        }

        let tail = h.slicedRows(start: n - 1, count: 1).reshaped(to: [hidden])
        let normed = finalNorm(tail, on: cmd)
        return lmHead(normed, on: cmd)
    }

    /// Embedding-input forward — the VLM splice path. Identical to
    /// `forward(tokenId:...)` minus the embedding gather: the `[hidden]`
    /// row is supplied directly (a vision-encoder token, or a text-token
    /// embedding the VL model looked up itself).
    public var supportsEmbeddingInput: Bool { true }

    public func forward(inputEmbedding: Tensor, position: Int,
                        caches: [any LayerCacheProtocol],
                        on cmd: MTLCommandBuffer, device: Device) -> Tensor {
        precondition(inputEmbedding.elementCount == hidden,
                     "Qwen3Model.forward(inputEmbedding:): expected [\(hidden)], "
                     + "got \(inputEmbedding.shape)")
        var h = inputEmbedding.reshaped(to: [hidden])
        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, position: position,
                              cache: caches[i] as! any KVCacheProtocol,
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

    // `forward`, `forwardSample`, `forwardSampleCategorical` come from
    // LanguageModel's default extension — they wrap `forward(...on cmd:)`
    // above with a 1-commit-per-token cmdbuf and the appropriate
    // output kernel.
}
