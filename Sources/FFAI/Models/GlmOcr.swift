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
// GlmOcr — THUDM's GLM-OCR vision-language model
// (`glm_ocr` model_type / `GlmOcrForConditionalGeneration` architecture).
//
// Composition:
//   • GLM-OCR vision tower — a dynamic-resolution ViT that shares the
//     Qwen-VL design:
//       – patch-embed via a flattened Conv3d (each input patch is a
//         `in_ch · temporalPatch · patch · patch` row, projected by a
//         single CPU GEMM),
//       – RMSNorm q_norm / k_norm on each attention head,
//       – fused QKV projection (+ bias) per attention block,
//       – bidirectional attention (no causal mask, no KV cache),
//       – SwiGLU MLP (gate_proj / up_proj / down_proj, all with bias),
//       – post-norm RMSNorm before the merger,
//       – spatial merge: Conv2d downsample (stride = spatialMergeSize)
//         then a PatchMerger (proj → GELU post-norm → gate-up-down SwiGLU).
//   • GLM-OCR text backbone — 16 decoder layers with a sandwiched
//     pre-post-norm scheme: input_layernorm → self_attn →
//     post_self_attn_layernorm → residual, then post_attention_layernorm →
//     mlp → post_mlp_layernorm → residual. M-RoPE drives position encoding;
//     approximated with sequential scalar positions for this coherence-first
//     port.
//   • Text weights live under `language_model.model.*` /
//     `language_model.lm_head.*`; vision weights under `model.visual.*`.
//     The checkpoint also ships a `num_nextn_predict_layers = 1` extra
//     prediction head at layer index `hiddenLayers`; that layer is skipped
//     during loading.
//
// Coherence-first port:
//   • Vision attention runs on CPU — patch counts are small (at most a few
//     thousand), so an O(n²·d) attention per head is cheap next to the GPU
//     projection GEMMs and is unambiguously correct. `DispatchQueue.
//     concurrentPerform` parallelises the per-head loops.
//   • Text M-RoPE is approximated by sequential scalar positions (same
//     approach as Qwen25VL, Qwen3VL, LFM2-VL in the agent branch).
//   • A head-dim-agnostic GPU vision SDPA and true text M-RoPE are later
//     performance / fidelity passes.
//
// The vision tower internals (GlmOcrTextLayer, GlmOcrModel, GlmOcrVisionTower,
// GlmOcrVisionBlock, CPU primitives, and bundle helpers) live in
// `Models/Vision/GlmOcrVision.swift`. This file is the family orchestrator
// (load entrypoint + the model-type / architecture identifiers).

import Foundation
import Metal

// ─── Errors ──────────────────────────────────────────────────────────

public enum GlmOcrError: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "GlmOcr: checkpoint config is missing required fields"
        case .missingTensor(let name):
            return "GlmOcr: checkpoint is missing tensor '\(name)'"
        }
    }
}

// ─── Family registry ─────────────────────────────────────────────────

public enum GlmOcr {
    /// `model_type` for GLM-OCR checkpoints.
    public static let modelTypes: Set<String> = ["glm_ocr"]
    /// Architecture string for GLM-OCR HF conversions.
    public static let architectures: Set<String> = [
        "GlmOcrForConditionalGeneration"
    ]
    /// Default `image_token_id` for GLM-OCR checkpoints.
    public static let defaultImageTokenId = 59280
    /// Default `eos_token_id` (first in list) for GLM-OCR checkpoints.
    public static let defaultEosTokenId = 59246

    /// Build a `GlmOcrModel` from a `GlmOcrForConditionalGeneration`
    /// checkpoint: the dynamic-resolution ViT + the GLM-OCR text backbone,
    /// joined by image-token injection at the prefill step.
    public static func load(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> GlmOcrModel {
        guard let textCfgRaw = config.nested("text_config"),
            let visCfgRaw = config.nested("vision_config")
        else {
            throw GlmOcrError.missingConfig
        }
        let textCfg = ModelConfig(
            architecture: nil, modelType: "glm_ocr_text",
            raw: textCfgRaw)
        let visCfg = ModelConfig(
            architecture: nil, modelType: "glm_ocr_vision",
            raw: visCfgRaw)

        // ── Text hyper-parameters ──
        guard let hidden = textCfg.hiddenSize,
            let nLayers = textCfg.numLayers,
            let nHeads = textCfg.numAttentionHeads,
            let headDim = textCfg.headDim,
            let vocab = textCfg.vocabSize,
            let intermed = textCfg.intermediateSize,
            let eps = textCfg.rmsNormEps
        else {
            throw GlmOcrError.missingConfig
        }
        let nKVHeads = textCfg.numKeyValueHeads ?? nHeads
        let tieEmbed = textCfg.tieWordEmbeddings
        let ropeTheta: Float = {
            if let p = textCfg.nested("rope_parameters") {
                return Float((p["rope_theta"] as? Double) ?? 10_000)
            }
            return 10_000
        }()
        let maxSeq = textCfg.int("max_position_embeddings") ?? 131_072
        // The checkpoint ships one extra "nextn-predict" layer at index
        // `nLayers`. Skip it during loading.
        let nextNLayers = textCfg.int("num_nextn_predict_layers") ?? 0
        let quant = config.quantization

        // Text weights sit under the `language_model.` prefix.
        let tw = weights.glmOcrPrefixed("language_model.")

        // Embedding
        let embedTokens = try loadEmbedding(
            base: "model.embed_tokens", in: tw, hidden: hidden, quantization: quant)

        // Decoder layers
        var layers: [GlmOcrTextLayer] = []
        layers.reserveCapacity(nLayers)
        for i in 0 ..< nLayers {
            let p = "model.layers.\(i)"
            let qProj = try loadLinear(base: "\(p).self_attn.q_proj", in: tw, quantization: quant)
            let kProj = try loadLinear(base: "\(p).self_attn.k_proj", in: tw, quantization: quant)
            let vProj = try loadLinear(base: "\(p).self_attn.v_proj", in: tw, quantization: quant)
            let oProj = try loadLinear(base: "\(p).self_attn.o_proj", in: tw, quantization: quant)
            let gateProj = try loadLinear(base: "\(p).mlp.gate_proj", in: tw, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: tw, quantization: quant)
            let downProj = try loadLinear(base: "\(p).mlp.down_proj", in: tw, quantization: quant)
            // Sandwiched pre-post-norm: 4 RMSNorm weights per layer.
            let inputNorm = RMSNorm(
                weight: try tw.tensor(named: "\(p).input_layernorm.weight"),
                eps: Float(eps))
            let postAttnNorm = RMSNorm(
                weight: try tw.tensor(named: "\(p).post_self_attn_layernorm.weight"),
                eps: Float(eps))
            let postAttnLN2 = RMSNorm(
                weight: try tw.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: Float(eps))
            let postMlpNorm = RMSNorm(
                weight: try tw.tensor(named: "\(p).post_mlp_layernorm.weight"),
                eps: Float(eps))
            layers.append(
                GlmOcrTextLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    postAttnLN2: postAttnLN2, postMlpNorm: postMlpNorm,
                    hidden: hidden, nHeads: nHeads, nKVHeads: nKVHeads,
                    headDim: headDim, intermediate: intermed,
                    ropeTheta: ropeTheta))
        }
        // Skip the nextn-predict layer(s) — they are not needed for inference.
        _ = nextNLayers

        let finalNorm = RMSNorm(
            weight: try tw.tensor(named: "model.norm.weight"),
            eps: Float(eps))

        // LM head — may be tied to embed_tokens.
        let lmHead: AnyLinear
        if !tieEmbed, tw.has("lm_head.weight") {
            lmHead = try loadLinear(base: "lm_head", in: tw, quantization: quant)
        } else if let q = quant, [3, 4, 5, 6, 8].contains(q.bits),
            tw.isQuantized("model.embed_tokens")
        {
            let t = try tw.quantizedTriplet("model.embed_tokens")
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: q.bits, groupSize: q.groupSize))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if tw.isQuantized("model.embed_tokens"),
            let scales = try? tw.tensor(named: "model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        // ── Vision tower ──
        // Vision weights sit under `model.visual.` in the mlx-community
        // GLM-OCR conversion.
        let vw = weights.glmOcrPrefixed("model.visual.")
        let visionTower = try GlmOcrVisionTower.load(
            cfg: visCfg, textHidden: hidden,
            weights: vw, dtype: activationDtype, device: device)

        let imageTokenId = config.int("image_token_id") ?? GlmOcr.defaultImageTokenId
        let eosTokenId = config.eosTokenId ?? GlmOcr.defaultEosTokenId

        return GlmOcrModel(
            embedTokens: embedTokens, layers: layers,
            finalNorm: finalNorm, lmHead: lmHead,
            visionTower: visionTower,
            hidden: hidden, nLayers: nLayers, nHeads: nHeads,
            nKVHeads: nKVHeads, headDim: headDim, vocab: vocab,
            maxSeq: maxSeq, ropeTheta: ropeTheta, dtype: activationDtype,
            imageTokenId: imageTokenId, eosTokenId: eosTokenId,
            kvCacheKind: options.kvCache)
    }
}
