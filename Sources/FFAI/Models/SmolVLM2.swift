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
// SmolVLM2 family — SmolVLM2 / Idefics3-style vision-language model.
//
// Architecture: SigLIP-style ViT vision encoder + pixel-shuffle connector +
// Llama-style language backbone. Config type is "smolvlm" with architecture
// string "SmolVLMForConditionalGeneration".
//
// Reference: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
// Upstream Python impl: transformers models/idefics3 (SmolVLM2 = Idefics3)
//
// SmolVLM2 is unusual — it does not go through VisionModel. The engine
// (`SmolVLM2Model`) implements `LanguageModel` directly and handles vision
// prefill internally. The vision tower internals (config structs, CPU vision
// ops, encoder layers, `SmolVLM2VisionEncoder`, `SmolVLM2Connector`,
// `SmolVLM2Model`) live in `Models/Vision/SmolVLM2Vision.swift`. This file
// is the public dispatch surface (registry metadata + `SmolVLM2Dense.loadModel`).

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────────────

public enum SmolVLM2 {
    public static let modelTypes: Set<String> = ["smolvlm"]
    public static let architectures: Set<String> = ["SmolVLMForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> SmolVLM2Dense.Type {
        return SmolVLM2Dense.self
    }
}

public enum SmolVLM2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingVisionConfig(String)
    case missingTextConfig(String)

    public var description: String {
        switch self {
        case .missingConfig(let f): return "SmolVLM2: missing top-level config field: \(f)"
        case .missingVisionConfig(let f): return "SmolVLM2: missing vision_config.\(f)"
        case .missingTextConfig(let f): return "SmolVLM2: missing text_config.\(f)"
        }
    }
}

// ─── Dense variant ───────────────────────────────────────────────────────────

public struct SmolVLM2Dense {
    /// Capabilities a SmolVLM2 checkpoint exposes. Text + image + video.
    ///
    /// SmolVLM2 does not declare a separate `video_token_id` in its
    /// config — the HF checkpoint (SmolVLM2-500M-Video-Instruct) reuses
    /// the same `<image>` placeholder (id 49190) for both image and video
    /// frames. Each video frame is encoded as an independent image through
    /// the SigLIP ViT + pixel-shuffle connector, producing
    /// `nPatches / scaleFactor²` tokens per frame. The caller should build
    /// a prompt with `frameCount × imageTokensPerFrame` image-token
    /// placeholders, then pass the concatenated per-frame embeddings to
    /// `SmolVLM2Model.prefillWithImage`.
    public static let availableCapabilities: Set<Capability> = [
        .textIn, .textOut, .imageIn, .videoIn,
    ]
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 1024,
        temperature: 0.0,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> SmolVLM2Model {
        // Parse sub-configs from the raw JSON
        guard let vcRaw = config.nested("vision_config") else {
            throw SmolVLM2Error.missingConfig("vision_config")
        }
        guard let tcRaw = config.nested("text_config") else {
            throw SmolVLM2Error.missingConfig("text_config")
        }
        let vc = try SmolVLM2VisionConfig(from: vcRaw)
        let tc = try SmolVLM2TextConfig(from: tcRaw)
        let smolCfg = try SmolVLM2Config(from: config.raw)

        // ─── Vision encoder & connector ─────────────────────────────────
        let visionEncoder = try SmolVLM2VisionEncoder(cfg: vc, weights: weights)
        let connector = try SmolVLM2Connector(cfg: smolCfg, weights: weights)

        // ─── Language backbone (Llama-style) ────────────────────────────
        let quant = config.quantization

        let embedTokens = try loadEmbedding(
            base: "language_model.embed_tokens", in: weights,
            hidden: tc.hiddenSize, quantization: quant
        )

        var llamaLayers: [LlamaLayer] = []
        llamaLayers.reserveCapacity(tc.numHiddenLayers)
        for i in 0 ..< tc.numHiddenLayers {
            let p = "language_model.layers.\(i)"
            let qProj = try loadLinear(
                base: "\(p).self_attn.q_proj", in: weights, quantization: quant)
            let kProj = try loadLinear(
                base: "\(p).self_attn.k_proj", in: weights, quantization: quant)
            let vProj = try loadLinear(
                base: "\(p).self_attn.v_proj", in: weights, quantization: quant)
            let oProj = try loadLinear(
                base: "\(p).self_attn.o_proj", in: weights, quantization: quant)
            let gateProj = try loadLinear(
                base: "\(p).mlp.gate_proj", in: weights, quantization: quant)
            let upProj = try loadLinear(base: "\(p).mlp.up_proj", in: weights, quantization: quant)
            let downProj = try loadLinear(
                base: "\(p).mlp.down_proj", in: weights, quantization: quant)
            let inputNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).input_layernorm.weight"),
                eps: tc.rmsNormEps)
            let postAttnNorm = RMSNorm(
                weight: try weights.tensor(named: "\(p).post_attention_layernorm.weight"),
                eps: tc.rmsNormEps)
            llamaLayers.append(
                LlamaLayer(
                    qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
                    gateProj: gateProj, upProj: upProj, downProj: downProj,
                    inputNorm: inputNorm, postAttnNorm: postAttnNorm,
                    hidden: tc.hiddenSize,
                    nHeads: tc.numAttentionHeads, nKVHeads: tc.numKeyValueHeads,
                    headDim: tc.headDim, intermediate: tc.intermediateSize,
                    ropeTheta: tc.ropeTheta,
                    ropeScaling: .none
                ))
        }

        let finalNorm = RMSNorm(
            weight: try weights.tensor(named: "language_model.norm.weight"),
            eps: tc.rmsNormEps)

        // LM head — SmolVLM2-500M uses tieWordEmbeddings == false and has lm_head.weight
        let lmHead: AnyLinear
        if !tc.tieWordEmbeddings, weights.has("language_model.lm_head.weight") {
            lmHead = try loadLinear(
                base: "language_model.lm_head", in: weights, quantization: quant)
        } else if let q = quant, weights.isQuantized("language_model.embed_tokens") {
            let t = try weights.quantizedTriplet("language_model.embed_tokens")
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: q.bits, groupSize: q.groupSize
                ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        let activationDtype: DType
        if weights.isQuantized("language_model.embed_tokens"),
            let scales = try? weights.tensor(named: "language_model.embed_tokens.scales")
        {
            activationDtype = scales.dtype
        } else {
            activationDtype = embedTokens.weight.dtype
        }

        let llamaModel = LlamaModel(
            embedTokens: embedTokens, layers: llamaLayers,
            finalNorm: finalNorm, lmHead: lmHead,
            hidden: tc.hiddenSize, nLayers: tc.numHiddenLayers,
            nHeads: tc.numAttentionHeads, nKVHeads: tc.numKeyValueHeads,
            headDim: tc.headDim, vocab: tc.vocabSize,
            maxSeq: tc.maxPositionEmbeddings, ropeTheta: tc.ropeTheta,
            dtype: activationDtype,
            kvCacheKind: options.kvCache
        )

        return SmolVLM2Model(
            llamaModel: llamaModel,
            visionEncoder: visionEncoder,
            connector: connector,
            cfg: smolCfg,
            device: device
        )
    }
}
