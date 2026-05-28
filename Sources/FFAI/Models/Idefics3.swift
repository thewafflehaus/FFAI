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
// Idefics3 family — HuggingFace Idefics3 / SmolVLM (the predecessor to SmolVLM2).
//
// Architecture: SigLIP-style ViT vision encoder + pixel-shuffle connector +
// Llama-style language backbone. Config type is "idefics3" with architecture
// string "Idefics3ForConditionalGeneration".
//
// Reference:
//   mlx-swift-lm: Libraries/MLXVLM/Models/Idefics3.swift
//   mlx-vlm: mlx_vlm/models/idefics3.py
//   HuggingFace: HuggingFaceM4/Idefics3-8B-Llama3
//
// Design mirrors SmolVLM2.swift exactly — Idefics3 is the ancestor and
// SmolVLM2 is a renamed descendant with a larger scale_factor (4 vs 2).
//
// Vision tower internals (config structs, CPU vision ops, encoder layers,
// Idefics3VisionEncoder, Idefics3Connector, Idefics3RemappedBundle,
// Idefics3Model) live in `Models/Vision/Idefics3Vision.swift`.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────────────

public enum Idefics3 {
    public static let modelTypes: Set<String> = ["idefics3"]
    public static let architectures: Set<String> = ["Idefics3ForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> Idefics3Dense.Type {
        return Idefics3Dense.self
    }
}

public enum Idefics3Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingVisionConfig(String)
    case missingTextConfig(String)

    public var description: String {
        switch self {
        case .missingConfig(let f): return "Idefics3: missing top-level config field: \(f)"
        case .missingVisionConfig(let f): return "Idefics3: missing vision_config.\(f)"
        case .missingTextConfig(let f): return "Idefics3: missing text_config.\(f)"
        }
    }
}

// ─── Dense variant ───────────────────────────────────────────────────────────

public struct Idefics3Dense {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut, .imageIn]
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
    ) throws -> Idefics3Model {
        guard let vcRaw = config.nested("vision_config") else {
            throw Idefics3Error.missingConfig("vision_config")
        }
        guard let tcRaw = config.nested("text_config") else {
            throw Idefics3Error.missingConfig("text_config")
        }
        let vc = try Idefics3VisionConfig(from: vcRaw)
        let tc = try Idefics3TextConfig(from: tcRaw)
        let idefCfg = try Idefics3Config(from: config.raw)

        // ─── Remap weight prefixes ────────────────────────────────────────────
        // HF Idefics3 stores weights as:
        //   model.text_model.*       → language_model.*
        //   model.vision_model.*     → vision_model.*
        //   model.connector.*        → connector.*
        //   lm_head.*                → language_model.lm_head.*  (when not tied)
        //
        // mlx-community conversions may already use the remapped form; the
        // `Idefics3RemappedBundle` wrapper transparently tries the flat key first,
        // then the HF form, making it idempotent for both checkpoint styles.
        let remapped = Idefics3RemappedBundle(weights)

        // ─── Vision encoder & connector ──────────────────────────────────────
        let visionEncoder = try Idefics3VisionEncoder(cfg: vc, weights: remapped)
        let connector = try Idefics3Connector(cfg: idefCfg, weights: remapped)

        // ─── Language backbone (Llama-style) ─────────────────────────────────
        let quant = config.quantization

        let embedTokens = try loadIdefics3Embedding(
            base: "language_model.embed_tokens", in: remapped,
            hidden: tc.hiddenSize, quantization: quant
        )

        var llamaLayers: [LlamaLayer] = []
        llamaLayers.reserveCapacity(tc.numHiddenLayers)
        for i in 0 ..< tc.numHiddenLayers {
            let p = "language_model.layers.\(i)"
            let qProj = try loadIdefics3Linear(
                base: "\(p).self_attn.q_proj", in: remapped, quantization: quant)
            let kProj = try loadIdefics3Linear(
                base: "\(p).self_attn.k_proj", in: remapped, quantization: quant)
            let vProj = try loadIdefics3Linear(
                base: "\(p).self_attn.v_proj", in: remapped, quantization: quant)
            let oProj = try loadIdefics3Linear(
                base: "\(p).self_attn.o_proj", in: remapped, quantization: quant)
            let gateProj = try loadIdefics3Linear(
                base: "\(p).mlp.gate_proj", in: remapped, quantization: quant)
            let upProj = try loadIdefics3Linear(
                base: "\(p).mlp.up_proj", in: remapped, quantization: quant)
            let downProj = try loadIdefics3Linear(
                base: "\(p).mlp.down_proj", in: remapped, quantization: quant)
            let inputNorm = RMSNorm(
                weight: try remapped.tensor(named: "\(p).input_layernorm.weight"),
                eps: tc.rmsNormEps)
            let postAttnNorm = RMSNorm(
                weight: try remapped.tensor(named: "\(p).post_attention_layernorm.weight"),
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
            weight: try remapped.tensor(named: "language_model.norm.weight"),
            eps: tc.rmsNormEps)

        // LM head — Idefics3-8B has tieWordEmbeddings == false and ships lm_head.weight.
        let lmHead: AnyLinear
        if !tc.tieWordEmbeddings, remapped.has("language_model.lm_head.weight") {
            lmHead = try loadIdefics3Linear(
                base: "language_model.lm_head", in: remapped, quantization: quant)
        } else if let q = quant, remapped.isQuantized("language_model.embed_tokens") {
            let t = try remapped.quantizedTriplet("language_model.embed_tokens")
            lmHead = AnyLinear(
                QuantizedLinear(
                    weight: t.weight, scales: t.scales, biases: t.biases,
                    bits: q.bits, groupSize: q.groupSize
                ))
        } else {
            lmHead = AnyLinear(Linear(weight: embedTokens.weight))
        }

        // Activation dtype
        let activationDtype: DType
        if remapped.isQuantized("language_model.embed_tokens"),
            let scales = try? remapped.tensor(named: "language_model.embed_tokens.scales")
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
            maxContextWindow: tc.maxPositionEmbeddings, ropeTheta: tc.ropeTheta,
            dtype: activationDtype,
            kvCacheKind: options.kvCache
        )

        return Idefics3Model(
            llamaModel: llamaModel,
            visionEncoder: visionEncoder,
            connector: connector,
            cfg: idefCfg,
            device: device
        )
    }
}
