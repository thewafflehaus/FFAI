// Paligemma family — Google PaliGemma (SigLIP vision encoder + Gemma backbone).
//
// Architecture: SigLIP ViT-So400m (1152-dim, 27 layers) → linear projector
// → Gemma text decoder (18 layers, 2048-dim). Image tokens (1024 of them for
// the 448×448 checkpoint) are injected into the text model's residual stream
// replacing the `<image>` token embeddings.
//
// Weight layout in the 8-bit mlx-community checkpoint:
//   vision_tower.vision_model.*          — SigLIP encoder
//   multi_modal_projector.linear.*       — projector
//   language_model.model.*               — Gemma text decoder
//   (no lm_head — tied to embed_tokens)
//
// Reference: mlx-swift-lm/Libraries/MLXVLM/Models/Paligemma.swift
//
// All inference dispatches to GPU via Ops.* kernels:
//   • Vision encoder projections (Q/K/V/O, fc1/fc2) — Ops.gemm (plain)
//     or Ops.dequantGemv (quantized).
//   • SigLIP attention core — Ops.sdpaBidirectional(headDim: 72).
//   • LayerNorms — GPU rms_norm / layer_norm kernels.
//   • Patch-embedding conv — load-time weight reshape (CPU); the
//     per-image projection itself is a GPU GEMM.
//   • Projector — CpuLinear.forwardBatch dispatches GPU GEMM; bias
//     broadcast via Ops.add over a tiled bias tensor.
//   • Gemma text backbone — same Ops.* kernels as Llama.
//   • setImagePixels(_:) is called before generation; it computes the
//     vision features and stores them as a GPU Tensor. forward()
//     substitutes the stored image embedding at image-token positions.
//
// Vision tower internals (SigLIP layer, CpuEmbedding, CpuLinear,
// PaligemmaModel — the "Cpu" prefix on those types is historical;
// projections dispatch to the GPU) live in
// `Models/Vision/PaligemmaVision.swift`.

import Foundation
import Metal

// ─── Family entry point ──────────────────────────────────────────────────────

public enum Paligemma {
    public static let modelTypes: Set<String> = ["paligemma"]
    public static let architectures: Set<String> = ["PaliGemmaForConditionalGeneration"]

    public static func variant(for config: ModelConfig) throws -> any PaligemmaVariant.Type {
        return PaligemmaStandard.self
    }
}

public protocol PaligemmaVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> PaligemmaModel
}

public enum PaligemmaError: Error, CustomStringConvertible {
    case missingConfig(String)
    case imageNotSet
    public var description: String {
        switch self {
        case .missingConfig(let f): return "Paligemma: required config field missing: \(f)"
        case .imageNotSet: return "Paligemma: setImagePixels(_:) must be called before generation"
        }
    }
}

// ─── Standard variant ────────────────────────────────────────────────────────

public struct PaligemmaStandard: PaligemmaVariant {
    public static let availableCapabilities: Set<Capability> = [.textIn, .textOut, .visionIn]

    /// PaliGemma defaults — conservative for VQA / captioning tasks.
    /// Temperature 0 (greedy) matches the reference implementation's
    /// recommended inference mode for caption / answer tasks.
    public static let defaultGenerationParameters = GenerationParameters(
        maxTokens: 256,
        prefillStepSize: 512,
        temperature: 0,
        topP: 1.0,
        topK: 0,
        repetitionPenalty: 1.0
    )

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> PaligemmaModel {
        // ── Text backbone — delegate to the shared Gemma 2 family ────────────
        //
        // PaliGemma 2 ships with a Gemma 2 2B text backbone (vs PaliGemma 1's
        // Gemma 1 backbone). We promote `text_config` to a synthetic
        // root-level ModelConfig and hand it to `Gemma2Dense.loadModel`,
        // pointed at the `language_model.`-prefixed weight sub-tree. The
        // resulting `Gemma2Model` handles all four Gemma 2 norms per layer,
        // alternating sliding / full attention, the sqrt(hidden) embed-scale,
        // tied LM head, and per-layer KV cache split — no PaliGemma-specific
        // forward code needed.
        //
        // Same pattern Gemma3VL uses for its text backbone (see
        // `Gemma3VL.load` + `gemma3TextConfigWithDefaults`).
        guard let textRaw = config.nested("text_config") else {
            throw PaligemmaError.missingConfig("text_config")
        }
        // PaliGemma 2's `quantization` block lives at the ROOT config (HF
        // convention for VLMs — the same quant scheme covers vision +
        // text), and the text_config in 8-bit / 4-bit conversions is
        // silent on it. Merge the root-level field into the synthetic
        // text-only config so `Gemma2Dense.loadModel` sees it and
        // routes through `QuantizedLinear` / `QuantizedEmbedding`
        // instead of trying to plain-`gather` a U32-packed weight.
        var mergedTextRaw = textRaw
        if let rootQuant = config.raw["quantization"],
           mergedTextRaw["quantization"] == nil {
            mergedTextRaw["quantization"] = rootQuant
        }
        let textConfig = ModelConfig(
            architecture: "Gemma2ForCausalLM",
            modelType: "gemma2",
            raw: mergedTextRaw)
        let textEngine = try Gemma2Dense.loadModel(
            config: textConfig,
            weights: weights.prefixed("language_model."),
            options: options, device: device)

        // ── Vision config ────────────────────────────────────────────────────
        guard let visionRaw = config.nested("vision_config") else {
            throw PaligemmaError.missingConfig("vision_config")
        }
        func visInt(_ key: String) -> Int? {
            if let v = visionRaw[key] as? Int { return v }
            if let v = visionRaw[key] as? Double { return Int(v) }
            return nil
        }
        func visDouble(_ key: String) -> Double? {
            if let v = visionRaw[key] as? Double { return v }
            if let v = visionRaw[key] as? Int { return Double(v) }
            return nil
        }
        guard let visHidden    = visInt("hidden_size"),
              let visNLayers   = visInt("num_hidden_layers"),
              let visNHeads    = visInt("num_attention_heads"),
              let visIntermed  = visInt("intermediate_size"),
              let visPatchSize = visInt("patch_size"),
              let visImgSize   = visInt("image_size")
        else {
            throw PaligemmaError.missingConfig("vision_config fields")
        }
        let visNumChannels = visInt("num_channels") ?? 3
        let visLayerNormEps = Float(visDouble("layer_norm_eps") ?? 1e-6)
        let numImageTokens = (visImgSize / visPatchSize) * (visImgSize / visPatchSize)

        // ── Projector config ─────────────────────────────────────────────────
        // projection_dim lives at top level in the PaliGemma config.
        let projDim = config.int("projection_dim") ?? textEngine.hidden
        let imageTokenIndex = config.int("image_token_index") ?? 257152

        // ── Quantization ─────────────────────────────────────────────────────
        let quant = config.quantization

        // ── Vision encoder weights (SigLIP ViT) ──────────────────────────────
        // These are loaded into plain CPU float arrays. The vision encoder
        // runs once before generation using DispatchQueue.concurrentPerform.
        let visPrefix = "vision_tower.vision_model"

        // Patch embedding: [out_channels, kH, kW, in_channels] (already
        // transposed from PyTorch by the mlx-community converter).
        let patchW = try weights.tensor(named: "\(visPrefix).embeddings.patch_embedding.weight")
        let patchB = try weights.tensor(named: "\(visPrefix).embeddings.patch_embedding.bias")

        // Position embedding: quantized [numImageTokens, visHidden].
        let posEmbedding: CpuEmbedding
        let posBase = "\(visPrefix).embeddings.position_embedding"
        if weights.isQuantized(posBase) {
            let t = try weights.quantizedTriplet(posBase)
            let gs = quant?.groupSize ?? 64
            posEmbedding = CpuEmbedding(
                quantized: t.weight, scales: t.scales, biases: t.biases,
                hidden: visHidden, bits: quant?.bits ?? 8, groupSize: gs
            )
        } else {
            posEmbedding = CpuEmbedding(weight: try weights.tensor(named: "\(posBase).weight"))
        }

        // Encoder layers.
        var visLayers: [SigLIPLayer] = []
        visLayers.reserveCapacity(visNLayers)
        for i in 0..<visNLayers {
            let p = "\(visPrefix).encoder.layers.\(i)"
            let layer = try SigLIPLayer.load(
                prefix: p, from: weights,
                hidden: visHidden, intermediate: visIntermed,
                numHeads: visNHeads, eps: visLayerNormEps,
                quantization: quant
            )
            visLayers.append(layer)
        }

        // Post layer norm: [visHidden] weight + bias (full precision F16 in
        // the checkpoint).
        let postLNW = try weights.tensor(named: "\(visPrefix).post_layernorm.weight")
        let postLNB = try weights.tensor(named: "\(visPrefix).post_layernorm.bias")

        // ── Projector weights ─────────────────────────────────────────────────
        // A single linear with bias: visHidden → projDim (= textHidden).
        let projBase = "multi_modal_projector.linear"
        let projLinear: CpuLinear
        if weights.isQuantized(projBase) {
            let t = try weights.quantizedTriplet(projBase)
            let gs = quant?.groupSize ?? 64
            projLinear = CpuLinear(
                quantized: t.weight, scales: t.scales, biases: t.biases,
                outDim: projDim, inDim: visHidden,
                bits: quant?.bits ?? 8, groupSize: gs,
                bias: try weights.tensor(named: "\(projBase).bias")
            )
        } else {
            projLinear = CpuLinear(
                weight: try weights.tensor(named: "\(projBase).weight"),
                bias: try weights.tensor(named: "\(projBase).bias")
            )
        }

        return PaligemmaModel(
            textEngine: textEngine,
            // Vision
            patchW: patchW, patchB: patchB, posEmbedding: posEmbedding,
            visLayers: visLayers, postLNW: postLNW, postLNB: postLNB,
            visHidden: visHidden, numImageTokens: numImageTokens,
            visNumChannels: visNumChannels, visPatchSize: visPatchSize, visImgSize: visImgSize,
            // Projector
            projLinear: projLinear, projDim: projDim,
            // Shared
            imageTokenIndex: imageTokenIndex
        )
    }
}
