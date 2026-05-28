// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Model — public entry point users interact with. Resolves a model
// id-or-path, downloads via HF if needed, decodes config, dispatches to
// the right family file, loads weights, and exposes a forward()/generate()
// surface.

import Foundation
import Metal
import Tokenizers

public enum ModelError: Error, CustomStringConvertible {
    case unsupportedArchitecture(String)
    case unsupportedModelType(String)
    case capabilityNotAvailable(Capability)
    case visionModelNotIntegrated(String)
    /// The model's weights + a minimal KV cache exceed the wired-memory
    /// budget — it can't run at any usable context on this machine /
    /// budget. Raised by the over-allocation guard (`MemoryBudget`).
    case insufficientMemory(weightBytes: Int, budgetBytes: Int, detail: String)
    /// An explicit `LoadOptions.wiredLimitBytes` exceeds the safe maximum
    /// (physical RAM − the OS reserve). Raised at load so a deliberate
    /// over-request is reported rather than silently clamped.
    case wiredLimitTooHigh(requestedBytes: Int, safeMaxBytes: Int, osReserveBytes: Int)

    public var description: String {
        switch self {
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        case .unsupportedModelType(let m): return "Unsupported model_type: \(m)"
        case .capabilityNotAvailable(let c): return "Capability not available: \(c)"
        case .visionModelNotIntegrated(let a):
            return "Vision-language checkpoint '\(a)' detected. The FFAI "
                + "vision foundation (VisionEncoder, ImagePreprocessing, "
                + "VisionModel cross-modal splice, conv2d/patch_embed/rope_2d "
                + "Ops) is in tree, but this VL family is not yet wired to "
                + "a checkpoint loader. Load the text-only checkpoint, or "
                + "compose a VisionModel directly from VisionEncoder + the text "
                + "engine."
        case .insufficientMemory(_, _, let detail):
            return "Insufficient memory: \(detail)"
        case .wiredLimitTooHigh(let requested, let safeMax, let reserve):
            let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1_073_741_824.0) }
            return "LoadOptions.wiredLimitBytes (\(gb(requested))) exceeds the safe "
                + "maximum \(gb(safeMax)) (physical RAM minus a \(gb(reserve)) OS "
                + "reserve). Lower wiredLimitBytes to \(gb(safeMax)) or less, or free "
                + "memory / use a machine with more RAM."
        }
    }
}

/// Architecture strings that identify a vision-language checkpoint. A
/// VL checkpoint carries a `vision_config` block and prefixes its text
/// weights under `language_model.*`; the registry recognizes these so a
/// VL load fails with an actionable `visionModelNotIntegrated` error
/// rather than a generic "unsupported architecture".
public enum VisionLanguageArchitectures {
    public static let architectures: Set<String> = [
        "Gemma3ForConditionalGeneration",
        "GlmOcrForConditionalGeneration",
        "Idefics3ForConditionalGeneration",
        "Lfm2VlForConditionalGeneration",
        "LlavaForConditionalGeneration",  // Pixtral-12B
        "LlavaQwen2ForCausalLM",  // FastVLM (Apple FastViTHD + Qwen2)
        "MiniCPMV4_6ForConditionalGeneration",
        "Mistral3ForConditionalGeneration",  // Mistral Small 3.1
        "PaliGemmaForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "SmolVLMForConditionalGeneration",
        // Note: `Gemma4ForConditionalGeneration` is intentionally NOT
        // listed — it is shared by text-only Gemma 4 checkpoints. The
        // `vision_config`-presence check below distinguishes the VL
        // conversion, which the dispatch routes to `Gemma4VL.load`.
    ]

    /// True if `config` describes a VL checkpoint — by architecture
    /// string, by the presence of a `vision_config` block, or by a
    /// VL-only `model_type` (e.g. `paligemma` checkpoints carry the
    /// model_type with or without a vision_config nested block on
    /// partial test fixtures).
    public static func isVisionLanguage(_ config: ModelConfig) -> Bool {
        if let arch = config.architecture, architectures.contains(arch) {
            return true
        }
        if config.nested("vision_config") != nil {
            return true
        }
        // VL-only model_types — none of these have a text-only
        // counterpart, so a bare model_type is enough signal.
        if let mt = config.modelType, Paligemma.modelTypes.contains(mt) {
            return true
        }
        return false
    }
}

/// True if `config` is a Nemotron Nano VL checkpoint — a VL checkpoint
/// (`vision_config` present) whose text backbone is a NemotronH hybrid
/// (its `text_config.model_type` is `nemotron_h`). The Nemotron Nano VL
/// conversion does not carry a single canonical top-level architecture
/// string, so the text backbone's `model_type` is the reliable signal.
func isNemotronVisionLanguage(_ config: ModelConfig) -> Bool {
    guard config.nested("vision_config") != nil,
        let tc = config.nested("text_config")
    else { return false }
    let textModelType = (tc["model_type"] as? String) ?? ""
    if NemotronH.modelTypes.contains(textModelType) { return true }
    let textArch = (tc["architectures"] as? [String])?.first ?? ""
    return NemotronH.architectures.contains(textArch)
}

/// True if `config` is a Nemotron-Labs-Diffusion VLM checkpoint — a VL
/// checkpoint (`vision_config` present) whose top-level model_type /
/// architecture declares the diffusion VLM. Unlike the hybrid Nemotron
/// VL above, the diffusion VLM ships a canonical top-level model_type
/// (`nemotron_labs_diffusion_vlm`) and architecture string
/// (`NemotronLabsDiffusionVLMModel`), so we route on those directly.
func isNemotronDiffusionVisionLanguage(_ config: ModelConfig) -> Bool {
    guard config.nested("vision_config") != nil else { return false }
    if let mt = config.modelType,
        NemotronDiffusionVL.modelTypes.contains(mt)
    {
        return true
    }
    if let arch = config.architecture,
        NemotronDiffusionVL.architectures.contains(arch)
    {
        return true
    }
    return false
}

/// Routes a config to the right family file. Family files declare which
/// architecture / model_type strings they handle. Add a new family by
/// extending `dispatchAndLoad` here.
public enum ModelRegistry {
    /// Engine + the variant-declared generation defaults. The defaults
    /// flow into the `Model` so callers can read them off without
    /// knowing the concrete family.
    public struct Loaded {
        public let engine: any LanguageModel
        public let defaultGenerationParameters: GenerationParameters
        /// Capabilities the loaded variant supports. Text-only families
        /// report `Capability.textOnly`; VL variants add `.imageIn`.
        public let availableCapabilities: Set<Capability>
        /// The composed vision-language model, when the checkpoint is a
        /// VLM. `nil` for text-only families. The `engine` is the VL
        /// model's text backbone, so text-only generation works
        /// regardless; `vlModel` adds the cross-modal image path.
        public let vlModel: VisionModel?

        public init(
            engine: any LanguageModel,
            defaultGenerationParameters: GenerationParameters,
            availableCapabilities: Set<Capability> = Capability.textOnly,
            vlModel: VisionModel? = nil
        ) {
            self.engine = engine
            self.defaultGenerationParameters = defaultGenerationParameters
            self.availableCapabilities = availableCapabilities
            self.vlModel = vlModel
        }
    }

    public static func dispatchAndLoad(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Loaded {
        // Vision-language checkpoints carry a nested `vision_config` and
        // prefix their text weights under `language_model.*`.
        if VisionLanguageArchitectures.isVisionLanguage(config) {
            // Gemma 3 VL — SigLIP tower + Gemma 3 text backbone. Fully
            // wired: the SigLIP architecture is exactly `VisionEncoder`.
            if config.architecture == "Gemma3ForConditionalGeneration" {
                let vlm = try Gemma3VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma3Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // LFM2-VL — SigLIP2 ViT + LiquidAI LFM2 conv+attention text
            // backbone, joined by a pixel-unshuffle + MLP projector that
            // collapses 256 patch tokens to 64 fused image tokens.
            if config.architecture == "Lfm2VlForConditionalGeneration" {
                let vlm = try LFM2VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                let lfm2DefaultParams = GenerationParameters(
                    maxTokens: 256, prefillStepSize: 1024,
                    temperature: 0.0, topP: 1.0, topK: 0,
                    minP: 0.0, repetitionPenalty: 1.0)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: lfm2DefaultParams,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Qwen 2.5-VL — dynamic-resolution windowed-attention ViT
            // tower + the Qwen 2.x text backbone (routed through the
            // Llama dense engine, which now supports embedding-input
            // forward for the VLM splice).
            if config.architecture == "Qwen2_5_VLForConditionalGeneration" {
                let vlm = try Qwen25VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Qwen25VL.availableCapabilities,
                    vlModel: vlm)
            }
            // Qwen 2-VL — dynamic-resolution full-attention ViT tower
            // (LayerNorm pre-norms, GELU MLP, pure M-RoPE, no windowing) +
            // the Qwen 2 text backbone (routed through the Llama dense engine).
            if config.architecture == "Qwen2VLForConditionalGeneration" {
                let vlm = try Qwen2VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Qwen2VL.availableCapabilities,
                    vlModel: vlm)
            }
            // Qwen 3-VL — dynamic-resolution full-attention ViT tower
            // (LayerNorm pre-norms, GELU MLP, learned position table) +
            // the Qwen 3 dense text backbone, joined by the splice.
            if config.architecture == "Qwen3VLForConditionalGeneration" {
                let vlm = try Qwen3VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen3Dense.defaultGenerationParameters,
                    availableCapabilities: Qwen3VL.availableCapabilities,
                    vlModel: vlm)
            }
            // Qwen 3-VL-MoE — the Qwen3-VL vision tower + the Qwen 3.5
            // mixture-of-experts hybrid text backbone (Gated Delta Net ↔
            // attention, block-sparse MoE FFN), joined by the splice.
            if config.architecture == "Qwen3VLMoeForConditionalGeneration" {
                let vlm = try Qwen3VLMoe.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                    availableCapabilities: Qwen3VLMoe.availableCapabilities,
                    vlModel: vlm)
            }
            // Gemma 4 VL — the bespoke Gemma 4 ViT tower (RoPE attention,
            // q/k/v norms, attention-pooling head) + multi-modal embedder
            // + the Gemma 4 text backbone, joined by the splice.
            if config.architecture == "Gemma4ForConditionalGeneration" {
                let vlm = try Gemma4VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Gemma4Dense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // FastVLM — Apple's FastViTHD vision tower + mlp2x_gelu projector
            // + Qwen2 text backbone. Architecture string is
            // `LlavaQwen2ForCausalLM`; model_type is `llava_qwen2`.
            // Dispatch via architecture first (more specific), then
            // model_type for forward compatibility.
            if let arch = config.architecture, FastVLM.architectures.contains(arch) {
                let vlm = try FastVLM.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            if let mt = config.modelType, FastVLM.modelTypes.contains(mt) {
                let vlm = try FastVLM.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Pixtral — Mistral AI's Pixtral-12B vision-language model.
            // The mlx-community conversion ships `model_type = "pixtral"`
            // and (from HF auto-model mapping) architecture
            // `LlavaForConditionalGeneration`. Dispatch via model_type so
            // the same code handles any future Pixtral variant regardless
            // of architecture string.
            if let mt = config.modelType, Pixtral.modelTypes.contains(mt) {
                let vlm = try Pixtral.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Mistral3 — Mistral Small 3.1 vision-language model. Shares
            // the Pixtral ViT tower but uses a different projector
            // (RMSNorm + patch merger 2×2 spatial unfold + linear + GELU
            // + linear). Dispatched via model_type "mistral3".
            if let mt = config.modelType, Mistral3.modelTypes.contains(mt) {
                let vlm = try Mistral3.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Idefics3 — HuggingFace Idefics3 (SmolVLM ancestor). SigLIP
            // encoder + pixel-shuffle connector + Llama text backbone.
            // The engine is itself an Idefics3Model exposing
            // `encodeImage(...)` + `prefillWithImage(...)` directly;
            // VisionModel adapter integration is a follow-up.
            // GLM-OCR — Zhipu's OCR-specialised VLM (GLM-Lite text backbone +
            // dynamic-resolution ViT). Engine is itself a GlmOcrModel
            // exposing generate(image:promptTokens:...) directly.
            if GlmOcr.modelTypes.contains(config.modelType ?? "")
                || GlmOcr.architectures.contains(config.architecture ?? "")
            {
                let m = try GlmOcr.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: LlamaDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]))
            }
            if Idefics3.modelTypes.contains(config.modelType ?? "")
                || Idefics3.architectures.contains(config.architecture ?? "")
            {
                let m = try Idefics3Dense.loadModel(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: Idefics3Dense.defaultGenerationParameters,
                    availableCapabilities: Idefics3Dense.availableCapabilities)
            }
            // PaliGemma — SigLIP + Gemma backbone. Engine is itself a
            // PaligemmaModel exposing setImagePixels(_:) + standard
            // LanguageModel forward; vision substitution happens inside
            // the forward at image-token positions.
            if Paligemma.modelTypes.contains(config.modelType ?? "")
                || Paligemma.architectures.contains(config.architecture ?? "")
            {
                let variant = try Paligemma.variant(for: config)
                let m = try variant.loadModel(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: m,
                    defaultGenerationParameters: variant.defaultGenerationParameters,
                    availableCapabilities: variant.availableCapabilities)
            }
            // MiniCPM-V 4.6 — SigLIP2-400M encoder + `vit_merger` (window
            // cross-attn merger injected after encoder layer 6 in the
            // default 16× mode) + final `merger` (2×2 reduction + project
            // to text hidden) + Qwen 3.5 text backbone, joined by the
            // splice. v1 ships a fixed 448×448 single-tile input.
            if config.architecture == "MiniCPMV4_6ForConditionalGeneration"
                || config.modelType == "minicpmv4_6"
            {
                let vlm = try MiniCPMV4_6.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                    availableCapabilities: MiniCPMV4_6.availableCapabilities,
                    vlModel: vlm)
            }
            // Nemotron-VLM — NVIDIA's Nemotron Nano VL: a ViT tower +
            // multi-modal projector + the NemotronH stack-interleaved
            // hybrid text backbone. Detected by a `text_config` whose
            // `model_type` is `nemotron_h` (the VL conversion does not
            // carry a single canonical top-level architecture string).
            if isNemotronVisionLanguage(config) {
                let vlm = try NemotronVL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: NemotronHHybrid.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Nemotron-Labs-Diffusion VLM — Pixtral ViT vision tower +
            // Mistral3-style patch-merger projector + the
            // NemotronDiffusion tri-mode text backbone. The diffusion
            // VLM ships with a canonical top-level model_type /
            // architecture so it routes directly (unlike the hybrid
            // Nemotron VL above, which relies on the `text_config`
            // sniff).
            if isNemotronDiffusionVisionLanguage(config) {
                let vlm = try NemotronDiffusionVL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: NemotronDiffusionDense.defaultGenerationParameters,
                    availableCapabilities: Capability.textOnly.union([.imageIn]),
                    vlModel: vlm)
            }
            // Qwen 3.5-VL — the VL variant of Qwen 3.5. Shares the
            // `Qwen3_5ForConditionalGeneration` architecture string with
            // the text-only Qwen 3.5 release; the only signal we have to
            // disambiguate is the presence of an actual vision tower in
            // the safetensors (vision_config alone is unreliable since
            // some text-only Qwen 3.5 checkpoints ship a vestigial copy).
            // Probe both published vision-tower layouts: `model.visual.*`
            // (raw HF release) and `vision_tower.*` (mlx-community
            // restructured conversions).
            if let arch = config.architecture, Qwen35.architectures.contains(arch),
                config.subConfig("vision_config") != nil,
                weights.has("model.visual.patch_embed.proj.weight")
                    || weights.has("vision_tower.patch_embed.proj.weight")
            {
                let vlm = try Qwen35VL.load(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: vlm.engine,
                    defaultGenerationParameters: Qwen35Hybrid.defaultGenerationParameters,
                    availableCapabilities: Qwen35VL.availableCapabilities,
                    vlModel: vlm)
            }
            // Text-only Qwen3.5 / 3.6 checkpoints with a vestigial
            // `vision_config` block (no vision tensors). The architecture
            // string is a Qwen3.5 MoE family name and we have a text-only
            // loader — route to it instead of throwing the
            // VL-not-integrated error.
            if let arch = config.architecture, Qwen35.architectures.contains(arch) {
                return try loadQwen35(
                    config: config, weights: weights,
                    options: options, device: device)
            }
            // SmolVLM2 — handles image+text internally via the
            // `prefillWithImage` API on its `LanguageModel` engine
            // (a Llama backbone wrapped with a CPU SigLIP vision tower
            // + pixel-shuffle connector). Does NOT use the VisionModel
            // splice; returns a plain `Loaded` whose `engine` is the
            // composite `SmolVLM2Model` and exposes `.imageIn`.
            if let arch = config.architecture, SmolVLM2.architectures.contains(arch) {
                let engine = try SmolVLM2Dense.loadModel(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: engine,
                    defaultGenerationParameters: SmolVLM2Dense.defaultGenerationParameters,
                    availableCapabilities: SmolVLM2Dense.availableCapabilities)
            }
            if let mt = config.modelType, SmolVLM2.modelTypes.contains(mt) {
                let engine = try SmolVLM2Dense.loadModel(
                    config: config, weights: weights,
                    options: options, device: device)
                return Loaded(
                    engine: engine,
                    defaultGenerationParameters: SmolVLM2Dense.defaultGenerationParameters,
                    availableCapabilities: SmolVLM2Dense.availableCapabilities)
            }
            // Other VL families — the FFAI vision foundation
            // (VisionEncoder, ImagePreprocessing, VisionModel splice,
            // conv2d/patch_embed/rope_2d Ops) is in tree, but these
            // towers are not yet wired to a checkpoint loader. Fail with
            // an actionable error rather than a generic "unsupported".
            throw ModelError.visionModelNotIntegrated(
                config.architecture ?? config.modelType ?? "<unknown>")
        }
        if let arch = config.architecture, Llama.architectures.contains(arch) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Llama.modelTypes.contains(mt) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Mistral and Llama share the same weight layout + forward shape.
        // The Mistral family enum routes through the Llama loader so
        // every Mistral 7B / Nemo / Small checkpoint Just Works without
        // a separate dense engine.
        if let arch = config.architecture, Mistral.architectures.contains(arch) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Mistral.modelTypes.contains(mt) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Llama-compatible families: SmolLM 1/2/3, OLMo 1/2, Granite 3,
        // Yi, InternLM 2. Same weight layout + forward shape as Llama 3;
        // optional QKV biases auto-detected by loadLinear. Each has its
        // own family root under `Models/` (per the universal "one file
        // per family root" rule) but they all dispatch through
        // `loadLlama` until / unless they diverge from the Llama-3 shape.
        //
        // NOTE: Starcoder 2 was previously in this set but is
        // structurally distinct (LayerNorm with bias, GELU-tanh
        // single-projection MLP with c_fc/c_proj names, `norm_epsilon`
        // config field). It now routes through its own loader; see
        // below.
        let llamaCompatibleArchs: Set<String> =
            SmolLM.architectures
            .union(OLMo.olmo1Architectures)
            .union(Yi.architectures)
        let llamaCompatibleTypes: Set<String> =
            SmolLM.modelTypes
            .union(OLMo.olmo1ModelTypes)
            .union(Yi.modelTypes)
        if let arch = config.architecture, llamaCompatibleArchs.contains(arch) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, llamaCompatibleTypes.contains(mt) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Granite 3.x — Llama-shaped, but trained with µP multipliers
        // (embedding / attention / residual / logits scaling) that the
        // plain Llama path silently drops, giving degenerate output.
        // `loadGranite3` folds them into the weights via
        // `LlamaDense.loadModel(muP:)`. See `Models/Granite3.swift`.
        if let arch = config.architecture, Granite3.architectures.contains(arch) {
            return try loadGranite3(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Granite3.modelTypes.contains(mt) {
            return try loadGranite3(
                config: config, weights: weights,
                options: options, device: device)
        }
        // InternLM2 — Llama-shaped, but with InternLM2-native tensor
        // names and a fused `wqkv` projection the plain Llama loader
        // can't read. `loadInternLM2` remaps the names + splits wqkv.
        // See `Models/Text/InternLM2Text.swift`.
        if let arch = config.architecture, InternLM2.architectures.contains(arch) {
            return try loadInternLM2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, InternLM2.modelTypes.contains(mt) {
            return try loadInternLM2(
                config: config, weights: weights,
                options: options, device: device)
        }
        // OLMo 2 — post-norm + q/k-norm, distinct from OLMo 1's pre-norm
        // Llama shape. `loadOLMo2` runs the dedicated decoder. See
        // `Models/Text/OLMo2Text.swift`.
        if let arch = config.architecture, OLMo.olmo2Architectures.contains(arch) {
            return try loadOLMo2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, OLMo.olmo2ModelTypes.contains(mt) {
            return try loadOLMo2(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Starcoder 2 — dedicated loader because the layout differs
        // from Llama in three places that the Llama loader can't
        // accommodate: LayerNorm-with-bias (not RMSNorm), single-
        // projection GELU MLP with `c_fc`/`c_proj` names (not the
        // SwiGLU triad), and `norm_epsilon` instead of
        // `rms_norm_eps`. See `Models/Text/Starcoder2Text.swift`.
        if let arch = config.architecture, Starcoder2.architectures.contains(arch) {
            return try loadStarcoder2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Starcoder2.modelTypes.contains(mt) {
            return try loadStarcoder2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let arch = config.architecture, Phi.architectures.contains(arch) {
            return try loadPhi(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Phi.modelTypes.contains(mt) {
            return try loadPhi(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Qwen 2 / 2.5 — Llama-shaped arch with QKV biases. The
        // bias-aware Linear in Layers.swift handles the layout
        // transparently; just route the dispatch.
        if let arch = config.architecture, Qwen2.architectures.contains(arch) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Qwen2.modelTypes.contains(mt) {
            return try loadLlama(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Gemma 2 — 2B / 9B / 27B text decoder. Ships under model_type
        // `gemma2`; the family file is the only variant. Checked before
        // Gemma 3 because Gemma 2's model_type is distinct (`gemma2` vs
        // `gemma3` / `gemma3_text`) — order matters only when the
        // architecture string disambiguates.
        if let arch = config.architecture, Gemma2.architectures.contains(arch) {
            return try loadGemma2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Gemma2.modelTypes.contains(mt) {
            return try loadGemma2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let arch = config.architecture, Gemma3.architectures.contains(arch) {
            return try loadGemma3(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Gemma3.modelTypes.contains(mt) {
            return try loadGemma3(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Gemma 4 — dense / PLE (E2B, E4B) / MoE (26B-A4B). All three
        // ship under the `gemma4` model_type; the family file picks the
        // variant from config. Checked before Qwen3 so the `gemma4`
        // model_type isn't shadowed.
        if let arch = config.architecture, Gemma4.architectures.contains(arch) {
            return try loadGemma4(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Gemma4.modelTypes.contains(mt) {
            return try loadGemma4(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let arch = config.architecture, Qwen3.architectures.contains(arch) {
            return try loadQwen3(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Qwen3.modelTypes.contains(mt) {
            return try loadQwen3(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let arch = config.architecture, Mamba2.architectures.contains(arch) {
            return try loadMamba2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Mamba2.modelTypes.contains(mt) {
            return try loadMamba2(
                config: config, weights: weights,
                options: options, device: device)
        }
        // FalconH1 — the first hybrid (Mamba 2 + attention in
        // every layer). Routes through its own family file + engine.
        if let arch = config.architecture, FalconH1.architectures.contains(arch) {
            return try loadFalconH1(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, FalconH1.modelTypes.contains(mt) {
            return try loadFalconH1(
                config: config, weights: weights,
                options: options, device: device)
        }
        // NemotronH — a stack-interleaved hybrid (Mamba 2 /
        // attention / dense-MLP layers selected per-layer by a
        // hybrid_override_pattern). Routes through its own family file.
        if let arch = config.architecture, NemotronH.architectures.contains(arch) {
            return try loadNemotronH(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, NemotronH.modelTypes.contains(mt) {
            return try loadNemotronH(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Granite4 — a stack-interleaved hybrid (Mamba 2
        // / attention layers selected by `layer_types`) with an MoE +
        // shared-expert feed-forward. Routes through its own family file.
        if let arch = config.architecture, Granite4.architectures.contains(arch) {
            return try loadGranite4(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Granite4.modelTypes.contains(mt) {
            return try loadGranite4(
                config: config, weights: weights,
                options: options, device: device)
        }
        // Jamba — a stack-interleaved hybrid (Mamba 1 / attention
        // layers selected by `layers_block_type`) with a dense SwiGLU or
        // MoE feed-forward. Routes through its own family file.
        if let arch = config.architecture, Jamba.architectures.contains(arch) {
            return try loadJamba(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Jamba.modelTypes.contains(mt) {
            return try loadJamba(
                config: config, weights: weights,
                options: options, device: device)
        }

        // Qwen3.5 — a stack-interleaved hybrid (Gated Delta Net /
        // full-attention layers alternating every `full_attention_interval`)
        // with a dense SwiGLU or MoE feed-forward. Routes through its own
        // family file.
        if let arch = config.architecture, Qwen35.architectures.contains(arch) {
            return try loadQwen35(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, Qwen35.modelTypes.contains(mt) {
            return try loadQwen35(
                config: config, weights: weights,
                options: options, device: device)
        }

        // GPT-OSS — a mixture-of-experts transformer with an alternating
        // sliding/full attention schedule, learned per-head attention
        // sinks, and bias-corrected projections. Routes through its own
        // family file.
        if let arch = config.architecture, GPTOSS.architectures.contains(arch) {
            return try loadGPTOSS(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, GPTOSS.modelTypes.contains(mt) {
            return try loadGPTOSS(
                config: config, weights: weights,
                options: options, device: device)
        }

        // LFM2 — LiquidAI's stack-interleaved hybrid (short-conv /
        // attention layers selected by `layer_types` / `full_attn_idxs`)
        // with a SwiGLU or block-sparse-MoE feed-forward. Also serves the
        // LFM2.5 collection (architecturally identical — same `lfm2`
        // model_type).
        if let arch = config.architecture, LFM2.architectures.contains(arch) {
            return try loadLFM2(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, LFM2.modelTypes.contains(mt) {
            return try loadLFM2(
                config: config, weights: weights,
                options: options, device: device)
        }

        // Nemotron-Labs-Diffusion — tri-mode (AR / diffusion /
        // self-speculation) dense transformer. Distinct from the
        // NemotronH stack-interleaved hybrid family above.
        if let arch = config.architecture, NemotronDiffusion.architectures.contains(arch) {
            return try loadNemotronDiffusion(
                config: config, weights: weights,
                options: options, device: device)
        }
        if let mt = config.modelType, NemotronDiffusion.modelTypes.contains(mt) {
            return try loadNemotronDiffusion(
                config: config, weights: weights,
                options: options, device: device)
        }
        throw ModelError.unsupportedArchitecture(
            config.architecture ?? config.modelType ?? "<unknown>"
        )
    }

    public static func loadLlama(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Llama.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    /// Granite 3.x dense (`GraniteForCausalLM`). Same shape as Llama 3
    /// dense, so it reuses `LlamaDense` — but with the four µP
    /// multipliers folded in via `loadModel(muP:)`. Granite 3 MoE
    /// (`GraniteMoeForCausalLM`) is a distinct arch and does not route
    /// here.
    public static func loadGranite3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let engine = try LlamaDense.loadModel(
            config: config, weights: weights, options: options,
            device: device, muP: Granite3.muP(from: config))
        return Loaded(
            engine: engine,
            defaultGenerationParameters: LlamaDense.defaultGenerationParameters)
    }

    /// InternLM2 (`InternLM2ForCausalLM`). Llama-shaped forward; the
    /// dedicated loader remaps InternLM2's tensor names and splits the
    /// fused `wqkv` into q/k/v. See `Models/Text/InternLM2Text.swift`.
    public static func loadInternLM2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let engine = try InternLM2Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: InternLM2Dense.defaultGenerationParameters)
    }

    /// OLMo 2 (`Olmo2ForCausalLM`). Post-norm + q/k-norm decoder; the
    /// dedicated engine lives in `Models/Text/OLMo2Text.swift`. OLMo 1
    /// stays on `loadLlama`.
    public static func loadOLMo2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let engine = try OLMo2Dense.loadModel(
            config: config, weights: weights, options: options, device: device)
        return Loaded(
            engine: engine,
            defaultGenerationParameters: OLMo2Dense.defaultGenerationParameters)
    }

    public static func loadStarcoder2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Starcoder2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters,
            availableCapabilities: variant.availableCapabilities)
    }

    public static func loadQwen3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadPhi(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Phi.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma3(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma3.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGemma4(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Gemma4.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadMamba2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Mamba2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadFalconH1(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try FalconH1.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadNemotronH(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronH.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGranite4(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Granite4.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadJamba(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Jamba.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadQwen35(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try Qwen35.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadGPTOSS(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try GPTOSS.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadLFM2(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try LFM2.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }

    public static func loadNemotronDiffusion(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> Loaded {
        let variant = try NemotronDiffusion.variant(for: config)
        let engine = try variant.loadModel(
            config: config, weights: weights,
            options: options, device: device
        )
        return Loaded(
            engine: engine,
            defaultGenerationParameters: variant.defaultGenerationParameters)
    }
}

/// High-level loaded model with tokenizer attached. The public API users
/// touch.
public final class Model: @unchecked Sendable {
    /// The concrete model engine (LlamaModel, Qwen3Model, …). For a VLM
    /// this is the text backbone — text-only generation works
    /// regardless of whether `.imageIn` is enabled.
    public let engine: any LanguageModel
    public let tokenizer: any Tokenizer
    public let config: ModelConfig
    public let modelDirectory: URL
    public let availableCapabilities: Set<Capability>

    /// The `LoadOptions` this model was loaded with. Retained so the
    /// memory-management path (`makeManagedCaches`) can honor
    /// `maxContextLength` / `preallocateKVCache` / `initialKVCacheCapacity`
    /// / `wiredLimitBytes` at cache-creation time, uniformly across
    /// every family.
    public let loadOptions: LoadOptions

    /// The composed vision-language model — `nil` unless the checkpoint
    /// is a VLM. Use `vlModel.generate(...)` for an image+text prompt;
    /// available only when `availableCapabilities` contains `.imageIn`.
    public let vlModel: VisionModel?

    /// Currently-enabled capabilities. Mutated via `enable(_:)` /
    /// `disable(_:)`; guarded by `capabilityLock` for thread safety.
    private var _enabledCapabilities: Set<Capability>
    private let capabilityLock = NSLock()

    /// Snapshot of the enabled-capability set.
    public var enabledCapabilities: Set<Capability> {
        capabilityLock.lock()
        defer { capabilityLock.unlock() }
        return _enabledCapabilities
    }
    /// Default generation parameters declared by the model's family
    /// variant. Use as-is, or call `.with { $0.maxTokens = ... }` to
    /// tweak a field without losing the family-tuned baseline.
    public let defaultGenerationParameters: GenerationParameters

    /// Convenience accessor for tests + tools that want the Llama-typed
    /// model. Returns nil if the loaded engine isn't Llama.
    public var llama: LlamaModel? { engine as? LlamaModel }

    /// Convenience accessor for the Qwen3 engine.
    public var qwen3: Qwen3Model? { engine as? Qwen3Model }

    /// Convenience accessor for the Mamba 2 engine.
    public var mamba2: Mamba2Model? { engine as? Mamba2Model }

    /// Convenience accessor for the SmolVLM2 engine.
    public var smolVLM2: SmolVLM2Model? { engine as? SmolVLM2Model }

    /// Convenience accessor for the FalconH1 hybrid engine.
    public var falconH1: FalconH1Model? { engine as? FalconH1Model }

    /// Convenience accessor for the NemotronH hybrid engine.
    public var nemotronH: NemotronHModel? { engine as? NemotronHModel }

    /// Convenience accessor for the Granite4 hybrid engine.
    public var graniteMoeHybrid: Granite4Model? {
        engine as? Granite4Model
    }

    /// Convenience accessor for the Jamba hybrid engine.
    public var jamba: JambaModel? { engine as? JambaModel }

    /// Convenience accessor for the Qwen3.5 hybrid engine.
    public var qwen35: Qwen35Model? { engine as? Qwen35Model }

    /// Convenience accessor for the Gemma 2 engine.
    public var gemma2: Gemma2Model? { engine as? Gemma2Model }

    /// Convenience accessor for the GPT-OSS MoE engine.
    public var gptOSS: GPTOSSModel? { engine as? GPTOSSModel }

    /// Convenience accessor for the LFM2 hybrid engine.
    public var lfm2: LFM2Model? { engine as? LFM2Model }

    /// Convenience accessor for the Nemotron-Labs-Diffusion engine.
    public var nemotronLabsDiffusion: NemotronDiffusionModel? {
        engine as? NemotronDiffusionModel
    }

    /// Convenience accessor for the Starcoder 2 engine. Returns nil if
    /// the loaded engine isn't Starcoder 2. Note: Starcoder 2 was
    /// previously misrouted through the Llama loader and `m.llama`
    /// returned a (broken) Llama wrapper — `m.starcoder2` is the
    /// correct accessor for Starcoder2ForCausalLM checkpoints.
    public var starcoder2: Starcoder2Model? { engine as? Starcoder2Model }

    /// Convenience accessor for the OLMo 2 engine. Returns nil if the
    /// loaded engine isn't OLMo 2. OLMo 2 (`Olmo2ForCausalLM`) is
    /// post-norm with q/k-norm and routes to its own decoder — not the
    /// Llama path — so `m.llama` is nil for OLMo 2 checkpoints. (OLMo 1
    /// stays Llama-shaped and does surface through `m.llama`.)
    public var olmo2: OLMo2Model? { engine as? OLMo2Model }

    private let stateLock = NSLock()
    private var _currentState: ModelLifecycleState = .ready

    public var currentState: ModelLifecycleState {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _currentState
    }

    /// Maximum number of lifecycle events buffered when no consumer is
    /// reading from `events`. The default `AsyncStream` policy is
    /// `.unbounded`, which leaks events forever if nobody subscribes
    /// (the common case — most callers don't attach an `events` task).
    /// 64 is well above the typical event count per generation
    /// (~6: idle → loading → ready → generating → idle, plus a few
    /// capability flips) but small enough that the unconsumed-events
    /// retention is bounded.
    public static let eventsBufferCapacity = 64

    public let events: AsyncStream<ModelLifecycleEvent>
    private let eventsContinuation: AsyncStream<ModelLifecycleEvent>.Continuation

    /// Weight buffers pinned into the device's residency set at load,
    /// released in `deinit` so the wired memory doesn't accumulate across
    /// models loaded in one process. Empty for models built outside the
    /// `Model.load` path (e.g. tests) — `unmarkWeightsResident([])` no-ops.
    private let residentWeightBuffers: [MTLBuffer]
    /// The device whose residency set holds `residentWeightBuffers`.
    private let residencyDevice: Device

    init(
        engine: any LanguageModel, tokenizer: any Tokenizer, config: ModelConfig,
        modelDirectory: URL,
        availableCapabilities: Set<Capability>,
        enabledCapabilities: Set<Capability>,
        defaultGenerationParameters: GenerationParameters,
        loadOptions: LoadOptions = LoadOptions(),
        vlModel: VisionModel? = nil,
        residentWeightBuffers: [MTLBuffer] = [],
        residencyDevice: Device = .shared
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.modelDirectory = modelDirectory
        self.availableCapabilities = availableCapabilities
        self.loadOptions = loadOptions
        self.vlModel = vlModel
        self.residentWeightBuffers = residentWeightBuffers
        self.residencyDevice = residencyDevice
        // textIn / textOut are universal — always enabled. Other
        // requested capabilities are honored only if the model declares
        // them available.
        self._enabledCapabilities =
            enabledCapabilities
            .union(Capability.textOnly)
            .intersection(availableCapabilities.union(Capability.textOnly))
        self.defaultGenerationParameters = defaultGenerationParameters
        // Bounded buffer — when no consumer is reading, keep the most
        // recent `eventsBufferCapacity` events and drop older ones.
        // Avoids the unbounded-growth leak from the default policy.
        let (stream, cont) = AsyncStream<ModelLifecycleEvent>.makeStream(
            bufferingPolicy: .bufferingNewest(Self.eventsBufferCapacity)
        )
        self.events = stream
        self.eventsContinuation = cont
    }

    deinit {
        // Release this model's weights from the device residency set so
        // the wired memory is freed — otherwise every model loaded in a
        // process stays resident and memory accumulates across loads.
        residencyDevice.unmarkWeightsResident(residentWeightBuffers)
        eventsContinuation.finish()
    }

    fileprivate func emit(_ event: ModelLifecycleEvent) {
        stateLock.lock()
        _currentState = event.state
        stateLock.unlock()
        eventsContinuation.yield(event)
    }

    // ─── Capability enable / disable ─────────────────────────────────

    /// Whether a capability is currently enabled.
    public func isEnabled(_ capability: Capability) -> Bool {
        enabledCapabilities.contains(capability)
    }

    /// Enable a capability at runtime — e.g. `enable(.imageIn)` lights
    /// up the vision path on a model loaded text-only. No-op if the
    /// capability isn't in `availableCapabilities` (a text-only model
    /// can't gain vision) or is already enabled. Emits a lifecycle
    /// event tagged with the capability so consumers can react.
    ///
    /// `textIn` / `textOut` are universal and always enabled — calling
    /// `enable` / `disable` on them is a harmless no-op.
    @discardableResult
    public func enable(_ capability: Capability) -> Bool {
        guard
            availableCapabilities.contains(capability)
                || Capability.textOnly.contains(capability)
        else { return false }
        capabilityLock.lock()
        let changed = !_enabledCapabilities.contains(capability)
        _enabledCapabilities.insert(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    /// Disable a capability at runtime. `textIn` / `textOut` are
    /// universal and cannot be disabled — those calls are a no-op.
    /// Emits a capability-tagged lifecycle event when the set changes.
    @discardableResult
    public func disable(_ capability: Capability) -> Bool {
        guard !Capability.textOnly.contains(capability) else { return false }
        capabilityLock.lock()
        let changed = _enabledCapabilities.contains(capability)
        _enabledCapabilities.remove(capability)
        capabilityLock.unlock()
        if changed {
            eventsContinuation.yield(
                ModelLifecycleEvent(capability: capability, state: currentState))
        }
        return changed
    }

    // ─── Top-level loader ────────────────────────────────────────────

    /// Resolve an id-or-path, download if needed, decode config, load
    /// weights, build the family-specific model, attach tokenizer.
    public static func load(
        _ idOrPath: String,
        options: LoadOptions = LoadOptions(),
        device: Device = .shared
    ) async throws -> Model {
        Debug.log(.load, "Model.load id-or-path=\(idOrPath)")
        // Fail fast on an explicit wired-memory ticket that would starve
        // the OS (physical RAM minus the 8 GB reserve), before any heavy
        // load work. No-op when wiredLimitBytes isn't set.
        try MemoryBudget.validateWiredLimit(options: options, device: device)
        let model = try await Profile.timeAsync("model_load") {
            try await Profile.signpostAsync("model_load") {
                let locator = ModelLocator(
                    downloader: ModelDownloader(cacheDirectory: options.cacheDirectory))
                let dir = try await locator.resolve(idOrPath: idOrPath, revision: options.revision)
                Debug.log(.loader, "resolved snapshot dir: \(dir.path)")
                let config = try ModelConfig.load(from: dir)
                Debug.log(
                    .load,
                    "config: arch=\(config.architecture ?? "?") model_type=\(config.modelType ?? "?") hidden=\(config.hiddenSize ?? 0) layers=\(config.numLayers ?? 0)"
                )
                let bundle = try SafeTensorsBundle(directory: dir, device: device)
                // Pin every weight buffer in a persistent MTLResidencySet
                // so prefill / decode dispatches skip per-allocation
                // residency tracking. macOS 15+ / iOS 18+; older OSes
                // and `FFAI_NO_RESIDENCY_SET=1` no-op via
                // `Device.markWeightsResident`.
                var weightBuffers: [MTLBuffer] = []
                for file in bundle.files {
                    for entry in file.entries.values {
                        weightBuffers.append(entry.buffer)
                    }
                }
                device.markWeightsResident(weightBuffers)
                let loaded = try ModelRegistry.dispatchAndLoad(
                    config: config, weights: bundle, options: options, device: device
                )
                // Nemotron-Labs-Diffusion ships an optional
                // `linear_spec_lora` adapter that sharpens the
                // self-speculation diffusion drafter — attach it if the
                // checkpoint included the subfolder.
                if let nd = loaded.engine as? NemotronDiffusionModel {
                    nd.attachLoRA(from: dir, device: device)
                }
                let tokenizer = try await TokenizerLoader().load(from: dir)
                return Model(
                    engine: loaded.engine, tokenizer: tokenizer, config: config,
                    modelDirectory: dir,
                    availableCapabilities: loaded.availableCapabilities,
                    enabledCapabilities: options.capabilities,
                    defaultGenerationParameters: loaded.defaultGenerationParameters,
                    loadOptions: options,
                    vlModel: loaded.vlModel,
                    residentWeightBuffers: weightBuffers,
                    residencyDevice: device
                )
            }
        }

        // Prewarm just touches the embedding lookup once so the PSO is
        // compiled before the first user-visible decode. Captured as a
        // separate phase so `--profiling 1` shows it broken out from
        // model_load.
        if options.prewarm {
            await Profile.timeAsync("prewarm") {
                await Profile.signpostAsync("prewarm") {
                    await model.prewarm()
                }
            }
        }

        model.emit(ModelLifecycleEvent(state: .ready))
        return model
    }

    /// Compile PSOs for the kernels we'll need during decode by running
    /// one no-op forward step. Costs ~100ms-1s on first load.
    ///
    /// The cache only ever holds the single position-0 token this
    /// forward appends, so it is sized to a tiny fixed depth rather
    /// than the model's full context window. Without the cap, a
    /// long-context model (e.g. Qwen3.6-27B at max_position_embeddings
    /// = 262144) would pre-allocate ~17 GB of KV cache here — at LOAD
    /// time, before the caller has even asked to generate — and could
    /// exhaust unified memory on its own. PSO compilation depends on
    /// kernel shapes, not cache depth, so a depth of a few slots
    /// compiles exactly the same pipeline states.
    public func prewarm() async {
        let cache = engine.makeLayerCaches(maxSeq: Self.prewarmCacheDepth)
        _ = engine.forward(tokenId: 0, position: 0, caches: cache)
    }

    /// Cache depth used by `prewarm()`. A single position-0 forward
    /// needs only one slot; the small margin is defensive headroom for
    /// any family whose forward touches position+1 internally.
    private static let prewarmCacheDepth = 8

    // ─── Managed KV-cache creation (the memory-management chokepoint) ──

    /// Resolve the KV-cache context ceiling for a session: the model's
    /// `max_position_embeddings`, narrowed by `LoadOptions.maxContextLength`,
    /// then by the optional `generationBudget` (prompt + maxTokens), then
    /// clamped by the over-allocation guard so weights + max-KV + a
    /// working-memory margin fit the wired-memory budget. Throws
    /// `ModelError.insufficientMemory` if the model can't run at any
    /// usable context on this machine / budget.
    ///
    /// Every cache-creation path (generation, CLI inspect/generate)
    /// routes through here, so the cap + guard apply uniformly across
    /// every model family without per-family wiring.
    public func resolvedKVContextCeiling(
        generationBudget: Int? = nil, device: Device = .shared
    ) throws -> Int {
        var ceiling = engine.maxContextWindow
        if let mc = loadOptions.maxContextLength { ceiling = Swift.min(ceiling, mc) }
        if let gb = generationBudget { ceiling = Swift.min(ceiling, gb) }
        ceiling = try MemoryBudget.clampContext(
            requestedCeiling: ceiling, engine: engine,
            options: loadOptions, device: device)
        return Swift.max(1, ceiling)
    }

    /// Build the per-layer caches for a session with memory management
    /// applied: the context ceiling is resolved + guarded (see
    /// `resolvedKVContextCeiling`), and `LoadOptions.preallocateKVCache`
    /// / `initialKVCacheCapacity` are honored by setting the KVCache
    /// initial-capacity knob around the (synchronous) construction.
    ///
    /// `preallocateKVCache` collapses to "start at the full ceiling"
    /// (no growth). The knob is saved + restored around the build so it
    /// doesn't leak into other models' cache creation.
    public func makeManagedCaches(
        generationBudget: Int? = nil, device: Device = .shared
    ) throws -> [any LayerCacheProtocol] {
        let ceiling = try resolvedKVContextCeiling(
            generationBudget: generationBudget, device: device)

        // Common path: no per-load initial-capacity override → build at
        // the global default, touching no shared state.
        guard loadOptions.preallocateKVCache || loadOptions.initialKVCacheCapacity != nil
        else {
            return engine.makeLayerCaches(maxSeq: ceiling, device: device)
        }

        // Override path: `preallocateKVCache` collapses to "start at the
        // full ceiling"; `initialKVCacheCapacity` sets a custom start.
        // We express both through the `KVCache.defaultInitialCapacity`
        // knob around the SYNCHRONOUS build, saving + restoring it so the
        // override doesn't leak to other models. (The save/restore window
        // is synchronous — no `await` — so a single call is self-consistent;
        // simultaneous generations on *different* models that both set an
        // override could still race the global, but that only affects the
        // initial allocation SIZE, never correctness, since growth
        // reconstructs the cache faithfully regardless of where it starts.)
        let saved = KVCache.defaultInitialCapacity
        KVCache.defaultInitialCapacity =
            loadOptions.preallocateKVCache
            ? ceiling
            : Swift.max(1, loadOptions.initialKVCacheCapacity ?? saved)
        defer { KVCache.defaultInitialCapacity = saved }
        return engine.makeLayerCaches(maxSeq: ceiling, device: device)
    }
}

// ─── Hot LoRA adapter management ─────────────────────────────────────

extension Model {
    /// Whether a LoRA adapter is currently attached. Always `false` for
    /// families that don't support adapters (only Nemotron-Labs-
    /// Diffusion does today).
    public var hasLoRA: Bool { nemotronLabsDiffusion?.hasLoRA ?? false }

    /// Hot-load a LoRA adapter at runtime. `directory` may be the model
    /// directory (the adapter is resolved under `linear_spec_lora/`) or
    /// a directory holding `adapter_model.safetensors` directly — so the
    /// same call swaps in the bundled adapter or an external one. Any
    /// currently-attached adapter is replaced. No-op on families that
    /// don't support adapters. Do not call during an active generate.
    public func loadLoRA(from directory: URL, device: Device = .shared) {
        guard let nd = nemotronLabsDiffusion else { return }
        nd.detachLoRA()
        nd.attachLoRA(from: directory, device: device)
    }

    /// Hot-unload the current LoRA adapter. No-op when none is attached.
    public func unloadLoRA() { nemotronLabsDiffusion?.detachLoRA() }
}
