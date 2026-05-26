// NemotronDiffusion family root — NVIDIA Nemotron-Labs-Diffusion, the
// "tri-mode" line that runs the same dense Ministral/Llama-shaped
// weights as autoregressive (AR), block-wise diffusion, or self-
// speculation.
//
// This file is the **main model interface** for the family:
//   • the family enum `NemotronDiffusion` (modelTypes, architectures,
//     variant dispatch — covers both the text-only checkpoint and the
//     VLM checkpoint whose text backbone uses the same `encoder.*` /
//     `diffusion_head.weight` layout),
//   • the `NemotronDiffusionVariant` protocol every concrete variant
//     conforms to,
//   • the `NemotronDiffusionError` type the loader / decode site
//     raises.
//
// Concrete variants + the tri-mode decoder + per-layer impl live under
// `Models/Text/NemotronDiffusionText.swift`:
//   - `NemotronDiffusionDense` — the dense Ministral/Llama-shaped
//     backbone with YaRN-scaled RoPE; AR drives the standard
//     `forward`, block-diffusion / self-speculation drive
//     `forwardBlock` and `GenerateDiffusion`.
//
// Note: this family is also exported through the unified `enum
// Nemotron` family root (`Models/Nemotron.swift`) which unions
// modelTypes / architectures across NemotronH, NemotronVL, and
// NemotronDiffusion for single-membership-check dispatch in the
// registry. This `Models/NemotronDiffusion.swift` anchor keeps the
// "every family has its own root file" rule uniform.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum NemotronDiffusion {
    // Both the text-only checkpoint and the VLM checkpoint share this
    // family — the VLM's text backbone uses the identical `encoder.*` /
    // `diffusion_head.weight` layout. Loading a VLM checkpoint here
    // brings up the tri-mode *text* backbone; the `vision_tower.*` /
    // `multi_modal_projector.*` tensors are left unreferenced until the
    // vision path lands.
    public static let modelTypes: Set<String> = [
        "nemotron_labs_diffusion", "nemotron_labs_diffusion_vlm",
    ]
    public static let architectures: Set<String> = [
        "NemotronDiffusionModel", "NemotronDiffusionVLMModel",
    ]

    public static func variant(
        for config: ModelConfig
    ) throws -> any NemotronDiffusionVariant.Type {
        return NemotronDiffusionDense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol NemotronDiffusionVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> NemotronDiffusionModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum NemotronDiffusionError: Error, CustomStringConvertible {
    case missingConfig

    public var description: String {
        switch self {
        case .missingConfig: return "NemotronDiffusion: required config field missing"
        }
    }
}
