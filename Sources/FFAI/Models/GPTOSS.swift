// GPTOSS family root — OpenAI's GPT-OSS open-weights MoE line
// (`gpt_oss` model_type, e.g. GPT-OSS-20B).
//
// This file is the **main model interface** for the family:
//   • the family enum `GPTOSS` (modelTypes, architectures, variant
//     dispatch),
//   • the `GPTOSSVariant` protocol every concrete variant conforms to,
//   • the `GPTOSSError` type the loader / decode site raises.
//
// Concrete variants + the MoE decoder + per-layer impl live under
// `Models/Text/GPTOSSText.swift`:
//   - `GPTOSSMoEVariant` — the 24-layer MoE transformer (~20B total /
//     ~3.6B active per token) with alternating sliding / full
//     attention, learned per-head attention sinks, biased Q/K/V/O
//     projections, and an MXFP4-sourced clipped-α-SwiGLU expert FFN
//     that's transcoded to FFAI's affine-int4 format at load time.
//     The `GPTOSSAttentionKind` tag, MXFP4 codec, `GPTOSSExpert`,
//     `GPTOSSMoELayer`, and `buildGPTOSSMoE` loader all live in the
//     Text file.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum GPTOSS {
    public static let modelTypes: Set<String> = ["gpt_oss"]
    public static let architectures: Set<String> = ["GptOssForCausalLM"]

    public static func variant(for _: ModelConfig) throws -> any GPTOSSVariant.Type {
        return GPTOSSMoEVariant.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol GPTOSSVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> GPTOSSModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum GPTOSSError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "GPT-OSS: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "GPT-OSS: unsupported config: \(m)"
        }
    }
}
