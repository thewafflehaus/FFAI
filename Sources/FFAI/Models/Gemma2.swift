// Gemma2 family root — Google's open Gemma 2 line (2B / 9B / 27B,
// `gemma2` / `gemma2_text` model_type).
//
// This file is the **main model interface** for the family:
//   • the family enum `Gemma2` (modelTypes, architectures, variant
//     dispatch),
//   • the `Gemma2Variant` protocol every concrete variant conforms to,
//   • the `Gemma2Error` type the loader / decode site raises.
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/Gemma2Text.swift`:
//   - `Gemma2Dense` — the 2B / 9B / 27B dense text decoder with
//     alternating sliding-window / full attention layers, GemmaRMSNorm
//     `+1.0` fold, and (deliberately) no q_norm / k_norm. Used as the
//     text backbone by Paligemma 2.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Gemma2 {
    public static let modelTypes: Set<String> = ["gemma2", "gemma2_text"]
    public static let architectures: Set<String> = [
        "Gemma2ForCausalLM"
    ]

    public static func variant(for config: ModelConfig) throws -> any Gemma2Variant.Type {
        return Gemma2Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Gemma2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Gemma2Error: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig:
            return "Gemma2: required config field missing"
        }
    }
}
