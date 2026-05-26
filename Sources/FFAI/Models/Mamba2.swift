// Mamba2 family root — the pure selective-SSM decoder line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Mamba2` (modelTypes, architectures, variant
//     dispatch),
//   • the `Mamba2Variant` protocol every concrete variant conforms to,
//   • the `Mamba2Error` type the loader / decode site raises.
//
// Concrete variants + the SSM decoder + per-layer impl live under
// `Models/Text/Mamba2Text.swift`:
//   - `Mamba2Dense` — the single dense variant (130m / 370m / 780m /
//     1.3b / 2.7b state-space decoder, `mamba2` model_type),
//   - `Mamba2Layer`, `Mamba2Model` — per-layer + full-model impl.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Mamba2 {
    public static let modelTypes: Set<String> = ["mamba2"]
    public static let architectures: Set<String> = ["Mamba2ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any Mamba2Variant.Type {
        return Mamba2Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Mamba2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Mamba2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Mamba2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "Mamba2: required config field missing: \(f)"
        case .unsupportedConfig(let m): return "Mamba2: unsupported config: \(m)"
        }
    }
}
