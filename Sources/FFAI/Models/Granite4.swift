// Granite4 family root — IBM's Granite 4 GraniteMoeHybrid line
// (`granitemoehybrid` model_type; H-350M / H-1B / H-Tiny / H-Small).
//
// This file is the **main model interface** for the family:
//   • the family enum `Granite4` (modelTypes, architectures, variant
//     dispatch),
//   • the `Granite4Variant` protocol every concrete variant conforms
//     to,
//   • the `Granite4Error` type the loader / decode site raises.
//
// Concrete variants + the hybrid decoder + per-layer impl live under
// `Models/Text/Granite4Text.swift`:
//   - `Granite4Hybrid` — the single stack-interleaved variant. A
//     `layer_types` schedule names each layer "mamba" or "attention";
//     the FFN half is either a dense SwiGLU MLP or a top-K MoE block
//     (plus an always-on shared SwiGLU expert). No RoPE — Mamba
//     carries sequence order.
//
// Related (separate family):
//   - Models/Granite3.swift — Granite v3 (Llama-shaped dense,
//                              `granite` model_type)

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Granite4 {
    public static let modelTypes: Set<String> = ["granitemoehybrid"]
    public static let architectures: Set<String> = ["Granite4ForCausalLM"]

    public static func variant(for _: ModelConfig) throws -> any Granite4Variant.Type {
        return Granite4Hybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Granite4Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Granite4Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Granite4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Granite4: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "Granite4: unsupported config: \(m)"
        }
    }
}
