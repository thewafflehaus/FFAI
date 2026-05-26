// FalconH1 family root — TII's Falcon-H1 parallel-hybrid line
// (Tiny-90M / 0.5B / 1.5B / 3B / 7B; `falcon_h1` model_type).
//
// This file is the **main model interface** for the family:
//   • the family enum `FalconH1` (modelTypes, architectures, variant
//     dispatch),
//   • the `FalconH1Variant` protocol every concrete variant conforms
//     to,
//   • the `FalconH1Error` type the loader / decode site raises.
//
// Concrete variants + the per-layer impl live under
// `Models/Text/FalconH1Text.swift`:
//   - `FalconH1Hybrid` — the single (and only) variant; every decoder
//     layer runs BOTH a Mamba 2 mixer and a GQA attention path on the
//     same normalized input, sums them into the residual, then runs a
//     SwiGLU MLP. Per-layer scalar / µP multipliers are folded into
//     projection weights at load time when the checkpoint isn't
//     already pre-sanitized.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum FalconH1 {
    public static let modelTypes: Set<String> = ["falcon_h1"]
    public static let architectures: Set<String> = ["FalconH1ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any FalconH1Variant.Type {
        return FalconH1Hybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol FalconH1Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> FalconH1Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum FalconH1Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f): return "FalconH1: required config field missing: \(f)"
        case .unsupportedConfig(let m): return "FalconH1: unsupported config: \(m)"
        }
    }
}
