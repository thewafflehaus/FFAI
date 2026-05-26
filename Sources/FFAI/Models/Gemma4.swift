// Gemma 4 family root — Google's Gemma 4 line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Gemma4` (modelTypes, architectures, variant
//     dispatch),
//   • the `Gemma4Variant` protocol every concrete variant conforms to,
//   • the unified `Gemma4Error` type every loader / decode site raises
//     (covers both the text path and the Gemma 4 VL path).
//
// Concrete variants + the dense / E / MoE decoder + per-layer impl live
// under `Models/Text/Gemma4Text.swift`:
//   - `Gemma4Dense` — 31B Gemma-style backbone, no PLE, dense MLP.
//   - `Gemma4E`     — E2B / E4B; adds Per-Layer Embeddings.
//   - `Gemma4MoE`   — 26B-A4B; mixture-of-experts feed-forward.
//   - `Gemma4Model` — the full LanguageModel decoder.
//
// The Gemma 4 VL vision-language orchestrator (`enum Gemma4VL`) — which
// ties the Gemma 4 text backbone to the bespoke Gemma 4 ViT vision
// tower + multi-modal embedder — lives in
// `Models/Vision/Gemma4Vision.swift` alongside the tower internals.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Gemma4 {
    public static let modelTypes: Set<String> = ["gemma4", "gemma4_text"]
    public static let architectures: Set<String> = [
        "Gemma4ForCausalLM", "Gemma4TextForCausalLM",
        "Gemma4ForConditionalGeneration",
    ]

    /// Resolve the concrete variant from config. MoE wins over PLE wins
    /// over plain dense.
    public static func variant(for config: ModelConfig) throws -> any Gemma4Variant.Type {
        let tc = Gemma4Config.textConfig(config)
        if (tc["enable_moe_block"] as? Bool) ?? false {
            return Gemma4MoE.self
        }
        if let ple = tc["hidden_size_per_layer_input"] as? Int, ple > 0 {
            return Gemma4E.self
        }
        return Gemma4Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Gemma4Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma4Model
}

public extension Gemma4Variant {
    static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    static var defaultGenerationParameters: GenerationParameters {
        // Gemma 4: 4096-token prefill chunk is the audited family
        // optimum (pure-attention backbone, no SSM bottleneck).
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Gemma 4 family error — raised by both the text loaders
/// (`Gemma4Dense` / `Gemma4E` / `Gemma4MoE.loadModel`) and the Gemma 4
/// VL orchestrator (`Gemma4VL.load` in `Models/Vision/Gemma4Vision.swift`).
public enum Gemma4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedHeadDim(Int)
    case unalignedNorm(Int)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Gemma4: required config field missing: \(f)"
        case .missingTensor(let name):
            return "Gemma4: checkpoint is missing tensor '\(name)'"
        case .unsupportedHeadDim(let d):
            return "Gemma4: head_dim \(d) unsupported (Ops.sdpaDecode needs 64/128/256/512)"
        case .unalignedNorm(let n):
            return "Gemma4: norm row size \(n) must be 128-aligned"
        }
    }
}
