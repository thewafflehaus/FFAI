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
// Starcoder 2 family root — BigCode's code dense decoder. Structurally
// distinct from the Llama dense family:
//
//   - LayerNorm with `.bias` (NOT RMSNorm)
//   - Single-projection GELU-tanh MLP with `c_fc` + `c_proj` names
//     (NOT the SwiGLU `gate_proj` + `up_proj` + `down_proj` triad)
//   - Attention biases on all four q/k/v/o projections
//     (`use_bias: true`; loadLinear's auto-detection handles them)
//   - Config field is `norm_epsilon` (NOT `rms_norm_eps`)
//
// Routes through its own `Starcoder2Dense` variant — earlier revisions
// misrouted this family through `llamaCompatibleArchs` in
// `Loader/Model.swift`, which threw `Llama: required config field
// missing` because Starcoder2 has `norm_epsilon` instead of
// `rms_norm_eps`. The dedicated loader lives in
// `Models/Text/Starcoder2Text.swift`.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Starcoder2 {
    public static let modelTypes: Set<String> = ["starcoder2"]
    public static let architectures: Set<String> = ["Starcoder2ForCausalLM"]

    /// Variant dispatch — Starcoder2 ships only the dense backbone
    /// today. Future MoE / instruction-tuned variants would branch here.
    public static func variant(
        for _: ModelConfig
    ) throws -> any Starcoder2Variant.Type {
        return Starcoder2Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

/// Concrete Starcoder2 backbones conform to this. `Starcoder2Dense` is
/// the only conformer today. Mirrors the per-family variant protocol
/// every other family root declares.
public protocol Starcoder2Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Starcoder2Model
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum Starcoder2Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Starcoder2: required config field missing: \(f)"
        case .unsupportedConfig(let f):
            return "Starcoder2: unsupported config: \(f)"
        }
    }
}
