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
// Llama family root — Meta's Llama 3.x line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Llama` (modelTypes, architectures, variant
//     dispatch),
//   • the `LlamaVariant` protocol every concrete variant conforms to,
//   • the `LlamaError` type every loader / decode site raises.
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/LlamaText.swift`:
//   - `LlamaDense` — Llama 3 / 3.1 / 3.2 dense GQA transformer (the
//     only variant shipped today; 1B / 3B / 8B / 70B; 405B with
//     quantization).
//   - `LlamaLayer`, `LlamaModel` — per-layer + full-model impl.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Llama {
    /// HuggingFace `model_type` strings this family handles.
    public static let modelTypes: Set<String> = ["llama"]
    /// HuggingFace `architectures[0]` strings this family handles.
    public static let architectures: Set<String> = ["LlamaForCausalLM"]

    /// Pick the concrete variant for a config. Only `LlamaDense`
    /// ships today; future variants (Llama 4 MoE, etc.) dispatch here.
    public static func variant(for config: ModelConfig) throws -> any LlamaVariant.Type {
        _ = config
        return LlamaDense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol LlamaVariant {
    /// Capabilities a checkpoint of this variant exposes.
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant. The user can override any
    /// field; absent overrides fall back to the values declared here.
    /// See `planning/roadmap.md` for which fields are honored today vs
    /// staged for planned (sampling kernels).
    static var defaultGenerationParameters: GenerationParameters { get }
    /// Build a `LlamaModel` decoder from a checkpoint.
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum LlamaError: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig: return "Llama: required config field missing"
        }
    }
}
