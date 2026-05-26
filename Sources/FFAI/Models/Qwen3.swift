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
// Qwen 3 family root — Alibaba's Qwen 3 line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Qwen3` (modelTypes, architectures, variant
//     dispatch),
//   • the `Qwen3Variant` protocol every concrete variant conforms to,
//   • the unified `Qwen3Error` type every loader / decode site raises
//     (covers both the text and Qwen3-VL paths).
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/Qwen3Text.swift`:
//   - `Qwen3Dense` — the dense Qwen 3 transformer with per-head q_norm /
//     k_norm. Future variants (Qwen 3.5 hybrid / MoE, Qwen 3.5-Omni)
//     plug in here by adding a struct + a `Qwen3.variant(for:)` arm.
//   - `Qwen3Layer`, `Qwen3Model` — per-layer + full-model impl.
//
// The Qwen 3-VL vision-language orchestrator (`enum Qwen3VL`) — which
// ties the Qwen 3 text backbone to the Qwen 3-VL ViT vision tower —
// lives in `Models/Vision/Qwen3Vision.swift` alongside the tower
// internals.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen3 {
    /// HuggingFace `model_type` strings this family handles. Qwen 3-VL
    /// ships only as an architecture string (no distinct model_type).
    public static let modelTypes: Set<String> = ["qwen3"]
    /// HuggingFace `architectures[0]` strings this family handles —
    /// the union of the dense text path and the Qwen 3-VL path.
    public static let architectures: Set<String> = [
        "Qwen3ForCausalLM", "Qwen3VLForConditionalGeneration",
    ]

    /// Pick the concrete variant for a config. Only `Qwen3Dense` ships
    /// today; future variants dispatch here.
    public static func variant(for config: ModelConfig) throws -> any Qwen3Variant.Type {
        _ = config
        return Qwen3Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Qwen3Variant {
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant. The user can override any
    /// field; absent overrides fall back to the values declared here.
    /// See planning/roadmap.md for which fields are honored today vs
    /// staged for planned (sampling kernels).
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen3Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Qwen 3 family error — raised by both the text loaders
/// (`Qwen3Dense.loadModel`) and the Qwen 3-VL orchestrator (`Qwen3VL.load`
/// in `Models/Vision/Qwen3Vision.swift`).
public enum Qwen3Error: Error, CustomStringConvertible {
    case missingConfig
    case missingTensor(String)

    public var description: String {
        switch self {
        case .missingConfig:
            return "Qwen3: required config field missing"
        case .missingTensor(let name):
            return "Qwen3: checkpoint is missing tensor '\(name)'"
        }
    }
}
