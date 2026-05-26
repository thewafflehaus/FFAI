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
// Qwen 3.5 family root — Alibaba's Qwen 3.5 hybrid line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Qwen35` (modelTypes, architectures, variant
//     dispatch),
//   • the `Qwen35Variant` protocol every concrete variant conforms to,
//   • the unified `Qwen35Error` type every loader / decode site raises
//     (covers both the text hybrid path and the Qwen3-VL-MoE path).
//
// Concrete variants + the hybrid decoder + per-layer impl live under
// `Models/Text/Qwen3xText.swift` (the "Qwen3x" file covers BOTH Qwen 3.5
// AND Qwen 3.6 — they share the same stack-interleaved GDN + attention
// hybrid architecture and the same `qwen3_5*` model_type strings):
//   - `Qwen35Hybrid` — the single variant; dense-vs-MoE is decided per
//     checkpoint inside `loadModel` from `num_experts`.
//   - `Qwen35Model` — the full LanguageModel decoder.
//
// The Qwen 3-VL-MoE vision-language orchestrator (`enum Qwen3VLMoe`) —
// which ties the Qwen 3.5-MoE text backbone to the shared Qwen 3-VL ViT
// vision tower — lives in `Models/Vision/Qwen3Vision.swift` alongside
// its dense Qwen3-VL sibling and the dense Qwen 3.5-VL orchestrator
// (`enum Qwen35VL`, same file). See `Models/Qwen36.swift` for the Qwen
// 3.6 root anchor; the underlying types are the same.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen35 {
    public static let modelTypes: Set<String> = [
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    ]
    public static let architectures: Set<String> = [
        "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM",
    ]

    public static func variant(for _: ModelConfig) throws -> any Qwen35Variant.Type {
        // A single variant covers all three forms — dense vs MoE is
        // decided per-checkpoint from `num_experts` inside `loadModel`.
        return Qwen35Hybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Qwen35Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Qwen35Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Qwen 3.5 family error — raised by the text loaders
/// (`Qwen35Hybrid.loadModel`), the Qwen 3-VL-MoE orchestrator
/// (`Qwen3VLMoe.load`), and the dense Qwen 3.5-VL orchestrator
/// (`Qwen35VL.load`) — both VL siblings in `Models/Vision/Qwen3Vision.swift`.
public enum Qwen35Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Qwen3.5: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "Qwen3.5: unsupported config: \(m)"
        }
    }
}
