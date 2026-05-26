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
// Gemma 3 family root — Google's Gemma 3 line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Gemma3` (modelTypes, architectures, variant
//     dispatch),
//   • the `Gemma3Variant` protocol every concrete variant conforms to,
//   • the unified `Gemma3Error` type every loader / decode site raises
//     (covers both the text and Gemma 3 VL paths — the existing
//     `Gemma3Error.missingConfig` already covers what the VL loader
//     needed).
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/Gemma3Text.swift`:
//   - `Gemma3Dense` — the 1B / 4B / 12B / 27B Gemma 3 text decoder.
//   - `Gemma3Model` — the full LanguageModel decoder.
//
// The Gemma 3 VL vision-language orchestrator (`enum Gemma3VL`) — which
// ties the Gemma 3 text backbone to the SigLIP ViT vision tower —
// lives in `Models/Vision/Gemma3Vision.swift` alongside the tower
// internals.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Gemma3 {
    public static let modelTypes: Set<String> = ["gemma3", "gemma3_text"]
    public static let architectures: Set<String> = [
        "Gemma3ForCausalLM", "Gemma3TextForCausalLM"
    ]

    public static func variant(for config: ModelConfig) throws -> any Gemma3Variant.Type {
        _ = config
        return Gemma3Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Gemma3Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Gemma3Model
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Gemma 3 family error — raised by both the text loaders
/// (`Gemma3Dense.loadModel`) and the Gemma 3 VL orchestrator
/// (`Gemma3VL.load` in `Models/Vision/Gemma3Vision.swift`).
public enum Gemma3Error: Error, CustomStringConvertible {
    case missingConfig
    public var description: String {
        switch self {
        case .missingConfig:
            return "Gemma3: required config field missing"
        }
    }
}
