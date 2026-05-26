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
// Jamba family root — AI21's Jamba stack-interleaved hybrid line
// (`jamba` model_type, e.g. Jamba v0.1 / Jamba-Reasoning-3B).
//
// This file is the **main model interface** for the family:
//   • the family enum `Jamba` (modelTypes, architectures, variant
//     dispatch),
//   • the `JambaVariant` protocol every concrete variant conforms to,
//   • the `JambaError` type the loader / decode site raises.
//
// Concrete variants + the hybrid decoder + per-layer impl live under
// `Models/Text/JambaText.swift`:
//   - `JambaHybrid` — the single stack-interleaved (Mamba 1 + attention
//     + MoE/MLP) variant; CPU-side selective scan for the Mamba mixer,
//     mid-layer command-buffer commits handled by `JambaModel`.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Jamba {
    public static let modelTypes: Set<String> = ["jamba"]
    public static let architectures: Set<String> = ["JambaForCausalLM"]

    public static func variant(for _: ModelConfig) throws -> any JambaVariant.Type {
        return JambaHybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol JambaVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> JambaModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum JambaError: Error, CustomStringConvertible {
    case missingConfig(String)
    case unsupportedConfig(String)
    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Jamba: required config field missing: \(f)"
        case .unsupportedConfig(let m):
            return "Jamba: unsupported config: \(m)"
        }
    }
}
