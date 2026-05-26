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
// Phi family root — Microsoft's Phi-3 / Phi-3.5 line.
//
// This file is the **main model interface** for the family:
//   • the family enum `Phi` (modelTypes, architectures, variant
//     dispatch),
//   • the `PhiVariant` protocol every concrete variant conforms to,
//   • the `PhiError` type the loader / decode site raises.
//
// Concrete variants + the dense decoder + per-layer impl live under
// `Models/Text/PhiText.swift`:
//   - `Phi3Dense` — Phi-3 / Phi-3.5 mini / small / medium dense GQA
//     transformer with fused QKV + fused gate/up projections row-sliced
//     into LlamaLayer views. Returns a `LlamaModel` engine.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Phi {
    public static let modelTypes: Set<String> = ["phi3"]
    public static let architectures: Set<String> = ["Phi3ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any PhiVariant.Type {
        return Phi3Dense.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol PhiVariant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> LlamaModel
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum PhiError: Error, CustomStringConvertible {
    case missingConfig
    case unsupportedRopeScaling(String)
    case quantizedFusedNotSupported
    public var description: String {
        switch self {
        case .missingConfig: return "Phi: required config field missing"
        case .unsupportedRopeScaling(let t):
            return
                "Phi: rope_scaling type '\(t)' not supported yet (SuScaledRoPE is a follow-up); use Phi-3-mini-4k-instruct or wait"
        case .quantizedFusedNotSupported:
            return
                "Phi: quantized fused qkv_proj / gate_up_proj not supported yet; load the raw bf16 / f16 checkpoint or convert with split projections"
        }
    }
}
