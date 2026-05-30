// Copyright 2026 Tom Turney (@TheTom)
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
// DeepSeek V4 family root — DeepSeek-AI's V4 line (DeepSeek-V4-Flash,
// DeepSeek-V4-Pro).
//
// This file is the **main model interface** for the family:
//   • the family enum `DeepSeekV4` (modelTypes, architectures, variant
//     dispatch),
//   • the `DeepSeekV4Variant` protocol every concrete variant conforms
//     to,
//   • the unified `DeepSeekV4Error` type every loader / decode site
//     raises.
//
// Concrete variants + the hybrid CSA/HCA/MLA decoder + per-layer impl
// live under `Models/Text/DeepSeekV4Text.swift`:
//   - `DeepSeekV4Flash` — 284B total / 13B active. 43 transformer
//     layers + 1 MTP head, interleaved full / CSA / HCA attention,
//     288-expert MoE with sigmoid+bias + Lightning Indexer routing.
//   - `DeepSeekV4Pro`   — same arch, ~1.6T / 49B active.
//   - `DeepSeekV4Model` — full LanguageModel decoder.
//
// **Status:** WIP. Family scaffold + config decoder + loader hook are
// in place so a `DeepSeek-V4-Flash` checkpoint (safetensors OR GGUF) is
// identified end-to-end; the forward path is stubbed and raises
// `DeepSeekV4Error.notYetImplemented` on load. The MLA / CSA / HCA /
// Lightning-Indexer kernel work lands in follow-ups (metaltile-side
// scope spans 4-5 distinct PRs). The companion GGUF reader in
// `Loader/GGUF/` ships in this same PR so the family namespace +
// load-from-GGUF infrastructure go in together.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum DeepSeekV4 {
    /// HuggingFace `model_type` strings this family handles.
    public static let modelTypes: Set<String> = ["deepseek_v4", "deepseek4"]

    /// HuggingFace `architectures[0]` strings this family handles. GGUF
    /// checkpoints carry the bare `deepseek4` string in
    /// `general.architecture`; the safetensors convention is
    /// `DeepseekV4ForCausalLM`.
    public static let architectures: Set<String> = [
        "DeepseekV4ForCausalLM",
        "DeepseekV4Model",
        "deepseek4",
    ]

    /// Resolve the concrete variant from config. Flash and Pro share
    /// the same architecture surface; the variant is picked by total
    /// parameter count (Pro: 1.6T; Flash: 284B). Defaults to Flash
    /// since that's the user-runnable size on Apple Silicon today.
    public static func variant(
        for config: ModelConfig
    ) throws -> any DeepSeekV4Variant.Type {
        let tc = DeepSeekV4Config.textConfig(config)
        // `num_hidden_layers` is the cheapest variant discriminator —
        // Pro layers ≈ 60+, Flash = 43. Fall back to Flash on any
        // ambiguity.
        if let layers = tc.int("num_hidden_layers"), layers > 50 {
            return DeepSeekV4Pro.self
        }
        return DeepSeekV4Flash.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol DeepSeekV4Variant {
    static var availableCapabilities: Set<Capability> { get }
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> DeepSeekV4Model

    /// GGUF entry point — parallel to `loadModel(weights:)` for the
    /// safetensors path. The default impl throws
    /// `notYetImplemented`; concrete variants override once the
    /// GGUF→tensor dequant + tokenizer reconstruction land.
    static func loadModelFromGGUF(
        config: ModelConfig,
        gguf: GGUFTensorBundle,
        options: LoadOptions,
        device: Device
    ) throws -> DeepSeekV4Model
}

extension DeepSeekV4Variant {
    public static var availableCapabilities: Set<Capability> {
        [.textIn, .textOut]
    }
    public static var defaultGenerationParameters: GenerationParameters {
        // DSv4-Flash: 1M context (256K from RoPE base + 4× YARN
        // extrapolation). The 4096-token prefill chunk matches the
        // Gemma 4 / Qwen 3 hybrid family defaults — large enough to
        // amortise the MLA absorb-W_UK setup over many positions,
        // small enough to fit on a 96 GB Apple Silicon machine.
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }

    public static func loadModelFromGGUF(
        config: ModelConfig,
        gguf: GGUFTensorBundle,
        options: LoadOptions,
        device: Device
    ) throws -> DeepSeekV4Model {
        _ = config; _ = gguf; _ = options; _ = device
        throw DeepSeekV4Error.notYetImplemented("GGUF forward path")
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

public enum DeepSeekV4Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedLayerType(String)
    case unsupportedRouterShape(String)
    case unsupportedQuantType(GGUFTensorType, tensor: String)
    case notYetImplemented(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "DeepSeekV4: required config field missing: \(f)"
        case .missingTensor(let name):
            return "DeepSeekV4: checkpoint is missing tensor '\(name)'"
        case .unsupportedLayerType(let t):
            return "DeepSeekV4: unknown layer kind '\(t)' (expected one of: full / csa / hca)"
        case .unsupportedRouterShape(let why):
            return "DeepSeekV4: MoE router shape unsupported: \(why)"
        case .unsupportedQuantType(let t, let tensor):
            return "DeepSeekV4: GGUF quant '\(t)' for tensor '\(tensor)' not yet supported"
        case .notYetImplemented(let what):
            return "DeepSeekV4: \(what) — WIP, not yet implemented"
        }
    }
}
