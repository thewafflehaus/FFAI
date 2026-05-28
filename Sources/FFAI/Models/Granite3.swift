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
// Granite 3 family — IBM's Granite v3 dense text models (granite-3.0,
// granite-3.1, granite-3.2). Llama-3-shaped weights with optional QKV
// biases that `loadLinear` auto-detects; the family root just declares
// the dispatch metadata and routes the loader through `LlamaDense`.
//
// Granite 4 (granite-4.0-h, GraniteMoeHybrid) is a different
// architecture — Mamba 2 / attention / MoE stack-interleaved hybrid —
// and lives in `Models/Text/Granite4Text.swift` with its own forward
// code.

import Foundation

public enum Granite3 {
    public static let modelTypes: Set<String> = ["granite"]
    public static let architectures: Set<String> = ["GraniteForCausalLM"]

    /// µP multipliers declared in a Granite 3.x `config.json`. Granite
    /// trains with maximal-update-parametrization, so these scale the
    /// embeddings, attention scores, residual branches, and final
    /// logits — all default to identity for checkpoints that omit them.
    /// `LlamaDense.loadModel(muP:)` folds them into the weights so the
    /// shared Llama forward needs no Granite-specific code.
    public static func muP(from config: ModelConfig) -> MuPMultipliers {
        MuPMultipliers(
            embedding: Float(config.float("embedding_multiplier") ?? 1.0),
            attention: config.float("attention_multiplier").map { Float($0) },
            residual: Float(config.float("residual_multiplier") ?? 1.0),
            logits: Float(config.float("logits_scaling") ?? 1.0))
    }
}
