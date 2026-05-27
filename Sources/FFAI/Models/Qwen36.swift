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
// Qwen 3.6 family root — Alibaba's Qwen 3.6 line.
//
// Qwen 3.6 ships under the *same* `qwen3_5*` HuggingFace `model_type`
// strings and the same `Qwen3_5*ForCausalLM` / `Qwen3_5*ForConditional-
// Generation` architectures as Qwen 3.5 — the architecture (stack-
// interleaved Gated Delta Net ↔ attention, dense SwiGLU or block-sparse
// MoE FFN with an always-on shared expert) is unchanged. The only
// practical 3.5 → 3.6 deltas are larger checkpoints, more MoE experts,
// and quantized embeddings — all of which the shared loader handles
// natively. This file mirrors `Qwen35.swift`'s shape so the Qwen 3.6
// entry point is discoverable from `Models/<Family>.swift` (the
// directory-scan rule the rest of the codebase follows) — the enum /
// variant-protocol / error types delegate to the Qwen 3.5 namespace so
// the actual backbone implementation lives in one place.
//
// Concrete variants + the hybrid decoder + per-layer impl live under
// `Models/Text/Qwen3xText.swift` (covers BOTH Qwen 3.5 AND Qwen 3.6):
//   - `Qwen35Hybrid` — the single variant; dense-vs-MoE is decided per
//     checkpoint inside `loadModel` from `num_experts`.
//   - `Qwen35Model` — the full LanguageModel decoder.
//
// The Qwen 3-VL-MoE vision-language orchestrator (`enum Qwen3VLMoe`)
// lives in `Models/Vision/Qwen3Vision.swift` alongside its dense
// Qwen3-VL sibling. The dense Qwen 3.5-VL orchestrator (`enum Qwen35VL`)
// lives in `Models/Vision/Qwen3xVision.swift` and will host the Qwen
// 3.6-VL orchestrator when that release ships — same naming convention
// as `Qwen3xText.swift` for the text variants.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Qwen36 {
    /// Qwen 3.6 ships under the same `model_type` strings as Qwen 3.5.
    /// Re-exported here so a caller probing `Qwen36.modelTypes` finds
    /// the right set, and so a future split (if HF ever differentiates
    /// 3.6 in config) only touches this constant.
    public static let modelTypes: Set<String> = Qwen35.modelTypes

    /// Architecture strings — same as Qwen 3.5 (`Qwen3_5ForCausalLM`,
    /// `Qwen3_5MoeForCausalLM`, `Qwen3_5ForConditionalGeneration`,
    /// `Qwen3_5MoeForConditionalGeneration`).
    public static let architectures: Set<String> = Qwen35.architectures

    /// Variant dispatch — delegates to `Qwen35.variant(for:)`. Both
    /// families resolve to `Qwen35Hybrid`; dense-vs-MoE is decided
    /// per-checkpoint from `num_experts` inside `loadModel`.
    public static func variant(for config: ModelConfig) throws -> any Qwen35Variant.Type {
        try Qwen35.variant(for: config)
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

/// Qwen 3.6's variant protocol — the same protocol as `Qwen35Variant`
/// (re-exported so a caller writing `func loadModel(...) -> Qwen36-
/// Variant` doesn't have to know that 3.5 and 3.6 share the implementation).
public typealias Qwen36Variant = Qwen35Variant

// ─── Errors ──────────────────────────────────────────────────────────

/// Qwen 3.6's error type — the same type as `Qwen35Error` (re-exported
/// so 3.6-specific call sites read naturally). Cases mirror `Qwen35Error`.
public typealias Qwen36Error = Qwen35Error
