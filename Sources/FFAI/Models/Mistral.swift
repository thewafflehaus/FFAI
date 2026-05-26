// Mistral family root — Mistral 7B and Nemo / Small dense decoders.
//
// This file is the **main model interface** for the family:
//   • the family enum `Mistral` (modelTypes, architectures, variant
//     dispatch). Mistral 7B / Nemo are byte-for-byte identical to
//     Llama 3 dense except for `rope_theta` and the
//     `max_position_embeddings` cap, both of which flow naturally
//     through `LlamaDense` from `config.json` — so `variant(for:)`
//     reuses `LlamaDense` rather than declaring a Mistral-specific
//     variant. No `MistralVariant` protocol or `MistralError` type
//     ships today.
//
// Concrete loader notes + the `MistralForCausalLM` weight-key contract
// live in `Models/Text/MistralText.swift`.
//
// Related (separate family):
//   - Models/Mistral3.swift — Mistral Small 3.1 vision-language
//                             (`mistral3` model_type, ViT + MLP
//                             projector + LlamaDense backbone)

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Mistral {
    public static let modelTypes: Set<String> = ["mistral"]
    public static let architectures: Set<String> = ["MistralForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any LlamaVariant.Type {
        // Mistral has no architectural variants relevant to the
        // dense-text path. Reuse Llama's dense loader.
        return LlamaDense.self
    }
}
