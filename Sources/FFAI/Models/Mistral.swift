// Mistral family — Mistral 7B and Nemo / Small variants. The
// architecture is byte-for-byte identical to Llama 3 except for the
// default rope_theta (1_000_000) and a slightly different
// max_position_embeddings cap. Both differences flow naturally through
// `LlamaDense` from `config.json`, so this file aliases the loader
// rather than duplicating it.
//
// Mistral checkpoints declare `model_type = "mistral"` and
// `architectures = ["MistralForCausalLM"]`. They use the same weight
// layout: `model.layers.{i}.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,
// down}_proj`, `input_layernorm`, `post_attention_layernorm`. The
// engine returned is therefore a `LlamaModel`.
//
// Mistral 3 (Mistral Small / Devstral) ships with a slightly richer
// config (vision encoders, etc.) — those variants surface as
// `Mistral3ForCausalLM` and route through their own family file when
// we add VL coverage in Phase 6.5.

import Foundation

public enum Mistral {
    public static let modelTypes: Set<String> = ["mistral"]
    public static let architectures: Set<String> = ["MistralForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any LlamaVariant.Type {
        // Mistral has no architectural variants relevant to the
        // dense-text path. Reuse Llama's dense loader.
        return LlamaDense.self
    }
}
