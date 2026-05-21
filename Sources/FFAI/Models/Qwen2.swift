// Qwen 2.x family — the 2-series and 2.5-series dense models. These
// use the 3-series Llama-style architecture with one structural
// difference: the QKV projections carry per-output biases, while
// o_proj does not. The 3-series uses no biases anywhere.
//
// `loadLinear` in `Sources/FFAI/Layers.swift` was extended to detect
// `<base>.bias` automatically, so Qwen 2.x routes through the same
// loader as the 3-series + Mistral7B with no extra plumbing. The only
// thing this file adds is the model-type dispatch entry.
//
// Quantized Qwen 2.x (mlx-community 4-bit) is supported transparently:
// `QuantizedLinear` doesn't carry biases (the int4 / int8 packed-row
// gemv kernel applies the dequant scale+bias per group, which is a
// different axis), so the conversion drops the original projection
// biases for quantized variants. That's consistent with how mlx-lm
// loads these checkpoints.

import Foundation

public enum Qwen2 {
    /// Both 2.x and 2.5.x ship `model_type = "qwen2"`. The 2.5 refresh
    /// changed weights + tokenizer but kept the same arch + config keys.
    public static let modelTypes: Set<String> = ["qwen2"]
    public static let architectures: Set<String> = ["Qwen2ForCausalLM"]

    public static func variant(for config: ModelConfig) throws -> any LlamaVariant.Type {
        // No Qwen2-specific architectural variants relevant to dense
        // text. The 3-series dense loader handles the shape + RoPE.
        return LlamaDense.self
    }
}
