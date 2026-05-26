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
// the mlx-community converter KEEPS the additive output biases on the
// QKV projections (`q_proj.bias` next to `q_proj.{weight,scales,biases}`)
// even after 4-bit conversion. `QuantizedLinear` carries an optional
// `additiveBias` that's folded in after `Ops.dequantGemv` /
// `Ops.dequantGemmDynamicM`, and `loadLinear` picks it up from the
// bundle when present. Dropping these biases gave silent degenerate
// output (DeepSeek-R1-Distill-Qwen-1.5B emitted token-15 "0" forever
// before this fix). The per-group quant `biases` (plural) are a
// different axis — the dequant offset applied during the gemv itself —
// and have always been handled.

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
