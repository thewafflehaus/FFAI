// OLMo family — Allen AI's open-source Llama-shaped dense decoder.
// OLMo 1 (olmo) and OLMo 2 (olmo2) both ship with byte-identical
// weights to Llama 3 dense + optional QKV biases that `loadLinear`
// auto-detects, so the family root just declares dispatch metadata
// and routes the loader through `LlamaDense`.

import Foundation

public enum OLMo {
    public static let modelTypes: Set<String> = ["olmo", "olmo2"]
    public static let architectures: Set<String> = [
        "OlmoForCausalLM",
        "Olmo2ForCausalLM",
    ]
}
