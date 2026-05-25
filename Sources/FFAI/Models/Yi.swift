// Yi family — 01.ai's Yi dense text models. Llama-3-shaped weights
// with optional QKV biases that `loadLinear` auto-detects; the family
// root just declares dispatch metadata and routes through `LlamaDense`.

import Foundation

public enum Yi {
    public static let modelTypes: Set<String> = ["yi"]
    public static let architectures: Set<String> = ["YiForCausalLM"]
}
