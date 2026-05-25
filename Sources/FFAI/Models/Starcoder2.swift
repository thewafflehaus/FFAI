// Starcoder 2 family — BigCode's Llama-shaped code dense decoder.
// Same weight layout + forward shape as Llama 3; optional QKV biases
// auto-detected by `loadLinear`. Family root just declares dispatch
// metadata and routes through `LlamaDense`.

import Foundation

public enum Starcoder2 {
    public static let modelTypes: Set<String> = ["starcoder2"]
    public static let architectures: Set<String> = ["Starcoder2ForCausalLM"]
}
