// InternLM 2 family — Shanghai AI Lab's InternLM v2 dense text models.
// Llama-3-shaped weights — some checkpoints use a fused `wqkv`
// projection that `loadLinear` handles transparently via the
// bias-aware Linear; the family root just declares dispatch metadata
// and routes through `LlamaDense`.

import Foundation

public enum InternLM2 {
    public static let modelTypes: Set<String> = ["internlm2"]
    public static let architectures: Set<String> = ["InternLM2ForCausalLM"]
}
