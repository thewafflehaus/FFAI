// Llama-compatible architectures — families that ship config.json
// architectures != "LlamaForCausalLM" but use a byte-for-byte
// compatible weight layout + forward shape. We add them as separate
// family enums so the model registry is self-documenting (each
// family lists the architectures + modelTypes it claims), then route
// through the existing Llama loader.
//
// Why a separate file: the weight-layout-compatible group is large
// and growing. Keeping them out of Llama.swift avoids that file
// becoming a registry. Each enum is a 6-line family declaration —
// no real loader code, just dispatch metadata.
//
// What "compatible" means here:
//   - q_proj / k_proj / v_proj / o_proj  (optional biases auto-picked
//     up by `loadLinear`)
//   - gate_proj / up_proj / down_proj    (SwiGLU MLP)
//   - input_layernorm + post_attention_layernorm  (RMSNorm)
//   - rope_theta / num_key_value_heads / etc. consumed from config.
//
// Anything beyond this — fused QKV (Phi-3), per-head q/k norms
// (Qwen 3, Gemma 3), MoE routing (Mixtral, DeepSeek MoE), tied
// parallel residual (Cohere), or rope quirks (yarn / longrope) —
// belongs in its own family file.
//
// The current entries cover the popular "Llama-3-style with maybe-a-
// bias" cluster: SmolLM 2/3, OLMo 2, Granite, Yi, Internlm 2,
// Starcoder 2, DeepSeek R1 Distill (Llama variant). DeepSeek R1
// Distill (Qwen variant) lives under Qwen2.

import Foundation

public enum LlamaCompatibles {
    /// Union of every model-type label that flows through the Llama
    /// loader without architectural mods.
    public static let modelTypes: Set<String> = [
        "smollm",        // SmolLM 1
        "smollm2",       // SmolLM 2 family
        "smollm3",       // SmolLM 3 family
        "olmo",          // OLMo 1
        "olmo2",         // OLMo 2
        "granite",       // Granite 3.x dense
        "yi",            // 01.ai Yi family
        "internlm2",     // InternLM 2 (uses fused wqkv on some; load_linear handles bias)
        "starcoder2",    // BigCode Starcoder 2
    ]

    /// Union of every architectures[] label that flows through the
    /// Llama loader without architectural mods.
    public static let architectures: Set<String> = [
        "SmolLMForCausalLM",
        "SmolLM2ForCausalLM",
        "SmolLM3ForCausalLM",
        "OlmoForCausalLM",
        "Olmo2ForCausalLM",
        "GraniteForCausalLM",
        "YiForCausalLM",
        "InternLM2ForCausalLM",
        "Starcoder2ForCausalLM",
    ]
}
