// Copyright 2026 Tom Turney (@TheTom)
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
// Step 3 text backbone — Step-3.5-Flash / Step-3.7-Flash decoder.
//
// **Status:** WIP scaffold. This file declares the static shape — the
// `Step3TextConfig` decoder, the `Step3Hybrid` variant, and the
// `Step3Model` placeholder — so the loader can identify a Step-3
// checkpoint and dispatch into the family. Concrete forward / per-layer
// code lands in follow-ups.
//
// ─── Architecture summary (from the upstream config) ─────────────────
//
// • 45 transformer layers, hidden=4096, vocab=128896.
// • **Hybrid attention pattern**: 1 full-attention layer for every 3
//   sliding-window-512 layers (`layer_types` repeats
//   `[full, swa, swa, swa]`). Full-attn layers carry the long-range
//   context; SWA layers cap KV at 512 tokens.
// • **Asymmetric GQA per layer-type**:
//     – Full-attn layers: 64 query heads / 8 KV groups, head_dim=128.
//     – SWA layers:        96 query heads / 8 KV groups, head_dim=128.
//   Routes to `Ops.sdpaDecode2Pass` at 256K context (token-parallel
//   2-pass FA-2 — required at this kv length).
// • **Partial RoPE on full-attn layers only** (`partial_rotary_factors`
//   = 0.5 on full, 1.0 on SWA). **YARN scaling applied only to
//   full-attn layers** (factor 2.0, base 131072 → 262144 = 256K
//   context). Per-layer-type RoPE base: 5M on full, 10K on SWA.
// • **Per-head Q/K RMSNorm** (`q_norm` / `k_norm` over the head_dim
//   row) — identical to Qwen 3's stabilizer pattern.
// • **Head-wise attention output gate** (`use_head_wise_attn_gate`) —
//   a `sigmoid(g_proj(x))` scalar per head multiplied into the SDPA
//   output before `o_proj`. New op; not in the existing surface.
// • **MoE on layers 3-44** (288 experts, top-k=8, expert intermediate
//   dim=1280, 1 always-on shared expert). **Dense MLP on layers 0-2**
//   (intermediate dim=11264).
// • **Sigmoid + router-bias gate** (DeepSeek-V3 style), `argpartition`
//   top-k, optional normalize, scale (`moe_router_scaling_factor=3.0`).
//   Distinct from softmax routing — needs a sigmoid-gate variant in
//   the MoE router op.
// • **Clamped SwiGLU on layers 43-44** (gpt-oss-style activation
//   clipping: `clip(silu(gate), max=L) * clip(x, -L, L)`). Plain
//   SwiGLU on all other layers — clipping is a per-layer override.
// • **MTP heads** present in the checkpoint (`num_nextn_predict_layers`
//   = 3) — wiring is a follow-up; the loader strips them today.

import Foundation

// ─── Step3TextConfig ─────────────────────────────────────────────────

/// Static text-decoder shape decoded from `text_config`. Verbose enough
/// to drive both the dense MLP layers (0-2) and the MoE layers (3-44),
/// and to apply the per-layer-type overrides (head count, RoPE base,
/// partial-rotary factor, YARN scope).
struct Step3TextConfig {
    let nLayers: Int
    let hidden: Int
    let vocab: Int
    let maxSeq: Int
    let rmsNormEps: Float

    // Default attention shape (the full-attn layers; SWA overrides
    // below).
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int

    // Per-layer overrides indexed by layer id — populated when
    // `attention_other_setting` is present in config and `layer_types`
    // marks a layer as `sliding_attention`. Empty for a uniform stack.
    let perLayerHeads: [Int: Int]
    let perLayerKVHeads: [Int: Int]
    let perLayerRopeTheta: [Int: Float]
    /// `partial_rotary_factors` as observed on the `layer_types` axis
    /// (full=0.5, SWA=1.0). The rotary dim is `headDim * factor`.
    let perLayerPartialRotaryFactor: [Int: Float]
    /// Window radius (token count) for sliding-window-attention layers.
    /// `0` means "this layer is full-attention".
    let perLayerSlidingWindow: [Int: Int]
    /// Layers (43-44 in the upstream checkpoint) where SwiGLU output is
    /// clipped to `± limit` — gpt-oss-style.
    let perLayerSwigluLimit: [Int: Float]
    /// Same as above, but for the always-on shared expert (limit 16 in
    /// the upstream checkpoint).
    let perLayerSharedSwigluLimit: [Int: Float]

    // RoPE — base values are layer-overridable; YARN scaling is only
    // applied to layers whose `layer_type` appears in `yarnLayerTypes`.
    let ropeTheta: Float
    let yarnFactor: Float
    let yarnOriginalContext: Int
    let yarnLayerTypes: Set<String>

    // MoE shape.
    let moeStartLayer: Int     // first MoE layer index (3 in 3.5/3.7)
    let nExperts: Int          // 288
    let nExpertsPerToken: Int  // 8
    let nSharedExperts: Int    // 1
    let moeIntermediate: Int   // 1280
    let sharedExpertIntermediate: Int
    let denseIntermediate: Int // 11264 on layers 0..moeStartLayer-1
    let routerBias: Bool       // true — sigmoid + bias (DeepSeek-V3 style)
    let routerNormTopK: Bool
    let routerScalingFactor: Float
    let needsFp32Gate: Bool

    /// Attention gate flag — true means an extra
    /// `sigmoid(g_proj(x))[..., None] * attn_out` term sits between
    /// SDPA and `o_proj`.
    let useHeadwiseAttnGate: Bool

    let tieWordEmbeddings: Bool

    static func decode(_ tc: ModelConfig) throws -> Step3TextConfig {
        guard
            let nLayers = tc.int("num_hidden_layers"),
            let hidden = tc.int("hidden_size"),
            let vocab = tc.int("vocab_size"),
            let nHeads = tc.int("num_attention_heads")
        else {
            throw Step3Error.missingConfig("text_config core attention shape")
        }
        let nKVHeads =
            tc.int("num_key_value_heads")
            ?? tc.int("num_attention_groups") ?? nHeads
        let headDim = tc.int("head_dim") ?? (hidden / nHeads)
        let maxSeq =
            tc.int("max_position_embeddings")
            ?? tc.int("model_max_length") ?? 262_144

        // Per-layer overrides — Step's `layer_types` is a list aligned
        // 1-1 with the transformer stack. SWA layers override
        // `n_heads`, optionally `rope_theta`, and the partial-rotary
        // factor.
        var perLayerHeads: [Int: Int] = [:]
        var perLayerKVHeads: [Int: Int] = [:]
        var perLayerRopeTheta: [Int: Float] = [:]
        var perLayerPartial: [Int: Float] = [:]
        var perLayerSWA: [Int: Int] = [:]

        let layerTypes = (tc.raw["layer_types"] as? [String]) ?? []
        let swaSize = tc.int("sliding_window") ?? 512

        let other = tc.nested("attention_other_setting")
        let otherHeads = other?["num_attention_heads"] as? Int
        let otherKVHeads = other?["num_key_value_heads"] as? Int
        let otherRopeTheta = (other?["rope_theta"] as? Double).map(Float.init)

        let fullPartial = Float(tc.float("partial_rotary_factors") ?? 0.5)
        let swaPartial: Float = 1.0

        for (i, lt) in layerTypes.enumerated() {
            switch lt {
            case "full_attention":
                perLayerPartial[i] = fullPartial
                perLayerSWA[i] = 0
            case "sliding_attention":
                if let h = otherHeads { perLayerHeads[i] = h }
                if let kv = otherKVHeads { perLayerKVHeads[i] = kv }
                if let rt = otherRopeTheta { perLayerRopeTheta[i] = rt }
                perLayerPartial[i] = swaPartial
                perLayerSWA[i] = swaSize
            default:
                throw Step3Error.unsupportedLayerType(lt)
            }
        }

        // Clamped SwiGLU — gpt-oss-style per-layer activation clip. The
        // upstream lists are length-num_hidden_layers, 0 means "no
        // clip"; preserve the sparse map shape.
        var perLayerSwiglu: [Int: Float] = [:]
        var perLayerSharedSwiglu: [Int: Float] = [:]
        if let limits = tc.raw["swiglu_limits"] as? [Double] {
            for (i, v) in limits.enumerated()
            where v > 0 && i < nLayers {
                perLayerSwiglu[i] = Float(v)
            }
        }
        if let limits = tc.raw["swiglu_limits_shared"] as? [Double] {
            for (i, v) in limits.enumerated()
            where v > 0 && i < nLayers {
                perLayerSharedSwiglu[i] = Float(v)
            }
        }

        // RoPE scaling — YARN applied selectively per layer-type.
        var yarnFactor: Float = 1.0
        var yarnOriginal: Int = maxSeq
        var yarnTypes: Set<String> = []
        if let rs = tc.nested("rope_scaling") {
            if let f = rs["factor"] as? Double { yarnFactor = Float(f) }
            if let o = rs["original_max_position_embeddings"] as? Int { yarnOriginal = o }
            if let only = rs["yarn_only_types"] as? [String] {
                yarnTypes = Set(only)
            } else {
                yarnTypes = ["full_attention", "sliding_attention"]
            }
        }

        // MoE.
        let moeStart = tc.int("moe_layer_start_index") ?? 3
        let nExperts = tc.int("num_experts") ?? tc.int("n_routed_experts") ?? 288
        let nExpertsPerToken =
            tc.int("num_experts_per_token")
            ?? tc.int("moe_topk") ?? 8
        let nShared = tc.int("num_shared_experts") ?? 1
        let moeIntermediate =
            tc.int("moe_intermediate_size")
            ?? tc.int("expert_dim") ?? 1280
        let sharedIntermediate =
            tc.int("share_expert_dim") ?? tc.int("shared_expert_intermediate_size") ?? moeIntermediate
        let denseIntermediate = tc.int("intermediate_size") ?? 11264

        return Step3TextConfig(
            nLayers: nLayers,
            hidden: hidden,
            vocab: vocab,
            maxSeq: maxSeq,
            rmsNormEps: Float(tc.float("rms_norm_eps") ?? 1e-6),
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            perLayerHeads: perLayerHeads,
            perLayerKVHeads: perLayerKVHeads,
            perLayerRopeTheta: perLayerRopeTheta,
            perLayerPartialRotaryFactor: perLayerPartial,
            perLayerSlidingWindow: perLayerSWA,
            perLayerSwigluLimit: perLayerSwiglu,
            perLayerSharedSwigluLimit: perLayerSharedSwiglu,
            ropeTheta: Float(tc.float("rope_theta") ?? 10_000),
            yarnFactor: yarnFactor,
            yarnOriginalContext: yarnOriginal,
            yarnLayerTypes: yarnTypes,
            moeStartLayer: moeStart,
            nExperts: nExperts,
            nExpertsPerToken: nExpertsPerToken,
            nSharedExperts: nShared,
            moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: sharedIntermediate,
            denseIntermediate: denseIntermediate,
            routerBias: tc.bool("router_bias") ?? true,
            routerNormTopK: tc.bool("norm_topk_prob") ?? tc.bool("norm_expert_weight") ?? true,
            routerScalingFactor: Float(tc.float("moe_router_scaling_factor") ?? 3.0),
            needsFp32Gate: tc.bool("need_fp32_gate") ?? false,
            useHeadwiseAttnGate: tc.bool("use_head_wise_attn_gate") ?? false,
            tieWordEmbeddings: tc.bool("tie_word_embeddings") ?? false)
    }
}

// ─── Step3Hybrid — the canonical Step-3.5/3.7 variant ────────────────

public enum Step3Hybrid: Step3Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }

    public static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Step3Model {
        let tc = Step3Config.textConfig(config)
        _ = try Step3TextConfig.decode(tc)
        // Forward path is the next deliverable. The config decode above
        // exercises the architecture-recognition surface and gives a
        // useful diagnostic on bad / partial checkpoints.
        throw Step3Error.notYetImplemented("Step3Hybrid decoder forward")
    }
}

// ─── Step3Model — placeholder ────────────────────────────────────────

/// Hybrid Step-3 decoder. **WIP** — the concrete `Module` conformance,
/// per-layer wiring, and `forward` paths land in follow-up commits. The
/// type exists so the family entry-point + loader dispatch type-check
/// today.
public final class Step3Model {
    let textConfig: Step3TextConfig
    init(textConfig: Step3TextConfig) {
        self.textConfig = textConfig
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Tiny shim that mirrors `Gemma4Config.textConfig` / `Qwen35Config.textConfig`
/// — returns the `text_config` sub-tree on a VL conversion, otherwise
/// the top-level config (text-only checkpoint).
enum Step3Config {
    static func textConfig(_ c: ModelConfig) -> ModelConfig {
        c.subConfig("text_config") ?? c
    }
}
