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
// DeepSeek V4 text backbone — DSv4-Flash / DSv4-Pro decoder config +
// variants.
//
// **Status:** WIP scaffold. This file declares the static shape — the
// `DeepSeekV4TextConfig` decoder, the two variants (`DeepSeekV4Flash`,
// `DeepSeekV4Pro`), and the `DeepSeekV4Model` placeholder — so the
// loader can identify a DSv4 checkpoint (safetensors or GGUF) and
// dispatch into the family. The forward path land in follow-up PRs
// per the multi-week metaltile kernel sequence (MLA decode, CSA
// sparse-gather SDPA, HCA compressed-stream SDPA, Lightning Indexer,
// FP4 / block-FP8 dequant).
//
// ─── Architecture summary (from the upstream config) ─────────────────
//
// • 43 transformer layers + 1 MTP head, hidden=4096, vocab=129,280.
// • Interleaved attention pattern (`compress_ratios` aligned 1-1 with
//   the layer stack): layers 0-1 = full attention; layers 2-41 =
//   alternating CSA(4×) / HCA(128×) pairs; layer 42 = full; layer 43
//   = MTP next-N predictor.
// • **MLA (Multi-head Latent Attention)** carried over from DSv3.
//   `head_dim=512` (MLA latent dim), `kv_heads=1` (single MLA cache),
//   `qk_rope_head_dim=64` decoupled-RoPE concat tail. Q is low-rank-
//   compressed to `q_lora_rank=1024`. O-projection is now ALSO
//   low-rank (`o_lora_rank=1024`, `o_groups=8`) — new in V4.
// • **CSA (Compressed Sparse Attention)** — 4× compressed KV stream.
//   A 64-head × 128-dim **Lightning Indexer** sub-network scores all
//   compressed entries; per query, top-`index_topk=512` selected +
//   128-token local sliding window. Scoring: `sqrtsoftplus`.
// • **HCA (Heavily Compressed Attention)** — 128× compressed KV
//   stream. Dense attention (no top-k) over the small compressed
//   buffer. Separate `compress_rope_theta=160_000` (vs `rope_theta=
//   10_000` for full + CSA layers).
// • MoE on every non-MTP layer (288 experts, top-k=6, expert
//   intermediate=2048, 1 always-on shared expert at dim=2048).
//   `noaux_tc` aux-loss-free + per-expert bias routing.
// • **`sqrtsoftplus` routing scoring** — replaces DSv3's sigmoid+bias
//   gate (companion `ffai_moe_router_sigmoid_bias` kernel from
//   the Step-3 series is the closest cousin; sqrtsoftplus is its
//   own small variant kernel).
// • **mHC (Manifold-Constrained Hyper-Connections)** — residual
//   projection matrix constrained to the Birkhoff polytope (doubly
//   stochastic) via 20 Sinkhorn-Knopp iterations. Folded into
//   weights at LOAD time, not per token — no runtime kernel needed.
// • **Clamped SwiGLU** — `swiglu_limit=10.0` applied to every MoE
//   expert MLP. The Step-3 `mt_clamped_swiglu` kernel is the
//   drop-in here.
// • **MTP head** — `num_nextn_predict_layers=1`. Carries the same
//   per-layer block shape as the main stack but its output flows
//   into a separate logits head used by speculative decoding.
//
// ─── Mixed precision (from the upstream config) ──────────────────────
//
// • MoE expert weights stored as **fp4** (FP4 e2m1, block 32, with
//   per-block fp8 e4m3 scales). New kernel family — not in
//   metaltile-std today.
// • Attention / router weights stored as **block-FP8** (FP8 e4m3,
//   block 128×128). Distinct from per-channel int4 affine; another
//   net-new dequant kernel.
// • Activations: bf16 throughout. Output norms: f32.

import Foundation

// ─── DeepSeekV4TextConfig ────────────────────────────────────────────

struct DeepSeekV4TextConfig {
    let nLayers: Int        // 43 (excludes MTP)
    let hidden: Int         // 4096
    let vocab: Int          // 129_280
    let maxSeq: Int         // 1_048_576 (1M)
    let rmsNormEps: Float

    // ── Attention shape ──
    let nHeads: Int          // 64
    let nKVHeads: Int        // 1 — MLA carries one logical KV head
    let headDim: Int         // 512 — MLA latent dim
    let qkRopeHeadDim: Int   // 64 — decoupled-RoPE concat tail
    let qLoraRank: Int       // 1024 — Q low-rank compression
    let oLoraRank: Int       // 1024 — O-projection low-rank (new in V4)
    let oGroups: Int         // 8 — O-projection group count

    // ── Per-layer compression schedule ──
    /// `compress_ratios` mirror of the layer stack (length =
    /// nLayers + MTP slot). Value semantics:
    ///   - 0 → full attention
    ///   - 4 → CSA (4× compression)
    ///   - 128 → HCA (128× compression)
    let layerCompressRatios: [Int]
    let slidingWindow: Int   // 128 — CSA local window

    // ── Lightning Indexer ──
    let indexerHeads: Int        // 64
    let indexerHeadDim: Int      // 128
    let indexerTopK: Int         // 512

    // ── mHC ──
    let hcMultiplier: Int        // 4
    let hcEpsilon: Float         // 1e-6
    let hcSinkhornIterations: Int // 20

    // ── RoPE ──
    let ropeTheta: Float          // 10_000 (full + CSA)
    let compressRopeTheta: Float  // 160_000 (HCA compressed stream)
    let yarnFactor: Float         // 16
    let yarnOriginalContext: Int  // 65_536

    // ── MoE ──
    let nExperts: Int             // 288
    let nExpertsPerToken: Int     // 6
    let nSharedExperts: Int       // 1
    let moeIntermediate: Int      // 2048
    let sharedExpertIntermediate: Int  // 2048
    let routerBias: Bool          // true (noaux_tc)
    let routerScalingFactor: Float // 1.5
    let routerScoringFunc: String  // "sqrtsoftplus"

    // ── Activation clip ──
    let swigluLimit: Float        // 10.0

    // ── MTP ──
    let nMTPLayers: Int           // 1

    static func decode(_ tc: ModelConfig) throws -> DeepSeekV4TextConfig {
        guard
            let nLayers = tc.int("num_hidden_layers"),
            let hidden = tc.int("hidden_size"),
            let vocab = tc.int("vocab_size"),
            let nHeads = tc.int("num_attention_heads")
        else {
            throw DeepSeekV4Error.missingConfig("text_config core attention shape")
        }
        let nKVHeads = tc.int("num_key_value_heads") ?? 1
        let headDim = tc.int("head_dim") ?? 512
        let qkRopeHeadDim = tc.int("qk_rope_head_dim") ?? 64
        let qLoraRank = tc.int("q_lora_rank") ?? 1024
        let oLoraRank = tc.int("o_lora_rank") ?? 1024
        let oGroups = tc.int("o_groups") ?? 8
        let maxSeq =
            tc.int("max_position_embeddings")
            ?? tc.int("model_max_length") ?? 1_048_576

        // `compress_ratios` schedule. If absent, default to a
        // uniformly-full attention stack (which is wrong for DSv4 but
        // safe — the loader will reject before forward is reached).
        let ratios: [Int] = tc.intArray("compress_ratios") ?? Array(repeating: 0, count: nLayers + 1)

        let slidingWindow = tc.int("sliding_window") ?? 128

        // Lightning indexer.
        let indexerHeads = tc.int("index_n_heads") ?? 64
        let indexerHeadDim = tc.int("index_head_dim") ?? 128
        let indexerTopK = tc.int("index_topk") ?? 512

        // mHC.
        let hcMult = tc.int("hc_mult") ?? 4
        let hcEps = Float(tc.float("hc_eps") ?? 1e-6)
        let hcIters = tc.int("hc_sinkhorn_iters") ?? 20

        // RoPE.
        let ropeTheta = Float(tc.float("rope_theta") ?? 10_000)
        let compressRopeTheta = Float(tc.float("compress_rope_theta") ?? 160_000)
        var yarnFactor: Float = 1.0
        var yarnOriginal = maxSeq
        if let rs = tc.nested("rope_scaling") {
            if let f = rs["factor"] as? Double { yarnFactor = Float(f) }
            if let o = rs["original_max_position_embeddings"] as? Int { yarnOriginal = o }
        }

        // MoE.
        let nExperts = tc.int("n_routed_experts") ?? tc.int("num_experts") ?? 288
        let nExpertsPerToken =
            tc.int("num_experts_per_tok")
            ?? tc.int("num_experts_per_token") ?? 6
        let nShared = tc.int("n_shared_experts") ?? 1
        let moeIntermediate =
            tc.int("moe_intermediate_size") ?? tc.int("expert_dim") ?? 2048
        let sharedIntermediate =
            tc.int("share_expert_dim")
            ?? tc.int("shared_expert_intermediate_size")
            ?? moeIntermediate
        let routerScoringFunc = tc.string("scoring_func") ?? "sqrtsoftplus"
        let routerScale = Float(tc.float("routed_scaling_factor") ?? 1.5)

        let swigluLimit = Float(tc.float("swiglu_limit") ?? 10.0)
        let nMTPLayers = tc.int("num_nextn_predict_layers") ?? 1

        return DeepSeekV4TextConfig(
            nLayers: nLayers,
            hidden: hidden,
            vocab: vocab,
            maxSeq: maxSeq,
            rmsNormEps: Float(tc.float("rms_norm_eps") ?? 1e-6),
            nHeads: nHeads,
            nKVHeads: nKVHeads,
            headDim: headDim,
            qkRopeHeadDim: qkRopeHeadDim,
            qLoraRank: qLoraRank,
            oLoraRank: oLoraRank,
            oGroups: oGroups,
            layerCompressRatios: ratios,
            slidingWindow: slidingWindow,
            indexerHeads: indexerHeads,
            indexerHeadDim: indexerHeadDim,
            indexerTopK: indexerTopK,
            hcMultiplier: hcMult,
            hcEpsilon: hcEps,
            hcSinkhornIterations: hcIters,
            ropeTheta: ropeTheta,
            compressRopeTheta: compressRopeTheta,
            yarnFactor: yarnFactor,
            yarnOriginalContext: yarnOriginal,
            nExperts: nExperts,
            nExpertsPerToken: nExpertsPerToken,
            nSharedExperts: nShared,
            moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: sharedIntermediate,
            routerBias: tc.bool("router_bias") ?? true,
            routerScalingFactor: routerScale,
            routerScoringFunc: routerScoringFunc,
            swigluLimit: swigluLimit,
            nMTPLayers: nMTPLayers)
    }
}

// ─── Variants ────────────────────────────────────────────────────────

/// 284B total / 13B active. The user-runnable size on Apple Silicon
/// today (4-bit weights + dequant load fits in 128 GB unified memory
/// at ~32K context).
public enum DeepSeekV4Flash: DeepSeekV4Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }

    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        let tc = DeepSeekV4Config.textConfig(config)
        _ = try DeepSeekV4TextConfig.decode(tc)
        throw DeepSeekV4Error.notYetImplemented("DeepSeekV4Flash safetensors forward path")
    }

    public static func loadModelFromGGUF(
        config: ModelConfig, gguf: GGUFTensorBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        let tc = DeepSeekV4Config.textConfig(config)
        _ = try DeepSeekV4TextConfig.decode(tc)
        // Architecture-string sanity-check now that the reader is
        // open. Either form is accepted upstream.
        if let arch = gguf.architecture,
            !DeepSeekV4.architectures.contains(arch),
            arch != "deepseek4"
        {
            throw DeepSeekV4Error.missingConfig(
                "general.architecture='\(arch)' not in DeepSeekV4 known set")
        }
        throw DeepSeekV4Error.notYetImplemented("DeepSeekV4Flash GGUF forward path")
    }
}

/// ~1.6T / 49B active. Same architecture as Flash, deeper + wider
/// MoE. Not currently runnable on Apple Silicon (weights alone need
/// ~480 GB at 4-bit). Kept here for completeness — the variant
/// dispatch + config decode work today; load throws.
public enum DeepSeekV4Pro: DeepSeekV4Variant {
    public static func loadModel(
        config: ModelConfig, weights: SafeTensorsBundle,
        options: LoadOptions, device: Device
    ) throws -> DeepSeekV4Model {
        _ = config; _ = weights; _ = options; _ = device
        throw DeepSeekV4Error.notYetImplemented("DeepSeekV4Pro safetensors forward path")
    }
}

// ─── DeepSeekV4Model — weight slots ──────────────────────────────────
//
// Tensor inventory mirrors the DSv4-Flash GGUF (see
// `Tests/ModelIntegrationTests/GGUFDsv4TensorMapTest.swift` for the
// full dump). Field names follow the GGUF tensor-name convention so
// the loader is a direct `bundle.tensor(named:"blk.\(N).\(suffix)")`
// dispatch.
//
// Architecture summary:
//
// - Each layer holds an mHC 4-channel residual state H[hidden, 4, t].
// - Attention sub-block: rms_norm → q_a → q_a_norm → q_b → per-head
//   Q-norm (eps-only) → partial RoPE on tail 64 dims of each head;
//   kv (single 512-d MQA head) → kv_a_norm → partial RoPE on tail 64
//   dims → optional FP8 quantize on first 448 dims → store in cache.
//   Softmax-attention with `attn_sinks` (per-head learnable extra
//   logit). Inverse partial RoPE on output. Grouped O-LoRA: reshape
//   to [4096, 8 groups] × [4096, 1024] per group → [8192] → wo_b →
//   [4096].
// - FFN sub-block: rms_norm → MoE (256 experts top-6, sqrt-softplus
//   routing OR precomputed hash routing via `ffn_gate_tid2eid` on
//   the first `n_hash_layers`) + shared-expert SwiGLU.
// - mHC: at each sub-block boundary, `hc_*_fn @ flatten(H)` produces
//   a 24-dim mix that splits as 4 `pre` (sigmoid+eps) + 4 `post`
//   (2·sigmoid) + 16 (=4×4) `comb` matrix (softmax + Sinkhorn-Knopp
//   row/col-normalized). `pre` collapses H → sub-block input;
//   `post` + `comb` expand the sub-block output back into H.
// - CSA (compress_ratio=4): adds a Lightning Indexer that scores all
//   compressed K-entries against a 64-head × 128-dim Q (sharing the
//   same `qr` from the main attn-path) + an attn compressor that
//   builds a 4×-pooled (overlap-2) compressed KV stream. Top-512
//   compressed slots feed the sparse-gather attention.
// - HCA (compress_ratio=128): only the attn compressor (no indexer);
//   dense attention over the small compressed stream.
// - Layer pattern per the GGUF: 0,1 = full; 2,4,…,42 = CSA;
//   3,5,…,41 = HCA. The `compress_ratios` array on the GGUF metadata
//   is authoritative.

/// One transformer block's worth of weights. Allocated per-layer
/// regardless of compression regime — the regime-specific tensors
/// (compressor, indexer) are nil for layers that don't use them.
final class DeepSeekV4Layer: @unchecked Sendable {
    let layerIndex: Int
    let compressRatio: Int  // 0 = full, 4 = CSA, 128 = HCA

    // ── Common attention path ──
    let attnNorm: Tensor          // f32 [hidden]
    let attnQA: Tensor            // q8_0 [hidden, q_lora_rank]
    let attnQANorm: Tensor        // f32 [q_lora_rank]
    let attnQB: Tensor            // q8_0 [q_lora_rank, n_heads * head_dim]
    let attnKV: Tensor            // q8_0 [hidden, head_dim] (MQA: 1 kv head)
    let attnKVANorm: Tensor       // f32 [head_dim]
    let attnSinks: Tensor         // f32 [n_heads]
    let attnOutputA: Tensor       // q8_0 [group_dim, n_groups * o_lora_rank]
    let attnOutputB: Tensor       // q8_0 [n_groups * o_lora_rank, hidden]

    // ── FFN path ──
    let ffnNorm: Tensor              // f32 [hidden]
    let ffnGateInp: Tensor           // f16 [hidden, n_experts]
    let ffnGateTid2Eid: Tensor?      // i32 [n_experts_per_token, vocab] — hash-route, nil past n_hash_layers
    let ffnGateExps: Tensor          // iq2_xxs [hidden, expert_intermediate, n_experts]
    let ffnUpExps: Tensor            // iq2_xxs [hidden, expert_intermediate, n_experts]
    let ffnDownExps: Tensor          // q2_K [expert_intermediate, hidden, n_experts]
    let ffnGateShexp: Tensor         // q8_0 [hidden, shared_expert_intermediate]
    let ffnUpShexp: Tensor           // q8_0 [hidden, shared_expert_intermediate]
    let ffnDownShexp: Tensor         // q8_0 [shared_expert_intermediate, hidden]
    let expProbsBias: Tensor?        // f32 [n_experts] — noaux_tc bias, only on non-hash layers

    // ── mHC weights (attn + ffn sub-blocks) ──
    let hcAttnBase: Tensor    // f32 [24]
    let hcAttnFn: Tensor      // f16 [hc_dim, 24]  where hc_dim = n_hc * hidden = 4*4096 = 16384
    let hcAttnScale: Tensor   // f32 [3]
    let hcFfnBase: Tensor     // f32 [24]
    let hcFfnFn: Tensor       // f16 [hc_dim, 24]
    let hcFfnScale: Tensor    // f32 [3]

    // ── CSA / HCA compressor (compress_ratio > 0) ──
    let attnCompressorAPE: Tensor?    // f16 [coff * head_dim, ratio]  (ratio=4 for CSA, =128 for HCA)
    let attnCompressorGate: Tensor?   // f16 [hidden, coff * head_dim]
    let attnCompressorKV: Tensor?     // f16 [hidden, coff * head_dim]
    let attnCompressorNorm: Tensor?   // f32 [head_dim]

    // ── CSA-only Lightning Indexer (compress_ratio == 4) ──
    let indexerAttnQB: Tensor?              // f16 [q_lora_rank, indexer_n_heads * indexer_head_size]
    let indexerProj: Tensor?                // f16 [hidden, indexer_n_heads]
    let indexerCompressorAPE: Tensor?       // f16 [coff * indexer_head_size, ratio]
    let indexerCompressorGate: Tensor?      // f16 [hidden, coff * indexer_head_size]
    let indexerCompressorKV: Tensor?        // f16 [hidden, coff * indexer_head_size]
    let indexerCompressorNorm: Tensor?      // f32 [indexer_head_size]

    init(
        layerIndex: Int, compressRatio: Int,
        attnNorm: Tensor, attnQA: Tensor, attnQANorm: Tensor, attnQB: Tensor,
        attnKV: Tensor, attnKVANorm: Tensor, attnSinks: Tensor,
        attnOutputA: Tensor, attnOutputB: Tensor,
        ffnNorm: Tensor, ffnGateInp: Tensor, ffnGateTid2Eid: Tensor?,
        ffnGateExps: Tensor, ffnUpExps: Tensor, ffnDownExps: Tensor,
        ffnGateShexp: Tensor, ffnUpShexp: Tensor, ffnDownShexp: Tensor,
        expProbsBias: Tensor?,
        hcAttnBase: Tensor, hcAttnFn: Tensor, hcAttnScale: Tensor,
        hcFfnBase: Tensor, hcFfnFn: Tensor, hcFfnScale: Tensor,
        attnCompressorAPE: Tensor? = nil, attnCompressorGate: Tensor? = nil,
        attnCompressorKV: Tensor? = nil, attnCompressorNorm: Tensor? = nil,
        indexerAttnQB: Tensor? = nil, indexerProj: Tensor? = nil,
        indexerCompressorAPE: Tensor? = nil, indexerCompressorGate: Tensor? = nil,
        indexerCompressorKV: Tensor? = nil, indexerCompressorNorm: Tensor? = nil
    ) {
        self.layerIndex = layerIndex
        self.compressRatio = compressRatio
        self.attnNorm = attnNorm; self.attnQA = attnQA; self.attnQANorm = attnQANorm
        self.attnQB = attnQB; self.attnKV = attnKV; self.attnKVANorm = attnKVANorm
        self.attnSinks = attnSinks
        self.attnOutputA = attnOutputA; self.attnOutputB = attnOutputB
        self.ffnNorm = ffnNorm; self.ffnGateInp = ffnGateInp; self.ffnGateTid2Eid = ffnGateTid2Eid
        self.ffnGateExps = ffnGateExps; self.ffnUpExps = ffnUpExps; self.ffnDownExps = ffnDownExps
        self.ffnGateShexp = ffnGateShexp; self.ffnUpShexp = ffnUpShexp; self.ffnDownShexp = ffnDownShexp
        self.expProbsBias = expProbsBias
        self.hcAttnBase = hcAttnBase; self.hcAttnFn = hcAttnFn; self.hcAttnScale = hcAttnScale
        self.hcFfnBase = hcFfnBase; self.hcFfnFn = hcFfnFn; self.hcFfnScale = hcFfnScale
        self.attnCompressorAPE = attnCompressorAPE; self.attnCompressorGate = attnCompressorGate
        self.attnCompressorKV = attnCompressorKV; self.attnCompressorNorm = attnCompressorNorm
        self.indexerAttnQB = indexerAttnQB; self.indexerProj = indexerProj
        self.indexerCompressorAPE = indexerCompressorAPE
        self.indexerCompressorGate = indexerCompressorGate
        self.indexerCompressorKV = indexerCompressorKV
        self.indexerCompressorNorm = indexerCompressorNorm
    }
}

/// DSv4 decoder. Holds the per-layer weight bundles plus the
/// shared embedding / output-norm / LM-head / output-mHC weights.
///
/// `forward(...)` lands incrementally — first the full-attn layer
/// type (layers 0, 1) end-to-end against a small prompt, then the
/// CSA and HCA paths. See `DeepSeekV4Forward.swift` (lands when the
/// first layer type clears local tests).
public final class DeepSeekV4Model: @unchecked Sendable {
    let textConfig: DeepSeekV4TextConfig

    // ── Non-block weights ──
    let tokenEmbd: Tensor         // f16 [hidden, vocab]
    let outputNorm: Tensor        // f32 [hidden]
    let outputHead: Tensor        // q8_0 [hidden, vocab]
    let outputHcBase: Tensor      // f32 [n_hc=4]
    let outputHcFn: Tensor        // f16 [hc_dim, n_hc]  (head mHC uses simpler decomposition)
    let outputHcScale: Tensor     // f32 [1]

    /// 43 per-layer bundles, indexed 0..<nLayers. CSA / HCA / full
    /// regime is recoverable via `layers[i].compressRatio`.
    let layers: [DeepSeekV4Layer]

    init(
        textConfig: DeepSeekV4TextConfig,
        tokenEmbd: Tensor, outputNorm: Tensor, outputHead: Tensor,
        outputHcBase: Tensor, outputHcFn: Tensor, outputHcScale: Tensor,
        layers: [DeepSeekV4Layer]
    ) {
        self.textConfig = textConfig
        self.tokenEmbd = tokenEmbd
        self.outputNorm = outputNorm
        self.outputHead = outputHead
        self.outputHcBase = outputHcBase
        self.outputHcFn = outputHcFn
        self.outputHcScale = outputHcScale
        self.layers = layers
    }
}

// ─── Config shim ─────────────────────────────────────────────────────

/// Returns the `text_config` sub-tree on a multimodal V4 conversion
/// (none ship today, but the slot exists upstream); otherwise the
/// top-level config (text-only checkpoint).
enum DeepSeekV4Config {
    static func textConfig(_ c: ModelConfig) -> ModelConfig {
        c.subConfig("text_config") ?? c
    }
}
