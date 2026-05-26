// Gemma4TextTests — unit coverage for `Sources/FFAI/Models/Text/Gemma4Text.swift`.
//
// Offline. Covers the three Gemma 4 variant structs (Dense / E / MoE)
// against the shared default surface from `Gemma4Variant` (capabilities
// + 4096-token prefill chunk), the `Gemma4Config.textConfig` helper
// that flattens VLM-nested `text_config` blocks, and the
// `Gemma4Params.previousKVs` KV-sharing donor-map for the
// `num_kv_shared_layers` tail re-use pattern.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma4 Text Variant Surface + Config Helpers")
struct Gemma4TextTests {

    // ─── Variant surface ─────────────────────────────────────────────

    @Test("Gemma4Dense / E / MoE all advertise text in/out capabilities")
    func variantCapabilities() {
        for caps in [
            Gemma4Dense.availableCapabilities,
            Gemma4E.availableCapabilities,
            Gemma4MoE.availableCapabilities,
        ] {
            #expect(caps.contains(.textIn))
            #expect(caps.contains(.textOut))
        }
    }

    /// All three variants share the protocol default — 4096-token
    /// prefill chunk per the Gemma 4 audit.
    @Test("default generation parameters declare a Gemma-style prefill chunk")
    func defaultGenerationParameters() {
        let dense = Gemma4Dense.defaultGenerationParameters
        let e = Gemma4E.defaultGenerationParameters
        let moe = Gemma4MoE.defaultGenerationParameters
        for p in [dense, e, moe] {
            #expect(p.maxTokens > 0)
            #expect(p.temperature >= 0)
            #expect(p.topP > 0 && p.topP <= 1.0)
            // Audited family optimum for pure-attention backbone.
            #expect(p.prefillStepSize == 4096)
        }
    }

    // ─── Gemma4Config.textConfig ─────────────────────────────────────

    @Test("Gemma4Config.textConfig flattens a nested text_config block")
    func textConfigNested() {
        let cfg = ModelConfig(
            architecture: "Gemma4ForConditionalGeneration",
            modelType: "gemma4",
            raw: ["text_config": ["hidden_size": 1152, "vocab_size": 262144]])
        let tc = Gemma4Config.textConfig(cfg)
        #expect((tc["hidden_size"] as? Int) == 1152)
        #expect((tc["vocab_size"] as? Int) == 262144)
    }

    @Test("Gemma4Config.textConfig falls back to the root when no text_config")
    func textConfigRootFallback() {
        // Plain text checkpoint — fields live at the top level.
        let cfg = ModelConfig(architecture: "Gemma4ForCausalLM",
                              modelType: "gemma4_text",
                              raw: ["hidden_size": 2048])
        let tc = Gemma4Config.textConfig(cfg)
        #expect((tc["hidden_size"] as? Int) == 2048)
    }

    // ─── Gemma4Params.previousKVs ────────────────────────────────────

    /// `numKvSharedLayers == 0` (the dense + MoE checkpoints) — every
    /// layer is its own donor.
    @Test("previousKVs is the identity map when no KV sharing")
    func previousKVsIdentity() throws {
        let raw: [String: Any] = [
            "hidden_size": 1152, "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "layer_types": ["sliding_attention", "sliding_attention",
                            "full_attention", "sliding_attention"],
        ]
        let cfg = ModelConfig(architecture: "Gemma4ForCausalLM",
                              modelType: "gemma4", raw: raw)
        let p = try Gemma4Params(cfg)
        #expect(p.numKvSharedLayers == 0)
        #expect(p.previousKVs == [0, 1, 2, 3])
        #expect(p.firstKvSharedIdx == 4)
    }

    /// Gemma4E ships `num_kv_shared_layers`. With 2 shared layers and
    /// a `[sliding, sliding, full, sliding, sliding, full]` schedule
    /// the last `full_attention` shared layer reuses layer 2's KV; the
    /// last `sliding_attention` shared layer reuses layer 3's KV.
    @Test("previousKVs routes shared-tail layers to the last donor of the same kind")
    func previousKVsShared() throws {
        let raw: [String: Any] = [
            "hidden_size": 1152, "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_kv_shared_layers": 2,
            "layer_types": [
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "full_attention",
            ],
        ]
        let cfg = ModelConfig(architecture: "Gemma4ForCausalLM",
                              modelType: "gemma4", raw: raw)
        let p = try Gemma4Params(cfg)
        #expect(p.numKvSharedLayers == 2)
        #expect(p.firstKvSharedIdx == 4)
        // Layer 4 (sliding) → last sliding donor in [0, 4) = layer 3.
        // Layer 5 (full)    → last full donor in [0, 4) = layer 2.
        #expect(p.previousKVs == [0, 1, 2, 3, 3, 2])
    }

    // ─── Gemma4Error ─────────────────────────────────────────────────

    @Test("Gemma4Error.missingTensor + unalignedNorm carry the offending value")
    func errorDescriptionsExtra() {
        #expect(Gemma4Error.missingTensor("model.embed_tokens").description
            .contains("embed_tokens"))
        #expect(Gemma4Error.unalignedNorm(960).description.contains("960"))
    }
}
