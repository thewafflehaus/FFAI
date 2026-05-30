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
// Step 3 family root — StepFun's Step-3 line (Step-3.5-Flash, Step-3.7-Flash, …).
//
// This file is the **main model interface** for the family:
//   • the family enum `Step3` (modelTypes, architectures, variant
//     dispatch),
//   • the `Step3Variant` protocol every concrete variant conforms to,
//   • the unified `Step3Error` type every loader / decode site raises
//     (covers both the text path and the Step-3 vision-language path).
//
// Concrete variants + the hybrid full/sliding-window decoder + per-layer
// impl live under `Models/Text/Step3Text.swift`:
//   - `Step3Hybrid` — the canonical Step-3.5/3.7 backbone (288-expert
//     MoE FFN on layers 3-44, dense MLP on layers 0-2, asymmetric GQA
//     per layer-type, 12 full-attention + 33 sliding-window-512 layers).
//   - `Step3Layer`, `Step3Model` — per-layer + full-model impl.
//
// The Step-3 vision-language orchestrator (`enum Step3VL`) — which ties
// the Step-3 text backbone to the bespoke Perception-Encoder vision
// tower + 2× strided patch-downsampler + projector — lives in
// `Models/Vision/Step3Vision.swift` alongside the tower internals.
//
// **Status:** WIP. Family scaffold + config decoders + loader hook are
// in place so a `Step3.7-Flash` checkpoint can be identified end-to-end;
// the forward path (Step3Model + Step3VLVisionModel) is stubbed and
// raises `Step3Error.notYetImplemented` on load. Concrete kernels and
// per-layer code land in follow-ups.

import Foundation

// ─── Family entry point ──────────────────────────────────────────────

public enum Step3 {
    /// HuggingFace `text_config.model_type` strings this family handles.
    /// Step-3.5-Flash and Step-3.7-Flash both carry `step3p5` in their
    /// text config; the bare `step3` value covers the original
    /// `stepfun-ai/step3` (321B) checkpoint.
    public static let modelTypes: Set<String> = ["step3", "step3p5", "step3p7"]

    /// HuggingFace `architectures[0]` strings this family handles — the
    /// union of the text-only path and the vision-language path. The
    /// `Step3p5` and `Step3p7` prefixes ride alongside the unversioned
    /// `Step3` strings so checkpoint-side naming drift doesn't sink the
    /// dispatch.
    public static let architectures: Set<String> = [
        "Step3ForCausalLM", "Step3p5ForCausalLM", "Step3p7ForCausalLM",
        "Step3ForConditionalGeneration",
        "Step3p5ForConditionalGeneration", "Step3p7ForConditionalGeneration",
        "Step3VLForConditionalGeneration",
    ]

    /// Vision-language architecture strings — a subset of
    /// [`architectures`] that always require a vision tower load.
    public static let vlArchitectures: Set<String> = [
        "Step3ForConditionalGeneration",
        "Step3p5ForConditionalGeneration",
        "Step3p7ForConditionalGeneration",
        "Step3VLForConditionalGeneration",
    ]

    /// Resolve the concrete variant for a config. Only `Step3Hybrid`
    /// ships today; future variants (smaller dense ports of the Step
    /// stack, MFA variants of the original 321B model) plug in here.
    public static func variant(for config: ModelConfig) throws -> any Step3Variant.Type {
        _ = config
        return Step3Hybrid.self
    }
}

// ─── Variant protocol ────────────────────────────────────────────────

public protocol Step3Variant {
    static var availableCapabilities: Set<Capability> { get }
    /// Generation defaults for this variant. The user can override any
    /// field; absent overrides fall back to the values declared here.
    static var defaultGenerationParameters: GenerationParameters { get }
    static func loadModel(
        config: ModelConfig,
        weights: SafeTensorsBundle,
        options: LoadOptions,
        device: Device
    ) throws -> Step3Model
}

extension Step3Variant {
    public static var availableCapabilities: Set<Capability> { [.textIn, .textOut] }
    public static var defaultGenerationParameters: GenerationParameters {
        // Step-3.x-Flash: 256K context with YARN on full-attention layers,
        // 512-token sliding window on the other 3-of-4. The
        // hybrid-cache shape dominates prefill costing; 4096-step
        // chunking matches the Gemma 4 hybrid family defaults.
        GenerationParameters(
            maxTokens: 256, prefillStepSize: 4096,
            temperature: 1.0, topP: 0.95, topK: 64,
            repetitionPenalty: 1.0)
    }
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Unified Step 3 family error — raised by both the text loaders
/// (`Step3Hybrid.loadModel`) and the Step-3 vision-language orchestrator
/// (`Step3VL.load` in `Models/Vision/Step3Vision.swift`).
public enum Step3Error: Error, CustomStringConvertible {
    case missingConfig(String)
    case missingTensor(String)
    case unsupportedHeadDim(Int)
    case unsupportedLayerType(String)
    case unsupportedRouterShape(String)
    case notYetImplemented(String)

    public var description: String {
        switch self {
        case .missingConfig(let f):
            return "Step3: required config field missing: \(f)"
        case .missingTensor(let name):
            return "Step3: checkpoint is missing tensor '\(name)'"
        case .unsupportedHeadDim(let d):
            return "Step3: head_dim \(d) unsupported (Ops.sdpaDecode2Pass needs 64/96/128/256/512)"
        case .unsupportedLayerType(let t):
            return "Step3: unknown layer_type '\(t)' (expected 'full_attention' or 'sliding_attention')"
        case .unsupportedRouterShape(let why):
            return "Step3: MoE router shape unsupported: \(why)"
        case .notYetImplemented(let what):
            return "Step3: \(what) — WIP, not yet implemented"
        }
    }
}
