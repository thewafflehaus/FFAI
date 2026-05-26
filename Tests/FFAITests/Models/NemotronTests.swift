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
// NemotronTests — root-file unit tests for `Sources/FFAI/Models/Nemotron.swift`.
//
// Offline. `enum Nemotron` is the unified family root that unions the
// per-variant metadata across NemotronH (text + VL), NemotronDiffusion
// (text + VL). These tests pin the union shape so the registry can
// rely on "is this any Nemotron?" with one membership lookup.
//
// They also cover the per-variant family entry points
// (`NemotronH`, `NemotronDiffusionVL`) declared in the same file and
// the unified `NemotronHError` description shape.

import Foundation
import Testing

@testable import FFAI

@Suite("Nemotron Unified Family Root")
struct NemotronUnifiedRootTests {

    @Test("Nemotron.modelTypes is the union over every variant")
    func modelTypesUnion() {
        let union = Nemotron.modelTypes
        #expect(NemotronH.modelTypes.isSubset(of: union))
        #expect(NemotronVL.modelTypes.isSubset(of: union))
        #expect(NemotronDiffusion.modelTypes.isSubset(of: union))
        #expect(NemotronDiffusionVL.modelTypes.isSubset(of: union))
    }

    @Test("Nemotron.architectures is the union over every variant")
    func architecturesUnion() {
        let union = Nemotron.architectures
        #expect(NemotronH.architectures.isSubset(of: union))
        #expect(NemotronDiffusion.architectures.isSubset(of: union))
        #expect(NemotronDiffusionVL.architectures.isSubset(of: union))
        // NemotronVL deliberately ships an empty architectures set (the
        // VL checkpoints route via the vision-config sniff) — but it
        // SHOULD still be a subset of an empty set.
    }

    @Test("NemotronH owns nemotron_h + NemotronHForCausalLM")
    func nemotronHRegistration() {
        #expect(NemotronH.modelTypes.contains("nemotron_h"))
        #expect(NemotronH.architectures.contains("NemotronHForCausalLM"))
    }

    @Test("NemotronH.variant(for:) returns NemotronHHybrid")
    func nemotronHVariant() throws {
        let cfg = ModelConfig(
            architecture: "NemotronHForCausalLM",
            modelType: "nemotron_h", raw: [:])
        let v = try NemotronH.variant(for: cfg)
        #expect(String(describing: v) == String(describing: NemotronHHybrid.self))
    }

    @Test("NemotronHError stringifies every case with its payload")
    func nemotronHErrorDescriptions() {
        #expect(NemotronHError.missingConfig("layers").description.contains("layers"))
        #expect(NemotronHError.missingTensor("foo").description.contains("foo"))
        #expect(NemotronHError.unsupportedConfig("bad").description.contains("bad"))
        #expect(NemotronHError.missingConfig("x").description.contains("NemotronH"))
    }

    @Test("NemotronDiffusionVL owns the VLM model_type + arch")
    func nemotronDiffusionVLRegistration() {
        #expect(NemotronDiffusionVL.modelTypes.contains("nemotron_labs_diffusion_vlm"))
        #expect(NemotronDiffusionVL.architectures.contains("NemotronLabsDiffusionVLMModel"))
    }

    @Test("NemotronDiffusionVL defaults match the shipped 8B config")
    func nemotronDiffusionVLDefaults() {
        #expect(NemotronDiffusionVL.defaultSpatialMergeSize == 2)
        #expect(NemotronDiffusionVL.defaultImageTokenId == 10)
    }
}
