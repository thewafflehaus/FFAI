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
// PhiTextTests — unit coverage for `Sources/FFAI/Models/Text/PhiText.swift`.
//
// Offline. Covers the `Phi3Dense` variant surface (capabilities +
// defaults) and the family-level `PhiError` description text for the
// cases the text loader raises — `unsupportedRopeScaling` and
// `quantizedFusedNotSupported` are both surfaced from this file's
// `loadModel` and need stable, descriptive messages.

import Foundation
import Testing
@testable import FFAI

@Suite("Phi3Dense Variant Surface")
struct PhiTextTests {

    @Test("Phi3Dense advertises text in/out capabilities")
    func capabilities() {
        #expect(Phi3Dense.availableCapabilities.contains(.textIn))
        #expect(Phi3Dense.availableCapabilities.contains(.textOut))
        #expect(!Phi3Dense.availableCapabilities.contains(.imageIn))
    }

    @Test("Phi3Dense default generation parameters mirror Llama dense")
    func defaultGenerationParameters() {
        let p = Phi3Dense.defaultGenerationParameters
        #expect(p.maxTokens > 0)
        #expect(p.temperature >= 0)
        #expect(p.topP > 0 && p.topP <= 1.0)
        #expect((p.prefillStepSize ?? 0) >= 256)
    }

    @Test("PhiError.unsupportedRopeScaling description names the bad rope type")
    func unsupportedRopeScalingDescription() {
        let desc = PhiError.unsupportedRopeScaling("longrope").description
        #expect(desc.contains("Phi"))
        #expect(desc.contains("longrope"))
    }

    @Test("PhiError.quantizedFusedNotSupported description mentions Phi + workaround")
    func quantizedFusedNotSupportedDescription() {
        let desc = PhiError.quantizedFusedNotSupported.description
        #expect(desc.contains("Phi"))
        // Hints the user toward the raw checkpoint workaround the
        // loader's row-slice path can handle.
        #expect(desc.contains("raw") || desc.contains("split"))
    }
}
