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
// Qwen2Tests — root-file unit tests for `Sources/FFAI/Models/Qwen2.swift`.
//
// Offline. `Qwen2` is a VL-only orchestrator (no `model_type` set);
// the file declares the image / video token defaults + capability set
// + the `Qwen2VLError` cases. The `vision_config` decode surface is
// already covered by `Tests/FFAITests/Models/Vision/Qwen2VisionConfigTests.swift`
// — this file focuses on the constants + error shape.

import Foundation
import Testing
@testable import FFAI

@Suite("Qwen2VL Family Root")
struct Qwen2VLRootTests {

    @Test("Qwen2VL exposes the canonical image / video placeholder ids")
    func tokenIdDefaults() {
        #expect(Qwen2VL.defaultImageTokenId == 151_655)
        #expect(Qwen2VL.defaultVideoTokenId == 151_656)
    }

    @Test("availableCapabilities advertise text + image + video in")
    func capabilities() {
        let caps = Qwen2VL.availableCapabilities
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(caps.contains(.visionIn))
        #expect(caps.contains(.videoIn))
    }

    @Test("Qwen2VLError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(Qwen2VLError.missingConfig.description.contains("Qwen2VL"))
        #expect(Qwen2VLError.missingTensor("vision_tower.x").description
            .contains("vision_tower.x"))
    }
}
