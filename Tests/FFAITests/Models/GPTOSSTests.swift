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
// GPTOSSTests — root-file unit tests for `Sources/FFAI/Models/GPTOSS.swift`.
//
// Offline. Covers GPT-OSS family metadata + variant dispatch
// (`GPTOSSMoEVariant` is the only variant) + every `GPTOSSError`
// case.

import Foundation
import Testing
@testable import FFAI

@Suite("GPTOSS Family Root")
struct GPTOSSRootTests {

    @Test("modelTypes advertises gpt_oss")
    func modelTypes() {
        #expect(GPTOSS.modelTypes.contains("gpt_oss"))
    }

    @Test("architectures advertises GptOssForCausalLM")
    func architectures() {
        #expect(GPTOSS.architectures.contains("GptOssForCausalLM"))
    }

    @Test("variant(for:) returns GPTOSSMoEVariant")
    func variantDispatch() throws {
        let cfg = ModelConfig(architecture: "GptOssForCausalLM",
                              modelType: "gpt_oss", raw: [:])
        let v = try GPTOSS.variant(for: cfg)
        #expect(String(describing: v) == String(describing: GPTOSSMoEVariant.self))
    }

    @Test("GPTOSSError stringifies every case with its payload")
    func errorDescriptions() {
        #expect(GPTOSSError.missingConfig("num_experts").description
            .contains("num_experts"))
        #expect(GPTOSSError.unsupportedConfig("bad").description.contains("bad"))
        #expect(GPTOSSError.missingConfig("x").description.contains("GPT-OSS"))
    }
}
