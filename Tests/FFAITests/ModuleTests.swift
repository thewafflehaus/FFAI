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
import Testing
@testable import FFAI

@Suite("Module")
struct ModuleTests {
    final class StubModule: Module {
        let params: [(String, Tensor)]
        init(_ params: [(String, Tensor)]) { self.params = params }
        func parameters() -> [(String, Tensor)] { params }
    }

    @Test("parameterSummary lines up name : shape dtype")
    func summaryShape() {
        let t1 = Tensor.empty(shape: [2, 3], dtype: .f32)
        let t2 = Tensor.empty(shape: [4], dtype: .f16)
        let mod = StubModule([("alpha.weight", t1), ("beta.bias", t2)])

        let summary = mod.parameterSummary()
        #expect(summary.contains("alpha.weight"))
        #expect(summary.contains("[2, 3]"))
        #expect(summary.contains("f32"))
        #expect(summary.contains("beta.bias"))
        #expect(summary.contains("[4]"))
        #expect(summary.contains("f16"))
    }

    @Test("empty parameter list summarizes to empty string")
    func emptyParams() {
        let mod = StubModule([])
        #expect(mod.parameterSummary() == "")
    }
}
