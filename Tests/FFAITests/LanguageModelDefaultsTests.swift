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
// LanguageModelDefaultsTests — verifies the default extension methods
// in `LanguageModel` compose correctly on top of the single primitive
// `forward(...on cmd:)`. The key contract being pinned:
//
//   - `forwardSample` queues forward + argmax on ONE command buffer.
//   - `forwardSampleCategorical` queues forward + softmax-categorical
//     on ONE command buffer.
//
// Before this refactor the default `forwardSampleCategorical` ran
// forward inside its own cmdbuf and then queued the sampler on a
// SECOND cmdbuf — 2 commits and 2 waits per token. Llama / Qwen3
// worked around it with hand-rolled overrides; Mamba 2 inherited the
// slow path. Now the default is fast and all three families share
// the same code path.

import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("LanguageModel default extension")
final class LanguageModelDefaultsTests {

    /// Minimal mock model. `forward(on cmd:)` is the only primitive;
    /// it records every command buffer it's invoked with so the test
    /// can prove the default impls reuse one cmdbuf instead of
    /// allocating a second.
    final class CountingModel: LanguageModel, @unchecked Sendable {
        let hidden = 4
        let nLayers = 0
        let nHeads = 1
        let nKVHeads = 1
        let headDim = 4
        let vocab = 8
        let maxSeq = 1
        let dtype: DType = .f32

        /// Tracks every cmdbuf passed into `forward(on cmd:)`.
        var observedCmdbufs: [ObjectIdentifier] = []

        func parameters() -> [(String, Tensor)] { [] }

        func makeLayerCaches(maxSeq _: Int? = nil, device _: Device) -> [any LayerCacheProtocol] {
            []
        }

        func forward(tokenId _: Int, position _: Int,
                     caches _: [any LayerCacheProtocol],
                     on cmd: MTLCommandBuffer, device: Device) -> Tensor {
            observedCmdbufs.append(ObjectIdentifier(cmd))
            // Return a tiny constant logits tensor. Argmax over [10, 1, 1, 1, …]
            // picks index 0 deterministically; softmax-categorical with
            // peaked logits will too (within any temperature/uniform).
            let t = Tensor.empty(shape: [vocab], dtype: .f32)
            var values = [Float](repeating: 1, count: vocab)
            values[0] = 100  // peak so argmax + categorical agree
            t.copyIn(from: values)
            return t
        }
    }

    @Test("forwardSampleCategorical uses exactly one command buffer")
    func categoricalIsFused() {
        let model = CountingModel()
        let token = model.forwardSampleCategorical(
            tokenId: 0, position: 0, caches: [],
            temperature: 1.0, uniformDraw: 0.5
        )
        // The peaked logits guarantee idx 0 wins.
        #expect(token == 0)
        // The whole point of this test: forward + sampler share a cmdbuf.
        #expect(model.observedCmdbufs.count == 1,
                "forward should be invoked exactly once (got \(model.observedCmdbufs.count))")
    }

    @Test("forwardSample uses exactly one command buffer")
    func sampleIsFused() {
        let model = CountingModel()
        let token = model.forwardSample(tokenId: 0, position: 0, caches: [])
        #expect(token == 0)
        #expect(model.observedCmdbufs.count == 1,
                "forward should be invoked exactly once (got \(model.observedCmdbufs.count))")
    }

    @Test("forward (default) wraps the primitive in one cmdbuf and waits")
    func forwardDefaultWaits() {
        let model = CountingModel()
        let logits = model.forward(tokenId: 0, position: 0, caches: [])
        #expect(logits.shape == [model.vocab])
        #expect(model.observedCmdbufs.count == 1)
        // Cmdbuf is committed + waited inside forward; the returned
        // tensor's contents are readable on CPU now.
        let arr = logits.toArray(as: Float.self)
        #expect(arr[0] == 100)
    }
}
